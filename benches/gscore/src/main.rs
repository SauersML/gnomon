#![feature(portable_simd)]
//! gscore: a standalone reference scorer for gnomon's `score` semantics on PLINK
//! .bed/.bim/.fam inputs. It exists to measure where the time goes and how close a
//! single pass can get to the physical bound, stage by stage.
//!
//! Semantics follow $SW/tools/score_oracle (gnomon at a5f27055): allele pairs match in
//! either orientation; a locus with one .bim row is simple, several rows make each
//! score line a complex application resolved per person by complex.rs's heuristic
//! chain; one denominator per distinct (row, score) plus one per application;
//! `_AVG = sum / (denominator - missing)`, `_MISSING_PCT` in f32.
//!
//! Arithmetic is exact. Every weight of score s is an integer multiple of
//! 2^exp[s] / lcm, so each person's sum is an integer. Up to six needed rows at a time
//! fold into one lookup table indexed by a person's joint calls. A table entry splits
//! into a low lane (with the missing count packed underneath) and a high lane that
//! accumulate as u64 and i64 without overflow for 2^g groups before carrying into i128
//! totals. The quotient is taken once in double-double, as the oracle takes it, so the
//! output is the oracle's value. Integer sums do not depend on partition or order, so
//! any thread count produces the same bytes.

use std::fs::File;
use std::io::Write;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::simd::prelude::*;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

fn fail(message: impl Into<String>) -> ! {
    eprintln!("gscore: {}", message.into());
    std::process::exit(2);
}

// ------------------------------------------------------------------------------ timers

struct Timer {
    start: Instant,
    last: Instant,
    laps: Vec<(&'static str, f64)>,
}

impl Timer {
    fn new() -> Self {
        let now = Instant::now();
        Timer { start: now, last: now, laps: Vec::new() }
    }

    fn lap(&mut self, name: &'static str) {
        let now = Instant::now();
        self.laps.push((name, (now - self.last).as_secs_f64() * 1e3));
        self.last = now;
    }

    fn report(&self) {
        for (name, ms) in &self.laps {
            eprintln!("TIMING\t{name}\t{ms:.3}");
        }
        eprintln!("TIMING\ttotal\t{:.3}", (self.last - self.start).as_secs_f64() * 1e3);
    }
}

// ------------------------------------------------------------------------------ files

unsafe extern "C" {
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
    fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
}

const MADV_WILLNEED: i32 = 3;

struct Map {
    ptr: *mut u8,
    len: usize,
    _file: File,
}

unsafe impl Send for Map {}
unsafe impl Sync for Map {}

impl Map {
    fn open(path: &Path) -> Map {
        let file = File::open(path).unwrap_or_else(|e| fail(format!("{}: {e}", path.display())));
        let len = file
            .metadata()
            .unwrap_or_else(|e| fail(format!("{}: {e}", path.display())))
            .len() as usize;
        if len == 0 {
            return Map { ptr: std::ptr::null_mut(), len: 0, _file: file };
        }
        // PROT_READ and MAP_PRIVATE have the same values on Linux and macOS.
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, 1, 2, file.as_raw_fd(), 0) };
        if ptr as isize == -1 {
            fail(format!("{}: mmap failed", path.display()));
        }
        Map { ptr, len, _file: file }
    }

    fn bytes(&self) -> &[u8] {
        if self.len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(self.ptr, self.len) } }
    }

    fn will_need(&self, start: usize, end: usize) {
        let start = start & !4095;
        let end = end.min(self.len);
        if end > start {
            unsafe { madvise(self.ptr.add(start), end - start, MADV_WILLNEED) };
        }
    }
}

impl Drop for Map {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe { munmap(self.ptr, self.len) };
        }
    }
}

// ------------------------------------------------------------------------------ parallel helpers

fn par_map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    if threads <= 1 || n <= 1 {
        return (0..n).map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..threads.min(n) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let value = f(i);
                *slots[i].lock().unwrap() = Some(value);
            });
        }
    });
    slots.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}

/// Splits `data` into about `parts` ranges that end on line boundaries.
fn split_lines(data: &[u8], parts: usize) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = 0;
    for i in 1..parts.max(1) {
        let mut cut = data.len() * i / parts;
        if cut <= start {
            continue;
        }
        cut = match data[cut..].iter().position(|&b| b == b'\n') {
            Some(offset) => cut + offset + 1,
            None => data.len(),
        };
        if cut > start {
            out.push(start..cut);
            start = cut;
        }
    }
    if start < data.len() || out.is_empty() {
        out.push(start..data.len());
    }
    out
}

/// Calls `f(offset, line)` for each line, split as `str::lines` splits.
#[inline]
fn for_each_line(data: &[u8], base: usize, mut f: impl FnMut(usize, &[u8])) {
    let mut start = 0;
    while start < data.len() {
        let end = match data[start..].iter().position(|&b| b == b'\n') {
            Some(i) => start + i,
            None => data.len(),
        };
        let mut line = &data[start..end];
        if let [rest @ .., b'\r'] = line {
            line = rest;
        }
        f(base + start, line);
        start = end + 1;
    }
}

fn merge_runs<T: Copy>(a: &[T], b: &[T], out: &mut [T], key: &impl Fn(&T) -> u64) {
    let (mut i, mut j, mut o) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        if key(&b[j]) < key(&a[i]) {
            out[o] = b[j];
            j += 1;
        } else {
            out[o] = a[i];
            i += 1;
        }
        o += 1;
    }
    out[o..o + a.len() - i].copy_from_slice(&a[i..]);
    o += a.len() - i;
    out[o..].copy_from_slice(&b[j..]);
}

/// Stable sort by key: sorted parts in parallel, then pairwise parallel merges.
fn par_sort_by_key<T: Copy + Send + Sync>(v: Vec<T>, threads: usize, key: impl Fn(&T) -> u64 + Sync) -> Vec<T> {
    if v.windows(2).all(|w| key(&w[0]) <= key(&w[1])) {
        return v;
    }
    let mut src = v;
    if threads <= 1 || src.len() < 1 << 15 {
        src.sort_by_key(|x| key(x));
        return src;
    }
    let mut width = src.len().div_ceil(threads.next_power_of_two());
    std::thread::scope(|s| {
        for part in src.chunks_mut(width) {
            let key = &key;
            s.spawn(move || part.sort_by_key(|x| key(x)));
        }
    });
    let mut dst = src.clone();
    while width < src.len() {
        std::thread::scope(|s| {
            for (out, pair) in dst.chunks_mut(2 * width).zip(src.chunks(2 * width)) {
                let key = &key;
                s.spawn(move || {
                    let split = width.min(pair.len());
                    merge_runs(&pair[..split], &pair[split..], out, key);
                });
            }
        });
        std::mem::swap(&mut src, &mut dst);
        width *= 2;
    }
    src
}

// ------------------------------------------------------------------------------ parsing

fn parse_chr(label: &[u8]) -> Option<u8> {
    let mut t = label.trim_ascii();
    if t.len() >= 3 && t[..3].eq_ignore_ascii_case(b"chr") {
        t = &t[3..];
    }
    if t.eq_ignore_ascii_case(b"X") {
        return Some(23);
    }
    if t.eq_ignore_ascii_case(b"Y") {
        return Some(24);
    }
    if t.eq_ignore_ascii_case(b"MT") {
        return Some(25);
    }
    std::str::from_utf8(t).ok()?.parse::<u8>().ok()
}

#[inline]
fn parse_key(chr: &[u8], pos: &[u8]) -> Option<u64> {
    let chr = parse_chr(chr)?;
    let pos: u32 = std::str::from_utf8(pos.trim_ascii()).ok()?.parse().ok()?;
    Some((u64::from(chr) << 32) | u64::from(pos))
}

#[inline]
fn pack(offset: usize, len: usize) -> u64 {
    if len >= 1 << 16 {
        fail(format!("allele of {len} bytes is longer than this prototype supports"));
    }
    ((offset as u64) << 16) | len as u64
}

#[inline]
fn span(data: &[u8], packed: u64) -> &[u8] {
    let start = (packed >> 16) as usize;
    &data[start..start + (packed & 0xffff) as usize]
}

struct Fam {
    map: Map,
    iids: Vec<(usize, usize)>,
}

fn read_fam(path: &Path, threads: usize) -> Fam {
    let map = Map::open(path);
    let iids = {
        let data = map.bytes();
        let parts = split_lines(data, threads * 4);
        par_map(parts.len(), threads, |i| {
            let range = parts[i].clone();
            let mut out = Vec::with_capacity(range.len() / 16 + 1);
            for_each_line(&data[range.clone()], range.start, |offset, line| {
                if line.is_empty() {
                    return;
                }
                let mut fields = line.split(|b| b.is_ascii_whitespace()).filter(|f| !f.is_empty());
                fields.next();
                let Some(iid) = fields.next() else {
                    fail(format!("{}: missing IID", path.display()));
                };
                out.push((offset + (iid.as_ptr() as usize - line.as_ptr() as usize), iid.len()));
            });
            out
        })
        .concat()
    };
    Fam { map, iids }
}

#[derive(Clone, Copy)]
enum Strategy {
    Original,
    Harmonized,
    Fallback,
}

struct Pgs {
    strategy: Strategy,
    chr: Option<usize>,
    pos: Option<usize>,
    hm_chr: Option<usize>,
    hm_pos: Option<usize>,
    ea: usize,
    ew: usize,
    oa: Option<usize>,
}

struct ScoreSrc {
    path: PathBuf,
    map: Map,
    body: usize,
    names: Vec<String>,
    pgs: Option<Pgs>,
}

fn normalize_build(value: &str) -> Option<u8> {
    match value.trim().to_lowercase().as_str() {
        "grch37" | "hg19" | "37" => Some(37),
        "grch38" | "hg38" | "38" => Some(38),
        _ => None,
    }
}

fn open_score(path: &Path) -> ScoreSrc {
    let map = Map::open(path);
    let (body, names, pgs) = {
        let data = map.bytes();
        let (mut orig, mut hm, mut pgs_id) = (None, None, None::<String>);
        let mut offset = 0;
        loop {
            if offset >= data.len() {
                fail(format!("{} has no header line", path.display()));
            }
            let end = data[offset..].iter().position(|&b| b == b'\n').map_or(data.len(), |i| offset + i);
            let line = std::str::from_utf8(&data[offset..end])
                .unwrap_or_else(|_| fail(format!("{}: header is not UTF-8", path.display())));
            let line = line.strip_suffix('\r').unwrap_or(line);
            let next = (end + 1).min(data.len());
            if line.starts_with('#') {
                let meta = line.trim_start_matches('#').trim();
                if let Some(v) = meta.strip_prefix("genome_build=") {
                    orig = normalize_build(v);
                } else if let Some(v) = meta.strip_prefix("HmPOS_build=") {
                    hm = normalize_build(v);
                } else if let Some(v) = meta.strip_prefix("pgs_id=") {
                    pgs_id = Some(v.to_string());
                }
                offset = next;
                continue;
            }
            if line.trim().is_empty() {
                offset = next;
                continue;
            }
            let header = line.trim();
            if header.starts_with("variant_id\teffect_allele\tother_allele") {
                break (next, header.split('\t').skip(3).map(str::to_string).collect::<Vec<_>>(), None);
            }
            let columns: Vec<&str> = header.split('\t').collect();
            // gnomon builds a HashMap from the header, so a repeated name maps to its last column.
            let find = |name: &str| columns.iter().rposition(|c| *c == name);
            let strategy = match (orig, hm) {
                (Some(o), Some(h)) if o == h => Strategy::Fallback,
                (None, None) => Strategy::Fallback,
                (Some(_), None) => Strategy::Original,
                (None, Some(_)) | (Some(_), Some(_)) => Strategy::Harmonized,
            };
            let ea = find("effect_allele").unwrap_or_else(|| fail(format!("{}: no effect_allele column", path.display())));
            let ew = find("effect_weight").unwrap_or_else(|| fail(format!("{}: no effect_weight column", path.display())));
            let label = pgs_id.filter(|id| !id.eq_ignore_ascii_case("PGS_SCORE")).unwrap_or_else(|| {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("PGS_SCORE").to_string();
                match path.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()) {
                    Some(parent) if parent != "." && parent != "/" => format!("{parent}_{stem}"),
                    _ => stem,
                }
            });
            let pgs = Pgs {
                strategy,
                chr: find("chr_name"),
                pos: find("chr_position"),
                hm_chr: find("hm_chr"),
                hm_pos: find("hm_pos"),
                ea,
                ew,
                oa: find("other_allele").or_else(|| find("hm_inferOtherAllele")),
            };
            break (next, vec![label], Some(pgs));
        }
    };
    ScoreSrc { path: path.to_path_buf(), map, body, names, pgs }
}

#[derive(Clone, Copy)]
struct SRow {
    key: u64,
    ea: u64,
    oa: u64,
}

struct SPart {
    rows: Vec<SRow>,
    weights: Vec<f64>,
    skipped: usize,
}

fn parse_weight(text: &[u8], src: &ScoreSrc) -> f64 {
    let text = std::str::from_utf8(text).unwrap_or_else(|_| fail(format!("{}: weight is not UTF-8", src.path.display())));
    let weight: f64 = text.parse().unwrap_or_else(|_| fail(format!("invalid weight '{text}' in {}", src.path.display())));
    if !weight.is_finite() {
        fail(format!("non-finite weight '{text}' in {}", src.path.display()));
    }
    weight
}

fn parse_score_part(src: &ScoreSrc, range: Range<usize>) -> SPart {
    let data = src.map.bytes();
    let k = src.names.len();
    let estimate = range.len() / 40 + 1;
    let mut part = SPart { rows: Vec::with_capacity(estimate), weights: Vec::with_capacity(estimate * k), skipped: 0 };
    for_each_line(&data[range.clone()], range.start, |offset, line| {
        let at = |field: &[u8]| offset + (field.as_ptr() as usize - line.as_ptr() as usize);
        match &src.pgs {
            None => {
                if line.trim_ascii().is_empty() || line[0] == b'#' {
                    return;
                }
                let mut fields = line.split(|&b| b == b'\t');
                let (Some(id), Some(effect), Some(other)) = (fields.next(), fields.next(), fields.next()) else {
                    part.skipped += 1;
                    return;
                };
                if id.is_empty() || effect.is_empty() || other.is_empty() {
                    part.skipped += 1;
                    return;
                }
                if other == b"N" {
                    fail(format!("{}: unknown other_allele 'N'", src.path.display()));
                }
                let key = match id.iter().position(|&b| b == b':') {
                    Some(c) => parse_key(&id[..c], &id[c + 1..]),
                    None => None,
                };
                let Some(key) = key else {
                    part.skipped += 1;
                    return;
                };
                let base = part.weights.len();
                part.weights.resize(base + k, f64::NAN);
                let mut any = false;
                for (i, text) in fields.enumerate() {
                    let text = text.trim_ascii();
                    if text.is_empty() || i >= k {
                        continue;
                    }
                    part.weights[base + i] = parse_weight(text, src);
                    any = true;
                }
                if !any {
                    part.weights.truncate(base);
                    part.skipped += 1;
                    return;
                }
                part.rows.push(SRow { key, ea: pack(at(effect), effect.len()), oa: pack(at(other), other.len()) });
            }
            Some(pgs) => {
                if line.is_empty() || line[0] == b'#' {
                    return;
                }
                let mut fields: [&[u8]; 64] = [&[]; 64];
                let mut count = 0;
                for field in line.split(|&b| b == b'\t') {
                    if count < 64 {
                        fields[count] = field;
                    }
                    count += 1;
                }
                let get = |i: Option<usize>| {
                    i.filter(|&i| i < count.min(64)).map(|i| fields[i].trim_ascii()).filter(|f| !f.is_empty())
                };
                let attempt = |c: Option<usize>, p: Option<usize>| match (get(c), get(p)) {
                    (Some(c), Some(p)) => parse_key(c, p),
                    _ => None,
                };
                let key = match pgs.strategy {
                    Strategy::Fallback => attempt(pgs.hm_chr, pgs.hm_pos).or_else(|| attempt(pgs.chr, pgs.pos)),
                    Strategy::Harmonized => attempt(pgs.hm_chr, pgs.hm_pos),
                    Strategy::Original => attempt(pgs.chr, pgs.pos),
                };
                let (Some(key), Some(effect), Some(weight), Some(other)) = (key, get(Some(pgs.ea)), get(Some(pgs.ew)), get(pgs.oa)) else {
                    part.skipped += 1;
                    return;
                };
                if other == b"N" {
                    fail(format!("{}: unknown other_allele 'N'", src.path.display()));
                }
                part.weights.push(parse_weight(weight, src));
                part.rows.push(SRow { key, ea: pack(at(effect), effect.len()), oa: pack(at(other), other.len()) });
            }
        }
    });
    part
}

#[derive(Clone, Copy)]
struct BRow {
    key: u64,
    row: u64,
    a1: u64,
    a2: u64,
}

#[inline]
fn contains(keys: &[u64], key: u64, hint: &mut usize) -> bool {
    let mut i = *hint;
    let mut steps = 0;
    while i < keys.len() && keys[i] < key && steps < 16 {
        i += 1;
        steps += 1;
    }
    if (i == keys.len() || keys[i] >= key) && (i == 0 || keys[i - 1] < key) {
        *hint = i;
        return i < keys.len() && keys[i] == key;
    }
    let r = keys.partition_point(|&k| k < key);
    *hint = r;
    r < keys.len() && keys[r] == key
}

fn parse_bim_part(data: &[u8], range: Range<usize>, keys: &[u64]) -> (Vec<BRow>, u64) {
    let mut out = Vec::with_capacity(range.len() / 64 + 1);
    let mut lines = 0u64;
    let mut hint = 0usize;
    for_each_line(&data[range.clone()], range.start, |offset, line| {
        let row = lines;
        lines += 1;
        let mut fields = line.split(|b| b.is_ascii_whitespace()).filter(|f| !f.is_empty());
        let chr = fields.next();
        fields.next();
        fields.next();
        let (Some(chr), Some(pos), Some(a1), Some(a2)) = (chr, fields.next(), fields.next(), fields.next()) else {
            return;
        };
        let Some(key) = parse_key(chr, pos) else {
            return;
        };
        if !contains(keys, key, &mut hint) {
            return;
        }
        let at = |field: &[u8]| offset + (field.as_ptr() as usize - line.as_ptr() as usize);
        out.push(BRow { key, row, a1: pack(at(a1), a1.len()), a2: pack(at(a2), a2.len()) });
    });
    (out, lines)
}

// ------------------------------------------------------------------------------ plan

struct FileData {
    k: usize,
    columns: Vec<u32>,
    rows: Vec<SRow>,
    weights: Vec<f64>,
}

#[derive(Clone, Copy)]
struct GRow {
    key: u64,
    ea: u64,
    oa: u64,
    file: u32,
    index: u32,
}

#[derive(Clone, Copy)]
struct Term {
    /// row << 16 | score
    order: u64,
    flipped: bool,
    weight: f64,
}

#[derive(Clone)]
struct App {
    score: u32,
    weight: f64,
    effect: Vec<u8>,
    other: Vec<u8>,
    contexts: Vec<(u64, Vec<u8>, Vec<u8>)>,
}

#[inline]
fn pair_matches(effect: &[u8], other: &[u8], a1: &[u8], a2: &[u8]) -> bool {
    (effect == a1 && other == a2) || (effect == a2 && other == a1)
}

fn join_part(rows: &[GRow], files: &[FileData], srcs: &[ScoreSrc], bims: &[BRow], bim: &[u8]) -> (Vec<Term>, Vec<App>) {
    let mut terms = Vec::with_capacity(rows.len());
    let mut apps = Vec::new();
    if rows.is_empty() {
        return (terms, apps);
    }
    let mut b = bims.partition_point(|r| r.key < rows[0].key);
    let mut i = 0;
    while i < rows.len() {
        let key = rows[i].key;
        let mut j = i + 1;
        while j < rows.len() && rows[j].key == key {
            j += 1;
        }
        while b < bims.len() && bims[b].key < key {
            b += 1;
        }
        let mut e = b;
        while e < bims.len() && bims[e].key == key {
            e += 1;
        }
        let group = &bims[b..e];
        for line in &rows[i..j] {
            if group.is_empty() {
                break;
            }
            let fd = &files[line.file as usize];
            let text = srcs[line.file as usize].map.bytes();
            let (effect, other) = (span(text, line.ea), span(text, line.oa));
            let weights = &fd.weights[line.index as usize * fd.k..(line.index as usize + 1) * fd.k];
            if group.len() == 1 {
                let record = &group[0];
                let (a1, a2) = (span(bim, record.a1), span(bim, record.a2));
                if !pair_matches(effect, other, a1, a2) {
                    continue;
                }
                let flipped = effect == a1;
                for (column, &weight) in weights.iter().enumerate() {
                    if !weight.is_nan() {
                        terms.push(Term { order: (record.row << 16) | u64::from(fd.columns[column]), flipped, weight });
                    }
                }
            } else {
                let contexts: Vec<(u64, Vec<u8>, Vec<u8>)> = group
                    .iter()
                    .filter(|r| pair_matches(effect, other, span(bim, r.a1), span(bim, r.a2)))
                    .map(|r| (r.row, span(bim, r.a1).to_vec(), span(bim, r.a2).to_vec()))
                    .collect();
                if contexts.is_empty() {
                    continue;
                }
                for (column, &weight) in weights.iter().enumerate() {
                    if !weight.is_nan() {
                        apps.push(App {
                            score: fd.columns[column],
                            weight,
                            effect: effect.to_vec(),
                            other: other.to_vec(),
                            contexts: contexts.clone(),
                        });
                    }
                }
            }
        }
        i = j;
    }
    (terms, apps)
}

struct Plan {
    names: Vec<String>,
    denominators: Vec<u32>,
    exp: Vec<i32>,
    lcm: i64,
    rows: Vec<u64>,
    row_off: Vec<u32>,
    e_score: Vec<u32>,
    e_alpha: Vec<i128>,
    e_beta: Vec<i128>,
    apps: Vec<App>,
}

/// A finite f64 as (odd signed mantissa, exponent); zero has no parts.
fn f64_parts(value: f64) -> Option<(i64, i32)> {
    let bits = value.to_bits();
    let field = ((bits >> 52) & 0x7ff) as i32;
    let fraction = (bits & ((1u64 << 52) - 1)) as i64;
    let (mantissa, exp) = if field == 0 { (fraction, -1074) } else { (fraction | (1i64 << 52), field - 1075) };
    if mantissa == 0 {
        return None;
    }
    let tz = mantissa.trailing_zeros() as i32;
    let odd = mantissa >> tz;
    Some((if bits >> 63 == 1 { -odd } else { odd }, exp + tz))
}

#[inline]
fn to_int(weight: f64, exp: i32) -> i128 {
    f64_parts(weight).map_or(0, |(m, e)| (m as i128) << (e - exp))
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn compile_plan(names: Vec<String>, terms: Vec<Term>, apps: Vec<App>) -> Plan {
    let k = names.len();
    let max_contexts = apps.iter().map(|a| a.contexts.len() as i64).max().unwrap_or(1).max(1);
    let lcm = (1..=max_contexts).fold(1i64, |acc, x| acc / gcd(acc, x) * x);
    let mut min_exp = vec![i32::MAX; k];
    let mut max_top = vec![i32::MIN; k];
    let mut count = vec![0u64; k];
    let mut note = |score: usize, weight: f64| {
        count[score] += 1;
        if let Some((m, e)) = f64_parts(weight) {
            min_exp[score] = min_exp[score].min(e);
            max_top[score] = max_top[score].max(e + (64 - m.unsigned_abs().leading_zeros()) as i32);
        }
    };
    for term in &terms {
        note((term.order & 0xffff) as usize, term.weight);
    }
    for app in &apps {
        note(app.score as usize, app.weight);
    }
    let exp: Vec<i32> = (0..k)
        .map(|s| {
            if min_exp[s] == i32::MAX {
                return 0;
            }
            let count_bits = 64 - (2 * lcm as u64 * count[s].max(1)).leading_zeros() as i32;
            if (max_top[s] - min_exp[s]) + count_bits >= 125 || min_exp[s] < -1022 {
                fail(format!("score '{}' weight range exceeds i128 fixed point", names[s]));
            }
            min_exp[s]
        })
        .collect();
    let mut plan = Plan {
        names,
        denominators: vec![0; k],
        exp,
        lcm,
        rows: Vec::new(),
        row_off: Vec::new(),
        e_score: Vec::with_capacity(terms.len()),
        e_alpha: Vec::with_capacity(terms.len()),
        e_beta: Vec::with_capacity(terms.len()),
        apps: Vec::new(),
    };
    let mut t = 0;
    while t < terms.len() {
        let order = terms[t].order;
        let (row, score) = (order >> 16, (order & 0xffff) as u32);
        let (mut alpha, mut beta) = (0i128, 0i128);
        while t < terms.len() && terms[t].order == order {
            let m = to_int(terms[t].weight, plan.exp[score as usize]) * lcm as i128;
            if terms[t].flipped {
                alpha -= m;
                beta += 2 * m;
            } else {
                alpha += m;
            }
            t += 1;
        }
        if plan.rows.last() != Some(&row) {
            plan.rows.push(row);
            plan.row_off.push(plan.e_score.len() as u32);
        }
        plan.e_score.push(score);
        plan.e_alpha.push(alpha);
        plan.e_beta.push(beta);
        plan.denominators[score as usize] += 1;
    }
    plan.row_off.push(plan.e_score.len() as u32);
    for app in &apps {
        plan.denominators[app.score as usize] += 1;
    }
    plan.apps = apps;
    plan
}

// ------------------------------------------------------------------------------ plan cache

fn hash_block(data: &[u8], seed: u64) -> u64 {
    const P: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut lanes = [
        seed ^ 0x243f_6a88_85a3_08d3,
        seed ^ 0x1319_8a2e_0370_7344,
        seed ^ 0xa409_3822_299f_31d0,
        seed ^ 0x082e_fa98_ec4e_6c89,
    ];
    let mut chunks = data.chunks_exact(32);
    for chunk in &mut chunks {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let w = u64::from_le_bytes(chunk[8 * i..8 * i + 8].try_into().unwrap());
            *lane = (*lane ^ w).wrapping_mul(P).rotate_left(29);
        }
    }
    let mut tail = [0u8; 32];
    tail[..chunks.remainder().len()].copy_from_slice(chunks.remainder());
    for (i, lane) in lanes.iter_mut().enumerate() {
        let w = u64::from_le_bytes(tail[8 * i..8 * i + 8].try_into().unwrap());
        *lane = (*lane ^ w).wrapping_mul(P).rotate_left(29);
    }
    let mut h = data.len() as u64;
    for lane in lanes {
        h = (h ^ lane).wrapping_mul(P).rotate_left(31);
    }
    h ^ (h >> 29)
}

fn hash_bytes(data: &[u8], threads: usize) -> u64 {
    const BLOCK: usize = 4 << 20;
    let blocks = data.len().div_ceil(BLOCK).max(1);
    par_map(blocks, threads, |i| hash_block(&data[(i * BLOCK).min(data.len())..((i + 1) * BLOCK).min(data.len())], i as u64))
        .into_iter()
        .fold(data.len() as u64, |h, x| (h ^ x).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(31))
}

fn put<T: Copy>(out: &mut Vec<u8>, values: &[T]) {
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) };
    out.extend_from_slice(bytes);
}

fn get<T: Copy>(data: &[u8], at: &mut usize) -> Vec<T> {
    let n = u64::from_le_bytes(data[*at..*at + 8].try_into().unwrap()) as usize;
    *at += 8;
    let bytes = n * std::mem::size_of::<T>();
    let mut values = Vec::<T>::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(data[*at..*at + bytes].as_ptr(), values.as_mut_ptr() as *mut u8, bytes);
        values.set_len(n);
    }
    *at += bytes;
    values
}

const CACHE_MAGIC: &[u8; 8] = b"GSPLAN01";

fn encode_plan(plan: &Plan, key: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + plan.rows.len() * 12 + plan.e_score.len() * 36);
    out.extend_from_slice(CACHE_MAGIC);
    out.extend_from_slice(&key.to_le_bytes());
    put(&mut out, plan.names.join("\n").as_bytes());
    put(&mut out, &plan.denominators);
    put(&mut out, &plan.exp);
    put(&mut out, &[plan.lcm]);
    put(&mut out, &plan.rows);
    put(&mut out, &plan.row_off);
    put(&mut out, &plan.e_score);
    put(&mut out, &plan.e_alpha);
    put(&mut out, &plan.e_beta);
    put(&mut out, &[plan.apps.len() as u64]);
    for app in &plan.apps {
        put(&mut out, &[app.score]);
        put(&mut out, &[app.weight]);
        put(&mut out, &app.effect);
        put(&mut out, &app.other);
        put(&mut out, &[app.contexts.len() as u64]);
        for (row, a1, a2) in &app.contexts {
            put(&mut out, &[*row]);
            put(&mut out, a1);
            put(&mut out, a2);
        }
    }
    out
}

fn decode_plan(data: &[u8], key: u64) -> Option<Plan> {
    if data.len() < 16 || &data[..8] != CACHE_MAGIC || u64::from_le_bytes(data[8..16].try_into().unwrap()) != key {
        return None;
    }
    let mut at = 16;
    let names: Vec<u8> = get(data, &mut at);
    let names = String::from_utf8(names).ok()?.split('\n').map(str::to_string).collect();
    let denominators = get(data, &mut at);
    let exp = get(data, &mut at);
    let lcm = get::<i64>(data, &mut at)[0];
    let rows = get(data, &mut at);
    let row_off = get(data, &mut at);
    let e_score = get(data, &mut at);
    let e_alpha = get(data, &mut at);
    let e_beta = get(data, &mut at);
    let napps = get::<u64>(data, &mut at)[0] as usize;
    let mut apps = Vec::with_capacity(napps);
    for _ in 0..napps {
        let score = get::<u32>(data, &mut at)[0];
        let weight = get::<f64>(data, &mut at)[0];
        let effect = get(data, &mut at);
        let other = get(data, &mut at);
        let n = get::<u64>(data, &mut at)[0] as usize;
        let mut contexts = Vec::with_capacity(n);
        for _ in 0..n {
            let row = get::<u64>(data, &mut at)[0];
            contexts.push((row, get(data, &mut at), get(data, &mut at)));
        }
        apps.push(App { score, weight, effect, other, contexts });
    }
    Some(Plan { names, denominators, exp, lcm, rows, row_off, e_score, e_alpha, e_beta, apps })
}

// ------------------------------------------------------------------------------ kernel

#[inline(always)]
fn transpose(bytes: [u8; 4]) -> [u8; 4] {
    let mut word = u32::from_le_bytes(bytes);
    let swap = (word ^ (word >> 6)) & 0x00cc_00cc;
    word ^= swap ^ (swap << 6);
    let swap = (word ^ (word >> 12)) & 0x0000_f0f0;
    word ^= swap ^ (swap << 12);
    word.to_le_bytes()
}

/// keys[p] = calls of people p in rows 0..4, two bits per row, row v at bits 2v.
fn keys4(src: [&[u8]; 4], n: usize, keys: &mut [u8]) {
    assert!(cfg!(target_endian = "little"));
    let full = n / 4;
    let (s0, s1, s2, s3) = (&src[0][..full], &src[1][..full], &src[2][..full], &src[3][..full]);
    let keys = &mut keys[..full.div_ceil(8) * 32 + 4];
    let lanes = full / 8 * 8;
    let mut j = 0;
    while j < lanes {
        let b0 = Simd::<u8, 8>::from_slice(&s0[j..j + 8]).cast::<u32>();
        let b1 = Simd::<u8, 8>::from_slice(&s1[j..j + 8]).cast::<u32>();
        let b2 = Simd::<u8, 8>::from_slice(&s2[j..j + 8]).cast::<u32>();
        let b3 = Simd::<u8, 8>::from_slice(&s3[j..j + 8]).cast::<u32>();
        let mut w = b0 | (b1 << Simd::splat(8)) | (b2 << Simd::splat(16)) | (b3 << Simd::splat(24));
        let swap = (w ^ (w >> Simd::splat(6))) & Simd::splat(0x00cc_00cc);
        w ^= swap ^ (swap << Simd::splat(6));
        let swap = (w ^ (w >> Simd::splat(12))) & Simd::splat(0x0000_f0f0);
        w ^= swap ^ (swap << Simd::splat(12));
        let out: [u32; 8] = w.to_array();
        keys[4 * j..4 * j + 32].copy_from_slice(unsafe { &*(out.as_ptr() as *const [u8; 32]) });
        j += 8;
    }
    while j < full {
        keys[4 * j..4 * j + 4].copy_from_slice(&transpose([s0[j], s1[j], s2[j], s3[j]]));
        j += 1;
    }
    for p in full * 4..n {
        let (b, shift) = (p / 4, 2 * (p % 4));
        keys[p] = ((src[0][b] >> shift) & 3)
            | (((src[1][b] >> shift) & 3) << 2)
            | (((src[2][b] >> shift) & 3) << 4)
            | (((src[3][b] >> shift) & 3) << 6);
    }
}

fn build_lut(tables: &[[i128; 4]], part: &[bool], e: &mut [i128], m: &mut [u8]) {
    for c in 0..4 {
        e[c] = tables[0][c];
        m[c] = u8::from(c == 1 && part[0]);
    }
    for v in 1..tables.len() {
        let pl = 1usize << (2 * v);
        for c in (1..4).rev() {
            let (add, miss) = (tables[v][c], u8::from(c == 1 && part[v]));
            let (prefix, rest) = e.split_at_mut(c * pl);
            for (d, s) in rest[..pl].iter_mut().zip(&prefix[..pl]) {
                *d = *s + add;
            }
            let (prefix, rest) = m.split_at_mut(c * pl);
            for (d, s) in rest[..pl].iter_mut().zip(&prefix[..pl]) {
                *d = *s + miss;
            }
        }
        let add = tables[v][0];
        for d in &mut e[..pl] {
            *d += add;
        }
    }
}

fn split_lut(e: &[i128], m: &[u8], l: u32, mb: u32, lh: &mut [u64]) {
    let mask = (1i128 << l) - 1;
    for ((d, &x), &miss) in lh.chunks_exact_mut(2).zip(e).zip(m) {
        d[0] = (((x & mask) as u64) << mb) | u64::from(miss);
        d[1] = (x >> l) as i64 as u64;
    }
}

#[inline(never)]
fn accumulate_u8(keys: &[u8], lh: &[u64; 512], acc: &mut [u64]) {
    for (a, &key) in acc.chunks_exact_mut(2).zip(keys) {
        let i = key as usize;
        a[0] = a[0].wrapping_add(lh[2 * i]);
        a[1] = a[1].wrapping_add(lh[2 * i + 1]);
    }
}

/// The same additions with both lanes loaded, added and stored as one 128-bit vector.
#[inline(never)]
fn accumulate_u8_pair(keys: &[u8], lh: &[u64; 512], acc: &mut [u64]) {
    let (table, _) = lh.as_chunks::<2>();
    let table: &[[u64; 2]; 256] = table.try_into().unwrap();
    let (pairs, _) = acc.as_chunks_mut::<2>();
    for (a, &key) in pairs.iter_mut().zip(keys) {
        *a = (Simd::from_array(*a) + Simd::<u64, 2>::from_array(table[key as usize])).to_array();
    }
}

/// Transposition and accumulation in one pass over the packed bytes: 32 people per step.
#[inline(never)]
fn fused_u8_pair(src: [&[u8]; 4], n: usize, lh: &[u64; 512], acc: &mut [u64]) {
    let (table, _) = lh.as_chunks::<2>();
    let table: &[[u64; 2]; 256] = table.try_into().unwrap();
    let (pairs, _) = acc.as_chunks_mut::<2>();
    let full = n / 4;
    let (s0, s1, s2, s3) = (&src[0][..full], &src[1][..full], &src[2][..full], &src[3][..full]);
    let lanes = full / 8 * 8;
    let mut j = 0;
    while j < lanes {
        let b0 = Simd::<u8, 8>::from_slice(&s0[j..j + 8]).cast::<u32>();
        let b1 = Simd::<u8, 8>::from_slice(&s1[j..j + 8]).cast::<u32>();
        let b2 = Simd::<u8, 8>::from_slice(&s2[j..j + 8]).cast::<u32>();
        let b3 = Simd::<u8, 8>::from_slice(&s3[j..j + 8]).cast::<u32>();
        let mut w = b0 | (b1 << Simd::splat(8)) | (b2 << Simd::splat(16)) | (b3 << Simd::splat(24));
        let swap = (w ^ (w >> Simd::splat(6))) & Simd::splat(0x00cc_00cc);
        w ^= swap ^ (swap << Simd::splat(6));
        let swap = (w ^ (w >> Simd::splat(12))) & Simd::splat(0x0000_f0f0);
        w ^= swap ^ (swap << Simd::splat(12));
        let keys: [u32; 8] = w.to_array();
        let people = &mut pairs[4 * j..4 * j + 32];
        for (lane, &word) in keys.iter().enumerate() {
            for (q, &key) in word.to_le_bytes().iter().enumerate() {
                let a = &mut people[4 * lane + q];
                *a = (Simd::from_array(*a) + Simd::<u64, 2>::from_array(table[key as usize])).to_array();
            }
        }
        j += 8;
    }
    for p in 4 * lanes..n {
        let (b, shift) = (p / 4, 2 * (p % 4));
        let key = ((src[0][b] >> shift) & 3)
            | (((src[1][b] >> shift) & 3) << 2)
            | (((src[2][b] >> shift) & 3) << 4)
            | (((src[3][b] >> shift) & 3) << 6);
        pairs[p] = (Simd::from_array(pairs[p]) + Simd::<u64, 2>::from_array(table[key as usize])).to_array();
    }
}

#[inline(never)]
fn accumulate_u16(keys: &[u16], lh: &[u64], acc: &mut [u64]) {
    let lh: &[u64; 8192] = lh[..8192].try_into().unwrap();
    for (a, &key) in acc.chunks_exact_mut(2).zip(keys) {
        let i = key as usize & 0xfff;
        a[0] = a[0].wrapping_add(lh[2 * i]);
        a[1] = a[1].wrapping_add(lh[2 * i + 1]);
    }
}

/// All of a person's 2K lanes load from one table row and add as one vector.
#[inline(never)]
fn accumulate_k8(keys: &[u8], lut: &[u64], lanes: usize, acc: &mut [u64]) {
    let lut = &lut[..256 * lanes];
    for (a, &key) in acc.chunks_exact_mut(lanes).zip(keys) {
        let row = &lut[key as usize * lanes..(key as usize + 1) * lanes];
        for (x, &y) in a.iter_mut().zip(row) {
            *x = x.wrapping_add(y);
        }
    }
}

#[inline(never)]
fn accumulate_k16(keys: &[u16], lut: &[u64], lanes: usize, acc: &mut [u64]) {
    let lut = &lut[..4096 * lanes];
    for (a, &key) in acc.chunks_exact_mut(lanes).zip(keys) {
        let key = key as usize & 0xfff;
        let row = &lut[key * lanes..(key + 1) * lanes];
        for (x, &y) in a.iter_mut().zip(row) {
            *x = x.wrapping_add(y);
        }
    }
}

/// Writes score s's two lanes of every table row.
fn split_into(e: &[i128], m: &[u8], l: u32, mb: u32, lut: &mut [u64], lanes: usize, s: usize) {
    let mask = (1i128 << l) - 1;
    for (key, (&x, &miss)) in e.iter().zip(m).enumerate() {
        lut[key * lanes + 2 * s] = (((x & mask) as u64) << mb) | u64::from(miss);
        lut[key * lanes + 2 * s + 1] = (x >> l) as i64 as u64;
    }
}

fn flush(acc: &mut [u64], tot: &mut [i128], miss: &mut [u32], l: u32, mb: u32) {
    let mask = (1u64 << mb) - 1;
    for ((a, t), m) in acc.chunks_exact_mut(2).zip(tot.iter_mut()).zip(miss.iter_mut()) {
        *t += ((a[1] as i64 as i128) << l) + i128::from(a[0] >> mb);
        *m += (a[0] & mask) as u32;
        a[0] = 0;
        a[1] = 0;
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kernel {
    Lut,
    Scalar,
    Touch,
    Catalog,
}

struct KernelOptions {
    kernel: Kernel,
    gv: usize,
    advise: bool,
    threads: usize,
    /// 0 = two scalar lanes, 1 = one 128-bit pair, 2 = pair fused with transposition.
    acc: u8,
    /// Read needed rows with positional reads into per-thread buffers instead of the mapping.
    pread: bool,
    partition: Partition,
    memory_budget: u64,
}

#[derive(Clone, Copy, PartialEq)]
enum Partition {
    Auto,
    Rows,
    People,
}

/// Returns (g, l, mb) for the packed-lane accumulator, or None when a group can overflow.
fn lane_layout(plan: &Plan, gv: usize) -> Option<(u32, u32, u32)> {
    let mut bound = 0u128;
    for (&a, &b) in plan.e_alpha.iter().zip(&plan.e_beta) {
        bound = bound.max(2 * a.unsigned_abs() + b.unsigned_abs());
    }
    let bits = 128 - (bound * gv as u128).leading_zeros();
    if bits > 123 {
        return None;
    }
    let g = ((123 - bits) / 3).min(8);
    Some((g, 60 - 2 * g, g + 4))
}

struct Totals {
    tot: Vec<i128>,
    miss: Vec<u32>,
    touched: u64,
}

fn simple_kernel(plan: &Plan, bed: &Map, bed_path: &Path, n: usize, options: &KernelOptions) -> Totals {
    let k = plan.names.len();
    let lanes = 2 * k;
    let bpr = n.div_ceil(4);
    let rows = &plan.rows;
    let gv = if options.kernel == Kernel::Lut { options.gv } else { 4 };
    let (g, l, mb) = match options.kernel {
        Kernel::Lut => lane_layout(plan, gv).unwrap_or_else(|| fail("weights too wide for packed lanes; use --kernel scalar")),
        _ => (0, 0, 0),
    };
    let threads = options.threads.max(1);
    // Row partitioning gives every thread whole rows but its own copy of the accumulators;
    // person partitioning shares one copy and gives each thread a slice of every row.
    let copies = (threads * n * k * 36) as u64;
    let people_mode = !options.pread
        && threads > 1
        && match options.partition {
            Partition::Rows => false,
            Partition::People => true,
            Partition::Auto => copies > options.memory_budget.min(256 << 20),
        };
    let target = (((if options.pread { 16usize } else { 64 }) << 20) / bpr.max(1)).max(gv);
    let file = options.pread.then(|| File::open(bed_path).unwrap_or_else(|e| fail(format!("{}: {e}", bed_path.display()))));
    let data = bed.bytes();

    // Scores people [p0, p1) over the rows of every chunk it pulls from `next`;
    // tot and miss are exactly those people's cells, person-major.
    let work = |p0: usize, p1: usize, chunk_rows: usize, next: &AtomicUsize, tot: &mut [i128], miss: &mut [u32]| -> u64 {
        let chunks = rows.len().div_ceil(chunk_rows);
        let np = p1 - p0;
        let (b0, b1) = (p0 / 4, p1.div_ceil(4));
        let mut acc = vec![0u64; if options.kernel == Kernel::Lut { np * lanes } else { 0 }];
        let mut touched = 0u64;
        let mut keys_a = vec![0u8; (b1 - b0) * 4 + 36];
        let mut keys_b = vec![0u8; if gv > 4 { (b1 - b0) * 4 + 36 } else { 0 }];
        let mut keys16 = vec![0u16; if gv > 4 { np } else { 0 }];
        let size = 1usize << (2 * gv);
        let (mut lut_e, mut lut_m) = (vec![0i128; size], vec![0u8; size]);
        let mut lut = vec![0u64; (size * lanes).max(8192)];
        let mut dirty = vec![false; k];
        let zero = vec![0u8; bpr];
        let mut buffer: Vec<u8> = Vec::new();
        let mut row_at: Vec<usize> = Vec::new();
        let mut groups = 0usize;
        let mut active: Vec<u32> = Vec::new();
        loop {
            let c = next.fetch_add(1, Ordering::Relaxed);
            if c >= chunks {
                break;
            }
            let (a, b) = (c * chunk_rows, ((c + 1) * chunk_rows).min(rows.len()));
            if options.advise {
                bed.will_need(3 + rows[a] as usize * bpr, 3 + (rows[b - 1] as usize + 1) * bpr);
            }
            if let Some(file) = &file {
                // Coalesce rows separated by at most 64 KiB of unneeded bytes into one read.
                let gap_rows = ((64usize << 10) / bpr.max(1)).max(1) as u64;
                row_at.clear();
                let mut runs: Vec<(u64, u64, usize)> = Vec::new();
                let mut filled = 0usize;
                let mut r = a;
                while r < b {
                    let first = rows[r];
                    let mut last = first;
                    let mut e = r + 1;
                    while e < b && rows[e] - last <= gap_rows {
                        last = rows[e];
                        e += 1;
                    }
                    runs.push((first, last, filled));
                    for q in r..e {
                        row_at.push(filled + (rows[q] - first) as usize * bpr);
                    }
                    filled += (last - first + 1) as usize * bpr;
                    r = e;
                }
                if buffer.len() < filled {
                    buffer.resize(filled, 0);
                }
                for &(first, last, at) in &runs {
                    let len = (last - first + 1) as usize * bpr;
                    file.read_exact_at(&mut buffer[at..at + len], 3 + first * bpr as u64)
                        .unwrap_or_else(|e| fail(format!("reading .bed rows {first}..={last}: {e}")));
                }
            }
            for g0 in (a..b).step_by(gv) {
                let g1 = (g0 + gv).min(b);
                let mut src: [&[u8]; 8] = [&zero[b0..b1]; 8];
                for v in 0..g1 - g0 {
                    let row: &[u8] = if file.is_some() {
                        let at = row_at[g0 + v - a];
                        &buffer[at..at + bpr]
                    } else {
                        let offset = 3 + rows[g0 + v] as usize * bpr;
                        &data[offset..offset + bpr]
                    };
                    src[v] = &row[b0..b1];
                }
                match options.kernel {
                    Kernel::Touch => {
                        for row in &src[..g1 - g0] {
                            touched = row.iter().fold(touched, |h, &x| h.wrapping_add(u64::from(x)));
                        }
                        continue;
                    }
                    Kernel::Scalar => {
                        for v in 0..g1 - g0 {
                            for e in plan.row_off[g0 + v] as usize..plan.row_off[g0 + v + 1] as usize {
                                let s = plan.e_score[e] as usize;
                                let (alpha, beta) = (plan.e_alpha[e], plan.e_beta[e]);
                                let table = [beta, 0, alpha + beta, 2 * alpha + beta];
                                for p in 0..np {
                                    let code = (src[v][p / 4] >> (2 * (p % 4))) & 3;
                                    tot[p * k + s] += table[code as usize];
                                    miss[p * k + s] += u32::from(code == 1);
                                }
                            }
                        }
                        continue;
                    }
                    Kernel::Lut | Kernel::Catalog => {}
                }
                let fused = gv == 4 && k == 1 && options.acc == 2;
                if !fused {
                    keys4([src[0], src[1], src[2], src[3]], np, &mut keys_a);
                }
                if gv > 4 {
                    keys4([src[4], src[5], src[6], src[7]], np, &mut keys_b);
                    for ((d, &x), &y) in keys16.iter_mut().zip(&keys_a[..np]).zip(&keys_b[..np]) {
                        *d = u16::from(x) | (u16::from(y) << 8);
                    }
                }
                active.clear();
                for e in plan.row_off[g0] as usize..plan.row_off[g1] as usize {
                    if !active.contains(&plan.e_score[e]) {
                        active.push(plan.e_score[e]);
                    }
                }
                for s in 0..k {
                    let now = active.contains(&(s as u32));
                    if dirty[s] && !now {
                        for key in 0..size {
                            lut[key * lanes + 2 * s] = 0;
                            lut[key * lanes + 2 * s + 1] = 0;
                        }
                    }
                    dirty[s] = now;
                }
                for &s in &active {
                    let mut tables = [[0i128; 4]; 8];
                    let mut part = [false; 8];
                    for v in 0..gv.min(g1 - g0) {
                        for e in plan.row_off[g0 + v] as usize..plan.row_off[g0 + v + 1] as usize {
                            if plan.e_score[e] == s {
                                let (alpha, beta) = (plan.e_alpha[e], plan.e_beta[e]);
                                tables[v] = [beta, 0, alpha + beta, 2 * alpha + beta];
                                part[v] = true;
                            }
                        }
                    }
                    build_lut(&tables[..gv], &part[..gv], &mut lut_e, &mut lut_m);
                    split_into(&lut_e, &lut_m, l, mb, &mut lut, lanes, s as usize);
                }
                if k == 1 && gv == 4 {
                    let lh: &[u64; 512] = lut[..512].try_into().unwrap();
                    match options.acc {
                        0 => accumulate_u8(&keys_a[..np], lh, &mut acc),
                        1 => accumulate_u8_pair(&keys_a[..np], lh, &mut acc),
                        _ => fused_u8_pair([src[0], src[1], src[2], src[3]], np, lh, &mut acc),
                    }
                } else if k == 1 {
                    accumulate_u16(&keys16, &lut, &mut acc);
                } else if gv == 4 {
                    accumulate_k8(&keys_a[..np], &lut, lanes, &mut acc);
                } else {
                    accumulate_k16(&keys16, &lut, lanes, &mut acc);
                }
                groups += 1;
                if groups == 1 << g {
                    flush(&mut acc, tot, miss, l, mb);
                    groups = 0;
                }
            }
        }
        if options.kernel == Kernel::Lut {
            flush(&mut acc, tot, miss, l, mb);
        }
        touched
    };

    let mut total = Totals { tot: vec![0i128; n * k], miss: vec![0u32; n * k], touched: 0 };
    if people_mode {
        let per = n.div_ceil(threads).div_ceil(4) * 4;
        let touched: Vec<u64> = std::thread::scope(|s| {
            let handles: Vec<_> = total
                .tot
                .chunks_mut(per * k)
                .zip(total.miss.chunks_mut(per * k))
                .enumerate()
                .map(|(t, (tot, miss))| {
                    let work = &work;
                    s.spawn(move || {
                        let p0 = t * per;
                        work(p0, (p0 + per).min(n), target.div_ceil(gv) * gv, &AtomicUsize::new(0), tot, miss)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        total.touched = touched.into_iter().fold(0, u64::wrapping_add);
    } else {
        let chunk_rows = target.min(rows.len().div_ceil(threads * 4).max(1)).div_ceil(gv) * gv;
        let next = AtomicUsize::new(0);
        let own = |next: &AtomicUsize| {
            let (mut tot, mut miss) = (vec![0i128; n * k], vec![0u32; n * k]);
            let touched = work(0, n, chunk_rows, next, &mut tot, &mut miss);
            (tot, miss, touched)
        };
        let parts: Vec<(Vec<i128>, Vec<u32>, u64)> = if threads == 1 {
            vec![own(&next)]
        } else {
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..threads).map(|_| s.spawn(|| own(&next))).collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            })
        };
        for (tot, miss, touched) in parts {
            total.touched = total.touched.wrapping_add(touched);
            for (d, x) in total.tot.iter_mut().zip(&tot) {
                *d += x;
            }
            for (d, x) in total.miss.iter_mut().zip(&miss) {
                *d += x;
            }
        }
    }
    total
}

/// Many sparse scores: every score folds only its own rows, four at a time, so a table
/// is never built for a row the score does not use. Tasks are (score, run of its rows)
/// and run in parallel; each merges its exact totals under a lock as it finishes.
fn catalog_kernel(plan: &Plan, bed: &Map, n: usize, threads: usize) -> Totals {
    let k = plan.names.len();
    let bpr = n.div_ceil(4);
    let (g, l, mb) = lane_layout(plan, 4).unwrap_or_else(|| fail("weights too wide for packed lanes; use --kernel scalar"));
    let mut entries: Vec<Vec<u32>> = vec![Vec::new(); k];
    for r in 0..plan.rows.len() {
        for e in plan.row_off[r] as usize..plan.row_off[r + 1] as usize {
            entries[plan.e_score[e] as usize].push(e as u32);
        }
    }
    let rows_of = |e: u32| {
        let e = e as usize;
        plan.rows[plan.row_off.partition_point(|&off| off as usize <= e) - 1]
    };
    let target = (((64usize << 20) / bpr.max(1)).max(4) / 4 * 4).max(4);
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
    for (s, list) in entries.iter().enumerate() {
        let mut a = 0;
        while a < list.len() {
            let b = (a + target).min(list.len());
            tasks.push((s, a, b));
            a = b;
        }
    }
    tasks.sort_by_key(|&(s, a, b)| (std::cmp::Reverse(b - a), s, a));
    let data = bed.bytes();
    let total = Mutex::new(Totals { tot: vec![0i128; n * k], miss: vec![0u32; n * k], touched: 0 });
    let next = AtomicUsize::new(0);
    let worker = || {
        let mut acc = vec![0u64; 2 * n];
        let mut tot = vec![0i128; n];
        let mut miss = vec![0u32; n];
        let mut keys = vec![0u8; bpr * 4 + 36];
        let (mut lut_e, mut lut_m, mut lut) = (vec![0i128; 256], vec![0u8; 256], vec![0u64; 512]);
        let zero = vec![0u8; bpr];
        loop {
            let t = next.fetch_add(1, Ordering::Relaxed);
            if t >= tasks.len() {
                break;
            }
            let (s, a, b) = tasks[t];
            let list = &entries[s][a..b];
            let mut groups = 0usize;
            for group in list.chunks(4) {
                let mut src: [&[u8]; 4] = [&zero; 4];
                let mut tables = [[0i128; 4]; 4];
                let mut part = [false; 4];
                for (v, &e) in group.iter().enumerate() {
                    let offset = 3 + rows_of(e) as usize * bpr;
                    src[v] = &data[offset..offset + bpr];
                    let (alpha, beta) = (plan.e_alpha[e as usize], plan.e_beta[e as usize]);
                    tables[v] = [beta, 0, alpha + beta, 2 * alpha + beta];
                    part[v] = true;
                }
                keys4(src, n, &mut keys);
                build_lut(&tables, &part, &mut lut_e, &mut lut_m);
                split_lut(&lut_e, &lut_m, l, mb, &mut lut);
                accumulate_u8_pair(&keys[..n], lut[..512].try_into().unwrap(), &mut acc);
                groups += 1;
                if groups == 1 << g {
                    flush(&mut acc, &mut tot, &mut miss, l, mb);
                    groups = 0;
                }
            }
            flush(&mut acc, &mut tot, &mut miss, l, mb);
            let mut total = total.lock().unwrap();
            for p in 0..n {
                total.tot[p * k + s] += tot[p];
                total.miss[p * k + s] += miss[p];
            }
            drop(total);
            tot.fill(0);
            miss.fill(0);
        }
    };
    if threads <= 1 {
        worker();
    } else {
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(&worker);
            }
        });
    }
    total.into_inner().unwrap()
}

// ------------------------------------------------------------------------------ complex loci (complex.rs)

type Context = (u64, Vec<u8>, Vec<u8>);

fn calculate_score_dosage(code: u8, a1: &[u8], a2: &[u8], app: &App) -> Option<i64> {
    if !pair_matches(&app.effect, &app.other, a1, a2) {
        return None;
    }
    let dosage_a1 = match code {
        0b00 => 2,
        0b10 => 1,
        _ => 0,
    };
    if a1 == app.effect.as_slice() {
        Some(dosage_a1)
    } else if a2 == app.effect.as_slice() {
        Some(2 - dosage_a1)
    } else {
        None
    }
}

fn interpret_person_alleles<'a>(code: u8, a1: &'a [u8], a2: &'a [u8]) -> Option<(&'a [u8], &'a [u8])> {
    match code {
        0b00 => Some((a1, a1)),
        0b10 => Some((a1, a2)),
        0b11 => Some((a2, a2)),
        _ => None,
    }
}

fn resolve(app: &App, conflicts: &[(u8, &Context)]) -> Option<(i64, i64)> {
    let (effect, other) = (app.effect.as_slice(), app.other.as_slice());
    let single = |(code, c): &(u8, &Context)| calculate_score_dosage(*code, &c.1, &c.2, app).map(|d| (d, 1));
    let exact: Vec<_> = conflicts.iter().filter(|(_, c)| pair_matches(effect, other, &c.1, &c.2)).collect();
    if exact.len() == 1
        && let Some(d) = single(exact[0])
    {
        return Some(d);
    }
    let unambiguous: Vec<_> = conflicts
        .iter()
        .filter(|(_, c)| (c.1 == effect || c.1 == other) && (c.2 == effect || c.2 == other))
        .collect();
    if unambiguous.len() == 1
        && let Some(d) = single(unambiguous[0])
    {
        return Some(d);
    }
    let structure: Vec<_> = conflicts
        .iter()
        .filter(|(code, c)| match interpret_person_alleles(*code, &c.1, &c.2) {
            None => false,
            Some((p1, p2)) => {
                (p1.len() == effect.len() && p2.len() == other.len()) || (p1.len() == other.len() && p2.len() == effect.len())
            }
        })
        .collect();
    if structure.len() == 1
        && let Some(d) = single(structure[0])
    {
        return Some(d);
    }
    let dosages: Option<Vec<i64>> = conflicts.iter().map(|(code, c)| calculate_score_dosage(*code, &c.1, &c.2, app)).collect();
    if let Some(d) = &dosages
        && d.iter().all(|&x| x == d[0])
    {
        return Some((d[0], 1));
    }
    let homozygous: Vec<&[u8]> = conflicts
        .iter()
        .filter_map(|(code, c)| match *code {
            0b00 => Some(c.1.as_slice()),
            0b11 => Some(c.2.as_slice()),
            _ => None,
        })
        .collect();
    if homozygous.len() >= 2 && homozygous.iter().any(|&a| a != homozygous[0]) {
        let shortest = *homozygous.iter().min_by_key(|a| a.len()).unwrap();
        let longest = *homozygous.iter().max_by_key(|a| a.len()).unwrap();
        if longest.starts_with(shortest)
            && homozygous.iter().all(|&a| a == shortest || a == longest)
            && ((effect == shortest && other == longest) || (effect == longest && other == shortest))
        {
            return Some((i64::from(effect == shortest) + i64::from(effect == longest), 1));
        }
    }
    let het: Vec<_> = conflicts.iter().filter(|(code, _)| *code == 0b10).collect();
    if het.len() == 1
        && conflicts.iter().any(|(code, _)| *code != 0b10)
        && let Some(d) = single(het[0])
    {
        return Some(d);
    }
    let mut hom_a1: Vec<&[u8]> = conflicts.iter().filter(|(code, _)| *code == 0b00).map(|(_, c)| c.1.as_slice()).collect();
    let mut hom_a2: Vec<&[u8]> = conflicts.iter().filter(|(code, _)| *code == 0b11).map(|(_, c)| c.2.as_slice()).collect();
    hom_a1.sort_unstable();
    hom_a1.dedup();
    hom_a2.sort_unstable();
    hom_a2.dedup();
    if hom_a1.len() == 1 && hom_a2.len() == 1 {
        let (from_00, from_11) = (hom_a1[0], hom_a2[0]);
        if from_00 != from_11 && ((effect == from_00 && other == from_11) || (effect == from_11 && other == from_00)) {
            return Some((i64::from(effect == from_00) + i64::from(effect == from_11), 1));
        }
    }
    if !conflicts.is_empty()
        && let Some(d) = dosages
    {
        return Some((d.iter().sum(), d.len() as i64));
    }
    None
}

/// Outcome of one application for every joint call of its contexts: the integer to
/// add, or 1 = missing, 2 = unresolvable.
struct Outcome {
    offsets: Vec<usize>,
    value: Vec<i128>,
    flag: Vec<u8>,
}

fn complex_kernel(plan: &Plan, bed: &[u8], n: usize, threads: usize, totals: &mut Totals) {
    if plan.apps.is_empty() {
        return;
    }
    let k = plan.names.len();
    let bpr = n.div_ceil(4);
    let outcomes: Vec<Outcome> = par_map(plan.apps.len(), threads, |a| {
        let app = &plan.apps[a];
        if app.contexts.len() > 8 {
            fail("complex application with more than 8 contexts is outside this prototype");
        }
        let m = to_int(app.weight, plan.exp[app.score as usize]);
        let size = 1usize << (2 * app.contexts.len());
        let mut outcome = Outcome {
            offsets: app.contexts.iter().map(|c| 3 + c.0 as usize * bpr).collect(),
            value: vec![0; size],
            flag: vec![0; size],
        };
        let mut valid: Vec<(u8, &Context)> = Vec::with_capacity(app.contexts.len());
        for key in 0..size {
            valid.clear();
            for (i, context) in app.contexts.iter().enumerate() {
                let code = ((key >> (2 * i)) & 3) as u8;
                if code != 0b01 {
                    valid.push((code, context));
                }
            }
            let dosage = match valid.len() {
                0 => None,
                1 => calculate_score_dosage(valid[0].0, &valid[0].1.1, &valid[0].1.2, app).map(|d| (d, 1)),
                _ => match resolve(app, &valid) {
                    Some(d) => Some(d),
                    None => {
                        outcome.flag[key] = 2;
                        continue;
                    }
                },
            };
            match dosage {
                None => outcome.flag[key] = 1,
                Some((num, den)) => outcome.value[key] = m * i128::from(plan.lcm / den) * i128::from(num),
            }
        }
        for &offset in &outcome.offsets {
            if offset + bpr > bed.len() {
                fail("a complex context row lies past the end of the .bed");
            }
        }
        outcome
    });
    let block = n.div_ceil(threads * 4).max(256);
    let blocks = par_map(n.div_ceil(block), threads, |b| {
        let (p0, p1) = (b * block, ((b + 1) * block).min(n));
        let width = p1 - p0;
        let mut tot = vec![0i128; k * width];
        let mut miss = vec![0u32; k * width];
        let mut fatal = None;
        for (app, outcome) in plan.apps.iter().zip(&outcomes) {
            let s = app.score as usize;
            for p in p0..p1 {
                let (byte, shift) = (p / 4, 2 * (p % 4));
                let mut key = 0usize;
                for (i, &offset) in outcome.offsets.iter().enumerate() {
                    key |= usize::from((bed[offset + byte] >> shift) & 3) << (2 * i);
                }
                match outcome.flag[key] {
                    0 => tot[(p - p0) * k + s] += outcome.value[key],
                    1 => miss[(p - p0) * k + s] += 1,
                    _ => fatal = fatal.or(Some(p)),
                }
            }
        }
        (tot, miss, fatal)
    });
    for (b, (tot, miss, fatal)) in blocks.into_iter().enumerate() {
        if let Some(p) = fatal {
            fail(format!("unresolvable complex locus for fam index {p}"));
        }
        let p0 = b * block;
        for (d, x) in totals.tot[p0 * k..p0 * k + tot.len()].iter_mut().zip(&tot) {
            *d += x;
        }
        for (d, x) in totals.miss[p0 * k..p0 * k + miss.len()].iter_mut().zip(&miss) {
            *d += x;
        }
    }
}

// ------------------------------------------------------------------------------ output

/// Double-double quotient (hi + lo) / d rounded to f64, as score_oracle takes it.
fn quotient(numerator: i128, d: f64) -> f64 {
    let hi = numerator as f64;
    let lo = (numerator - hi as i128) as f64;
    let q1 = hi / d;
    let product = q1 * d;
    let error = q1.mul_add(d, -product);
    let r = ((hi - product) - error) + lo;
    q1 + r / d
}

fn write_sscore(path: &Path, fam: &Fam, plan: &Plan, totals: &Totals, threads: usize) {
    let (n, k) = (fam.iids.len(), plan.names.len());
    let fam_bytes = fam.map.bytes();
    let mut header = String::from("#IID");
    for name in &plan.names {
        header.push_str(&format!("\t{name}_AVG\t{name}_MISSING_PCT"));
    }
    header.push('\n');
    let block = 8192;
    let pieces = par_map(n.div_ceil(block), threads, |b| {
        let (p0, p1) = (b * block, ((b + 1) * block).min(n));
        let mut out = Vec::with_capacity((p1 - p0) * (16 + 32 * k));
        let (mut avg_text, mut pct_text) = (ryu::Buffer::new(), ryu::Buffer::new());
        for p in p0..p1 {
            let (start, len) = fam.iids[p];
            out.extend_from_slice(&fam_bytes[start..start + len]);
            for s in 0..k {
                let (total, missing) = (plan.denominators[s], totals.miss[p * k + s]);
                let used = total.saturating_sub(missing);
                let avg = if used == 0 {
                    0.0
                } else {
                    let scale = f64::from_bits(((1023 + i64::from(plan.exp[s])) as u64) << 52);
                    quotient(totals.tot[p * k + s], plan.lcm as f64 * f64::from(used)) * scale
                };
                let pct = if total > 0 { (missing as f32 / total as f32) * 100.0 } else { 0.0 };
                out.push(b'\t');
                out.extend_from_slice(avg_text.format(avg).as_bytes());
                out.push(b'\t');
                out.extend_from_slice(pct_text.format(pct).as_bytes());
            }
            out.push(b'\n');
        }
        out
    });
    let mut file = File::create(path).unwrap_or_else(|e| fail(format!("{}: {e}", path.display())));
    file.write_all(header.as_bytes()).unwrap_or_else(|e| fail(format!("{}: {e}", path.display())));
    for piece in pieces {
        file.write_all(&piece).unwrap_or_else(|e| fail(format!("{}: {e}", path.display())));
    }
}

// ------------------------------------------------------------------------------ memory

/// min(MemAvailable, cgroup limit - usage) in bytes, when either is visible.
fn available_memory() -> Option<u64> {
    let mut best: Option<u64> = None;
    let mut take = |value: u64| best = Some(best.map_or(value, |b| b.min(value)));
    if let Ok(text) = std::fs::read_to_string("/proc/meminfo")
        && let Some(line) = text.lines().find(|l| l.starts_with("MemAvailable:"))
        && let Some(kb) = line.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok())
    {
        take(kb * 1024);
    }
    if let Ok(groups) = std::fs::read_to_string("/proc/self/cgroup") {
        for line in groups.lines() {
            let mut fields = line.splitn(3, ':');
            let (_, controllers, path) = (fields.next(), fields.next().unwrap_or(""), fields.next().unwrap_or(""));
            let (limit, usage) = if controllers.is_empty() {
                (format!("/sys/fs/cgroup{path}/memory.max"), format!("/sys/fs/cgroup{path}/memory.current"))
            } else if controllers.split(',').any(|c| c == "memory") {
                (
                    format!("/sys/fs/cgroup/memory{path}/memory.limit_in_bytes"),
                    format!("/sys/fs/cgroup/memory{path}/memory.usage_in_bytes"),
                )
            } else {
                continue;
            };
            if let (Ok(limit), Ok(usage)) = (std::fs::read_to_string(limit), std::fs::read_to_string(usage))
                && let (Ok(limit), Ok(usage)) = (limit.trim().parse::<u64>(), usage.trim().parse::<u64>())
            {
                take(limit.saturating_sub(usage));
            }
        }
    }
    best
}

// ------------------------------------------------------------------------------ main

fn score_command(args: &[String]) {
    let mut timer = Timer::new();
    let mut positional = Vec::new();
    let (mut out, mut threads, mut cache_dir, mut no_cache) = (None, None, None, false);
    let mut options = KernelOptions {
        kernel: Kernel::Lut,
        gv: 4,
        advise: false,
        threads: 1,
        acc: 1,
        pread: false,
        partition: Partition::Auto,
        memory_budget: u64::MAX,
    };
    let mut i = 0;
    while i < args.len() {
        let value = || args.get(i + 1).cloned().unwrap_or_else(|| fail(format!("{} needs a value", args[i])));
        match args[i].as_str() {
            "--out" => {
                out = Some(PathBuf::from(value()));
                i += 1;
            }
            "--threads" => {
                threads = Some(value().parse::<usize>().unwrap_or_else(|_| fail("--threads needs a number")));
                i += 1;
            }
            "--cache-dir" => {
                cache_dir = Some(PathBuf::from(value()));
                i += 1;
            }
            "--no-cache" => no_cache = true,
            "--advise" => options.advise = true,
            "--partition" => {
                options.partition = match value().as_str() {
                    "auto" => Partition::Auto,
                    "rows" => Partition::Rows,
                    "people" => Partition::People,
                    _ => fail("--partition takes auto, rows or people"),
                };
                i += 1;
            }
            "--io" => {
                options.pread = match value().as_str() {
                    "mmap" => false,
                    "pread" => true,
                    _ => fail("--io takes mmap or pread"),
                };
                i += 1;
            }
            "--acc" => {
                options.acc = match value().as_str() {
                    "plain" => 0,
                    "pair" => 1,
                    "fused" => 2,
                    _ => fail("--acc takes plain, pair or fused"),
                };
                i += 1;
            }
            "--gv" => {
                options.gv = value().parse().ok().filter(|g| (4..=6).contains(g)).unwrap_or_else(|| fail("--gv takes 4, 5 or 6"));
                i += 1;
            }
            "--kernel" => {
                options.kernel = match value().as_str() {
                    "lut" => Kernel::Lut,
                    "scalar" => Kernel::Scalar,
                    "touch" => Kernel::Touch,
                    "catalog" => Kernel::Catalog,
                    _ => fail("--kernel takes lut, scalar, touch or catalog"),
                };
                i += 1;
            }
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    if positional.len() != 2 {
        fail("usage: gscore score <score file>[,...] <bfile prefix> --out FILE [--threads N] [--cache-dir D | --no-cache] [--kernel lut|scalar|touch] [--gv 4|5|6] [--advise]");
    }
    let out = out.unwrap_or_else(|| fail("--out is required"));
    let threads = threads.unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get())).max(1);
    let prefix = positional[1].trim_end_matches(".bed").to_string();
    let mut score_paths: Vec<PathBuf> = Vec::new();
    for arg in positional[0].split(',') {
        let path = PathBuf::from(arg);
        if path.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(&path)
                .unwrap_or_else(|e| fail(format!("{}: {e}", path.display())))
                .filter_map(|entry| entry.ok())
                // d_type from the directory listing; a stat per file costs about 0.3 ms on network filesystems.
                .filter(|e| e.file_type().is_ok_and(|t| t.is_file() || t.is_symlink()))
                .map(|e| e.path())
                .collect();
            files.sort();
            score_paths.extend(files);
        } else {
            score_paths.push(path);
        }
    }
    timer.lap("startup");

    let fam = read_fam(Path::new(&format!("{prefix}.fam")), threads);
    let n = fam.iids.len();
    timer.lap("fam");
    let bim = Map::open(Path::new(&format!("{prefix}.bim")));
    let srcs: Vec<ScoreSrc> = par_map(score_paths.len(), threads, |i| open_score(&score_paths[i]));
    timer.lap("open");

    let cache_path = (!no_cache).then(|| {
        let dir = cache_dir.clone().unwrap_or_else(|| out.parent().map_or_else(|| PathBuf::from("."), Path::to_path_buf));
        let mut key = hash_block(b"gscore plan v1", 0);
        for src in &srcs {
            key = (key ^ hash_bytes(src.map.bytes(), threads)).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(17);
        }
        key = (key ^ hash_bytes(bim.bytes(), threads)).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(17);
        (dir.join(format!("gscore-{key:016x}.plan")), key)
    });
    timer.lap("hash");

    let cached = cache_path
        .as_ref()
        .and_then(|(path, key)| std::fs::read(path).ok().and_then(|data| decode_plan(&data, *key)));
    let (plan, cache_writer) = if let Some(plan) = cached {
        timer.lap("cache_load");
        eprintln!("gscore: prepared plan loaded from cache");
        (plan, None)
    } else {
        timer.lap("cache_miss");
        let parts: Vec<(usize, Range<usize>)> = srcs
            .iter()
            .enumerate()
            .flat_map(|(f, src)| {
                let body = src.body;
                let len = src.map.len;
                let pieces = if len - body > 1 << 20 { threads * 4 } else { 1 };
                split_lines(&src.map.bytes()[body..], pieces).into_iter().map(move |r| (f, r.start + body..r.end + body))
            })
            .collect();
        let parsed = par_map(parts.len(), threads, |i| parse_score_part(&srcs[parts[i].0], parts[i].1.clone()));
        let mut files: Vec<FileData> =
            srcs.iter().map(|src| FileData { k: src.names.len(), columns: Vec::new(), rows: Vec::new(), weights: Vec::new() }).collect();
        let mut skipped = 0usize;
        for ((f, _), part) in parts.iter().zip(parsed) {
            files[*f].rows.extend_from_slice(&part.rows);
            files[*f].weights.extend_from_slice(&part.weights);
            skipped += part.skipped;
        }
        if skipped > 0 {
            eprintln!("gscore: skipped {skipped} score line(s) across {} file(s)", srcs.len());
        }
        timer.lap("score_parse");
        let mut names: Vec<String> = srcs.iter().flat_map(|s| s.names.iter().cloned()).collect();
        names.sort();
        if names.windows(2).any(|w| w[0] == w[1]) || names.iter().any(String::is_empty) {
            fail("empty or duplicate score id");
        }
        if names.len() >= 1 << 16 {
            fail("this prototype supports fewer than 65,536 scores");
        }
        for (fd, src) in files.iter_mut().zip(&srcs) {
            fd.columns = src.names.iter().map(|name| names.binary_search(name).unwrap() as u32).collect();
        }
        let mut all = Vec::with_capacity(files.iter().map(|f| f.rows.len()).sum());
        for (f, fd) in files.iter().enumerate() {
            for (index, row) in fd.rows.iter().enumerate() {
                all.push(GRow { key: row.key, ea: row.ea, oa: row.oa, file: f as u32, index: index as u32 });
            }
        }
        let all = par_sort_by_key(all, threads, |r| r.key);
        let mut keys: Vec<u64> = all.iter().map(|r| r.key).collect();
        keys.dedup();
        timer.lap("score_sort");

        let bim_bytes = bim.bytes();
        let parts = split_lines(bim_bytes, threads * 4);
        let parsed = par_map(parts.len(), threads, |i| parse_bim_part(bim_bytes, parts[i].clone(), &keys));
        let mut bims = Vec::with_capacity(parsed.iter().map(|p| p.0.len()).sum());
        let mut base = 0u64;
        for (records, lines) in parsed {
            bims.extend(records.into_iter().map(|r| BRow { row: r.row + base, ..r }));
            base += lines;
        }
        let bims = par_sort_by_key(bims, threads, |r| r.key);
        timer.lap("bim_parse");

        let mut cuts = vec![0usize];
        let nparts = threads * 4;
        for p in 1..nparts {
            let mut cut = all.len() * p / nparts;
            while cut > 0 && cut < all.len() && all[cut].key == all[cut - 1].key {
                cut += 1;
            }
            if cut > *cuts.last().unwrap() && cut < all.len() {
                cuts.push(cut);
            }
        }
        cuts.push(all.len());
        let joined = par_map(cuts.len() - 1, threads, |i| join_part(&all[cuts[i]..cuts[i + 1]], &files, &srcs, &bims, bim_bytes));
        let mut terms = Vec::with_capacity(joined.iter().map(|j| j.0.len()).sum());
        let mut apps = Vec::new();
        for (t, a) in joined {
            terms.extend_from_slice(&t);
            apps.extend(a);
        }
        drop(files);
        let terms = par_sort_by_key(terms, threads, |t| t.order);
        timer.lap("join");
        let plan = compile_plan(names, terms, apps);
        timer.lap("plan");
        let writer = cache_path.as_ref().map(|(path, key)| {
            let bytes = encode_plan(&plan, *key);
            let (path, tmp) = (path.clone(), path.with_extension(format!("tmp{}", std::process::id())));
            std::thread::spawn(move || {
                if std::fs::write(&tmp, &bytes).is_ok() {
                    let _ = std::fs::rename(&tmp, &path);
                }
            })
        });
        timer.lap("cache_encode");
        (plan, writer)
    };

    let bed = Map::open(Path::new(&format!("{prefix}.bed")));
    if bed.bytes().get(..3) != Some(&[0x6c, 0x1b, 0x01][..]) {
        fail(format!("{prefix}.bed is not a variant-major PLINK .bed"));
    }
    let bpr = n.div_ceil(4);
    if let Some(&last) = plan.rows.last()
        && 3 + (last as usize + 1) * bpr > bed.len
    {
        fail(format!("{prefix}.bed ends before needed row {last}"));
    }
    let k = plan.names.len();
    options.threads = threads;
    options.memory_budget = available_memory().map_or(256 << 20, |bytes| bytes / 4);
    timer.lap("bed_open");

    // Sparse many-score panels take the score-major catalog kernel.
    if options.kernel == Kernel::Lut && k > 64 && (plan.e_score.len() as f64) < 0.25 * (plan.rows.len() * k) as f64 {
        options.kernel = Kernel::Catalog;
    }
    let mut totals = if options.kernel == Kernel::Catalog {
        catalog_kernel(&plan, &bed, n, options.threads)
    } else {
        simple_kernel(&plan, &bed, Path::new(&format!("{prefix}.bed")), n, &options)
    };
    timer.lap("kernel");
    if options.kernel == Kernel::Touch {
        eprintln!("gscore: touched {} rows x {bpr} bytes, checksum {:x}", plan.rows.len(), totals.touched);
        timer.report();
        return;
    }
    complex_kernel(&plan, bed.bytes(), n, options.threads, &mut totals);
    timer.lap("complex");
    write_sscore(&out, &fam, &plan, &totals, threads);
    timer.lap("output");
    if let Some(writer) = cache_writer {
        writer.join().unwrap();
        timer.lap("cache_write");
    }
    eprintln!(
        "gscore: {n} people x {k} score(s); {} rows, {} (row, score) entries, {} complex applications; threads {} (kernel {}), lanes {:?}; denominators {:?}",
        plan.rows.len(),
        plan.e_score.len(),
        plan.apps.len(),
        threads,
        options.threads,
        if options.kernel == Kernel::Lut { lane_layout(&plan, options.gv) } else { None },
        &plan.denominators[..k.min(8)]
    );
    timer.report();
    let _ = std::io::stderr().flush();
    std::process::exit(0);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("score") => score_command(&args[2..]),
        _ => fail("usage: gscore score <score file>[,...] <bfile prefix> --out FILE [options]"),
    }
}
