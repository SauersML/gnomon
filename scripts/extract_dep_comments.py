#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path


FN_RE = re.compile(r"\bfn\s+[A-Za-z0-9_]+")


class ScanState:
    def __init__(self):
        self.in_block_comment = False
        self.in_string = False
        self.in_char = False
        self.in_raw_string = False
        self.raw_hashes = 0


def is_ident_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def detect_raw_string_start(line: str, idx: int):
    # Handles r"..." or r#"..."# or br"..." etc.
    if line[idx] == "r":
        start = idx
    elif line[idx] == "b" and idx + 1 < len(line) and line[idx + 1] == "r":
        start = idx + 1
    else:
        return None
    j = start + 1
    hashes = 0
    while j < len(line) and line[j] == "#":
        hashes += 1
        j += 1
    if j < len(line) and line[j] == "\"":
        return start, hashes, j
    return None


def scan_line_for_comments_and_code(line: str, state: ScanState):
    comments = []
    code_chars = []
    i = 0
    while i < len(line):
        ch = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""

        if state.in_block_comment:
            end_idx = line.find("*/", i)
            if end_idx == -1:
                comments.append(line[i:])
                return comments, "".join(code_chars), state
            comments.append(line[i : end_idx + 2])
            state.in_block_comment = False
            i = end_idx + 2
            continue

        if state.in_raw_string:
            end_pat = "\"" + ("#" * state.raw_hashes)
            end_idx = line.find(end_pat, i)
            if end_idx == -1:
                i = len(line)
                continue
            state.in_raw_string = False
            state.raw_hashes = 0
            i = end_idx + len(end_pat)
            continue

        if state.in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == "\"":
                state.in_string = False
            i += 1
            continue

        if state.in_char:
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                state.in_char = False
            i += 1
            continue

        raw_start = detect_raw_string_start(line, i)
        if raw_start:
            _, hashes, quote_idx = raw_start
            state.in_raw_string = True
            state.raw_hashes = hashes
            i = quote_idx + 1
            continue

        if ch == "b" and nxt == "\"":
            state.in_string = True
            i += 2
            continue

        if ch == "\"":
            state.in_string = True
            i += 1
            continue

        if ch == "'" and not is_ident_char(nxt):
            state.in_char = True
            i += 1
            continue

        if ch == "/" and nxt == "/":
            comments.append(line[i:])
            return comments, "".join(code_chars), state

        if ch == "/" and nxt == "*":
            state.in_block_comment = True
            comments.append(line[i:])
            return comments, "".join(code_chars), state

        code_chars.append(ch)
        i += 1

    return comments, "".join(code_chars), state


def find_sig_terminator(line: str, state: ScanState):
    i = 0
    while i < len(line):
        ch = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""

        if state.in_block_comment:
            end_idx = line.find("*/", i)
            if end_idx == -1:
                return None, None, state
            state.in_block_comment = False
            i = end_idx + 2
            continue

        if state.in_raw_string:
            end_pat = "\"" + ("#" * state.raw_hashes)
            end_idx = line.find(end_pat, i)
            if end_idx == -1:
                return None, None, state
            state.in_raw_string = False
            state.raw_hashes = 0
            i = end_idx + len(end_pat)
            continue

        if state.in_string:
            if ch == "\\":
                i += 2
                continue
            if ch == "\"":
                state.in_string = False
            i += 1
            continue

        if state.in_char:
            if ch == "\\":
                i += 2
                continue
            if ch == "'":
                state.in_char = False
            i += 1
            continue

        raw_start = detect_raw_string_start(line, i)
        if raw_start:
            _, hashes, quote_idx = raw_start
            state.in_raw_string = True
            state.raw_hashes = hashes
            i = quote_idx + 1
            continue

        if ch == "b" and nxt == "\"":
            state.in_string = True
            i += 2
            continue

        if ch == "\"":
            state.in_string = True
            i += 1
            continue

        if ch == "'" and not is_ident_char(nxt):
            state.in_char = True
            i += 1
            continue

        if ch == "/" and nxt == "/":
            return None, None, state

        if ch == "/" and nxt == "*":
            state.in_block_comment = True
            i += 2
            continue

        if ch == "{" or ch == ";":
            return i, ch, state

        i += 1

    return None, None, state


def count_braces_in_code(line: str, state: ScanState):
    brace_delta = 0
    code_state = state
    comments, code, code_state = scan_line_for_comments_and_code(line, code_state)
    for ch in code:
        if ch == "{":
            brace_delta += 1
        elif ch == "}":
            brace_delta -= 1
    return brace_delta, code_state


CRATE_STEM_RE = re.compile(r"^(.+)-([0-9].+)$")


def version_key_from_str(version: str):
    parts = re.split(r"[.-]", version)
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return key


def iter_crate_archives():
    cache_root = Path.home() / ".cargo/registry/cache"
    for path in cache_root.rglob("*.crate"):
        stem = path.stem
        match = CRATE_STEM_RE.match(stem)
        if not match:
            continue
        name, version = match.groups()
        yield name, version, path


def find_crate_archives(dep_name: str, include_prefix: bool) -> list[Path]:
    selected = {}
    for name, version, path in iter_crate_archives():
        if name != dep_name and not (include_prefix and name.startswith(f"{dep_name}-")):
            continue
        key = version_key_from_str(version)
        current = selected.get(name)
        if current is None or key > current[0]:
            selected[name] = (key, path)

    if not selected:
        cache_root = Path.home() / ".cargo/registry/cache"
        raise FileNotFoundError(f"No crate archive found for {dep_name} in {cache_root}")

    return [path for _, path in sorted(selected.items())]


def extract_crate(crate_path: Path, dest_dir: Path) -> Path:
    with tarfile.open(crate_path, "r:gz") as tf:
        tf.extractall(dest_dir)
    roots = [p for p in dest_dir.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"unexpected crate layout in {dest_dir}")
    return roots[0]


def collect_function_signature(lines, start_idx):
    fn_lines = []
    sig_state = ScanState()
    i = start_idx
    while i < len(lines):
        line = lines[i]
        idx, ch, sig_state = find_sig_terminator(line, sig_state)
        if idx is not None:
            fn_lines.append(line[: idx + 1])
            return fn_lines, i + 1
        fn_lines.append(line)
        i += 1
    return fn_lines, i


def extract_lines_in_order(path: Path):
    out_lines = []
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    state = ScanState()

    while i < len(lines):
        line = lines[i]
        comments, code, state = scan_line_for_comments_and_code(line, state)

        if FN_RE.search(code):
            fn_lines, next_idx = collect_function_signature(lines, i)
            out_lines.extend(fn_lines)
            i = next_idx
            continue

        for c in comments:
            if c.strip():
                out_lines.append(c)

        i += 1

    return out_lines


def main():
    parser = argparse.ArgumentParser(description="Extract comments and function definitions from a crate.")
    parser.add_argument("dep_name", help="Cargo dependency name, e.g. faer")
    parser.add_argument(
        "--include-prefix",
        action="store_true",
        help="Also scan crates whose names start with <dep_name>- (e.g. burn-core, burn-tensor).",
    )
    parser.add_argument(
        "--out",
        help="Output path (default: <dep_name>_comments.txt)",
        default=None,
    )
    args = parser.parse_args()

    crate_paths = find_crate_archives(args.dep_name, args.include_prefix)
    out_path = Path(args.out or f"{args.dep_name}_comments.txt")

    with tempfile.TemporaryDirectory(prefix=f"{args.dep_name}-src-") as tmpdir:
        out_lines = []
        for crate_path in crate_paths:
            extract_root = Path(tmpdir) / crate_path.stem
            extract_root.mkdir(parents=True, exist_ok=True)
            root = extract_crate(crate_path, extract_root)
            rs_files = sorted(root.rglob("*.rs"))
            for rs in rs_files:
                lines = extract_lines_in_order(rs)
                if not lines:
                    continue
                out_lines.extend(lines)

        out_text = "\n".join(out_lines).rstrip() + "\n"
        out_path.write_text(out_text)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
