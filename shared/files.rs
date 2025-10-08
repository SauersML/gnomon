use crate::score::pipeline::PipelineError;
use flate2::read::MultiGzDecoder;
use google_cloud_auth::credentials::{Credentials, anonymous::Builder as AnonymousCredentials};
use google_cloud_storage::client::{Storage, StorageControl};
use google_cloud_storage::model_ext::ReadRange;
use log::{debug, warn};
use memmap2::Mmap;
use natord::compare;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use tokio::runtime::Runtime;

/// The number of variants to process locally before updating the global atomic counter.
/// A power of 2 is often efficient
pub const PROGRESS_UPDATE_BATCH_SIZE: u64 = 1024;

const REMOTE_BLOCK_SIZE: usize = 8 * 1024 * 1024;
const REMOTE_CACHE_CAPACITY: usize = 8;

static RUNTIME_MANAGER: OnceLock<Arc<Runtime>> = OnceLock::new();

pub fn get_shared_runtime() -> Result<Arc<Runtime>, PipelineError> {
    if let Some(runtime) = RUNTIME_MANAGER.get() {
        return Ok(Arc::clone(runtime));
    }

    let runtime = Arc::new(
        Runtime::new()
            .map_err(|e| PipelineError::Io(format!("Failed to initialize Tokio runtime: {e}")))?,
    );

    match RUNTIME_MANAGER.set(Arc::clone(&runtime)) {
        Ok(()) => Ok(runtime),
        Err(_) => Ok(RUNTIME_MANAGER
            .get()
            .cloned()
            .expect("Tokio runtime should be initialized")),
    }
}

pub fn gcs_billing_project_from_env() -> Option<String> {
    for key in [
        "GOOGLE_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "CLOUDSDK_CORE_PROJECT",
    ] {
        if let Ok(v) = env::var(key) {
            if !v.trim().is_empty() {
                return Some(v);
            }
        }
    }
    None
}

pub fn load_adc_credentials() -> Result<Credentials, PipelineError> {
    let mut builder = google_cloud_auth::credentials::Builder::default();
    if let Some(project) = gcs_billing_project_from_env() {
        builder = builder.with_quota_project_id(project);
    }

    let runtime = if tokio::runtime::Handle::try_current().is_err() {
        Some(get_shared_runtime()?)
    } else {
        None
    };
    let runtime_guard = runtime.as_ref().map(|rt| rt.enter());

    let credentials = builder
        .build()
        .map_err(|e| PipelineError::Io(format!("Failed to load ADC credentials: {e}")))?;

    drop(runtime_guard);

    Ok(credentials)
}

/// A trait that abstracts sequential, line-oriented access to text data such as
/// `.bim` and `.fam` files, regardless of the underlying storage medium.
pub trait TextSource: Send {
    fn len(&self) -> Option<u64> {
        None
    }

    fn next_line<'a>(&'a mut self) -> Result<Option<&'a [u8]>, PipelineError>;
}

fn augment_pipeline_error(err: PipelineError, context: &str) -> PipelineError {
    match err {
        PipelineError::Io(msg) => PipelineError::Io(format!("{context}: {msg}")),
        PipelineError::Producer(msg) => PipelineError::Producer(format!("{context}: {msg}")),
        PipelineError::Compute(msg) => PipelineError::Compute(format!("{context}: {msg}")),
    }
}

/// A trait that abstracts byte-range access for `.bed` data, regardless of the
/// underlying storage mechanism.
pub trait ByteRangeSource: Send + Sync {
    fn len(&self) -> u64;
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError>;
}

/// A convenience wrapper that bundles the generic byte source with an optional
/// local memory map when the data lives on disk.
#[derive(Clone)]
pub struct BedSource {
    byte_source: Arc<dyn ByteRangeSource>,
    mmap: Option<Arc<Mmap>>,
}

impl BedSource {
    fn new(byte_source: Arc<dyn ByteRangeSource>, mmap: Option<Arc<Mmap>>) -> Self {
        Self { byte_source, mmap }
    }

    pub fn byte_source(&self) -> Arc<dyn ByteRangeSource> {
        Arc::clone(&self.byte_source)
    }

    pub fn mmap(&self) -> Option<Arc<Mmap>> {
        self.mmap.as_ref().map(Arc::clone)
    }

    pub fn len(&self) -> u64 {
        self.byte_source.len()
    }

    pub fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
        self.byte_source.read_at(offset, dst)
    }
}

impl std::fmt::Debug for BedSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedSource")
            .field("len", &self.len())
            .field("has_mmap", &self.mmap.is_some())
            .finish()
    }
}

/// Creates a `BedSource` for the provided `.bed` path. Local paths are
/// memory-mapped, while `gs://` locations stream data directly from Cloud
/// Storage.
pub fn open_bed_source(path: &Path) -> Result<BedSource, PipelineError> {
    if is_gcs_path(path) {
        let uri = path
            .to_str()
            .ok_or_else(|| PipelineError::Io("Invalid UTF-8 in path".to_string()))?;
        let (bucket, object) = parse_gcs_uri(uri)?;
        let remote = RemoteByteRangeSource::new(&bucket, &object)?;
        Ok(BedSource::new(Arc::new(remote), None))
    } else {
        let file = File::open(path)
            .map_err(|e| PipelineError::Io(format!("Opening {}: {e}", path.display())))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| PipelineError::Io(e.to_string()))? };
        mmap.advise(memmap2::Advice::Sequential)
            .map_err(|e| PipelineError::Io(e.to_string()))?;
        let mmap = Arc::new(mmap);
        let byte_source = Arc::new(MmapByteRangeSource::new(Arc::clone(&mmap)));
        Ok(BedSource::new(byte_source, Some(mmap)))
    }
}

struct PreparedBcfReader {
    reader: Box<dyn Read + Send>,
    len: u64,
    skip_header: bool,
}

/// Provides sequential access to one or more BCF streams. The data is
/// transparently decompressed when the underlying files are BGZF-compressed.
pub struct BcfSource {
    readers: VecDeque<PreparedBcfReader>,
    current: Option<Box<dyn Read + Send>>,
    total_len: Option<u64>,
}

impl BcfSource {
    fn from_readers(mut readers: Vec<PreparedBcfReader>) -> Result<Self, PipelineError> {
        if readers.is_empty() {
            return Err(PipelineError::Io(
                "No BCF files were discovered for the requested input".to_string(),
            ));
        }

        let total_len = readers.iter().try_fold(0u64, |acc, r| {
            acc.checked_add(r.len).ok_or_else(|| {
                PipelineError::Io("Combined BCF length exceeds u64::MAX".to_string())
            })
        })?;

        let mut queue = VecDeque::with_capacity(readers.len());
        for (idx, mut reader) in readers.drain(..).enumerate() {
            reader.skip_header = idx > 0;
            queue.push_back(reader);
        }

        Ok(Self {
            readers: queue,
            current: None,
            total_len: Some(total_len),
        })
    }

    fn read_internal(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        loop {
            if !self.ensure_current_reader()? {
                return Ok(0);
            }

            if let Some(reader) = self.current.as_mut() {
                match reader.read(buf) {
                    Ok(0) => {
                        self.current = None;
                        continue;
                    }
                    other => return other,
                }
            }
        }
    }

    fn ensure_current_reader(&mut self) -> io::Result<bool> {
        while self.current.is_none() {
            let prepared = match self.readers.pop_front() {
                Some(reader) => reader,
                None => return Ok(false),
            };

            let mut reader = prepared.reader;
            if prepared.skip_header {
                skip_bcf_header(reader.as_mut())?;
                self.total_len = None;
            }

            self.current = Some(reader);
        }

        Ok(true)
    }

    /// Returns the combined compressed length of the underlying BCF objects
    /// when known.
    pub fn len(&self) -> Option<u64> {
        self.total_len
    }

    /// Reads decompressed bytes from the concatenated BCF streams into `buf`.
    pub fn read_chunk(&mut self, buf: &mut [u8]) -> Result<usize, PipelineError> {
        self.read_internal(buf)
            .map_err(|e| PipelineError::Io(format!("Error reading BCF stream: {e}")))
    }
}

impl Read for BcfSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.read_internal(buf)
    }
}

fn skip_bcf_header(reader: &mut dyn Read) -> io::Result<u64> {
    use noodles_bcf::io::Reader as BcfReader;

    struct CountingReader<'a> {
        inner: &'a mut dyn Read,
        count: u64,
    }

    impl<'a> CountingReader<'a> {
        fn new(inner: &'a mut dyn Read) -> Self {
            Self { inner, count: 0 }
        }

        fn into_inner(self) -> (&'a mut dyn Read, u64) {
            (self.inner, self.count)
        }
    }

    impl Read for CountingReader<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = self.inner.read(buf)?;
            self.count += n as u64;
            Ok(n)
        }
    }

    let counting_reader = CountingReader::new(reader);
    let mut bcf_reader = BcfReader::from(counting_reader);
    bcf_reader.read_header()?;
    let counting_reader = bcf_reader.into_inner();
    let (_, consumed) = counting_reader.into_inner();
    Ok(consumed)
}

pub fn open_text_source(path: &Path) -> Result<Box<dyn TextSource>, PipelineError> {
    if is_gcs_path(path) {
        let uri = path
            .to_str()
            .ok_or_else(|| PipelineError::Io("Invalid UTF-8 in path".to_string()))?;
        let (bucket, object) = parse_gcs_uri(uri)?;
        let remote = RemoteByteRangeSource::new(&bucket, &object)?;
        Ok(Box::new(RemoteTextSource::new(
            format!("gs://{bucket}/{object}"),
            Arc::new(remote),
        )))
    } else {
        let file = File::open(path)
            .map_err(|e| PipelineError::Io(format!("Opening {}: {e}", path.display())))?;
        Ok(Box::new(LocalTextSource::new(path, file)?))
    }
}

pub fn open_bcf_source(path: &Path) -> Result<BcfSource, PipelineError> {
    if is_gcs_path(path) {
        return open_remote_bcf_source(path);
    }

    if path.is_dir() {
        let mut entries = gather_local_bcf_files(path)?;
        entries.sort_by(|a, b| compare_paths(a, b));
        let readers = entries
            .into_iter()
            .map(|p| open_local_bcf_reader(&p))
            .collect::<Result<Vec<_>, _>>()?;
        BcfSource::from_readers(readers)
    } else if has_bcf_extension(path) {
        let reader = open_local_bcf_reader(path)?;
        BcfSource::from_readers(vec![reader])
    } else {
        Err(PipelineError::Io(format!(
            "Path {} is not a .bcf file or directory",
            path.display()
        )))
    }
}

fn is_gcs_path(path: &Path) -> bool {
    path.to_str()
        .map(|s| s.starts_with("gs://"))
        .unwrap_or(false)
}

fn has_bcf_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("bcf"))
        .unwrap_or(false)
}

fn compare_paths(a: &Path, b: &Path) -> std::cmp::Ordering {
    let a_str = a
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| a.to_string_lossy().into_owned());
    let b_str = b
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| b.to_string_lossy().into_owned());
    compare(&a_str, &b_str)
}

fn gather_local_bcf_files(dir: &Path) -> Result<Vec<PathBuf>, PipelineError> {
    let read_dir = fs::read_dir(dir)
        .map_err(|e| PipelineError::Io(format!("Unable to list {}: {e}", dir.display())))?;

    let mut files = Vec::new();
    for entry in read_dir {
        let entry = entry.map_err(|e| {
            PipelineError::Io(format!(
                "Unable to read directory entry in {}: {e}",
                dir.display()
            ))
        })?;
        let path = entry.path();
        if path.is_file() && has_bcf_extension(&path) {
            files.push(path);
        }
    }

    if files.is_empty() {
        return Err(PipelineError::Io(format!(
            "No .bcf files found in directory {}",
            dir.display()
        )));
    }

    Ok(files)
}

fn open_local_bcf_reader(path: &Path) -> Result<PreparedBcfReader, PipelineError> {
    let mut file = File::open(path)
        .map_err(|e| PipelineError::Io(format!("Opening {}: {e}", path.display())))?;
    let metadata = file
        .metadata()
        .map_err(|e| PipelineError::Io(format!("Metadata for {}: {e}", path.display())))?;
    let len = metadata.len();

    let mut magic = [0u8; 2];
    let bytes_read = file
        .read(&mut magic)
        .map_err(|e| PipelineError::Io(format!("Reading {}: {e}", path.display())))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|e| PipelineError::Io(format!("Seeking {}: {e}", path.display())))?;

    let reader: Box<dyn Read + Send> = if bytes_read == 2 && is_gzip_magic(&magic) {
        Box::new(MultiGzDecoder::new(BufReader::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    Ok(PreparedBcfReader {
        reader,
        len,
        skip_header: false,
    })
}

fn is_gzip_magic(magic: &[u8; 2]) -> bool {
    magic[0] == 0x1F && magic[1] == 0x8B
}

fn parse_gcs_uri(uri: &str) -> Result<(String, String), PipelineError> {
    let without_scheme = uri.trim_start_matches("gs://");
    let mut parts = without_scheme.splitn(2, '/');
    let bucket = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| PipelineError::Io(format!("Malformed GCS URI '{uri}': missing bucket")))?;
    let object = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| PipelineError::Io(format!("Malformed GCS URI '{uri}': missing object")))?;
    Ok((bucket.to_string(), object.to_string()))
}

fn normalize_gcs_prefix(object: &str) -> String {
    if object.is_empty() {
        String::new()
    } else if object.ends_with('/') {
        object.to_string()
    } else {
        format!("{object}/")
    }
}

fn is_not_found_error(err: &PipelineError) -> bool {
    match err {
        PipelineError::Io(msg) => {
            let lower = msg.to_lowercase();
            lower.contains("not found") || lower.contains("no such object") || lower.contains("404")
        }
        _ => false,
    }
}

fn convert_list_error(
    bucket: &str,
    prefix: &str,
    user_project: Option<String>,
    err: google_cloud_storage::Error,
) -> PipelineError {
    let location = if prefix.is_empty() {
        format!("gs://{bucket}")
    } else {
        format!("gs://{bucket}/{prefix}")
    };
    let msg = err.to_string();
    if user_project.is_none() && msg.to_lowercase().contains("requester pays") {
        PipelineError::Io(format!(
            "Requester Pays bucket requires a billing project. Set GOOGLE_PROJECT (or run `gcloud config set project ...`). Original error while listing {location}: {msg}"
        ))
    } else {
        PipelineError::Io(format!(
            "Failed to list BCF objects under {location}: {msg}"
        ))
    }
}

fn fetch_bcf_object_names(
    runtime: &Arc<Runtime>,
    control: &StorageControl,
    bucket_path: &str,
    prefix: &str,
) -> Result<Vec<String>, google_cloud_storage::Error> {
    let mut page_token: Option<String> = None;
    let mut names = Vec::new();
    loop {
        let mut request = control
            .list_objects()
            .set_parent(bucket_path.to_string())
            .set_prefix(prefix.to_string());
        if let Some(token) = page_token.take() {
            request = request.set_page_token(token);
        }
        let response = runtime.block_on(request.send())?;
        names.extend(
            response
                .objects
                .into_iter()
                .filter(|o| !o.name.ends_with('/'))
                .filter(|o| {
                    o.name
                        .rsplit('.')
                        .next()
                        .map(|ext| ext.eq_ignore_ascii_case("bcf"))
                        .unwrap_or(false)
                })
                .map(|o| o.name),
        );
        if response.next_page_token.is_empty() {
            break;
        }
        page_token = Some(response.next_page_token);
    }
    Ok(names)
}

fn list_remote_bcf_objects(bucket: &str, prefix: &str) -> Result<Vec<String>, PipelineError> {
    let runtime = get_shared_runtime()?;
    let (_, control) = RemoteByteRangeSource::create_clients(&runtime, None)?;
    let bucket_path = format!("projects/_/buckets/{bucket}");
    let user_project = gcs_billing_project_from_env();

    let attempt =
        |control: &StorageControl| fetch_bcf_object_names(&runtime, control, &bucket_path, prefix);

    match attempt(&control) {
        Ok(names) => Ok(names),
        Err(err) if RemoteByteRangeSource::is_authentication_error(&err) => {
            let (_, fallback_control) = RemoteByteRangeSource::create_clients(
                &runtime,
                Some(AnonymousCredentials::new().build()),
            )?;
            drop(fallback_storage);
            attempt(&fallback_control).map_err(|retry_err| {
                convert_list_error(bucket, prefix, user_project.clone(), retry_err)
            })
        }
        Err(err) => Err(convert_list_error(bucket, prefix, user_project, err)),
    }
}

fn open_remote_bcf_prefix(
    bucket: &str,
    prefix: &str,
) -> Result<Vec<PreparedBcfReader>, PipelineError> {
    let mut objects = list_remote_bcf_objects(bucket, prefix)?;
    if objects.is_empty() {
        let location = if prefix.is_empty() {
            format!("gs://{bucket}")
        } else {
            format!("gs://{bucket}/{prefix}")
        };
        return Err(PipelineError::Io(format!(
            "No .bcf objects found under {location}"
        )));
    }

    objects.sort_by(|a, b| compare(a, b));

    let mut readers = Vec::with_capacity(objects.len());
    for object in objects {
        let source = RemoteByteRangeSource::new(bucket, &object)?;
        readers.push(prepare_remote_bcf_reader(bucket, &object, source)?);
    }

    Ok(readers)
}

fn open_remote_bcf_source(path: &Path) -> Result<BcfSource, PipelineError> {
    let uri = path
        .to_str()
        .ok_or_else(|| PipelineError::Io("Invalid UTF-8 in path".to_string()))?;
    let normalized = if uri.ends_with("/*") {
        &uri[..uri.len() - 1]
    } else {
        uri
    };
    let (bucket, object) = parse_gcs_uri(normalized)?;
    let lower = object.to_lowercase();
    let treat_as_directory =
        normalized.ends_with('/') || object.is_empty() || !lower.ends_with(".bcf");

    if !treat_as_directory {
        match RemoteByteRangeSource::new(&bucket, &object) {
            Ok(source) => {
                let reader = prepare_remote_bcf_reader(&bucket, &object, source)?;
                return BcfSource::from_readers(vec![reader]);
            }
            Err(err) => {
                if is_not_found_error(&err) {
                    let prefix = normalize_gcs_prefix(&object);
                    let readers = open_remote_bcf_prefix(&bucket, &prefix)?;
                    return BcfSource::from_readers(readers);
                } else {
                    return Err(err);
                }
            }
        }
    }

    let prefix = normalize_gcs_prefix(&object);
    let readers = open_remote_bcf_prefix(&bucket, &prefix)?;
    BcfSource::from_readers(readers)
}

struct MmapByteRangeSource {
    mmap: Arc<Mmap>,
}

impl MmapByteRangeSource {
    fn new(mmap: Arc<Mmap>) -> Self {
        Self { mmap }
    }
}

impl ByteRangeSource for MmapByteRangeSource {
    fn len(&self) -> u64 {
        self.mmap.len() as u64
    }

    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
        let offset = offset as usize;
        let end = offset
            .checked_add(dst.len())
            .ok_or_else(|| PipelineError::Io("Offset overflow while reading mmap".to_string()))?;
        if end > self.mmap.len() {
            return Err(PipelineError::Io(
                "Attempted to read past end of local .bed mapping".to_string(),
            ));
        }
        dst.copy_from_slice(&self.mmap[offset..end]);
        Ok(())
    }
}

struct LocalTextSource {
    reader: BufReader<File>,
    line: Vec<u8>,
    line_active: bool,
    len: u64,
    path_display: String,
}

impl LocalTextSource {
    fn new(path: &Path, file: File) -> Result<Self, PipelineError> {
        let len = file
            .metadata()
            .map_err(|e| PipelineError::Io(format!("Metadata for {}: {e}", path.display())))?
            .len();
        Ok(Self {
            reader: BufReader::new(file),
            line: Vec::with_capacity(1024),
            line_active: false,
            len,
            path_display: path.display().to_string(),
        })
    }
}

impl TextSource for LocalTextSource {
    fn len(&self) -> Option<u64> {
        Some(self.len)
    }

    fn next_line<'a>(&'a mut self) -> Result<Option<&'a [u8]>, PipelineError> {
        if self.line_active {
            self.line.clear();
            self.line_active = false;
        }

        let bytes_read = self
            .reader
            .read_until(b'\n', &mut self.line)
            .map_err(|e| PipelineError::Io(format!("Error reading {}: {e}", self.path_display)))?;

        if bytes_read == 0 {
            return Ok(None);
        }

        if self.line.last() == Some(&b'\n') {
            self.line.pop();
        }
        if self.line.last() == Some(&b'\r') {
            self.line.pop();
        }

        self.line_active = true;
        Ok(Some(&self.line))
    }
}

struct RemoteTextSource {
    path_display: String,
    source: Arc<RemoteByteRangeSource>,
    offset: u64,
    len: u64,
    buffer: Vec<u8>,
    cursor: usize,
    valid: usize,
    carry: Vec<u8>,
    carry_active: bool,
}

impl RemoteTextSource {
    fn new(path_display: String, source: Arc<RemoteByteRangeSource>) -> Self {
        let len = source.len();
        Self {
            path_display,
            source,
            offset: 0,
            len,
            buffer: Vec::with_capacity(REMOTE_BLOCK_SIZE),
            cursor: 0,
            valid: 0,
            carry: Vec::new(),
            carry_active: false,
        }
    }

    fn fill_buffer(&mut self) -> Result<bool, PipelineError> {
        if self.offset >= self.len {
            return Ok(false);
        }

        let remaining = self.len - self.offset;
        let to_read = remaining.min(REMOTE_BLOCK_SIZE as u64) as usize;
        if self.buffer.len() < to_read {
            self.buffer.resize(to_read, 0);
        }

        self.source
            .read_at(self.offset, &mut self.buffer[..to_read])
            .map_err(|e| augment_pipeline_error(e, &self.path_display))?;
        self.offset += to_read as u64;
        self.cursor = 0;
        self.valid = to_read;
        Ok(true)
    }
}

impl TextSource for RemoteTextSource {
    fn len(&self) -> Option<u64> {
        Some(self.len)
    }

    fn next_line<'a>(&'a mut self) -> Result<Option<&'a [u8]>, PipelineError> {
        if self.carry_active {
            self.carry.clear();
            self.carry_active = false;
        }

        loop {
            if self.cursor >= self.valid {
                if !self.fill_buffer()? {
                    if self.carry.is_empty() {
                        return Ok(None);
                    }
                    if self.carry.last() == Some(&b'\r') {
                        self.carry.pop();
                    }
                    self.carry_active = true;
                    return Ok(Some(&self.carry));
                }
            }

            if let Some(rel_pos) = self.buffer[self.cursor..self.valid]
                .iter()
                .position(|&b| b == b'\n')
            {
                let line_end = self.cursor + rel_pos;
                if self.carry.is_empty() {
                    let mut slice = &self.buffer[self.cursor..line_end];
                    if slice.last() == Some(&b'\r') {
                        slice = &slice[..slice.len() - 1];
                    }
                    self.cursor = line_end + 1;
                    return Ok(Some(slice));
                } else {
                    self.carry
                        .extend_from_slice(&self.buffer[self.cursor..line_end]);
                    if self.carry.last() == Some(&b'\r') {
                        self.carry.pop();
                    }
                    self.cursor = line_end + 1;
                    self.carry_active = true;
                    return Ok(Some(&self.carry));
                }
            } else {
                self.carry
                    .extend_from_slice(&self.buffer[self.cursor..self.valid]);
                self.cursor = self.valid;
            }
        }
    }
}

struct RemoteByteRangeSource {
    runtime: Arc<Runtime>,
    storage: Storage,
    bucket_path: String,
    object: String,
    user_project: Option<String>,
    len: u64,
    cache: Mutex<RemoteCache>,
}

impl RemoteByteRangeSource {
    fn new(bucket: &str, object: &str) -> Result<Self, PipelineError> {
        let runtime = get_shared_runtime()?;
        let (mut storage, control) = Self::create_clients(&runtime, None)?;
        let bucket_path = format!("projects/_/buckets/{bucket}");
        let user_project = gcs_billing_project_from_env();
        let metadata = match Self::fetch_object_metadata(&runtime, &control, &bucket_path, object) {
            Ok(metadata) => metadata,
            Err(err) if Self::is_authentication_error(&err) => {
                let err_msg = err.to_string();
                debug!(
                    "Retrying metadata fetch for gs://{bucket}/{object} with anonymous credentials after authentication failure: {err_msg}"
                );
                let (fallback_storage, fallback_control) = Self::create_clients(
                    &runtime,
                    Some(AnonymousCredentials::new().build()),
                )
                .map_err(|client_err| {
                    PipelineError::Io(format!(
                        "Failed to initialize Cloud Storage clients with anonymous credentials after authentication failure: {client_err} (initial error: {err_msg})"
                    ))
                })?;
                storage = fallback_storage;
                Self::fetch_object_metadata(&runtime, &fallback_control, &bucket_path, object)
                    .map_err(|retry_err| {
                        PipelineError::Io(format!(
                            "Failed to fetch metadata for gs://{bucket}/{object} with anonymous credentials: {retry_err} (initial error: {err_msg})"
                        ))
                    })?
            }
            Err(err) => {
                return Err(PipelineError::Io(format!(
                    "Failed to fetch metadata for gs://{bucket}/{object}: {err}"
                )));
            }
        };
        if metadata.size < 0 {
            return Err(PipelineError::Io(format!(
                "Remote object gs://{bucket}/{object} reported negative size"
            )));
        }
        let len = metadata.size as u64;
        debug!(
            "Initialized remote BED source gs://{bucket}/{object} ({} bytes)",
            len
        );
        Ok(Self {
            runtime,
            storage,
            bucket_path,
            object: object.to_string(),
            user_project,
            len,
            cache: Mutex::new(RemoteCache::new(REMOTE_CACHE_CAPACITY)),
        })
    }

    fn create_clients(
        runtime: &Arc<Runtime>,
        credentials: Option<Credentials>,
    ) -> Result<(Storage, StorageControl), PipelineError> {
        let base_credentials = match credentials {
            Some(creds) => creds,
            None => load_adc_credentials()?,
        };

        let storage_credentials = base_credentials.clone();
        let storage = runtime.block_on(async move {
            Storage::builder()
                .with_credentials(storage_credentials)
                .build()
                .await
                .map_err(|e| {
                    PipelineError::Io(format!("Failed to create Cloud Storage client: {e}"))
                })
        })?;

        let control = runtime.block_on(async move {
            StorageControl::builder()
                .with_credentials(base_credentials)
                .build()
                .await
                .map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to create Cloud Storage control client: {e}"
                    ))
                })
        })?;

        Ok((storage, control))
    }

    fn fetch_object_metadata(
        runtime: &Arc<Runtime>,
        control: &StorageControl,
        bucket_path: &str,
        object: &str,
    ) -> Result<google_cloud_storage::model::Object, google_cloud_storage::Error> {
        runtime.block_on(
            control
                .get_object()
                .set_bucket(bucket_path.to_string())
                .set_object(object.to_string())
                .send(),
        )
    }

    fn is_authentication_error(error: &google_cloud_storage::Error) -> bool {
        let message = error.to_string();
        message.contains("cannot create the authentication headers")
    }

    fn block_length(&self, start: u64) -> usize {
        let remaining = self.len.saturating_sub(start);
        remaining.min(REMOTE_BLOCK_SIZE as u64) as usize
    }

    fn ensure_block(&self, start: u64) -> Result<Arc<Vec<u8>>, PipelineError> {
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(block) = cache.get(start) {
                return Ok(block);
            }
        }

        let length = self.block_length(start);
        if length == 0 {
            return Err(PipelineError::Io(
                "Requested block beyond end of object".to_string(),
            ));
        }
        let data = self.fetch_block(start, length)?;
        let mut cache = self.cache.lock().unwrap();
        cache.insert(start, Arc::clone(&data));
        Ok(data)
    }

    fn fetch_block(&self, start: u64, length: usize) -> Result<Arc<Vec<u8>>, PipelineError> {
        let bucket_path = self.bucket_path.clone();
        let object = self.object.clone();
        let bucket_for_log = bucket_path.clone();
        let object_for_log = object.clone();
        let storage = self.storage.clone();
        let runtime = Arc::clone(&self.runtime);
        let user_project = self.user_project.clone();
        let mut data = runtime.block_on(async move {
            let mut response = storage
                .read_object(bucket_path.clone(), object.clone())
                .set_read_range(ReadRange::segment(start, length as u64))
                .send()
                .await
                .map_err(|e| {
                    let msg = e.to_string();
                    if user_project.is_none() && msg.to_lowercase().contains("requester pays") {
                        PipelineError::Io(format!(
                            "Requester Pays bucket requires a billing project. Set GOOGLE_PROJECT (or run `gcloud config set project ...`) and re-run. Original error: {msg}"
                        ))
                    } else {
                        PipelineError::Io(format!(
                            "Failed to start range read at offset {start} for gs://{}/{object}: {msg}",
                            bucket_path
                        ))
                    }
                })?;
            let mut buffer = Vec::with_capacity(length);
            while let Some(chunk) = response.next().await {
                let chunk = chunk.map_err(|e| {
                    PipelineError::Io(format!(
                        "Error streaming data from gs://{}/{object}: {e}",
                        bucket_path
                    ))
                })?;
                buffer.extend_from_slice(&chunk);
            }
            Ok::<Vec<u8>, PipelineError>(buffer)
        })?;

        if data.len() != length {
            warn!(
                "Remote range read returned {} bytes, expected {} (gs://{}/{})",
                data.len(),
                length,
                bucket_for_log,
                object_for_log
            );
            if data.len() < length {
                return Err(PipelineError::Io(format!(
                    "Remote range read from gs://{}/{object_for_log} truncated: expected {length} bytes, received {}",
                    bucket_for_log,
                    data.len()
                )));
            }
            data.truncate(length);
        }
        Ok(Arc::new(data))
    }
}

impl ByteRangeSource for RemoteByteRangeSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
        if dst.is_empty() {
            return Ok(());
        }
        if offset >= self.len {
            return Err(PipelineError::Io(format!(
                "Attempted to read past end of remote object at offset {offset}"
            )));
        }
        let end = offset.checked_add(dst.len() as u64).ok_or_else(|| {
            PipelineError::Io("Offset overflow while reading remote object".to_string())
        })?;
        if end > self.len {
            return Err(PipelineError::Io(format!(
                "Attempted to read past end of remote object (offset {offset}, len {})",
                dst.len()
            )));
        }

        let mut remaining = dst.len();
        let mut cursor = 0usize;
        let mut current_offset = offset;
        let block_size = REMOTE_BLOCK_SIZE as u64;

        while remaining > 0 {
            let block_start = (current_offset / block_size) * block_size;
            let block = self.ensure_block(block_start)?;
            let within_block = (current_offset - block_start) as usize;
            if within_block >= block.len() {
                return Err(PipelineError::Io(format!(
                    "Computed block offset {within_block} exceeds block size {}",
                    block.len()
                )));
            }
            let available = block.len() - within_block;
            let to_copy = available.min(remaining);
            dst[cursor..cursor + to_copy]
                .copy_from_slice(&block[within_block..within_block + to_copy]);
            cursor += to_copy;
            remaining -= to_copy;
            current_offset += to_copy as u64;

            if within_block + to_copy == block.len() {
                let next_start = block_start + block_size;
                if next_start < self.len {
                    let _ = self.ensure_block(next_start);
                }
            }
        }

        Ok(())
    }
}

struct RemoteStreamingReader {
    path_display: String,
    source: Arc<RemoteByteRangeSource>,
    offset: u64,
    len: u64,
    buffer: Vec<u8>,
    cursor: usize,
    valid: usize,
}

impl RemoteStreamingReader {
    fn new(path_display: String, source: Arc<RemoteByteRangeSource>) -> Self {
        let len = source.len();
        Self {
            path_display,
            source,
            offset: 0,
            len,
            buffer: Vec::with_capacity(REMOTE_BLOCK_SIZE),
            cursor: 0,
            valid: 0,
        }
    }

    fn ensure_buffer(&mut self) -> Result<bool, PipelineError> {
        if self.cursor < self.valid {
            return Ok(true);
        }
        if self.offset >= self.len {
            return Ok(false);
        }

        let remaining = self.len - self.offset;
        let to_read = remaining.min(REMOTE_BLOCK_SIZE as u64) as usize;
        if self.buffer.len() < to_read {
            self.buffer.resize(to_read, 0);
        }

        self.source
            .read_at(self.offset, &mut self.buffer[..to_read])
            .map_err(|e| augment_pipeline_error(e, &self.path_display))?;
        self.offset += to_read as u64;
        self.cursor = 0;
        self.valid = to_read;
        Ok(true)
    }

    fn peek_gzip_magic(&mut self) -> Result<bool, PipelineError> {
        if self.valid.saturating_sub(self.cursor) < 2 {
            while self.valid.saturating_sub(self.cursor) < 2 {
                if !self.ensure_buffer()? {
                    break;
                }
                if self.valid.saturating_sub(self.cursor) >= 2 {
                    break;
                }
            }
        }

        if self.valid.saturating_sub(self.cursor) >= 2 {
            Ok(self.buffer[self.cursor] == 0x1F && self.buffer[self.cursor + 1] == 0x8B)
        } else {
            Ok(false)
        }
    }
}

impl Read for RemoteStreamingReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        let mut total = 0;
        while total < buf.len() {
            if self.cursor >= self.valid {
                if !self
                    .ensure_buffer()
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?
                {
                    break;
                }
                if self.cursor >= self.valid {
                    continue;
                }
            }

            let available = self.valid - self.cursor;
            let to_copy = available.min(buf.len() - total);
            if to_copy == 0 {
                break;
            }

            buf[total..total + to_copy]
                .copy_from_slice(&self.buffer[self.cursor..self.cursor + to_copy]);
            self.cursor += to_copy;
            total += to_copy;
        }

        Ok(total)
    }
}

fn prepare_remote_bcf_reader(
    bucket: &str,
    object: &str,
    source: RemoteByteRangeSource,
) -> Result<PreparedBcfReader, PipelineError> {
    let len = source.len();
    let source = Arc::new(source);
    let mut reader =
        RemoteStreamingReader::new(format!("gs://{bucket}/{object}"), Arc::clone(&source));
    let is_gzip = reader.peek_gzip_magic()?;
    let reader: Box<dyn Read + Send> = if is_gzip {
        Box::new(MultiGzDecoder::new(reader))
    } else {
        Box::new(reader)
    };
    Ok(PreparedBcfReader {
        reader,
        len,
        skip_header: false,
    })
}

struct RemoteCache {
    capacity: usize,
    blocks: HashMap<u64, Arc<Vec<u8>>>,
    order: VecDeque<u64>,
}

impl RemoteCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            blocks: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, key: u64) -> Option<Arc<Vec<u8>>> {
        if let Some(value) = self.blocks.get(&key).cloned() {
            self.touch(key);
            Some(value)
        } else {
            None
        }
    }

    fn insert(&mut self, key: u64, value: Arc<Vec<u8>>) {
        if self.blocks.contains_key(&key) {
            self.blocks.insert(key, value);
            self.touch(key);
            return;
        }
        if self.order.len() == self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.blocks.remove(&oldest);
            }
        }
        self.order.push_back(key);
        self.blocks.insert(key, value);
    }

    fn touch(&mut self, key: u64) {
        self.order.retain(|&k| k != key);
        self.order.push_back(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    const PUBLIC_BUCKET: &str = "genomics-public-data";
    const PUBLIC_REFERENCE_OBJECT: &str = "references/hg38/v0/Homo_sapiens_assembly38.fasta";

    fn should_skip_remote_test(err: &PipelineError) -> bool {
        matches!(err,
            PipelineError::Io(msg)
                if msg.contains("The service is currently unavailable")
                    || msg.contains("tcp connect error")
                    || msg.contains("cannot create the authentication headers")
                    || msg.contains("dns error")
        )
    }

    fn remote_source_or_skip(
        test_name: &str,
    ) -> Result<Option<RemoteByteRangeSource>, PipelineError> {
        match RemoteByteRangeSource::new(PUBLIC_BUCKET, PUBLIC_REFERENCE_OBJECT) {
            Ok(source) => Ok(Some(source)),
            Err(err) => {
                if should_skip_remote_test(&err) {
                    eprintln!("Skipping {test_name}: {err}");
                    Ok(None)
                } else {
                    Err(err)
                }
            }
        }
    }

    #[test]
    fn remote_reference_exposes_length_and_initial_bytes() -> Result<(), PipelineError> {
        let source =
            match remote_source_or_skip("remote_reference_exposes_length_and_initial_bytes")? {
                Some(source) => source,
                None => return Ok(()),
            };
        assert!(
            source.len() > 0,
            "Remote reference must report a non-zero length"
        );

        let mut buffer = vec![0u8; 4096];
        source.read_at(0, &mut buffer)?;
        assert!(
            buffer.iter().any(|&byte| byte != 0),
            "Expected streamed data to contain non-zero bytes"
        );

        Ok(())
    }

    #[test]
    fn remote_reference_streams_across_multiple_blocks() -> Result<(), PipelineError> {
        let source = match remote_source_or_skip("remote_reference_streams_across_multiple_blocks")?
        {
            Some(source) => source,
            None => return Ok(()),
        };
        let read_len = REMOTE_BLOCK_SIZE + 1024;
        let mut buffer = vec![0u8; read_len];
        source.read_at(0, &mut buffer)?;
        assert!(
            buffer.iter().any(|&byte| byte != 0),
            "Expected streamed range to contain remote data"
        );

        Ok(())
    }

    #[test]
    fn remote_reference_supports_boundary_reads() -> Result<(), PipelineError> {
        let source = match remote_source_or_skip("remote_reference_supports_boundary_reads")? {
            Some(source) => source,
            None => return Ok(()),
        };
        let len = source.len();
        assert!(len > REMOTE_BLOCK_SIZE as u64);

        let offset = REMOTE_BLOCK_SIZE as u64 - 256;
        let mut buffer = vec![0u8; 1024];
        source.read_at(offset, &mut buffer)?;
        assert!(
            buffer.iter().any(|&byte| byte != 0),
            "Expected boundary read to yield streamed data"
        );

        let tail_offset = len.saturating_sub(512);
        let mut tail_buffer = vec![0u8; (len - tail_offset) as usize];
        source.read_at(tail_offset, &mut tail_buffer)?;
        assert!(
            tail_buffer.iter().any(|&byte| byte != 0),
            "Expected trailing read to yield streamed data"
        );

        Ok(())
    }

    fn env_mutex() -> &'static Mutex<()> {
        static ENV_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_MUTEX.get_or_init(|| Mutex::new(()))
    }

    struct EnvVarGuard {
        key: &'static str,
        original: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let original = env::var(key).ok();
            unsafe { env::set_var(key, value) };
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(original) = &self.original {
                unsafe { env::set_var(self.key, original) };
            } else {
                unsafe { env::remove_var(self.key) };
            }
        }
    }

    fn write_stub_adc_file(path: &Path) {
        const AUTHORIZED_USER: &str = r#"{
  "client_id": "test-client-id",
  "client_secret": "test-client-secret",
  "refresh_token": "test-refresh-token",
  "token_uri": "https://oauth2.googleapis.com/token",
  "type": "authorized_user"
}"#;
        std::fs::write(path, AUTHORIZED_USER).expect("failed to write stub ADC file");
    }

    #[test]
    fn load_adc_credentials_without_runtime_supports_storage_client() -> Result<(), PipelineError> {
        let env_lock = env_mutex().lock().unwrap();

        let adc_file = NamedTempFile::new().expect("failed to create temp ADC file");
        write_stub_adc_file(adc_file.path());

        let adc_guard = EnvVarGuard::set(
            "GOOGLE_APPLICATION_CREDENTIALS",
            adc_file
                .path()
                .to_str()
                .expect("temporary path should be valid UTF-8"),
        );

        assert!(tokio::runtime::Handle::try_current().is_err());

        let credentials = load_adc_credentials()?;

        let runtime = get_shared_runtime()?;
        let storage_control = runtime.block_on(async move {
            StorageControl::builder()
                .with_credentials(credentials.clone())
                .build()
                .await
                .map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to create Cloud Storage control client for test: {e}"
                    ))
                })
        })?;

        // Ensure the credentials are exercised in the same way as the CLI, which
        // constructs a storage client after loading ADC credentials without an
        // existing runtime.
        let _ = storage_control;

        drop(adc_guard);
        drop(env_lock);

        Ok(())
    }
}
