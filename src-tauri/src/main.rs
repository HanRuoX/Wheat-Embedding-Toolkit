// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use btleplug::api::Peripheral;
use btleplug::api::{Central, CentralEvent, Manager as _, ScanFilter};
use btleplug::platform::{Adapter, Manager};
use futures::stream::StreamExt;
use serialport::{available_ports, DataBits, FlowControl, Parity, StopBits};
use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::mpsc::{self, Sender, SyncSender, TryRecvError};
use std::sync::{Mutex, OnceLock};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant, SystemTime};
use tauri::{Emitter, Listener, Manager as _};

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct BleDevice {
    pub address: String,
    pub local_name: String,
    pub rssi: i16,
    pub manufacturer_data: HashMap<u16, Vec<u8>>,
    pub services: Vec<String>,
    pub service_data: HashMap<String, Vec<u8>>,
    pub adv: Vec<u8>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct FileInfo {
    name: String,
    is_dir: bool,
    is_file: bool,
    len: u64,
    create_time: u64,
}

#[derive(serde::Serialize, Debug, Clone)]
struct AudioInfo {
    path: String,
    file_name: String,
    format_name: String,
    format_long_name: String,
    duration: f64,
    size: u64,
    overall_bit_rate: Option<u64>,
    codec_name: String,
    codec_long_name: String,
    sample_rate: Option<u32>,
    channels: Option<u32>,
    channel_layout: String,
    stream_bit_rate: Option<u64>,
    tags: HashMap<String, String>,
}

#[derive(serde::Serialize, Debug, Clone)]
struct AudioConversionResult {
    output_path: String,
    info: AudioInfo,
}

#[derive(serde::Serialize, Debug, Clone)]
struct AudioSourceResult {
    playback_path: String,
    info: AudioInfo,
}

#[derive(Default, Clone)]
struct BinaryPathState {
    resource_dir: Option<PathBuf>,
    workspace_staged_bin_dir: Option<PathBuf>,
    workspace_source_bin_dir: Option<PathBuf>,
}

static BINARY_PATH_STATE: OnceLock<BinaryPathState> = OnceLock::new();

fn log_audio(message: impl AsRef<str>) {
    println!("[audio] {}", message.as_ref());
}

fn detect_workspace_staged_bin_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(current_dir) = env::current_dir() {
        candidates.push(current_dir.join("src-tauri").join("binaries"));
        candidates.push(current_dir.join("binaries"));
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(manifest_dir.join("binaries"));

    candidates.into_iter().find(|dir| dir.exists())
}

fn detect_workspace_source_bin_dir() -> Option<PathBuf> {
    let mut candidates = Vec::new();

    if let Ok(current_dir) = env::current_dir() {
        candidates.push(current_dir.join("src-tauri").join("bin"));
        candidates.push(current_dir.join("bin"));
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(manifest_dir.join("bin"));

    candidates.into_iter().find(|dir| dir.exists())
}

fn source_binary_relative_paths(binary_name: &str) -> Vec<PathBuf> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        vec![
            PathBuf::from("macos-arm64").join(binary_name),
            PathBuf::from("macos-x64").join(binary_name),
            PathBuf::from(binary_name),
        ]
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        vec![
            PathBuf::from("macos-x64").join(binary_name),
            PathBuf::from("macos-arm64").join(binary_name),
            PathBuf::from(binary_name),
        ]
    } else {
        vec![PathBuf::from(binary_name)]
    }
}

fn sidecar_binary_name(binary_name: &str) -> String {
    #[cfg(target_os = "windows")]
    {
        format!("{binary_name}-{}.exe", env!("TAURI_ENV_TARGET_TRIPLE"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        format!("{binary_name}-{}", env!("TAURI_ENV_TARGET_TRIPLE"))
    }
}

struct SerialSession {
    command_tx: SyncSender<SerialCommand>,
    stop_tx: Sender<()>,
    closing: Arc<AtomicBool>,
    reader_handle: Option<thread::JoinHandle<()>>,
    writer_handle: Option<thread::JoinHandle<()>>,
}

#[derive(Default)]
struct SerialAssistantState {
    sessions: Mutex<HashMap<String, SerialSession>>,
}

#[derive(serde::Serialize, Clone)]
struct SerialAssistantEvent {
    kind: String,
    text: String,
    hex: String,
}

enum SerialCommand {
    Send {
        data: Vec<u8>,
        response_tx: Sender<Result<usize, String>>,
    },
    SetSignals {
        rts: bool,
        dtr: bool,
        response_tx: Sender<Result<(), String>>,
    },
    Shutdown,
}

impl SerialAssistantEvent {
    fn status(text: impl Into<String>) -> Self {
        Self {
            kind: "status".to_string(),
            text: text.into(),
            hex: String::new(),
        }
    }

    fn error(text: impl Into<String>) -> Self {
        Self {
            kind: "error".to_string(),
            text: text.into(),
            hex: String::new(),
        }
    }

    fn data(bytes: &[u8]) -> Self {
        let hex = bytes
            .iter()
            .map(|b| format!("{b:02X}"))
            .collect::<Vec<_>>()
            .join(" ");
        let text = String::from_utf8_lossy(bytes).to_string();
        Self {
            kind: "data".to_string(),
            text,
            hex,
        }
    }
}

fn parse_data_bits(bits: u8) -> Result<DataBits, String> {
    match bits {
        5 => Ok(DataBits::Five),
        6 => Ok(DataBits::Six),
        7 => Ok(DataBits::Seven),
        8 => Ok(DataBits::Eight),
        _ => Err("data_bits must be one of: 5, 6, 7, 8".to_string()),
    }
}

fn parse_stop_bits(bits: u8) -> Result<StopBits, String> {
    match bits {
        1 => Ok(StopBits::One),
        2 => Ok(StopBits::Two),
        _ => Err("stop_bits must be one of: 1, 2".to_string()),
    }
}

fn parse_parity(parity: &str) -> Result<Parity, String> {
    match parity.to_ascii_lowercase().as_str() {
        "none" => Ok(Parity::None),
        "odd" => Ok(Parity::Odd),
        "even" => Ok(Parity::Even),
        _ => Err("parity must be one of: none, odd, even".to_string()),
    }
}

fn parse_flow_control(flow_control: &str) -> Result<FlowControl, String> {
    match flow_control.to_ascii_lowercase().as_str() {
        "none" => Ok(FlowControl::None),
        "software" => Ok(FlowControl::Software),
        "hardware" => Ok(FlowControl::Hardware),
        _ => Err("flow_control must be one of: none, software, hardware".to_string()),
    }
}

fn close_serial_session(state: &SerialAssistantState, label: &str) -> Result<(), String> {
    let session = {
        let mut sessions = state
            .sessions
            .lock()
            .map_err(|_| "failed to lock serial sessions".to_string())?;
        sessions.remove(label)
    };

    if let Some(session) = session {
        shutdown_serial_session(session, true);
    }

    Ok(())
}

fn shutdown_serial_session(mut session: SerialSession, wait_for_threads: bool) {
    session.closing.store(true, Ordering::SeqCst);
    let _ = session.command_tx.send(SerialCommand::Shutdown);
    let _ = session.stop_tx.send(());

    let writer_handle = session.writer_handle.take();
    let reader_handle = session.reader_handle.take();

    if wait_for_threads {
        if let Some(handle) = writer_handle {
            let _ = handle.join();
        }
        if let Some(handle) = reader_handle {
            let _ = handle.join();
        }
        return;
    }

    thread::spawn(move || {
        if let Some(handle) = writer_handle {
            let _ = handle.join();
        }
        if let Some(handle) = reader_handle {
            let _ = handle.join();
        }
    });
}

fn wait_for_output_drain(
    writer_port: &mut Box<dyn serialport::SerialPort>,
    closing: &AtomicBool,
) -> Result<(), String> {
    let start = Instant::now();
    let max_wait = Duration::from_millis(300);

    loop {
        if closing.load(Ordering::SeqCst) {
            return Err("serial is closing".to_string());
        }

        match writer_port.bytes_to_write() {
            Ok(0) => return Ok(()),
            Ok(_) => {}
            Err(_) => return Ok(()),
        }

        if start.elapsed() >= max_wait {
            return Ok(());
        }

        thread::sleep(Duration::from_millis(2));
    }
}

fn resolve_binary_path(binary: &str) -> Result<String, String> {
    let env_key = format!("{}_BIN", binary.to_ascii_uppercase());
    if let Ok(path) = env::var(&env_key) {
        if !path.trim().is_empty() {
            log_audio(format!("resolved {binary} from env {env_key}: {path}"));
            return Ok(path);
        }
    }

    let executable_name = {
        #[cfg(target_os = "windows")]
        {
            format!("{binary}.exe")
        }
        #[cfg(not(target_os = "windows"))]
        {
            binary.to_string()
        }
    };

    let mut bundled_candidates: Vec<PathBuf> = Vec::new();

        if let Some(state) = BINARY_PATH_STATE.get() {
        if let Some(resource_dir) = &state.resource_dir {
            bundled_candidates.push(resource_dir.join(sidecar_binary_name(binary)));
            bundled_candidates.push(resource_dir.join("binaries").join(sidecar_binary_name(binary)));
            bundled_candidates.push(resource_dir.join("bin").join(&executable_name));
            bundled_candidates.push(resource_dir.join(&executable_name));
        }
        if let Some(workspace_staged_bin_dir) = &state.workspace_staged_bin_dir {
            bundled_candidates.push(workspace_staged_bin_dir.join(sidecar_binary_name(binary)));
        }
        if let Some(workspace_source_bin_dir) = &state.workspace_source_bin_dir {
            for relative_path in source_binary_relative_paths(&executable_name) {
                bundled_candidates.push(workspace_source_bin_dir.join(relative_path));
            }
        }
    }

    if let Some(candidate) = bundled_candidates
        .into_iter()
        .find(|candidate| candidate.exists())
    {
        let resolved = candidate.display().to_string();
        log_audio(format!("resolved {binary} from bundled path: {resolved}"));
        return Ok(resolved);
    }

    let path_probe = Command::new(&executable_name).arg("-version").output();
    if let Ok(output) = path_probe {
        if output.status.success() {
            log_audio(format!("resolved {binary} from PATH: {executable_name}"));
            return Ok(executable_name);
        }
    }

    Err(format!(
        "{binary} not found. Install system {binary} or bundle `{}` into `src-tauri/bin` so the packaged app can use it without a local install.",
        executable_name
    ))
}

#[cfg(target_os = "macos")]
fn should_run_under_rosetta(binary_path: &str) -> bool {
    if !cfg!(target_arch = "aarch64") || !Path::new(binary_path).is_file() {
        return false;
    }

    let output = Command::new("/usr/bin/file").arg(binary_path).output();
    let Ok(output) = output else {
        return false;
    };
    if !output.status.success() {
        return false;
    }

    let description = String::from_utf8_lossy(&output.stdout);
    description.contains("Mach-O") && description.contains("x86_64") && !description.contains("arm64")
}

fn build_binary_command(binary_path: &str) -> Command {
    #[cfg(target_os = "macos")]
    {
        if should_run_under_rosetta(binary_path) {
            log_audio(format!("launching x86_64 binary via Rosetta: {binary_path}"));
            let mut command = Command::new("arch");
            command.arg("-x86_64").arg(binary_path);
            return command;
        }
    }

    Command::new(binary_path)
}

fn parse_optional_u64(value: Option<&Value>) -> Option<u64> {
    value
        .and_then(Value::as_str)
        .and_then(|raw| raw.parse::<u64>().ok())
        .or_else(|| value.and_then(Value::as_u64))
}

fn parse_optional_u32(value: Option<&Value>) -> Option<u32> {
    value
        .and_then(Value::as_str)
        .and_then(|raw| raw.parse::<u32>().ok())
        .or_else(|| value.and_then(Value::as_u64).and_then(|raw| u32::try_from(raw).ok()))
}

fn parse_duration(value: Option<&Value>) -> f64 {
    value
        .and_then(Value::as_str)
        .and_then(|raw| raw.parse::<f64>().ok())
        .or_else(|| value.and_then(Value::as_f64))
        .unwrap_or(0.0)
}

fn extract_tags(value: Option<&Value>) -> HashMap<String, String> {
    value
        .and_then(Value::as_object)
        .map(|tags| {
            tags.iter()
                .map(|(key, value)| {
                    let rendered = value
                        .as_str()
                        .map(|item| item.to_string())
                        .unwrap_or_else(|| value.to_string());
                    (key.to_string(), rendered)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn build_audio_info(path: &str, probe_json: &Value) -> AudioInfo {
    let null = Value::Null;
    let format = probe_json.get("format").unwrap_or(&null);
    let stream = probe_json
        .get("streams")
        .and_then(Value::as_array)
        .and_then(|streams| {
            streams
                .iter()
                .find(|item| item.get("codec_type").and_then(Value::as_str) == Some("audio"))
        })
        .unwrap_or(&null);

    AudioInfo {
        path: path.to_string(),
        file_name: Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(path)
            .to_string(),
        format_name: format
            .get("format_name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        format_long_name: format
            .get("format_long_name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        duration: parse_duration(format.get("duration")),
        size: parse_optional_u64(format.get("size")).unwrap_or(0),
        overall_bit_rate: parse_optional_u64(format.get("bit_rate")),
        codec_name: stream
            .get("codec_name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        codec_long_name: stream
            .get("codec_long_name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        sample_rate: parse_optional_u32(stream.get("sample_rate")),
        channels: parse_optional_u32(stream.get("channels")),
        channel_layout: stream
            .get("channel_layout")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        stream_bit_rate: parse_optional_u64(stream.get("bit_rate")),
        tags: extract_tags(format.get("tags")),
    }
}

fn read_audio_info(path: &str) -> Result<AudioInfo, String> {
    if !Path::new(path).exists() {
        return Err(format!("audio file does not exist: {path}"));
    }

    let ffprobe = resolve_binary_path("ffprobe")?;
    log_audio(format!("reading audio info via ffprobe: input={path}"));
    let mut command = build_binary_command(&ffprobe);
    let output = command
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ])
        .output()
        .map_err(|e| format!("failed to execute ffprobe ({ffprobe}): {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        log_audio(format!(
            "ffprobe failed for input={path}: {}",
            if stderr.is_empty() {
                "ffprobe failed to read audio info"
            } else {
                stderr.as_str()
            }
        ));
        return Err(if stderr.is_empty() {
            "ffprobe failed to read audio info".to_string()
        } else {
            stderr
        });
    }

    let probe_json: Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("failed to parse ffprobe output: {e}"))?;

    log_audio(format!("ffprobe succeeded: input={path}"));
    Ok(build_audio_info(path, &probe_json))
}

fn normalize_input_format(input_format: Option<&str>, path: &str) -> Option<String> {
    input_format
        .map(|item| item.trim().to_ascii_lowercase())
        .filter(|item| !item.is_empty())
        .or_else(|| {
            Path::new(path)
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.trim().to_ascii_lowercase())
                .filter(|ext| !ext.is_empty())
        })
}

fn validate_pcm_params(sample_rate: Option<u32>, channels: Option<u32>) -> Result<(u32, u32), String> {
    let sample_rate = sample_rate.ok_or_else(|| "pcm input requires sample rate".to_string())?;
    let channels = channels.ok_or_else(|| "pcm input requires channel count".to_string())?;
    if sample_rate == 0 {
        return Err("sample_rate must be greater than 0".to_string());
    }
    if channels == 0 {
        return Err("channels must be greater than 0".to_string());
    }
    Ok((sample_rate, channels))
}

fn resolve_input_audio_info(
    path: &str,
    input_format: Option<&str>,
    sample_rate: Option<u32>,
    channels: Option<u32>,
) -> Result<AudioInfo, String> {
    match normalize_input_format(input_format, path).as_deref() {
        Some("pcm") => {
            let (sample_rate, channels) = validate_pcm_params(sample_rate, channels)?;
            build_pcm_audio_info(path, sample_rate, channels)
        }
        _ => read_audio_info(path),
    }
}

fn conversion_codec_args(format: &str) -> Result<Vec<&'static str>, String> {
    match format.to_ascii_lowercase().as_str() {
        "mp3" => Ok(vec!["-codec:a", "libmp3lame"]),
        "wav" => Ok(vec!["-codec:a", "pcm_s16le"]),
        "flac" => Ok(vec!["-codec:a", "flac"]),
        "ogg" => Ok(vec!["-codec:a", "libvorbis"]),
        "opus" => Ok(vec!["-codec:a", "libopus", "-f", "opus"]),
        "pcm" => Ok(vec!["-codec:a", "pcm_s16le", "-f", "s16le"]),
        "aac" => Ok(vec!["-codec:a", "aac"]),
        "m4a" => Ok(vec!["-codec:a", "aac", "-movflags", "+faststart"]),
        other => Err(format!("unsupported output format: {other}")),
    }
}

fn run_ffmpeg_audio_job(
    input_args: &[String],
    input_path: &str,
    output_path: &str,
    codec_args: &[&str],
    extra_args: &[String],
) -> Result<(), String> {
    let output_parent = Path::new(output_path)
        .parent()
        .ok_or_else(|| "invalid output path".to_string())?;

    if !output_parent.exists() {
        fs::create_dir_all(output_parent)
            .map_err(|e| format!("failed to create output directory: {e}"))?;
    }

    let ffmpeg = resolve_binary_path("ffmpeg")?;
    log_audio(format!(
        "starting ffmpeg job: input={input_path}, output={output_path}, input_args={input_args:?}, extra_args={extra_args:?}, codec_args={codec_args:?}"
    ));
    let mut command = build_binary_command(&ffmpeg);
    command.arg("-y");
    for arg in input_args {
        command.arg(arg);
    }
    command.arg("-i");
    command.arg(input_path);
    for arg in extra_args {
        command.arg(arg);
    }
    command.arg("-vn");
    command.args(codec_args);
    command.arg(output_path);

    let output = command
        .output()
        .map_err(|e| format!("failed to execute ffmpeg ({ffmpeg}): {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        log_audio(format!(
            "ffmpeg job failed: input={input_path}, output={output_path}, {}",
            if stderr.is_empty() {
                "ffmpeg job failed"
            } else {
                stderr.as_str()
            }
        ));
        return Err(if stderr.is_empty() {
            "ffmpeg job failed".to_string()
        } else {
            stderr
        });
    }

    log_audio(format!("ffmpeg job succeeded: input={input_path}, output={output_path}"));
    Ok(())
}

fn build_audio_transform_args(
    sample_rate: Option<u32>,
    channels: Option<u32>,
) -> Result<Vec<String>, String> {
    let mut args = Vec::new();

    if let Some(sample_rate) = sample_rate {
        if sample_rate == 0 {
            return Err("sample_rate must be greater than 0".to_string());
        }
        args.push("-ar".to_string());
        args.push(sample_rate.to_string());
    }

    if let Some(channels) = channels {
        if channels == 0 {
            return Err("channels must be greater than 0".to_string());
        }
        args.push("-ac".to_string());
        args.push(channels.to_string());
    }

    Ok(args)
}

fn build_input_audio_args(
    input_format: Option<&str>,
    path: &str,
    sample_rate: Option<u32>,
    channels: Option<u32>,
) -> Result<Vec<String>, String> {
    match normalize_input_format(input_format, path).as_deref() {
        Some("pcm") => {
            let (sample_rate, channels) = validate_pcm_params(sample_rate, channels)?;
            Ok(vec![
                "-f".to_string(),
                "s16le".to_string(),
                "-ar".to_string(),
                sample_rate.to_string(),
                "-ac".to_string(),
                channels.to_string(),
            ])
        }
        _ => Ok(Vec::new()),
    }
}

fn build_pcm_audio_info(
    path: &str,
    sample_rate: u32,
    channels: u32,
) -> Result<AudioInfo, String> {
    let metadata = fs::metadata(path).map_err(|e| format!("failed to read output file metadata: {e}"))?;
    let size = metadata.len();
    let bytes_per_second = u64::from(sample_rate) * u64::from(channels) * 2;
    let duration = if bytes_per_second == 0 {
        0.0
    } else {
        size as f64 / bytes_per_second as f64
    };

    Ok(AudioInfo {
        path: path.to_string(),
        file_name: Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(path)
            .to_string(),
        format_name: "s16le".to_string(),
        format_long_name: "PCM signed 16-bit little-endian".to_string(),
        duration,
        size,
        overall_bit_rate: Some(bytes_per_second * 8),
        codec_name: "pcm_s16le".to_string(),
        codec_long_name: "PCM signed 16-bit little-endian".to_string(),
        sample_rate: Some(sample_rate),
        channels: Some(channels),
        channel_layout: match channels {
            1 => "mono".to_string(),
            2 => "stereo".to_string(),
            _ => String::new(),
        },
        stream_bit_rate: Some(bytes_per_second * 8),
        tags: HashMap::new(),
    })
}

fn create_pcm_wav_preview(
    app_handle: &tauri::AppHandle,
    input_path: &str,
    sample_rate: u32,
    channels: u32,
) -> Result<String, String> {
    let metadata = fs::metadata(input_path)
        .map_err(|e| format!("failed to inspect pcm file metadata: {e}"))?;
    let mut hasher = DefaultHasher::new();
    input_path.hash(&mut hasher);
    sample_rate.hash(&mut hasher);
    channels.hash(&mut hasher);
    metadata.len().hash(&mut hasher);
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
        .hash(&mut hasher);
    let hash = hasher.finish();

    let app_dir = app_handle.path().app_data_dir().unwrap_or_else(|_| {
        let mut fallback = env::temp_dir();
        fallback.push("wheat-embedding-toolkit");
        fallback
    });
    let preview_dir = app_dir.join("audio").join("preview");
    if !preview_dir.exists() {
        fs::create_dir_all(&preview_dir)
            .map_err(|e| format!("failed to create pcm preview directory: {e}"))?;
    }

    let base_name = Path::new(input_path)
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("audio");
    let output_path = preview_dir.join(format!("{base_name}-{hash:016x}.wav"));
    let data_len = metadata.len();
    let riff_chunk_size = 36u64 + data_len;
    let byte_rate = u64::from(sample_rate) * u64::from(channels) * 2;
    let block_align = channels * 2;

    let mut output = fs::File::create(&output_path)
        .map_err(|e| format!("failed to create pcm preview wav: {e}"))?;
    output
        .write_all(b"RIFF")
        .and_then(|_| output.write_all(&(riff_chunk_size as u32).to_le_bytes()))
        .and_then(|_| output.write_all(b"WAVE"))
        .and_then(|_| output.write_all(b"fmt "))
        .and_then(|_| output.write_all(&16u32.to_le_bytes()))
        .and_then(|_| output.write_all(&1u16.to_le_bytes()))
        .and_then(|_| output.write_all(&(channels as u16).to_le_bytes()))
        .and_then(|_| output.write_all(&sample_rate.to_le_bytes()))
        .and_then(|_| output.write_all(&(byte_rate as u32).to_le_bytes()))
        .and_then(|_| output.write_all(&(block_align as u16).to_le_bytes()))
        .and_then(|_| output.write_all(&16u16.to_le_bytes()))
        .and_then(|_| output.write_all(b"data"))
        .and_then(|_| output.write_all(&(data_len as u32).to_le_bytes()))
        .map_err(|e| format!("failed to write wav header: {e}"))?;

    let mut input = fs::File::open(input_path)
        .map_err(|e| format!("failed to open pcm source file: {e}"))?;
    std::io::copy(&mut input, &mut output)
        .map_err(|e| format!("failed to copy pcm payload into wav preview: {e}"))?;

    Ok(output_path.display().to_string())
}

fn build_output_audio_info(
    output_path: &str,
    output_format: &str,
    sample_rate: Option<u32>,
    channels: Option<u32>,
    source_info: &AudioInfo,
) -> Result<AudioInfo, String> {
    match output_format.to_ascii_lowercase().as_str() {
        "pcm" => {
            let resolved_sample_rate = sample_rate
                .or(source_info.sample_rate)
                .ok_or_else(|| "pcm output requires a known sample rate".to_string())?;
            let resolved_channels = channels
                .or(source_info.channels)
                .ok_or_else(|| "pcm output requires a known channel count".to_string())?;
            build_pcm_audio_info(output_path, resolved_sample_rate, resolved_channels)
        }
        _ => read_audio_info(output_path),
    }
}

fn prepare_audio_source_impl(
    app_handle: &tauri::AppHandle,
    path: &str,
    input_format: Option<&str>,
    sample_rate: Option<u32>,
    channels: Option<u32>,
) -> Result<AudioSourceResult, String> {
    log_audio(format!(
        "preparing audio source: input={path}, input_format={:?}, sample_rate={:?}, channels={:?}",
        input_format, sample_rate, channels
    ));
    let info = resolve_input_audio_info(path, input_format, sample_rate, channels)?;
    let playback_path = match normalize_input_format(input_format, path).as_deref() {
        Some("pcm") => {
            let (sample_rate, channels) = validate_pcm_params(sample_rate, channels)?;
            create_pcm_wav_preview(app_handle, path, sample_rate, channels)?
        }
        _ => path.to_string(),
    };

    log_audio(format!(
        "audio source prepared: input={path}, playback_path={playback_path}"
    ));
    Ok(AudioSourceResult { playback_path, info })
}

async fn get_central(manager: &Manager) -> Result<Adapter, String> {
    let adapters = manager
        .adapters()
        .await
        .map_err(|e| format!("failed to get bluetooth adapters: {e}"))?;
    adapters
        .into_iter()
        .next()
        .ok_or_else(|| "no bluetooth adapter found".to_string())
}

#[tauri::command]
async fn start_ble_advertisement_scan(window: tauri::Window) -> Result<(), String> {
    let (tx, rx) = mpsc::channel();

    let manager = Manager::new()
        .await
        .map_err(|e| format!("failed to create bluetooth manager: {e}"))?;

    // get the first bluetooth adapter
    // connect to the adapter
    let central = get_central(&manager).await?;

    // Each adapter has an event stream, we fetch via events(),
    // simplifying the type, this will return what is essentially a
    // Future<Result<Stream<Item=CentralEvent>>>.
    let mut events = central
        .events()
        .await
        .map_err(|e| format!("failed to subscribe bluetooth events: {e}"))?;

    // start scanning for devices
    central
        .start_scan(ScanFilter::default())
        .await
        .map_err(|e| format!("failed to start bluetooth scan: {e}"))?;

    let listen = window.listen("stop_ble_advertisement_scan", move |_event| {
        let _ = tx.send(());
    });
    let mut listener_active = true;

    // Print based on whatever the event receiver outputs. Note that the event
    // receiver blocks, so in a real program, this should be run in its own
    // thread (not task, as this library does not yet use async channels).
    while let Some(event) = events.next().await {
        match rx.try_recv() {
            Ok(_) | Err(TryRecvError::Disconnected) => {
                window.unlisten(listen);
                listener_active = false;
                break;
            }
            Err(TryRecvError::Empty) => {}
        }
        match event {
            CentralEvent::DeviceConnected(_id) => {}
            CentralEvent::DeviceDisconnected(_id) => {}
            CentralEvent::DeviceDiscovered(id)
            | CentralEvent::ManufacturerDataAdvertisement { id, .. }
            | CentralEvent::ServiceDataAdvertisement { id, .. }
            | CentralEvent::ServicesAdvertisement { id, .. } => {
                let peripheral = central.peripheral(&id).await;
                match peripheral {
                    Ok(peripheral) => {
                        let device = match peripheral.properties().await {
                            Ok(Some(device)) => device,
                            _ => continue,
                        };
                        let addr = id.to_string();
                        let mr = BleDevice {
                            address: addr,
                            local_name: device.local_name.unwrap_or(String::from("")),
                            rssi: device.rssi.unwrap_or(0),
                            manufacturer_data: device.manufacturer_data,
                            services: device.services.iter().map(|x| x.to_string()).collect(),
                            service_data: device
                                .service_data
                                .iter()
                                .map(|(x, y)| (x.to_string(), y.clone()))
                                .collect(),
                            adv: device
                                .service_data
                                .iter()
                                .flat_map(|x| x.1.clone())
                                .collect(),
                        };

                        if let Ok(payload) = serde_json::to_string(&mr) {
                            let _ = window.emit("ble_advertisement_scan_event", payload);
                        }
                    }
                    Err(_) => {}
                }
            }
            _ => {}
        }

        // thread::sleep(Duration::from_millis(100));
    }

    if listener_active {
        window.unlisten(listen);
    }
    Ok(())
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
fn get_serial_port_list() -> Vec<String> {
    match available_ports() {
        Ok(port_info_list) => port_info_list
            .iter()
            .map(|x| x.port_name.to_string())
            .collect(),
        Err(_) => Vec::new(),
    }
}

#[tauri::command]
fn serial_assistant_open(
    window: tauri::Window,
    state: tauri::State<SerialAssistantState>,
    port: String,
    baud_rate: u32,
    data_bits: u8,
    stop_bits: u8,
    parity: String,
    flow_control: String,
) -> Result<(), String> {
    if port.is_empty() {
        return Err("port is required".to_string());
    }

    close_serial_session(&state, window.label())?;

    let data_bits = parse_data_bits(data_bits)?;
    let stop_bits = parse_stop_bits(stop_bits)?;
    let parity = parse_parity(&parity)?;
    let flow_control = parse_flow_control(&flow_control)?;

    let serial = serialport::new(&port, baud_rate)
        .data_bits(data_bits)
        .stop_bits(stop_bits)
        .parity(parity)
        .flow_control(flow_control)
        .timeout(Duration::from_millis(100))
        .open()
        .map_err(|e| format!("failed to open serial port: {e}"))?;

    let mut reader_port = serial
        .try_clone()
        .map_err(|e| format!("failed to clone serial port: {e}"))?;
    let mut writer_port = serial;
    let (command_tx, command_rx) = mpsc::sync_channel::<SerialCommand>(1);
    let (stop_tx, stop_rx) = mpsc::channel();
    let closing = Arc::new(AtomicBool::new(false));
    let writer_closing = Arc::clone(&closing);
    let cloned_window = window.clone();
    let port_for_reader = port.clone();

    let reader_handle = thread::spawn(move || {
        let mut buffer = [0u8; 4096];
        loop {
            match stop_rx.try_recv() {
                Ok(_) | Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => {}
            }

            match reader_port.read(&mut buffer) {
                Ok(size) if size > 0 => {
                    let payload = SerialAssistantEvent::data(&buffer[..size]);
                    let _ = cloned_window.emit("serial_assistant_event", payload);
                }
                Ok(_) => {}
                Err(err) if err.kind() == std::io::ErrorKind::TimedOut => {}
                Err(err) => {
                    let payload = SerialAssistantEvent::error(format!(
                        "serial read failed ({port_for_reader}): {err}"
                    ));
                    let _ = cloned_window.emit("serial_assistant_event", payload);
                    break;
                }
            }
        }
    });

    let writer_handle = thread::spawn(move || {
        while let Ok(command) = command_rx.recv() {
            match command {
                SerialCommand::Send { data, response_tx } => {
                    let result = writer_port
                        .write_all(&data)
                        .map_err(|e| format!("failed to write serial data: {e}"))
                        .and_then(|_| wait_for_output_drain(&mut writer_port, writer_closing.as_ref()))
                        .map(|_| data.len())
                        .map_err(|e| e.to_string());
                    let _ = response_tx.send(result);
                }
                SerialCommand::SetSignals {
                    rts,
                    dtr,
                    response_tx,
                } => {
                    let result = writer_port
                        .write_request_to_send(rts)
                        .map_err(|e| format!("failed to set RTS: {e}"))
                        .and_then(|_| {
                            writer_port
                                .write_data_terminal_ready(dtr)
                                .map_err(|e| format!("failed to set DTR: {e}"))
                        });
                    let _ = response_tx.send(result);
                }
                SerialCommand::Shutdown => break,
            }
        }
    });

    {
        let mut sessions = state
            .sessions
            .lock()
            .map_err(|_| "failed to lock serial sessions".to_string())?;
        sessions.insert(
            window.label().to_string(),
            SerialSession {
                command_tx,
                stop_tx,
                closing,
                reader_handle: Some(reader_handle),
                writer_handle: Some(writer_handle),
            },
        );
    }

    let _ = window.emit(
        "serial_assistant_event",
        SerialAssistantEvent::status(format!("serial connected: {port} @ {baud_rate}")),
    );
    Ok(())
}

#[tauri::command]
fn serial_assistant_set_signals(
    window: tauri::Window,
    state: tauri::State<SerialAssistantState>,
    rts: bool,
    dtr: bool,
) -> Result<(), String> {
    let command_tx = {
        let sessions = state
            .sessions
            .lock()
            .map_err(|_| "failed to lock serial sessions".to_string())?;
        let session = sessions
            .get(window.label())
            .ok_or_else(|| "serial is not connected".to_string())?;
        session.command_tx.clone()
    };

    let (response_tx, response_rx) = mpsc::channel();
    command_tx
        .send(SerialCommand::SetSignals {
            rts,
            dtr,
            response_tx,
        })
        .map_err(|_| "serial writer is unavailable".to_string())?;

    response_rx
        .recv()
        .map_err(|_| "serial writer did not respond".to_string())?
}

#[tauri::command]
fn serial_assistant_send(
    window: tauri::Window,
    state: tauri::State<SerialAssistantState>,
    data: Vec<u8>,
) -> Result<usize, String> {
    if data.is_empty() {
        return Ok(0);
    }

    let command_tx = {
        let sessions = state
            .sessions
            .lock()
            .map_err(|_| "failed to lock serial sessions".to_string())?;
        let session = sessions
            .get(window.label())
            .ok_or_else(|| "serial is not connected".to_string())?;
        session.command_tx.clone()
    };

    let (response_tx, response_rx) = mpsc::channel();
    command_tx
        .send(SerialCommand::Send { data, response_tx })
        .map_err(|_| "serial writer is unavailable".to_string())?;

    response_rx
        .recv()
        .map_err(|_| "serial writer did not respond".to_string())?
}

#[tauri::command]
fn serial_assistant_close(
    window: tauri::Window,
    state: tauri::State<SerialAssistantState>,
) -> Result<(), String> {
    let session = {
        let mut sessions = state
            .sessions
            .lock()
            .map_err(|_| "failed to lock serial sessions".to_string())?;
        sessions.remove(window.label())
    };
    if let Some(session) = session {
        shutdown_serial_session(session, false);
    }
    let _ = window.emit(
        "serial_assistant_event",
        SerialAssistantEvent::status("serial disconnected"),
    );
    Ok(())
}

#[tauri::command]
fn serial_assistant_is_open(
    window: tauri::Window,
    state: tauri::State<SerialAssistantState>,
) -> Result<bool, String> {
    let sessions = state
        .sessions
        .lock()
        .map_err(|_| "failed to lock serial sessions".to_string())?;
    Ok(sessions.contains_key(window.label()))
}

#[tauri::command]
fn get_audio_info(path: &str) -> Result<AudioInfo, String> {
    log_audio(format!("invoke get_audio_info: input={path}"));
    read_audio_info(path)
}

#[tauri::command]
fn prepare_audio_source(
    app_handle: tauri::AppHandle,
    path: &str,
    input_format: Option<String>,
    sample_rate: Option<u32>,
    channels: Option<u32>,
) -> Result<AudioSourceResult, String> {
    log_audio(format!(
        "invoke prepare_audio_source: input={path}, input_format={:?}, sample_rate={:?}, channels={:?}",
        input_format, sample_rate, channels
    ));
    prepare_audio_source_impl(
        &app_handle,
        path,
        input_format.as_deref(),
        sample_rate,
        channels,
    )
}

#[tauri::command]
fn convert_audio_format(
    input_path: &str,
    output_path: &str,
    output_format: &str,
    input_format: Option<String>,
    input_sample_rate: Option<u32>,
    input_channels: Option<u32>,
    sample_rate: Option<u32>,
    channels: Option<u32>,
) -> Result<AudioConversionResult, String> {
    log_audio(format!(
        "invoke convert_audio_format: input={}, output={}, format={}, input_format={:?}, input_sample_rate={:?}, input_channels={:?}, sample_rate={:?}, channels={:?}",
        input_path,
        output_path,
        output_format,
        input_format,
        input_sample_rate,
        input_channels,
        sample_rate,
        channels
    ));
    if !Path::new(input_path).exists() {
        return Err(format!("input audio file does not exist: {input_path}"));
    }
    if output_path.trim().is_empty() {
        return Err("output path is required".to_string());
    }

    let source_info =
        resolve_input_audio_info(
            input_path,
            input_format.as_deref(),
            input_sample_rate,
            input_channels,
        )?;
    let input_args =
        build_input_audio_args(
            input_format.as_deref(),
            input_path,
            input_sample_rate,
            input_channels,
        )?;
    let codec_args = conversion_codec_args(output_format)?;
    let transform_args = build_audio_transform_args(sample_rate, channels)?;
    run_ffmpeg_audio_job(&input_args, input_path, output_path, &codec_args, &transform_args)?;

    log_audio(format!(
        "convert_audio_format finished: output={}, resolved_format={}",
        output_path, output_format
    ));
    Ok(AudioConversionResult {
        output_path: output_path.to_string(),
        info: build_output_audio_info(output_path, output_format, sample_rate, channels, &source_info)?,
    })
}

#[tauri::command]
fn clip_audio_segment(
    input_path: &str,
    output_path: &str,
    output_format: &str,
    input_format: Option<String>,
    sample_rate: Option<u32>,
    channels: Option<u32>,
    start_time: f64,
    end_time: f64,
) -> Result<AudioConversionResult, String> {
    log_audio(format!(
        "invoke clip_audio_segment: input={}, output={}, format={}, input_format={:?}, sample_rate={:?}, channels={:?}, start_time={:.3}, end_time={:.3}",
        input_path,
        output_path,
        output_format,
        input_format,
        sample_rate,
        channels,
        start_time,
        end_time
    ));
    if !Path::new(input_path).exists() {
        return Err(format!("input audio file does not exist: {input_path}"));
    }
    if output_path.trim().is_empty() {
        return Err("output path is required".to_string());
    }
    if !start_time.is_finite() || !end_time.is_finite() || start_time < 0.0 || end_time <= start_time {
        return Err("invalid clip range".to_string());
    }

    let source_info =
        resolve_input_audio_info(input_path, input_format.as_deref(), sample_rate, channels)?;
    let input_args =
        build_input_audio_args(input_format.as_deref(), input_path, sample_rate, channels)?;
    let codec_args = conversion_codec_args(output_format)?;
    let extra_args = vec![
        "-ss".to_string(),
        format!("{start_time:.3}"),
        "-t".to_string(),
        format!("{:.3}", end_time - start_time),
    ];
    run_ffmpeg_audio_job(&input_args, input_path, output_path, &codec_args, &extra_args)?;

    log_audio(format!(
        "clip_audio_segment finished: output={}, start_time={:.3}, end_time={:.3}",
        output_path, start_time, end_time
    ));
    Ok(AudioConversionResult {
        output_path: output_path.to_string(),
        info: build_output_audio_info(output_path, output_format, None, None, &source_info)?,
    })
}

#[tauri::command]
fn get_current_dir(app_handle: tauri::AppHandle) -> String {
    let path = app_handle.path().app_data_dir().unwrap_or_else(|_| {
        let mut fallback = env::temp_dir();
        fallback.push("wheat-embedding-toolkit");
        fallback
    });
    path.display().to_string()
}

#[tauri::command]
fn spawn_new_instance() -> Result<(), String> {
    let current_exe =
        env::current_exe().map_err(|e| format!("failed to get current executable: {e}"))?;

    let mut command = Command::new(&current_exe);
    if let Ok(current_dir) = env::current_dir() {
        command.current_dir(current_dir);
    }

    command
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("failed to spawn new instance: {e}"))
}

#[tauri::command]
fn open_file_in_explorer(path: &str) {
    #[cfg(target_os = "windows")]
    {
        let file_path = format!(r#"{}"#, path);
        let _ = Command::new("explorer")
            .arg("/select,")
            .arg(file_path)
            .status();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg("-R").arg(path).status();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("xdg-open")
            .arg(Path::new(path).parent().unwrap_or(Path::new(path)))
            .status();
    }
}

#[tauri::command]
fn open_directory_in_explorer(path: &str) {
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("explorer").arg(path).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(path).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("xdg-open").arg(path).spawn();
    }
}

#[tauri::command]
fn get_file_info(path: &str) -> FileInfo {
    let name = Path::new(path)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(path)
        .to_string();

    match fs::metadata(path) {
        Ok(metadata) => {
            let create_time = metadata
                .created()
                .or_else(|_| metadata.modified())
                .ok()
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());

            FileInfo {
                name,
                is_dir: metadata.is_dir(),
                is_file: metadata.is_file(),
                len: metadata.len(),
                create_time,
            }
        }
        Err(_) => FileInfo {
            name,
            is_dir: false,
            is_file: false,
            len: 0,
            create_time: 0,
        },
    }
}

fn main() {
    tauri::Builder::default()
        .manage(SerialAssistantState::default())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_notification::init())
        .setup(|app| {
            let _ = app.get_webview_window("main");
            let resource_dir = app.path().resource_dir().ok();
            let workspace_staged_bin_dir = detect_workspace_staged_bin_dir();
            let workspace_source_bin_dir = detect_workspace_source_bin_dir();
            log_audio(format!(
                "binary path state initialized: resource_dir={:?}, workspace_staged_bin_dir={:?}, workspace_source_bin_dir={:?}",
                resource_dir, workspace_staged_bin_dir, workspace_source_bin_dir
            ));
            let _ = BINARY_PATH_STATE.set(BinaryPathState {
                resource_dir,
                workspace_staged_bin_dir,
                workspace_source_bin_dir,
            });

            let app_dir = app.path().app_data_dir().unwrap_or_else(|_| {
                let mut fallback = env::temp_dir();
                fallback.push("wheat-embedding-toolkit");
                fallback
            });
            if !app_dir.exists() {
                let _ = fs::create_dir_all(&app_dir);
            }

            for item in ["firmware", "partitions", "audio"].iter() {
                let dir = app_dir.join(item);
                if !dir.exists() {
                    let _ = fs::create_dir_all(&dir);
                }
            }

            let chip_list = app_dir.join("chip.list.json");
            if !chip_list.exists() {
                let data = "[\"ESP32\",\"ESP32C2\",\"ESP32C3\",\"ESP32C6\",\"ESP32S2\",\"ESP32S3\",\"ESP32H2\",\"ESP8266\"]";
                let _ = fs::write(&chip_list, data);
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            get_serial_port_list,
            get_current_dir,
            spawn_new_instance,
            open_file_in_explorer,
            open_directory_in_explorer,
            get_file_info,
            start_ble_advertisement_scan,
            serial_assistant_open,
            serial_assistant_send,
            serial_assistant_close,
            serial_assistant_is_open,
            serial_assistant_set_signals,
            get_audio_info,
            prepare_audio_source,
            convert_audio_format,
            clip_audio_segment
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
