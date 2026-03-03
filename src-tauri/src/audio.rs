use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::Path;
use std::time::SystemTime;

use ogg::{PacketWriteEndInfo, PacketWriter};
use opus::{Application, Bitrate as OpusBitrate, Channels as OpusChannels, Encoder as OpusEncoder, Signal as OpusSignal};
use shine_rs::{encode_pcm_to_mp3, Mp3EncoderConfig, StereoMode};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CodecParameters, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::default::{get_codecs, get_probe};
use tauri::Manager;

#[derive(serde::Serialize, Debug, Clone)]
pub struct AudioInfo {
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
pub struct AudioConversionResult {
    output_path: String,
    info: AudioInfo,
}

#[derive(serde::Serialize, Debug, Clone)]
pub struct AudioSourceResult {
    playback_path: String,
    info: AudioInfo,
}

struct DecodedAudio {
    info: AudioInfo,
    sample_rate: u32,
    channels: usize,
    samples: Vec<f32>,
}

pub fn get_audio_info(path: &str) -> Result<AudioInfo, String> {
    log_audio(format!("invoke get_audio_info: input={path}"));
    resolve_input_audio_info(path, None, None, None)
}

pub fn prepare_audio_source(
    app_handle: &tauri::AppHandle,
    path: &str,
    input_format: Option<&str>,
    sample_rate: Option<u32>,
    channels: Option<u32>,
) -> Result<AudioSourceResult, String> {
    log_audio(format!(
        "invoke prepare_audio_source: input={path}, input_format={input_format:?}, sample_rate={sample_rate:?}, channels={channels:?}"
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

pub fn convert_audio_format(
    input_path: &str,
    output_path: &str,
    output_format: &str,
    input_format: Option<&str>,
    input_sample_rate: Option<u32>,
    input_channels: Option<u32>,
    sample_rate: Option<u32>,
    channels: Option<u32>,
) -> Result<AudioConversionResult, String> {
    log_audio(format!(
        "invoke convert_audio_format: input={input_path}, output={output_path}, format={output_format}, input_format={input_format:?}, input_sample_rate={input_sample_rate:?}, input_channels={input_channels:?}, sample_rate={sample_rate:?}, channels={channels:?}"
    ));
    ensure_output_path(output_path)?;

    let decoded = decode_audio(input_path, input_format, input_sample_rate, input_channels)?;
    let transformed = transform_audio(decoded, sample_rate, channels)?;
    let info = write_audio_output(output_path, output_format, &transformed)?;

    log_audio(format!(
        "convert_audio_format finished: output={output_path}, resolved_format={output_format}"
    ));
    Ok(AudioConversionResult {
        output_path: output_path.to_string(),
        info,
    })
}

pub fn clip_audio_segment(
    input_path: &str,
    output_path: &str,
    output_format: &str,
    input_format: Option<&str>,
    sample_rate: Option<u32>,
    channels: Option<u32>,
    start_time: f64,
    end_time: f64,
) -> Result<AudioConversionResult, String> {
    log_audio(format!(
        "invoke clip_audio_segment: input={input_path}, output={output_path}, format={output_format}, input_format={input_format:?}, sample_rate={sample_rate:?}, channels={channels:?}, start_time={start_time:.3}, end_time={end_time:.3}"
    ));
    ensure_output_path(output_path)?;
    if !start_time.is_finite() || !end_time.is_finite() || start_time < 0.0 || end_time <= start_time {
        return Err("invalid clip range".to_string());
    }

    let decoded = decode_audio(input_path, input_format, sample_rate, channels)?;
    let clipped = clip_audio(decoded, start_time, end_time);
    let info = write_audio_output(output_path, output_format, &clipped)?;

    log_audio(format!(
        "clip_audio_segment finished: output={output_path}, start_time={start_time:.3}, end_time={end_time:.3}"
    ));
    Ok(AudioConversionResult {
        output_path: output_path.to_string(),
        info,
    })
}

fn log_audio(message: impl AsRef<str>) {
    println!("[audio] {}", message.as_ref());
}

fn ensure_output_path(output_path: &str) -> Result<(), String> {
    if output_path.trim().is_empty() {
        return Err("output path is required".to_string());
    }
    let output_parent = Path::new(output_path)
        .parent()
        .ok_or_else(|| "invalid output path".to_string())?;
    if !output_parent.exists() {
        fs::create_dir_all(output_parent)
            .map_err(|e| format!("failed to create output directory: {e}"))?;
    }
    Ok(())
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
        file_name: file_name(path),
        format_name: "pcm".to_string(),
        format_long_name: "PCM signed 16-bit little-endian".to_string(),
        duration,
        size,
        overall_bit_rate: Some(bytes_per_second * 8),
        codec_name: "pcm_s16le".to_string(),
        codec_long_name: "PCM signed 16-bit little-endian".to_string(),
        sample_rate: Some(sample_rate),
        channels: Some(channels),
        channel_layout: channel_layout_name(channels),
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
    let bytes = fs::read(input_path).map_err(|e| format!("failed to read pcm source file: {e}"))?;
    write_wav_file_bytes(&output_path, sample_rate, channels as usize, &bytes)?;

    Ok(output_path.display().to_string())
}

fn read_audio_info(path: &str) -> Result<AudioInfo, String> {
    if !Path::new(path).exists() {
        return Err(format!("audio file does not exist: {path}"));
    }

    log_audio(format!("reading audio info via embedded decoder: input={path}"));
    let probed = open_audio_probe(path)?;
    let size = fs::metadata(path)
        .map_err(|e| format!("failed to inspect audio file metadata: {e}"))?
        .len();
    let sample_rate = probed.codec_params.sample_rate;
    let channels = probed
        .codec_params
        .channels
        .map(|value| value.count() as u32);
    let duration = probed
        .codec_params
        .sample_rate
        .and_then(|rate| duration_from_codec_params(&probed.codec_params, rate))
        .unwrap_or(0.0);
    let stream_bit_rate = match (sample_rate, channels) {
        (Some(rate), Some(channel_count)) => Some(u64::from(rate) * u64::from(channel_count) * 16),
        _ => None,
    };
    let overall_bit_rate = if duration > 0.0 {
        Some(((size as f64 * 8.0) / duration).round() as u64)
    } else {
        stream_bit_rate
    };

    let info = AudioInfo {
        path: path.to_string(),
        file_name: file_name(path),
        format_name: probed.format_name,
        format_long_name: probed.format_long_name,
        duration,
        size,
        overall_bit_rate,
        codec_name: codec_name(&probed.codec_params, path),
        codec_long_name: codec_long_name(&probed.codec_params, path),
        sample_rate,
        channels,
        channel_layout: channels
            .map(channel_layout_name)
            .unwrap_or_default(),
        stream_bit_rate,
        tags: HashMap::new(),
    };
    log_audio(format!("embedded decoder read audio info: input={path}"));
    Ok(info)
}

struct ProbedAudio {
    codec_params: CodecParameters,
    format_name: String,
    format_long_name: String,
}

fn open_audio_probe(path: &str) -> Result<ProbedAudio, String> {
    let source = fs::File::open(path).map_err(|e| format!("failed to open audio file: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(source), Default::default());

    let mut hint = Hint::new();
    if let Some(extension) = Path::new(path).extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(extension);
    }

    let probed = get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(symphonia_error_to_string)?;
    let format = probed.format;
    let track = format
        .default_track()
        .or_else(|| format.tracks().iter().find(|track| track.codec_params.codec != CODEC_TYPE_NULL))
        .ok_or_else(|| "no playable audio track found".to_string())?;

    Ok(ProbedAudio {
        codec_params: track.codec_params.clone(),
        format_name: container_format_name(path),
        format_long_name: container_format_long_name(path),
    })
}

fn decode_audio(
    path: &str,
    input_format: Option<&str>,
    sample_rate: Option<u32>,
    channels: Option<u32>,
) -> Result<DecodedAudio, String> {
    match normalize_input_format(input_format, path).as_deref() {
        Some("pcm") => decode_pcm_audio(path, sample_rate, channels),
        _ => decode_container_audio(path),
    }
}

fn decode_pcm_audio(
    path: &str,
    sample_rate: Option<u32>,
    channels: Option<u32>,
) -> Result<DecodedAudio, String> {
    let (sample_rate, channels) = validate_pcm_params(sample_rate, channels)?;
    let bytes = fs::read(path).map_err(|e| format!("failed to read pcm source file: {e}"))?;
    let mut samples = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push(sample as f32 / 32768.0);
    }

    Ok(DecodedAudio {
        info: build_pcm_audio_info(path, sample_rate, channels)?,
        sample_rate,
        channels: channels as usize,
        samples,
    })
}

fn decode_container_audio(path: &str) -> Result<DecodedAudio, String> {
    let info = read_audio_info(path)?;

    let source = fs::File::open(path).map_err(|e| format!("failed to open audio file: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(source), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = Path::new(path).extension().and_then(|ext| ext.to_str()) {
        hint.with_extension(extension);
    }

    let probed = get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(symphonia_error_to_string)?;
    let mut format = probed.format;
    let (track_id, codec_params) = {
        let track = format
            .default_track()
            .or_else(|| format.tracks().iter().find(|track| track.codec_params.codec != CODEC_TYPE_NULL))
            .ok_or_else(|| "no playable audio track found".to_string())?;
        (track.id, track.codec_params.clone())
    };
    let mut decoder = get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(symphonia_error_to_string)?;
    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| "audio source is missing sample rate information".to_string())?;
    let channels = codec_params
        .channels
        .map(|value| value.count())
        .ok_or_else(|| "audio source is missing channel information".to_string())?;
    let mut samples = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break,
            Err(error) => return Err(symphonia_error_to_string(error)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
                buffer.copy_interleaved_ref(decoded);
                samples.extend_from_slice(buffer.samples());
            }
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(_)) => break,
            Err(error) => return Err(symphonia_error_to_string(error)),
        }
    }

    Ok(DecodedAudio {
        info,
        sample_rate,
        channels,
        samples,
    })
}

fn transform_audio(
    decoded: DecodedAudio,
    target_sample_rate: Option<u32>,
    target_channels: Option<u32>,
) -> Result<DecodedAudio, String> {
    if let Some(sample_rate) = target_sample_rate {
        if sample_rate == 0 {
            return Err("sample_rate must be greater than 0".to_string());
        }
    }
    if let Some(channels) = target_channels {
        if channels == 0 {
            return Err("channels must be greater than 0".to_string());
        }
    }

    let next_channels = target_channels
        .map(|value| value as usize)
        .unwrap_or(decoded.channels);
    let remixed = remix_channels(&decoded.samples, decoded.channels, next_channels);
    let next_sample_rate = target_sample_rate.unwrap_or(decoded.sample_rate);
    let resampled = if next_sample_rate != decoded.sample_rate {
        resample_linear(&remixed, next_channels, decoded.sample_rate, next_sample_rate)
    } else {
        remixed
    };

    Ok(DecodedAudio {
        info: decoded.info,
        sample_rate: next_sample_rate,
        channels: next_channels,
        samples: resampled,
    })
}

fn clip_audio(decoded: DecodedAudio, start_time: f64, end_time: f64) -> DecodedAudio {
    let frame_count = decoded.samples.len() / decoded.channels.max(1);
    let start_frame = ((start_time * decoded.sample_rate as f64).floor() as usize).min(frame_count);
    let end_frame = ((end_time * decoded.sample_rate as f64).ceil() as usize).min(frame_count);
    let sample_start = start_frame * decoded.channels;
    let sample_end = end_frame * decoded.channels;

    DecodedAudio {
        info: decoded.info,
        sample_rate: decoded.sample_rate,
        channels: decoded.channels,
        samples: if sample_end > sample_start {
            decoded.samples[sample_start..sample_end].to_vec()
        } else {
            Vec::new()
        },
    }
}

fn write_audio_output(
    output_path: &str,
    output_format: &str,
    decoded: &DecodedAudio,
) -> Result<AudioInfo, String> {
    let frame_count = decoded.samples.len() / decoded.channels.max(1);
    match output_format.to_ascii_lowercase().as_str() {
        "wav" => {
            write_wav_file(output_path, decoded.sample_rate, decoded.channels, &decoded.samples)?;
            build_generated_audio_info(output_path, "wav", decoded.sample_rate, decoded.channels, frame_count)
        }
        "pcm" => {
            write_pcm_file(output_path, &decoded.samples)?;
            build_generated_audio_info(output_path, "pcm", decoded.sample_rate, decoded.channels, frame_count)
        }
        "mp3" => {
            write_mp3_file(output_path, decoded.sample_rate, decoded.channels, &decoded.samples)?;
            build_generated_audio_info(output_path, "mp3", decoded.sample_rate, decoded.channels, frame_count)
        }
        "opus" => {
            write_opus_file(output_path, decoded.sample_rate, decoded.channels, &decoded.samples)?;
            build_generated_audio_info(output_path, "opus", decoded.sample_rate, decoded.channels, frame_count)
        }
        other => Err(format!(
            "unsupported output format for embedded audio engine: {other}. Currently supported: wav, pcm, mp3, opus"
        )),
    }
}

fn build_generated_audio_info(
    path: &str,
    format: &str,
    sample_rate: u32,
    channels: usize,
    frame_count: usize,
) -> Result<AudioInfo, String> {
    let metadata = fs::metadata(path).map_err(|e| format!("failed to inspect output file metadata: {e}"))?;
    let size = metadata.len();
    let duration = if sample_rate == 0 || channels == 0 {
        0.0
    } else {
        frame_count as f64 / sample_rate as f64
    };
    let overall_bit_rate = if duration > 0.0 {
        Some(((size as f64 * 8.0) / duration).round() as u64)
    } else {
        None
    };

    let (format_name, format_long_name, codec_name, codec_long_name, stream_bit_rate) = match format {
        "wav" => (
            "wav".to_string(),
            "WAV audio".to_string(),
            "pcm_s16le".to_string(),
            "PCM signed 16-bit little-endian".to_string(),
            Some(u64::from(sample_rate) * channels as u64 * 16),
        ),
        "pcm" => (
            "pcm".to_string(),
            "PCM signed 16-bit little-endian".to_string(),
            "pcm_s16le".to_string(),
            "PCM signed 16-bit little-endian".to_string(),
            Some(u64::from(sample_rate) * channels as u64 * 16),
        ),
        "mp3" => (
            "mp3".to_string(),
            "MP3 audio".to_string(),
            "mp3".to_string(),
            "MPEG Layer III".to_string(),
            overall_bit_rate,
        ),
        "opus" => (
            "opus".to_string(),
            "Opus audio".to_string(),
            "opus".to_string(),
            "Opus".to_string(),
            overall_bit_rate,
        ),
        _ => (
            format.to_string(),
            format.to_string(),
            format.to_string(),
            format.to_string(),
            overall_bit_rate,
        ),
    };

    Ok(AudioInfo {
        path: path.to_string(),
        file_name: file_name(path),
        format_name,
        format_long_name,
        duration,
        size,
        overall_bit_rate,
        codec_name,
        codec_long_name,
        sample_rate: Some(sample_rate),
        channels: Some(channels as u32),
        channel_layout: channel_layout_name(channels as u32),
        stream_bit_rate,
        tags: HashMap::new(),
    })
}

fn write_wav_file(
    output_path: &str,
    sample_rate: u32,
    channels: usize,
    samples: &[f32],
) -> Result<(), String> {
    let mut pcm_bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        pcm_bytes.extend_from_slice(&float_to_i16(*sample).to_le_bytes());
    }
    write_wav_file_bytes(Path::new(output_path), sample_rate, channels, &pcm_bytes)
}

fn write_wav_file_bytes(
    output_path: &Path,
    sample_rate: u32,
    channels: usize,
    pcm_bytes: &[u8],
) -> Result<(), String> {
    let riff_chunk_size = 36u64 + pcm_bytes.len() as u64;
    let byte_rate = u64::from(sample_rate) * channels as u64 * 2;
    let block_align = channels as u16 * 2;

    let mut output = fs::File::create(output_path)
        .map_err(|e| format!("failed to create wav output file: {e}"))?;
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
        .and_then(|_| output.write_all(&block_align.to_le_bytes()))
        .and_then(|_| output.write_all(&16u16.to_le_bytes()))
        .and_then(|_| output.write_all(b"data"))
        .and_then(|_| output.write_all(&(pcm_bytes.len() as u32).to_le_bytes()))
        .and_then(|_| output.write_all(pcm_bytes))
        .map_err(|e| format!("failed to write wav output file: {e}"))?;
    Ok(())
}

fn write_pcm_file(output_path: &str, samples: &[f32]) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend_from_slice(&float_to_i16(*sample).to_le_bytes());
    }
    fs::write(output_path, bytes).map_err(|e| format!("failed to write pcm output file: {e}"))
}

fn write_mp3_file(
    output_path: &str,
    sample_rate: u32,
    channels: usize,
    samples: &[f32],
) -> Result<(), String> {
    let bitrate = if channels == 1 { 64 } else { 128 };
    let stereo_mode = if channels == 1 {
        StereoMode::Mono
    } else {
        StereoMode::JointStereo
    };
    let pcm_samples = samples_to_i16(samples);
    let config = Mp3EncoderConfig::new()
        .sample_rate(sample_rate)
        .bitrate(bitrate)
        .channels(channels as u8)
        .stereo_mode(stereo_mode);
    let mp3_data = encode_pcm_to_mp3(config, &pcm_samples)
        .map_err(|e| format!("failed to encode mp3 output: {e}"))?;
    fs::write(output_path, mp3_data).map_err(|e| format!("failed to write mp3 output file: {e}"))
}

fn write_opus_file(
    output_path: &str,
    sample_rate: u32,
    channels: usize,
    samples: &[f32],
) -> Result<(), String> {
    let encoded_rate = resolve_opus_sample_rate(sample_rate);
    let resampled = if encoded_rate != sample_rate {
        resample_linear(samples, channels, sample_rate, encoded_rate)
    } else {
        samples.to_vec()
    };
    let pcm_samples = samples_to_i16(&resampled);
    let opus_channels = opus_channels(channels)?;
    let mut encoder = OpusEncoder::new(encoded_rate, opus_channels, Application::Audio)
        .map_err(|e| format!("failed to initialize opus encoder: {e}"))?;
    let bitrate = if channels == 1 { 64_000 } else { 128_000 };
    encoder
        .set_bitrate(OpusBitrate::Bits(bitrate))
        .map_err(|e| format!("failed to configure opus bitrate: {e}"))?;
    encoder
        .set_signal(OpusSignal::Music)
        .map_err(|e| format!("failed to configure opus signal mode: {e}"))?;
    encoder
        .set_vbr(true)
        .map_err(|e| format!("failed to enable opus vbr: {e}"))?;

    let lookahead = encoder
        .get_lookahead()
        .map_err(|e| format!("failed to read opus lookahead: {e}"))?
        .max(0) as u32;
    let pre_skip = scale_opus_granule(lookahead as usize, encoded_rate);
    let total_frames = pcm_samples.len() / channels.max(1);
    let frames_per_packet = ((encoded_rate as usize) / 50).max(1);
    let serial = ogg_stream_serial(output_path);
    let vendor = b"Wheat Embedding Toolkit";
    let mut writer = PacketWriter::new(fs::File::create(output_path).map_err(|e| {
        format!("failed to create opus output file: {e}")
    })?);

    writer
        .write_packet(build_opus_head(channels as u8, pre_skip as u16, encoded_rate), serial, PacketWriteEndInfo::EndPage, 0)
        .map_err(|e| format!("failed to write opus header page: {e}"))?;
    writer
        .write_packet(build_opus_tags(vendor), serial, PacketWriteEndInfo::EndPage, 0)
        .map_err(|e| format!("failed to write opus tags page: {e}"))?;

    let mut frame_index = 0usize;
    while frame_index < total_frames {
        let next_frame = (frame_index + frames_per_packet).min(total_frames);
        let actual_frames = next_frame - frame_index;
        let mut packet_input = vec![0i16; frames_per_packet * channels];
        let sample_start = frame_index * channels;
        let sample_end = next_frame * channels;
        packet_input[..sample_end - sample_start].copy_from_slice(&pcm_samples[sample_start..sample_end]);
        let mut packet = vec![0u8; 4000];
        let encoded_len = encoder
            .encode(&packet_input, &mut packet)
            .map_err(|e| format!("failed to encode opus packet: {e}"))?;
        packet.truncate(encoded_len);

        let is_last = next_frame >= total_frames;
        let granule_position = if is_last {
            (pre_skip + scale_opus_granule(total_frames, encoded_rate)) as u64
        } else {
            (pre_skip + scale_opus_granule(frame_index + actual_frames, encoded_rate)) as u64
        };
        writer
            .write_packet(
                packet,
                serial,
                if is_last {
                    PacketWriteEndInfo::EndStream
                } else {
                    PacketWriteEndInfo::NormalPacket
                },
                granule_position,
            )
            .map_err(|e| format!("failed to write opus packet: {e}"))?;

        frame_index = next_frame;
    }

    Ok(())
}

fn resolve_opus_sample_rate(sample_rate: u32) -> u32 {
    match sample_rate {
        8000 | 12000 | 16000 | 24000 | 48000 => sample_rate,
        _ => 48000,
    }
}

fn scale_opus_granule(samples: usize, sample_rate: u32) -> usize {
    if sample_rate == 0 {
        return 0;
    }
    samples.saturating_mul(48_000usize) / sample_rate as usize
}

fn ogg_stream_serial(seed: &str) -> u32 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    (hasher.finish() & 0xffff_ffff) as u32
}

fn build_opus_head(channels: u8, pre_skip: u16, sample_rate: u32) -> Vec<u8> {
    let mut packet = Vec::with_capacity(19);
    packet.extend_from_slice(b"OpusHead");
    packet.push(1);
    packet.push(channels);
    packet.extend_from_slice(&pre_skip.to_le_bytes());
    packet.extend_from_slice(&sample_rate.to_le_bytes());
    packet.extend_from_slice(&0u16.to_le_bytes());
    packet.push(0);
    packet
}

fn build_opus_tags(vendor: &[u8]) -> Vec<u8> {
    let mut packet = Vec::with_capacity(16 + vendor.len());
    packet.extend_from_slice(b"OpusTags");
    packet.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    packet.extend_from_slice(vendor);
    packet.extend_from_slice(&0u32.to_le_bytes());
    packet
}

fn opus_channels(channels: usize) -> Result<OpusChannels, String> {
    match channels {
        1 => Ok(OpusChannels::Mono),
        2 => Ok(OpusChannels::Stereo),
        _ => Err("opus output currently supports mono or stereo only".to_string()),
    }
}

fn samples_to_i16(samples: &[f32]) -> Vec<i16> {
    samples.iter().map(|sample| float_to_i16(*sample)).collect()
}

fn remix_channels(samples: &[f32], source_channels: usize, target_channels: usize) -> Vec<f32> {
    if source_channels == 0 || target_channels == 0 {
        return Vec::new();
    }
    if source_channels == target_channels {
        return samples.to_vec();
    }

    let frames = samples.len() / source_channels;
    let mut output = Vec::with_capacity(frames * target_channels);

    for frame_index in 0..frames {
        let frame = &samples[frame_index * source_channels..(frame_index + 1) * source_channels];
        match (source_channels, target_channels) {
            (1, 2) => {
                output.push(frame[0]);
                output.push(frame[0]);
            }
            (2, 1) => {
                output.push((frame[0] + frame[1]) * 0.5);
            }
            (_, 1) => {
                let sum = frame.iter().copied().sum::<f32>();
                output.push(sum / source_channels as f32);
            }
            (1, target) => {
                for _ in 0..target {
                    output.push(frame[0]);
                }
            }
            (_, target) => {
                for channel_index in 0..target {
                    output.push(frame[channel_index.min(source_channels - 1)]);
                }
            }
        }
    }

    output
}

fn resample_linear(
    samples: &[f32],
    channels: usize,
    input_rate: u32,
    output_rate: u32,
) -> Vec<f32> {
    if channels == 0 || input_rate == 0 || output_rate == 0 {
        return Vec::new();
    }
    if input_rate == output_rate {
        return samples.to_vec();
    }

    let input_frames = samples.len() / channels;
    if input_frames == 0 {
        return Vec::new();
    }

    let output_frames = ((input_frames as u64 * output_rate as u64) / input_rate as u64) as usize;
    let output_frames = output_frames.max(1);
    let ratio = input_rate as f64 / output_rate as f64;
    let mut output = vec![0.0f32; output_frames * channels];

    for output_frame in 0..output_frames {
        let position = output_frame as f64 * ratio;
        let left_frame = position.floor() as usize;
        let right_frame = (left_frame + 1).min(input_frames - 1);
        let factor = (position - left_frame as f64) as f32;

        for channel in 0..channels {
            let left = samples[left_frame * channels + channel];
            let right = samples[right_frame * channels + channel];
            output[output_frame * channels + channel] = left + (right - left) * factor;
        }
    }

    output
}

fn float_to_i16(sample: f32) -> i16 {
    let clamped = sample.clamp(-1.0, 1.0);
    if clamped <= -1.0 {
        i16::MIN
    } else {
        (clamped * i16::MAX as f32).round() as i16
    }
}

fn duration_from_codec_params(codec_params: &CodecParameters, sample_rate: u32) -> Option<f64> {
    if let (Some(frames), Some(time_base)) = (codec_params.n_frames, codec_params.time_base) {
        let time = time_base.calc_time(frames);
        return Some(time.seconds as f64 + time.frac);
    }
    codec_params
        .n_frames
        .map(|frames| frames as f64 / sample_rate as f64)
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(path)
        .to_string()
}

fn channel_layout_name(channels: u32) -> String {
    match channels {
        1 => "mono".to_string(),
        2 => "stereo".to_string(),
        _ => String::new(),
    }
}

fn container_format_name(path: &str) -> String {
    match normalize_input_format(None, path).as_deref() {
        Some("wav") => "wav".to_string(),
        Some("flac") => "flac".to_string(),
        Some("mp3") => "mp3".to_string(),
        Some("ogg") => "ogg".to_string(),
        Some("m4a") | Some("mp4") => "mp4".to_string(),
        Some("aac") => "aac".to_string(),
        Some("pcm") => "pcm".to_string(),
        Some(other) => other.to_string(),
        None => "audio".to_string(),
    }
}

fn container_format_long_name(path: &str) -> String {
    match normalize_input_format(None, path).as_deref() {
        Some("wav") => "WAV audio".to_string(),
        Some("flac") => "FLAC audio".to_string(),
        Some("mp3") => "MP3 audio".to_string(),
        Some("ogg") => "Ogg audio".to_string(),
        Some("m4a") | Some("mp4") => "MPEG-4 audio".to_string(),
        Some("aac") => "AAC audio".to_string(),
        Some("pcm") => "PCM signed 16-bit little-endian".to_string(),
        Some(other) => other.to_ascii_uppercase(),
        None => "Audio".to_string(),
    }
}

fn codec_name(codec_params: &CodecParameters, path: &str) -> String {
    if let Some(name) = codec_name_from_type(codec_params) {
        return name;
    }

    match normalize_input_format(None, path).as_deref() {
        Some("m4a") | Some("aac") | Some("mp4") => "aac".to_string(),
        Some("ogg") => "vorbis".to_string(),
        Some("wav") => "pcm".to_string(),
        Some("pcm") => "pcm_s16le".to_string(),
        Some(other) => other.to_string(),
        None => "audio".to_string(),
    }
}

fn codec_long_name(codec_params: &CodecParameters, path: &str) -> String {
    match codec_name(codec_params, path).as_str() {
        "aac" => "Advanced Audio Coding".to_string(),
        "flac" => "Free Lossless Audio Codec".to_string(),
        "mp3" => "MPEG Layer III".to_string(),
        "vorbis" => "Vorbis".to_string(),
        "pcm" | "pcm_s16le" => "PCM signed 16-bit little-endian".to_string(),
        other => other.to_ascii_uppercase(),
    }
}

fn codec_name_from_type(codec_params: &CodecParameters) -> Option<String> {
    let rendered = format!("{:?}", codec_params.codec);
    if rendered == "CODEC_TYPE_NULL" {
        return None;
    }
    let normalized = rendered
        .trim_start_matches("CODEC_TYPE_")
        .to_ascii_lowercase();

    let mapped = match normalized.as_str() {
        "mp3" => "mp3",
        "aac" => "aac",
        "flac" => "flac",
        "vorbis" => "vorbis",
        "pcm_s16le" => "pcm_s16le",
        "pcm_s16be" => "pcm_s16be",
        "pcm_f32le" => "pcm_f32le",
        _ => normalized.as_str(),
    };

    Some(mapped.to_string())
}

fn symphonia_error_to_string(error: SymphoniaError) -> String {
    match error {
        SymphoniaError::IoError(error) => format!("audio io error: {error}"),
        SymphoniaError::DecodeError(error) => format!("audio decode error: {error}"),
        SymphoniaError::SeekError(error) => format!("audio seek error: {error:?}"),
        SymphoniaError::Unsupported(feature) => format!("unsupported audio feature: {feature}"),
        SymphoniaError::LimitError(error) => format!("audio limit error: {error}"),
        SymphoniaError::ResetRequired => "audio decoder reset required".to_string(),
    }
}
