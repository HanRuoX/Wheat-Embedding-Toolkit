import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";

export interface AudioInfo {
  path: string;
  fileName: string;
  formatName: string;
  formatLongName: string;
  duration: number;
  size: number;
  overallBitRate?: number | null;
  codecName: string;
  codecLongName: string;
  sampleRate?: number | null;
  channels?: number | null;
  channelLayout: string;
  streamBitRate?: number | null;
  tags: Record<string, string>;
}

export interface AudioConversionResult {
  outputPath: string;
  info: AudioInfo;
}

export interface AudioSourceResult {
  info: AudioInfo;
  playbackPath: string;
}

function normalizeAudioInfo(payload: any): AudioInfo {
  return {
    path: payload.path,
    fileName: payload.file_name,
    formatName: payload.format_name,
    formatLongName: payload.format_long_name,
    duration: payload.duration,
    size: payload.size,
    overallBitRate: payload.overall_bit_rate,
    codecName: payload.codec_name,
    codecLongName: payload.codec_long_name,
    sampleRate: payload.sample_rate,
    channels: payload.channels,
    channelLayout: payload.channel_layout,
    streamBitRate: payload.stream_bit_rate,
    tags: payload.tags ?? {},
  };
}

export async function getAudioInfo(path: string) {
  const payload = await invoke("get_audio_info", { path });
  return normalizeAudioInfo(payload);
}

export async function prepareAudioSource(
  path: string,
  inputFormat?: string,
  sampleRate?: number,
  channels?: number
) {
  const payload = (await invoke("prepare_audio_source", {
    path,
    inputFormat: inputFormat ?? null,
    sampleRate: sampleRate ?? null,
    channels: channels ?? null,
  })) as any;

  return {
    info: normalizeAudioInfo(payload.info),
    playbackPath: payload.playback_path,
  } satisfies AudioSourceResult;
}

export async function convertAudioFormat(
  inputPath: string,
  outputPath: string,
  outputFormat: string,
  inputFormat?: string,
  inputSampleRate?: number,
  inputChannels?: number,
  sampleRate?: number,
  channels?: number
) {
  const payload = (await invoke("convert_audio_format", {
    inputPath,
    outputPath,
    outputFormat,
    inputFormat: inputFormat ?? null,
    inputSampleRate: inputSampleRate ?? null,
    inputChannels: inputChannels ?? null,
    sampleRate: sampleRate ?? null,
    channels: channels ?? null,
  })) as any;

  return {
    outputPath: payload.output_path,
    info: normalizeAudioInfo(payload.info),
  } satisfies AudioConversionResult;
}

export async function clipAudioSegment(
  inputPath: string,
  outputPath: string,
  outputFormat: string,
  inputFormat: string | undefined,
  sampleRate: number | undefined,
  channels: number | undefined,
  startTime: number,
  endTime: number
) {
  const payload = (await invoke("clip_audio_segment", {
    inputPath,
    outputPath,
    outputFormat,
    inputFormat: inputFormat ?? null,
    sampleRate: sampleRate ?? null,
    channels: channels ?? null,
    startTime,
    endTime,
  })) as any;

  return {
    outputPath: payload.output_path,
    info: normalizeAudioInfo(payload.info),
  } satisfies AudioConversionResult;
}

export async function saveAudioFileDialog(
  defaultPath: string,
  format: string
) {
  const normalizedFormat = format.toLowerCase();
  const filePath = await save({
    defaultPath,
    filters: [
      {
        name: normalizedFormat.toUpperCase(),
        extensions: [normalizedFormat],
      },
    ],
  });

  if (!filePath) {
    return null;
  }

  const suffix = `.${normalizedFormat}`;
  if (filePath.toLowerCase().endsWith(suffix)) {
    return filePath;
  }

  return `${filePath}${suffix}`;
}
