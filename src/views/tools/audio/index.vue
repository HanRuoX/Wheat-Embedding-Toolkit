<template>
  <div class="audio-page">
    <div class="audio-grid">
      <section class="audio-panel">
        <div class="audio-panel-header">
          <div>
            <h3>{{ $t("audio.sourceTitle") }}</h3>
            <p>{{ $t("audio.sourceDescription") }}</p>
          </div>
          <div class="audio-panel-actions">
            <a-button @click="selectAudioFile">{{ $t("audio.selectFile") }}</a-button>
            <a-button :disabled="!audioPath || loadingInfo" @click="reloadAudio">
              {{ $t("audio.refreshInfo") }}
            </a-button>
          </div>
        </div>

        <Upload
          :title="$t('audio.dropTitle')"
          :subtitle="$t('audio.dropSubtitle')"
          @open="onAudioSelected"
          @drop="onAudioSelected"
        />

        <div v-if="isPcmSource" class="source-config-card">
          <div class="selection-card-header">
            <div class="selection-card-title">{{ $t("audio.pcmSourceTitle") }}</div>
            <div class="selection-card-description">{{ $t("audio.pcmSourceDescription") }}</div>
          </div>
          <div class="convert-form">
            <div class="convert-row">
              <label>{{ $t("audio.sampleRate") }}</label>
              <a-select v-model:value="sourceSampleRate" :options="pcmSampleRateOptions" />
            </div>
            <div class="convert-row">
              <label>{{ $t("audio.channels") }}</label>
              <a-select v-model:value="sourceChannels" :options="pcmChannelOptions" />
            </div>
          </div>
        </div>

        <div class="wave-shell">
          <div ref="waveContainer" class="wave-container" />
          <div v-if="!audioPath" class="wave-placeholder">
            {{ $t("audio.wavePlaceholder") }}
          </div>
          <div v-else-if="waveLoading" class="wave-placeholder">
            {{ $t("audio.waveLoading") }}
          </div>
        </div>

        <div class="audio-player-bar">
          <div class="audio-player-actions">
            <a-button type="primary" :disabled="!waveReady" @click="togglePlayback">
              {{ isPlaying ? $t("audio.pause") : $t("audio.play") }}
            </a-button>
            <a-button :disabled="!waveReady" @click="stopPlayback">
              {{ $t("audio.stop") }}
            </a-button>
          </div>
          <div class="audio-player-stats">
            <span>{{ formattedCurrentTime }}</span>
            <span>/</span>
            <span>{{ formattedDuration }}</span>
          </div>
        </div>

        <div class="selection-card">
          <div class="selection-card-header">
            <div class="selection-card-title">{{ $t("audio.selectionTitle") }}</div>
            <div class="selection-card-description">{{ $t("audio.selectionDescription") }}</div>
          </div>

          <div v-if="!hasSelection" class="selection-empty">
            {{ $t("audio.selectionEmpty") }}
          </div>
          <template v-else>
            <div class="selection-metrics">
              <div class="selection-metric">
                <span>{{ $t("audio.selectionStart") }}</span>
                <strong>{{ formattedSelectionStart }}</strong>
              </div>
              <div class="selection-metric">
                <span>{{ $t("audio.selectionEnd") }}</span>
                <strong>{{ formattedSelectionEnd }}</strong>
              </div>
              <div class="selection-metric">
                <span>{{ $t("audio.selectionDuration") }}</span>
                <strong>{{ formattedSelectionDuration }}</strong>
              </div>
            </div>

            <div class="selection-actions">
              <a-button :disabled="!hasSelection" @click="previewSelection">
                {{ $t("audio.previewSelection") }}
              </a-button>
              <a-button :disabled="!hasSelection" @click="clearSelection">
                {{ $t("audio.clearSelection") }}
              </a-button>
              <a-button
                type="primary"
                :loading="clipping"
                :disabled="!hasSelection"
                @click="exportSelection"
              >
                {{ $t("audio.exportSelection") }}
              </a-button>
            </div>
          </template>
        </div>
      </section>

      <section class="audio-panel">
        <div class="audio-panel-header">
          <div>
            <h3>{{ $t("audio.infoTitle") }}</h3>
            <p>{{ $t("audio.infoDescription") }}</p>
          </div>
        </div>

        <a-spin :spinning="loadingInfo">
          <a-empty v-if="!audioInfo" :description="$t('audio.infoEmpty')" />
          <template v-else>
            <a-descriptions :column="1" bordered size="small">
              <a-descriptions-item :label="$t('audio.fileName')">
                {{ audioInfo.fileName }}
              </a-descriptions-item>
              <a-descriptions-item :label="$t('audio.containerFormat')">
                {{ audioInfo.formatLongName || audioInfo.formatName || "-" }}
              </a-descriptions-item>
              <a-descriptions-item :label="$t('audio.codec')">
                {{ audioInfo.codecLongName || audioInfo.codecName || "-" }}
              </a-descriptions-item>
              <a-descriptions-item :label="$t('audio.duration')">
                {{ formattedInfoDuration }}
              </a-descriptions-item>
              <a-descriptions-item :label="$t('audio.fileSize')">
                {{ formattedSize }}
              </a-descriptions-item>
              <a-descriptions-item :label="$t('audio.sampleRate')">
                {{ formattedSampleRate }}
              </a-descriptions-item>
              <a-descriptions-item :label="$t('audio.channels')">
                {{ formattedChannels }}
              </a-descriptions-item>
              <a-descriptions-item :label="$t('audio.bitRate')">
                {{ formattedBitRate }}
              </a-descriptions-item>
            </a-descriptions>

            <div v-if="tagEntries.length" class="tag-section">
              <div class="tag-section-title">{{ $t("audio.tags") }}</div>
              <div class="tag-list">
                <a-tag v-for="tag in tagEntries" :key="tag.key" color="blue">
                  {{ tag.key }}: {{ tag.value }}
                </a-tag>
              </div>
            </div>
          </template>
        </a-spin>
      </section>

      <section class="audio-panel audio-convert-panel">
        <div class="audio-panel-header">
          <div>
            <h3>{{ $t("audio.convertTitle") }}</h3>
            <p>{{ $t("audio.convertDescription") }}</p>
          </div>
        </div>

        <div class="convert-form">
          <div class="convert-row">
            <label>{{ $t("audio.targetFormat") }}</label>
            <a-select v-model:value="outputFormat" :options="formatOptions" />
          </div>
          <div class="convert-row">
            <label>{{ $t("audio.targetSampleRate") }}</label>
            <a-select
              v-model:value="conversionSampleRate"
              :options="sampleRateOptions"
            />
          </div>
          <div class="convert-row">
            <label>{{ $t("audio.targetChannels") }}</label>
            <a-select
              v-model:value="conversionChannels"
              :options="channelOptions"
            />
          </div>
          <div class="convert-actions">
            <a-button
              type="primary"
              :loading="converting"
              :disabled="!audioPath"
              @click="convertAudio"
            >
              {{ $t("audio.startConvert") }}
            </a-button>
          </div>
        </div>

        <a-alert
          v-if="conversionResult"
          type="success"
          show-icon
          :message="$t('audio.convertSuccess')"
        />
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import WaveSurfer from "wavesurfer.js";
import Hover from "wavesurfer.js/dist/plugins/hover.esm.js";
import Regions from "wavesurfer.js/dist/plugins/regions.esm.js";
import { convertFileSrc } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { message } from "ant-design-vue";
import prettyBytes from "pretty-bytes";
import Upload from "@/components/Upload.vue";
import i18n from "@/locales/i18n";
import {
  clipAudioSegment,
  convertAudioFormat,
  prepareAudioSource,
  saveAudioFileDialog,
  type AudioConversionResult,
  type AudioInfo,
} from "@/utils/audio";

const waveContainer = ref<HTMLElement | null>(null);
const audioPath = ref("");
const outputFormat = ref("mp3");
const sourceSampleRate = ref(16000);
const sourceChannels = ref(1);
const conversionSampleRate = ref<number | "keep">("keep");
const conversionChannels = ref<number | "keep">("keep");
const audioInfo = ref<AudioInfo | null>(null);
const conversionResult = ref<AudioConversionResult | null>(null);
const loadingInfo = ref(false);
const converting = ref(false);
const clipping = ref(false);
const waveLoading = ref(false);
const waveReady = ref(false);
const isPlaying = ref(false);
const currentTime = ref(0);
const totalDuration = ref(0);
const selectionStart = ref(0);
const selectionEnd = ref(0);

let waveSurfer: WaveSurfer | null = null;
let regionsPlugin: any = null;
let activeRegion: any = null;
let disableRegionDragSelection: (() => void) | null = null;

const formatOptions = [
  { label: "MP3", value: "mp3" },
  { label: "WAV", value: "wav" },
  { label: "FLAC", value: "flac" },
  { label: "PCM", value: "pcm" },
  { label: "M4A", value: "m4a" },
  { label: "OGG", value: "ogg" },
  { label: "OPUS", value: "opus" },
  { label: "AAC", value: "aac" },
];
const isPcmSource = computed(() => /\.pcm$/i.test(audioPath.value));
const sourceInputFormat = computed(() => (isPcmSource.value ? "pcm" : undefined));
const pcmSampleRateOptions = [
  { label: "8000 Hz", value: 8000 },
  { label: "16000 Hz", value: 16000 },
  { label: "22050 Hz", value: 22050 },
  { label: "32000 Hz", value: 32000 },
  { label: "44100 Hz", value: 44100 },
  { label: "48000 Hz", value: 48000 },
];
const pcmChannelOptions = computed(() => [
  { label: i18n.global.t("audio.mono"), value: 1 },
  { label: i18n.global.t("audio.stereo"), value: 2 },
]);
const sampleRateOptions = computed(() => {
  const options = [
    { label: i18n.global.t("audio.keepOriginal"), value: "keep" },
    { label: "8000 Hz", value: 8000 },
    { label: "16000 Hz", value: 16000 },
    { label: "22050 Hz", value: 22050 },
    { label: "32000 Hz", value: 32000 },
    { label: "44100 Hz", value: 44100 },
    { label: "48000 Hz", value: 48000 },
  ];

  const current = audioInfo.value?.sampleRate;
  if (current && !options.some((option) => option.value === current)) {
    options.splice(1, 0, { label: `${current} Hz`, value: current });
  }

  return options;
});
const channelOptions = computed(() => [
  { label: i18n.global.t("audio.keepOriginal"), value: "keep" },
  { label: i18n.global.t("audio.mono"), value: 1 },
  { label: i18n.global.t("audio.stereo"), value: 2 },
]);

const tagEntries = computed(() =>
  Object.entries(audioInfo.value?.tags ?? {}).map(([key, value]) => ({ key, value }))
);
const formattedCurrentTime = computed(() => formatDuration(currentTime.value));
const formattedDuration = computed(() => formatDuration(totalDuration.value));
const formattedInfoDuration = computed(() =>
  audioInfo.value ? formatDuration(audioInfo.value.duration) : "-"
);
const hasSelection = computed(() => selectionEnd.value > selectionStart.value);
const formattedSelectionStart = computed(() => formatHoverTime(selectionStart.value));
const formattedSelectionEnd = computed(() => formatHoverTime(selectionEnd.value));
const formattedSelectionDuration = computed(() =>
  formatHoverTime(Math.max(0, selectionEnd.value - selectionStart.value))
);
const formattedSize = computed(() =>
  audioInfo.value ? prettyBytes(audioInfo.value.size || 0) : "-"
);
const formattedSampleRate = computed(() =>
  audioInfo.value?.sampleRate ? `${audioInfo.value.sampleRate} Hz` : "-"
);
const formattedChannels = computed(() => {
  if (!audioInfo.value?.channels) {
    return "-";
  }
  const suffix = audioInfo.value.channelLayout
    ? ` (${audioInfo.value.channelLayout})`
    : "";
  return `${audioInfo.value.channels}${suffix}`;
});
const formattedBitRate = computed(() => {
  const bitRate = audioInfo.value?.streamBitRate || audioInfo.value?.overallBitRate;
  if (!bitRate) {
    return "-";
  }
  return `${(bitRate / 1000).toFixed(0)} kbps`;
});

watch(outputFormat, () => {
  conversionResult.value = null;
});

watch([sourceSampleRate, sourceChannels], async () => {
  if (!isPcmSource.value || !audioPath.value) {
    return;
  }
  await loadAudio(audioPath.value);
});

async function ensureWaveSurfer() {
  if (waveSurfer || !waveContainer.value) {
    return;
  }

  waveSurfer = WaveSurfer.create({
    container: waveContainer.value,
    waveColor: "#8db7ff",
    progressColor: "#4a6fd2",
    cursorColor: "#f4f7ff",
    barWidth: 3,
    barGap: 2,
    barRadius: 2,
    height: 180,
    normalize: true,
    url: undefined,
    plugins: [
      Hover.create({
        lineColor: "#ff0000",
        lineWidth: 2,
        labelBackground: "#555",
        labelColor: "#fff",
        labelSize: "11px",
        labelPreferLeft: false,
        formatTimeCallback: formatHoverTime,
      }),
      Regions.create(),
    ],
  });
  regionsPlugin = waveSurfer.getActivePlugins().find((plugin: any) => {
    return typeof plugin?.addRegion === "function" && typeof plugin?.enableDragSelection === "function";
  });
  disableRegionDragSelection = regionsPlugin?.enableDragSelection(
    {
      color: "rgba(74, 111, 210, 0.22)",
      drag: true,
      resize: true,
      resizeStart: true,
      resizeEnd: true,
      minLength: 0.05,
    },
    2
  ) ?? null;

  regionsPlugin?.on("region-created", (region: any) => {
    setActiveRegion(region);
  });
  regionsPlugin?.on("region-updated", (region: any) => {
    if (activeRegion?.id === region.id) {
      syncSelection(region);
    }
  });
  regionsPlugin?.on("region-removed", (region: any) => {
    if (activeRegion?.id === region.id) {
      activeRegion = null;
      selectionStart.value = 0;
      selectionEnd.value = 0;
    }
  });

  waveSurfer.on("ready", () => {
    waveLoading.value = false;
    waveReady.value = true;
    totalDuration.value = waveSurfer?.getDuration() ?? 0;
  });
  waveSurfer.on("play", () => {
    isPlaying.value = true;
  });
  waveSurfer.on("pause", () => {
    isPlaying.value = false;
  });
  waveSurfer.on("finish", () => {
    isPlaying.value = false;
    currentTime.value = 0;
  });
  waveSurfer.on("timeupdate", (time: number) => {
    currentTime.value = time;
  });
  waveSurfer.on("error", (error: Error) => {
    waveLoading.value = false;
    waveReady.value = false;
    isPlaying.value = false;
    message.error(error.message || i18n.global.t("audio.waveLoadFailed"));
  });
}

async function selectAudioFile() {
  const selected = await open({
    multiple: false,
    filters: [
      {
        name: "Audio",
        extensions: ["mp3", "wav", "flac", "m4a", "ogg", "opus", "aac", "pcm"],
      },
    ],
  });

  if (typeof selected === "string") {
    await loadAudio(selected);
  }
}

async function onAudioSelected(value: string | string[]) {
  const nextPath = Array.isArray(value) ? value[0] : value;
  if (typeof nextPath === "string") {
    await loadAudio(nextPath);
  }
}

async function loadAudio(path: string) {
  audioPath.value = path;
  conversionResult.value = null;
  waveReady.value = false;
  waveLoading.value = true;
  isPlaying.value = false;
  currentTime.value = 0;
  totalDuration.value = 0;
  clearSelection();

  await ensureWaveSurfer();
  await refreshAudioInfo();

  if (!waveSurfer) {
    waveLoading.value = false;
    return;
  }

  try {
    const source = await refreshAudioInfo();
    if (!source) {
      waveLoading.value = false;
      waveReady.value = false;
      return;
    }
    await waveSurfer.load(convertFileSrc(source.playbackPath));
  } catch (error: any) {
    waveLoading.value = false;
    waveReady.value = false;
    message.error(error?.message || i18n.global.t("audio.waveLoadFailed"));
  }
}

async function refreshAudioInfo() {
  if (!audioPath.value) {
    audioInfo.value = null;
    return null;
  }

  loadingInfo.value = true;
  try {
    const source = await prepareAudioSource(
      audioPath.value,
      sourceInputFormat.value,
      isPcmSource.value ? sourceSampleRate.value : undefined,
      isPcmSource.value ? sourceChannels.value : undefined
    );
    audioInfo.value = source.info;
    return source;
  } catch (error: any) {
    audioInfo.value = null;
    message.error(error?.message || i18n.global.t("audio.infoLoadFailed"));
    return null;
  } finally {
    loadingInfo.value = false;
  }
}

async function reloadAudio() {
  if (!audioPath.value) {
    return;
  }
  await loadAudio(audioPath.value);
}

function togglePlayback() {
  if (!waveSurfer || !waveReady.value) {
    return;
  }
  waveSurfer.playPause();
}

function stopPlayback() {
  if (!waveSurfer || !waveReady.value) {
    return;
  }
  waveSurfer.stop();
  isPlaying.value = false;
  currentTime.value = 0;
}

function setActiveRegion(region: any) {
  if (activeRegion && activeRegion.id !== region.id) {
    activeRegion.remove();
  }
  activeRegion = region;
  syncSelection(region);
}

function syncSelection(region: any) {
  selectionStart.value = Math.max(0, Number(region?.start ?? 0));
  selectionEnd.value = Math.max(selectionStart.value, Number(region?.end ?? 0));
}

function clearSelection() {
  if (activeRegion) {
    activeRegion.remove();
    activeRegion = null;
  }
  selectionStart.value = 0;
  selectionEnd.value = 0;
  regionsPlugin?.clearRegions?.();
}

async function previewSelection() {
  if (!activeRegion || !hasSelection.value) {
    message.warning(i18n.global.t("audio.selectionRequired"));
    return;
  }
  await activeRegion.play(true);
}

async function convertAudio() {
  if (!audioPath.value) {
    return;
  }

  const selected = await saveAudioFileDialog(
    buildSuggestedOutputPath(audioPath.value, outputFormat.value),
    outputFormat.value
  );
  if (!selected) {
    return;
  }

  converting.value = true;
  try {
    conversionResult.value = await convertAudioFormat(
      audioPath.value,
      selected,
      outputFormat.value,
      sourceInputFormat.value,
      isPcmSource.value ? sourceSampleRate.value : undefined,
      isPcmSource.value ? sourceChannels.value : undefined,
      conversionSampleRate.value === "keep" ? undefined : conversionSampleRate.value,
      conversionChannels.value === "keep" ? undefined : conversionChannels.value
    );
    message.success(i18n.global.t("audio.convertSuccess"));
  } catch (error: any) {
    message.error(error?.message || i18n.global.t("audio.convertFailed"));
  } finally {
    converting.value = false;
  }
}

async function exportSelection() {
  if (!audioPath.value || !hasSelection.value) {
    message.warning(i18n.global.t("audio.selectionRequired"));
    return;
  }

  const selected = await saveAudioFileDialog(
    buildSuggestedClipPath(audioPath.value, outputFormat.value),
    outputFormat.value
  );
  if (!selected) {
    return;
  }

  clipping.value = true;
  try {
    conversionResult.value = await clipAudioSegment(
      audioPath.value,
      selected,
      outputFormat.value,
      sourceInputFormat.value,
      isPcmSource.value ? sourceSampleRate.value : undefined,
      isPcmSource.value ? sourceChannels.value : undefined,
      selectionStart.value,
      selectionEnd.value
    );
    message.success(i18n.global.t("audio.clipSuccess"));
  } catch (error: any) {
    message.error(error?.message || i18n.global.t("audio.clipFailed"));
  } finally {
    clipping.value = false;
  }
}

function buildSuggestedOutputPath(path: string, format: string) {
  if (!path) {
    return "";
  }
  const directory = dirname(path);
  const baseName = basename(path).replace(/\.[^.]+$/, "");
  return `${directory}/${baseName}-converted.${format}`;
}

function buildSuggestedClipPath(path: string, format: string) {
  if (!path) {
    return "";
  }
  const directory = dirname(path);
  const baseName = basename(path).replace(/\.[^.]+$/, "");
  return `${directory}/${baseName}-clip.${format}`;
}

function dirname(path: string) {
  return path.replace(/[\\/][^\\/]+$/, "");
}

function basename(path: string) {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

function formatDuration(value: number) {
  if (!Number.isFinite(value) || value <= 0) {
    return "00:00";
  }

  const totalSeconds = Math.floor(value);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return [hours, minutes, seconds]
      .map((item) => String(item).padStart(2, "0"))
      .join(":");
  }

  return [minutes, seconds]
    .map((item) => String(item).padStart(2, "0"))
    .join(":");
}

function formatHoverTime(value: number) {
  if (!Number.isFinite(value) || value < 0) {
    return "00:00.000";
  }

  const totalMilliseconds = Math.floor(value * 1000);
  const hours = Math.floor(totalMilliseconds / 3600000);
  const minutes = Math.floor((totalMilliseconds % 3600000) / 60000);
  const seconds = Math.floor((totalMilliseconds % 60000) / 1000);
  const milliseconds = totalMilliseconds % 1000;

  const timeParts =
    hours > 0
      ? [hours, minutes, seconds]
      : [minutes, seconds];

  return `${timeParts
    .map((item) => String(item).padStart(2, "0"))
    .join(":")}.${String(milliseconds).padStart(3, "0")}`;
}

onMounted(async () => {
  await nextTick();
  await ensureWaveSurfer();
});

onBeforeUnmount(() => {
  disableRegionDragSelection?.();
  disableRegionDragSelection = null;
  regionsPlugin = null;
  activeRegion = null;
  waveSurfer?.destroy();
  waveSurfer = null;
});
</script>

<style scoped>
.audio-page {
  height: 100%;
  min-height: 0;
  padding: 4px 4px 4px 0;
  box-sizing: border-box;
  overflow: auto;
}

.audio-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.95fr);
  gap: 12px;
  align-items: start;
}

.audio-panel {
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
  border-radius: 12px;
  padding: 16px;
  box-sizing: border-box;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
}

.audio-convert-panel {
  grid-column: 1 / -1;
}

.audio-panel-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 14px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--panel-border);
}

.audio-panel-header h3 {
  margin: 0 0 4px;
  color: var(--text-primary);
  font-size: 16px;
  font-weight: 600;
}

.audio-panel-header p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1.5;
}

.audio-panel-actions {
  display: flex;
  gap: 8px;
  flex-shrink: 0;
}

.wave-shell {
  position: relative;
  min-height: 200px;
  margin-top: 16px;
  border-radius: 10px;
  border: 1px solid var(--panel-border);
  background: var(--panel-bg-strong);
  overflow: hidden;
}

.wave-container {
  width: 100%;
  min-height: 200px;
  padding: 8px 0;
  box-sizing: border-box;
}

.wave-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-secondary);
  font-size: 14px;
  background: color-mix(in srgb, var(--panel-bg-strong) 88%, transparent);
}

.audio-player-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 14px;
  padding: 12px 14px;
  border: 1px solid var(--panel-border);
  border-radius: 10px;
  background: var(--panel-bg-strong);
}

.audio-player-actions {
  display: flex;
  gap: 8px;
}

.audio-player-stats {
  display: flex;
  gap: 6px;
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}

.tag-section {
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid var(--panel-border);
}

.selection-card {
  margin-top: 14px;
  padding: 12px 14px;
  border: 1px solid var(--panel-border);
  border-radius: 10px;
  background: var(--panel-bg-strong);
}

.source-config-card {
  margin-top: 14px;
  padding: 12px 14px;
  border: 1px solid var(--panel-border);
  border-radius: 10px;
  background: var(--panel-bg-strong);
}

.selection-card-header {
  margin-bottom: 12px;
}

.selection-card-title {
  margin-bottom: 4px;
  color: var(--text-primary);
  font-size: 14px;
  font-weight: 600;
}

.selection-card-description,
.selection-empty {
  color: var(--text-secondary);
  font-size: 12px;
  line-height: 1.5;
}

.selection-metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.selection-metric {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 10px 12px;
  border: 1px solid var(--panel-border);
  border-radius: 8px;
  background: var(--panel-bg);
}

.selection-metric span {
  color: var(--text-secondary);
  font-size: 12px;
}

.selection-metric strong {
  color: var(--text-primary);
  font-size: 14px;
  font-variant-numeric: tabular-nums;
}

.selection-actions {
  display: flex;
  gap: 10px;
  margin-top: 12px;
}

.tag-section-title {
  margin-bottom: 10px;
  color: var(--text-primary);
  font-size: 14px;
  font-weight: 600;
}

.tag-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.convert-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.convert-row {
  display: grid;
  grid-template-columns: 110px minmax(0, 1fr);
  gap: 12px;
  align-items: center;
}

.convert-row label {
  color: var(--text-secondary);
  font-size: 14px;
}

.convert-actions {
  display: flex;
  gap: 10px;
}

.audio-page :deep(.ant-empty-description),
.audio-page :deep(.ant-descriptions-item-label),
.audio-page :deep(.ant-descriptions-item-content),
.audio-page :deep(.ant-select-selection-item),
.audio-page :deep(.ant-input),
.audio-page :deep(.ant-alert-message),
.audio-page :deep(.ant-alert-description) {
  color: var(--text-primary);
}

.audio-page :deep(.ant-input),
.audio-page :deep(.ant-select-selector) {
  background: var(--panel-bg-strong) !important;
  border-color: var(--panel-border) !important;
  border-radius: 8px !important;
}

.audio-page :deep(.ant-input[readonly]) {
  color: var(--text-secondary);
}

.audio-page :deep(.ant-input::placeholder) {
  color: var(--text-secondary);
}

.audio-page :deep(.ant-select-selector:hover),
.audio-page :deep(.ant-input:hover),
.audio-page :deep(.ant-input:focus),
.audio-page :deep(.ant-select-focused .ant-select-selector) {
  border-color: var(--accent) !important;
  box-shadow: none !important;
}

.audio-page :deep(.ant-descriptions) {
  background: var(--panel-bg);
}

.audio-page :deep(.ant-descriptions-view) {
  border-color: var(--panel-border);
}

.audio-page :deep(.ant-descriptions-row > th),
.audio-page :deep(.ant-descriptions-row > td) {
  border-color: var(--panel-border);
}

.audio-page :deep(.ant-descriptions-item-label) {
  background: var(--panel-bg-strong);
  color: var(--text-secondary);
}

.audio-page :deep(.ant-descriptions-item-content) {
  background: var(--panel-bg);
}

.audio-page :deep(.ant-tag) {
  margin-right: 0;
  padding: 4px 8px;
  border-radius: 999px;
}

.audio-page :deep(.ant-alert.ant-alert-success) {
  background: color-mix(in srgb, var(--accent-soft) 55%, var(--panel-bg));
  border-color: color-mix(in srgb, var(--accent) 26%, var(--panel-border));
}

.audio-page :deep(.ant-empty) {
  margin: 32px 0;
}

@media (max-width: 1100px) {
  .audio-grid {
    grid-template-columns: 1fr;
  }

  .audio-convert-panel {
    grid-column: auto;
  }
}

@media (max-width: 720px) {
  .audio-panel {
    padding: 14px;
  }

  .audio-panel-header,
  .audio-player-bar {
    flex-direction: column;
    align-items: stretch;
  }

  .convert-row {
    grid-template-columns: 1fr;
  }

  .selection-metrics {
    grid-template-columns: 1fr;
  }

  .audio-panel-actions,
  .audio-player-actions,
  .convert-actions,
  .selection-actions {
    width: 100%;
  }

  .audio-panel-actions :deep(.ant-btn),
  .audio-player-actions :deep(.ant-btn),
  .convert-actions :deep(.ant-btn),
  .selection-actions :deep(.ant-btn) {
    flex: 1;
  }
}
</style>
