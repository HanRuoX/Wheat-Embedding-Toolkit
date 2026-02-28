<template>
  <Teleport to="#main-header-actions">
    <a-popover
      v-model:open="exportPopoverOpen"
      trigger="click"
      placement="bottomRight"
      @openChange="onExportPopoverOpenChange"
    >
      <template #content>
        <div class="export-popover">
          <div class="export-popover-title">{{ $t("serial.exportOptions") }}</div>
          <a-checkbox v-model:checked="exportReceiveHex">{{ $t("serial.displayHex") }}</a-checkbox>
          <a-checkbox v-model:checked="exportShowTimestamp">{{ $t("serial.displayTime") }}</a-checkbox>
          <a-checkbox v-model:checked="exportShowTxRx">{{ $t("serial.displayTxRx") }}</a-checkbox>
          <a-button type="primary" size="small" :disabled="historyRecords.length === 0" @click="confirmExportLogs">
            {{ $t("serial.exportNow") }}
          </a-button>
        </div>
      </template>
      <a-button type="default" :disabled="historyRecords.length === 0">
        {{ $t("serial.export") }}
      </a-button>
    </a-popover>
  </Teleport>
  <div class="serial-page" :class="{ 'is-light': preferenceStore.resolvedTheme === 'light' }">
    <div class="serial-main">
      <div class="left-panel">
        <div class="terminal-wrap">
          <div v-if="searchBarVisible" class="terminal-search">
            <a-input
              ref="searchInputRef"
              v-model:value="findKeyword"
              size="small"
              @pressEnter="onSearchInputEnter"
            />
            <a-button
              class="search-toggle"
              size="small"
              :type="searchCaseSensitive ? 'primary' : 'default'"
              :title="$t('serial.findCaseSensitive')"
              @click="toggleSearchCaseSensitive"
            >
              Aa
            </a-button>
            <span class="search-count">
              {{ currentSearchDisplayIndex }}/{{ searchMatches.length }}
            </span>
            <a-button class="search-nav" size="small" :title="$t('serial.findPrev')" @click="jumpToPreviousSearchMatch">
              ↑
            </a-button>
            <a-button class="search-nav" size="small" :title="$t('serial.findNext')" @click="jumpToNextSearchMatch">
              ↓
            </a-button>
            <button class="search-close" type="button" @click="closeSearchBar">×</button>
          </div>
          <div ref="terminalContainer" class="terminal-view" />
        </div>
        <div class="bottom-bar">
          <a-textarea
            class="send-input"
            v-model:value="sendInput"
            :auto-size="{ minRows: 4, maxRows: 8 }"
            :placeholder="sendPlaceholder"
            @keydown="onSendInputKeydown"
          />
          <div class="bottom-actions">
            <a-button type="primary" @click="clearTerminal">{{ $t("serial.clear") }}</a-button>
            <a-button
              type="primary"
              :danger="periodicSend && periodicRunning"
              :disabled="!connected"
              @click="onSendButtonClick"
            >
              {{ sendButtonText }}
            </a-button>
          </div>
        </div>
      </div>
      <div class="right-panel">
        <div class="settings-card">
          <div class="setting-row">
            <label>{{ $t("serial.port") }}:</label>
            <a-select
              v-model:value="selectedPort"
              :options="serialPortOptions"
              show-search
              placement="bottomRight"
              :popup-match-select-width="false"
              :dropdown-match-select-width="false"
              popup-class-name="serial-port-dropdown"
              :title="selectedPort"
              @focus="refreshPorts"
            />
          </div>
          <div class="setting-row">
            <label>{{ $t("serial.baudRate") }}:</label>
            <a-select v-model:value="selectedBaudRate" :options="baudRateOptions" />
          </div>
          <div class="setting-row">
            <label>{{ $t("serial.parity") }}:</label>
            <a-select v-model:value="selectedParity" :options="parityOptions" />
          </div>
          <div class="setting-row">
            <label>{{ $t("serial.dataBits") }}:</label>
            <a-select v-model:value="selectedDataBits" :options="dataBitsOptions" />
          </div>
          <div class="setting-row">
            <label>{{ $t("serial.stopBits") }}:</label>
            <a-select v-model:value="selectedStopBits" :options="stopBitsOptions" />
          </div>
          <div class="setting-row">
            <label>{{ $t("serial.flowControl") }}:</label>
            <a-select v-model:value="selectedFlowControl" :options="flowControlOptions" />
          </div>
        </div>
        <a-button class="open-btn" type="primary" :danger="connected" @click="toggleConnection">
          {{ connectButtonText }}
        </a-button>

        <a-card size="small" class="panel-card" :title="$t('serial.sendPanel')">
          <div class="check-row">
            <a-checkbox v-model:checked="sendHex">{{ $t("serial.sendHex") }}</a-checkbox>
            <a-checkbox v-model:checked="sendNewline">{{ $t("serial.sendNewline") }}</a-checkbox>
          </div>
          <div class="check-row">
            <a-checkbox v-model:checked="periodicSend">{{ $t("serial.periodicSend") }}</a-checkbox>
            <a-input-number
              v-model:value="periodicInterval"
              :min="10"
              :step="10"
              size="small"
              style="width: 112px"
              :disabled="!periodicSend"
            />
            <span>{{ $t("serial.ms") }}</span>
          </div>
          <div class="check-row">
            <a-checkbox v-model:checked="rts" :disabled="!connected">RTS</a-checkbox>
            <a-checkbox v-model:checked="dtr" :disabled="!connected">DTR</a-checkbox>
          </div>
        </a-card>

        <a-card size="small" class="panel-card" :title="$t('serial.receivePanel')">
          <div class="check-row">
            <a-checkbox v-model:checked="receiveHex">{{ $t("serial.displayHex") }}</a-checkbox>
          </div>
          <div class="check-row">
            <a-checkbox v-model:checked="showTimestamp">{{ $t("serial.displayTime") }}</a-checkbox>
            <a-checkbox v-model:checked="showTxRx">{{ $t("serial.displayTxRx") }}</a-checkbox>
          </div>
        </a-card>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { message } from "ant-design-vue";
import { Terminal } from "xterm";
import { FitAddon } from "xterm-addon-fit";
import moment from "moment";
import "xterm/css/xterm.css";
import { getSerialPortList, saveTextFileDialog, writeAllText } from "@/utils/common";
import {
  SerialAssistantEventPayload,
  serialAssistantClose,
  serialAssistantIsOpen,
  serialAssistantOpen,
  serialAssistantSend,
  serialAssistantSetSignals,
} from "@/utils/serial";
import i18n from "@/locales/i18n";
import { usePreferenceStore, type ResolvedTheme } from "@/stores/Preference";

type SelectOption = { label: string; value: string };
type SerialHistoryRecord = {
  kind: "data" | "info" | "error";
  direction?: "TX" | "RX";
  text: string;
  hex: string;
  timestamp: number;
};

const terminalContainer = ref<HTMLElement | null>(null);
const preferenceStore = usePreferenceStore();
const serialPortOptions = ref<SelectOption[]>([]);
const selectedPort = ref<string | undefined>(localStorage.getItem("port") ?? undefined);
const selectedBaudRate = ref("115200");
const selectedParity = ref("none");
const selectedDataBits = ref("8");
const selectedStopBits = ref("1");
const selectedFlowControl = ref("none");

const getStoredBoolean = (key: string, fallback: boolean) => {
  const value = localStorage.getItem(key);
  if (value === null) {
    return fallback;
  }
  return value === "1";
};

const connected = ref(false);
const sendHex = ref(false);
const sendNewline = ref(getStoredBoolean("serial.sendNewline", true));
const periodicSend = ref(false);
const periodicRunning = ref(false);
const periodicInterval = ref(1000);
const rts = ref(false);
const dtr = ref(false);
const receiveHex = ref(false);
const showTimestamp = ref(false);
const showTxRx = ref(false);
const exportReceiveHex = ref(false);
const exportShowTimestamp = ref(false);
const exportShowTxRx = ref(false);
const exportPopoverOpen = ref(false);
const findKeyword = ref("");
const searchCaseSensitive = ref(false);
const searchBarVisible = ref(false);
const activeSearchKeyword = ref("");
const currentSearchMatchIndex = ref(-1);
const sendInput = ref("");
const logEntries = ref<string[]>([]);
const historyRecords = ref<SerialHistoryRecord[]>([]);
const searchMatches = ref<SearchMatch[]>([]);
const searchInputRef = ref<{ focus: () => void } | null>(null);
type SearchMatch = {
  lineIndex: number;
  column: number;
  length: number;
};

const baudRateOptions: SelectOption[] = [
  "9600",
  "19200",
  "38400",
  "57600",
  "115200",
  "230400",
  "460800",
  "921600",
  "1152000",
].map((value) => ({ label: value, value }));
const dataBitsOptions: SelectOption[] = ["5", "6", "7", "8"].map((value) => ({ label: value, value }));
const stopBitsOptions: SelectOption[] = ["1", "2"].map((value) => ({ label: value, value }));
const parityOptions: SelectOption[] = [
  { label: i18n.global.t("serial.parityNone"), value: "none" },
  { label: i18n.global.t("serial.parityOdd"), value: "odd" },
  { label: i18n.global.t("serial.parityEven"), value: "even" },
];
const flowControlOptions: SelectOption[] = [
  { label: i18n.global.t("serial.flowNone"), value: "none" },
  { label: i18n.global.t("serial.flowSoftware"), value: "software" },
  { label: i18n.global.t("serial.flowHardware"), value: "hardware" },
];

const connectButtonText = computed(() =>
  connected.value ? i18n.global.t("serial.disconnect") : i18n.global.t("serial.connect")
);
const sendButtonText = computed(() => {
  if (periodicSend.value && periodicRunning.value) {
    return i18n.global.locale.value === "zh" ? "停止" : "Stop";
  }
  return i18n.global.t("serial.send");
});
const sendPlaceholder = computed(() =>
  sendHex.value ? i18n.global.t("serial.sendHexPlaceholder") : i18n.global.t("serial.sendTextPlaceholder")
);

const fitAddon = new FitAddon();
let terminal: Terminal | null = null;
let periodicTimer: ReturnType<typeof setTimeout> | null = null;
let unlistenSerial: UnlistenFn | null = null;
let resizeHandler: (() => void) | null = null;
let terminalResizeObserver: ResizeObserver | null = null;
let periodicRunId = 0;
const sendInFlight = ref(false);
let isDisplayLineOpen = false;
let lastDisplayDirection: "TX" | "RX" | null = null;
let pendingRxEspColor: string | null = null;
let renderSearchOccurrenceCursor = 0;
let pendingSearchScrollFrame: number | null = null;

const getTerminalTheme = (themeMode: ResolvedTheme) =>
  themeMode === "light"
    ? {
        background: "#f5f7fb",
        foreground: "#1f2a37",
        cursor: "#2563eb",
        cursorAccent: "#f5f7fb",
        selectionForeground: "#111827",
        selectionBackground: "#ffd24d",
        selectionInactiveBackground: "#ffd24d",
        black: "#1f2937",
        red: "#b42318",
        green: "#157347",
        yellow: "#9a6700",
        blue: "#1d4ed8",
        magenta: "#9f2a8a",
        cyan: "#0b7285",
        white: "#d0d7de",
        brightBlack: "#6b7280",
        brightRed: "#d92d20",
        brightGreen: "#2f9e44",
        brightYellow: "#b58105",
        brightBlue: "#2563eb",
        brightMagenta: "#b83280",
        brightCyan: "#1098ad",
        brightWhite: "#111827",
      }
    : {
        background: "#1d1f27",
        foreground: "#f2f2f2",
        cursor: "#3a8bff",
        cursorAccent: "#1d1f27",
        selectionForeground: "#111111",
        selectionBackground: "#ffb020",
        selectionInactiveBackground: "#ffb020",
      };

const ansi = {
  reset: "\x1b[0m",
  dim: "\x1b[2m",
  info: "\x1b[1;32m",
  warn: "\x1b[1;33m",
  error: "\x1b[1;31m",
  debug: "\x1b[1;36m",
  verbose: "\x1b[0;37m",
  tag: "\x1b[1;34m",
  tx: "\x1b[1;35m",
  rx: "\x1b[1;34m",
  searchStart: "\x1b[7m",
  searchEnd: "\x1b[27m",
  currentSearchStart: "\x1b[48;5;220m\x1b[30m\x1b[1m",
  currentSearchEnd: "\x1b[49m\x1b[39m\x1b[22m",
};
const ansiRegex = /\x1b\[[0-9;]*m/g;
const espLogHeaderRegex = /^([IWEVD]) \((\d+)\)\s+([^:]+):\s?(.*)$/;

const normalizeAnsiEscapes = (text: string) =>
  text.replace(/\\033\[/g, "\x1b[").replace(/\\x1b\[/gi, "\x1b[");

const hasAnsiEscapes = (text: string) =>
  /\x1b\[[0-9;]*m/.test(normalizeAnsiEscapes(text));

const bytesToHex = (data: number[]) =>
  data.map((item) => item.toString(16).padStart(2, "0")).join(" ").toUpperCase();

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const getSearchMatcher = (keyword: string, global = true) => {
  const flags = `${global ? "g" : ""}${searchCaseSensitive.value ? "" : "i"}`;
  return new RegExp(escapeRegExp(keyword), flags);
};

const applySearchHighlight = (text: string) => {
  const keyword = activeSearchKeyword.value.trim();
  if (!keyword) {
    return text;
  }

  const matcher = getSearchMatcher(keyword);
  return text
    .split(/(\x1b\[[0-9;]*m)/g)
    .map((part) => {
      if (/^\x1b\[[0-9;]*m$/.test(part)) {
        return part;
      }
      return part.replace(matcher, (matched) => {
        const occurrenceIndex = renderSearchOccurrenceCursor;
        renderSearchOccurrenceCursor += 1;
        return occurrenceIndex === currentSearchMatchIndex.value
          ? `${ansi.currentSearchStart}${matched}${ansi.currentSearchEnd}`
          : `${ansi.searchStart}${matched}${ansi.searchEnd}`;
      });
    })
    .join("");
};

const getEspLevelColor = (level: string) =>
  level === "I"
    ? ansi.info
    : level === "W"
      ? ansi.warn
      : level === "E"
        ? ansi.error
        : level === "D"
          ? ansi.debug
          : ansi.verbose;

const colorizeEspLogLine = (line: string) => {
  if (!line || hasAnsiEscapes(line)) {
    return line;
  }

  const match = line.match(espLogHeaderRegex);
  if (!match) {
    return line;
  }

  const [, level, ms, tag, messageText] = match;
  const levelColor = getEspLevelColor(level);

  return `${levelColor}${level}${ansi.reset} (${ansi.dim}${ms}${ansi.reset}) ${ansi.tag}${tag}${ansi.reset}${levelColor}: ${messageText}${ansi.reset}`;
};

const colorizeEspLogText = (text: string) => {
  const normalizedText = normalizeAnsiEscapes(text);
  if (!normalizedText) {
    return normalizedText;
  }

  return normalizedText
    .split(/(\r\n|\n|\r)/)
    .map((segment) => {
      if (segment === "\r\n" || segment === "\n" || segment === "\r") {
        pendingRxEspColor = null;
        return segment;
      }

      if (!segment) {
        return segment;
      }

      if (hasAnsiEscapes(segment)) {
        return segment;
      }

      const headerMatch = segment.match(espLogHeaderRegex);
      if (headerMatch) {
        const [, level, ms, tag, messageText] = headerMatch;
        const levelColor = getEspLevelColor(level);
        pendingRxEspColor = levelColor;
        return `${levelColor}${level}${ansi.reset} (${ansi.dim}${ms}${ansi.reset}) ${ansi.tag}${tag}${ansi.reset}${levelColor}: ${messageText}${ansi.reset}`;
      }

      if (pendingRxEspColor) {
        return `${pendingRxEspColor}${segment}${ansi.reset}`;
      }

      return segment;
    })
    .join("");
};

const stripAnsiForExport = (text: string) => normalizeAnsiEscapes(text).replace(ansiRegex, "");

const buildExportPrefix = (record: SerialHistoryRecord) => {
  const parts: string[] = [];

  if (exportShowTimestamp.value) {
    parts.push(`[${moment(record.timestamp).format("HH:mm:ss.SSS")}]`);
  }

  if (record.kind === "data" && record.direction && exportShowTxRx.value) {
    parts.push(`[${record.direction}]`);
  }

  if (record.kind === "info") {
    parts.push("[INFO]");
  }

  if (record.kind === "error") {
    parts.push("[ERROR]");
  }

  return parts.join(" ");
};

const formatExportRecord = (record: SerialHistoryRecord) => {
  const prefix = buildExportPrefix(record);
  const content =
    record.kind === "data" && exportReceiveHex.value ? record.hex : stripAnsiForExport(record.text);
  const normalized = content.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const lines = normalized.split("\n");

  if (lines.length === 0) {
    return prefix;
  }

  if (!prefix) {
    return lines.join("\n");
  }

  return lines
    .map((line, index) => (index === 0 ? `${prefix} ${line}` : `${" ".repeat(prefix.length + 1)}${line}`))
    .join("\n");
};

const buildExportContent = () => {
  const exportedAt = moment().format("YYYY-MM-DD HH:mm:ss");
  const header = [
    `# Serial Assistant Export`,
    `# Exported At: ${exportedAt}`,
    `# Port: ${selectedPort.value ?? "-"}`,
    `# Baud Rate: ${selectedBaudRate.value}`,
    "",
  ];

  return `${header.join("\n")}${historyRecords.value.map((record) => formatExportRecord(record)).join("\n")}\n`;
};

const parseHexInput = (value: string): number[] => {
  const normalized = value.replace(/0x/gi, "").replace(/\s+/g, "");
  if (normalized.length === 0) {
    return [];
  }
  if (normalized.length % 2 !== 0) {
    throw new Error(i18n.global.t("serial.hexLengthError"));
  }
  if (!/^[0-9a-fA-F]+$/.test(normalized)) {
    throw new Error(i18n.global.t("serial.hexFormatError"));
  }

  const data: number[] = [];
  for (let index = 0; index < normalized.length; index += 2) {
    data.push(parseInt(normalized.slice(index, index + 2), 16));
  }
  return data;
};

const splitAndSave = (text: string) => {
  text
    .replace(/\\033\[/g, "")
    .replace(/\\x1b\[/gi, "")
    .replace(ansiRegex, "")
    .replace(/\r\n/g, "\n")
    .split("\n")
    .filter((line) => line.length > 0)
    .forEach((line) => {
      logEntries.value.push(line);
      if (logEntries.value.length > 5000) {
        logEntries.value.shift();
      }
    });
};

const shouldStickToBottom = () => {
  if (!terminal) {
    return true;
  }
  const buffer = terminal.buffer.active;
  return buffer.baseY - buffer.viewportY <= 1;
};

const fitTerminal = () => {
  if (!terminal || !terminalContainer.value) {
    return;
  }

  nextTick(() => {
    requestAnimationFrame(() => {
      fitAddon.fit();
      terminal.scrollToBottom();
    });
  });
};

const focusSearchInput = () => {
  nextTick(() => {
    searchInputRef.value?.focus?.();
    const input = document.querySelector(".terminal-search .ant-input") as HTMLInputElement | null;
    input?.select();
  });
};

const openSearchBar = () => {
  searchBarVisible.value = true;
  focusSearchInput();
};

const openSearchBarWithInitialKeyword = (keyword?: string) => {
  openSearchBar();
  const normalizedKeyword = (keyword ?? "").trim();
  if (!normalizedKeyword) {
    return;
  }
  if (findKeyword.value !== normalizedKeyword) {
    findKeyword.value = normalizedKeyword;
    return;
  }
  findKeyword.value = normalizedKeyword;
  updateSearchResults(normalizedKeyword, true);
};

const closeSearchBar = () => {
  searchBarVisible.value = false;
  activeSearchKeyword.value = "";
  currentSearchMatchIndex.value = -1;
  replayHistory();
};

const updateSearchResults = (keyword: string, resetCurrent = false) => {
  const normalizedKeyword = keyword.trim();
  if (!normalizedKeyword) {
    activeSearchKeyword.value = "";
    currentSearchMatchIndex.value = -1;
    searchMatches.value = [];
    replayHistory();
    return;
  }

  const keywordChanged = normalizedKeyword !== activeSearchKeyword.value;
  activeSearchKeyword.value = normalizedKeyword;
  refreshSearchMatches();
  if (searchMatches.value.length === 0) {
    currentSearchMatchIndex.value = -1;
    replayHistory();
    return;
  }

  if (resetCurrent || keywordChanged) {
    currentSearchMatchIndex.value = 0;
  } else if (currentSearchMatchIndex.value >= searchMatches.value.length) {
    currentSearchMatchIndex.value = 0;
  }

  replayHistory();
};

const getTerminalSearchMatches = (keyword: string) => {
  if (!terminal) {
    return [] as SearchMatch[];
  }

  const normalizedKeyword = keyword.trim();
  if (!normalizedKeyword) {
    return [] as SearchMatch[];
  }

  const matcher = getSearchMatcher(normalizedKeyword);
  const matches: SearchMatch[] = [];
  const buffer = terminal.buffer.active;

  for (let lineIndex = 0; lineIndex < buffer.length; lineIndex += 1) {
    const line = buffer.getLine(lineIndex)?.translateToString(true) ?? "";
    matcher.lastIndex = 0;
    let matched: RegExpExecArray | null = null;
    while ((matched = matcher.exec(line)) !== null) {
      matches.push({
        lineIndex,
        column: matched.index,
        length: matched[0].length,
      });
      if (matched[0].length === 0) {
        matcher.lastIndex += 1;
      }
    }
  }

  return matches;
};

const currentSearchDisplayIndex = computed(() =>
  searchMatches.value.length === 0 || currentSearchMatchIndex.value < 0 ? 0 : currentSearchMatchIndex.value + 1
);

const refreshSearchMatches = () => {
  searchMatches.value = getTerminalSearchMatches(activeSearchKeyword.value || findKeyword.value);
};

const selectSearchMatch = (match: SearchMatch) => {
  if (!terminal) {
    return;
  }

  if (pendingSearchScrollFrame !== null) {
    cancelAnimationFrame(pendingSearchScrollFrame);
  }

  const targetLine = Math.max(match.lineIndex - 1, 0);
  pendingSearchScrollFrame = requestAnimationFrame(() => {
    terminal?.scrollToLine(targetLine);
    pendingSearchScrollFrame = null;
  });
};

const syncCurrentSearchSelection = () => {
  if (!activeSearchKeyword.value.trim()) {
    currentSearchMatchIndex.value = -1;
    searchMatches.value = [];
    return;
  }

  if (searchMatches.value.length === 0) {
    currentSearchMatchIndex.value = -1;
    return;
  }

  if (currentSearchMatchIndex.value < 0) {
    return;
  }

  const boundedIndex = Math.min(Math.max(currentSearchMatchIndex.value, 0), searchMatches.value.length - 1);
  currentSearchMatchIndex.value = boundedIndex;
  selectSearchMatch(searchMatches.value[boundedIndex]);
};

const moveSearchMatch = (step: 1 | -1) => {
  const keyword = activeSearchKeyword.value.trim() || findKeyword.value.trim();
  if (!keyword) {
    message.warning(i18n.global.t("serial.findKeywordRequired"));
    return;
  }

  activeSearchKeyword.value = keyword;
  refreshSearchMatches();
  if (searchMatches.value.length === 0) {
    currentSearchMatchIndex.value = -1;
    replayHistory();
    message.info(i18n.global.t("serial.findNotFound"));
    return;
  }

  if (currentSearchMatchIndex.value < 0) {
    currentSearchMatchIndex.value = step > 0 ? 0 : searchMatches.value.length - 1;
  } else {
    currentSearchMatchIndex.value =
      (currentSearchMatchIndex.value + step + searchMatches.value.length) % searchMatches.value.length;
  }

  replayHistory();
};

const jumpToNextSearchMatch = () => {
  moveSearchMatch(1);
};

const jumpToPreviousSearchMatch = () => {
  moveSearchMatch(-1);
};

const toggleSearchCaseSensitive = () => {
  searchCaseSensitive.value = !searchCaseSensitive.value;
  if (!searchBarVisible.value) {
    openSearchBar();
  }
  if (!activeSearchKeyword.value.trim() && !findKeyword.value.trim()) {
    return;
  }
  currentSearchMatchIndex.value = -1;
  replayHistory();
  if (findKeyword.value.trim()) {
    activeSearchKeyword.value = findKeyword.value.trim();
    jumpToNextSearchMatch();
  }
};

const onSearchInputEnter = (event: KeyboardEvent) => {
  if (event.shiftKey) {
    jumpToPreviousSearchMatch();
    return;
  }
  jumpToNextSearchMatch();
};

const onWindowKeydown = (event: KeyboardEvent) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "f") {
    event.preventDefault();
    const selectedText = terminal?.getSelection()?.trim() ?? "";
    openSearchBarWithInitialKeyword(selectedText);
    return;
  }

  if (!searchBarVisible.value) {
    return;
  }

  if (event.key === "Escape") {
    event.preventDefault();
    closeSearchBar();
  }
};

const writeRaw = (text: string) => {
  if (!terminal || !text) {
    return;
  }
  const stickToBottom = shouldStickToBottom();
  terminal.write(normalizeAnsiEscapes(text));
  if (stickToBottom) {
    terminal.scrollToBottom();
  }
  splitAndSave(text);
  isDisplayLineOpen = !/[\r\n]$/.test(text);
};

const writeLine = (text: string) => {
  if (!terminal) {
    return;
  }
  const stickToBottom = shouldStickToBottom();
  terminal.writeln(normalizeAnsiEscapes(text));
  if (stickToBottom) {
    terminal.scrollToBottom();
  }
  splitAndSave(text);
  isDisplayLineOpen = false;
};

const startNewDisplayLineIfNeeded = () => {
  if (!terminal || !isDisplayLineOpen) {
    return;
  }
  const stickToBottom = shouldStickToBottom();
  terminal.write("\r\n");
  if (stickToBottom) {
    terminal.scrollToBottom();
  }
  isDisplayLineOpen = false;
};

const formatPrefix = (direction: "TX" | "RX", timestamp = Date.now()) => {
  let prefix = "";
  if (showTimestamp.value) {
    prefix += `${ansi.dim}[${moment(timestamp).format("HH:mm:ss.SSS")}]${ansi.reset} `;
  }
  if (showTxRx.value) {
    const ioColor = direction === "TX" ? ansi.tx : ansi.rx;
    prefix += `${ioColor}[${direction}]${ansi.reset} `;
  }
  return prefix;
};

const shouldRenderRecordOnOwnLine = () => {
  return receiveHex.value || showTimestamp.value || showTxRx.value;
};

const renderHistoryRecord = (record: SerialHistoryRecord) => {
  if (record.kind === "error") {
    writeLine(applySearchHighlight(`${ansi.error}[ERROR]${ansi.reset} ${record.text}`));
    return;
  }

  if (record.kind === "info") {
    writeLine(applySearchHighlight(`${ansi.info}[INFO]${ansi.reset} ${record.text}`));
    return;
  }

  if (!record.direction) {
    return;
  }

  const prefix = formatPrefix(record.direction, record.timestamp);
  const displayText =
    record.direction === "RX"
      ? applySearchHighlight(`${prefix}${colorizeEspLogText(record.text)}`)
      : applySearchHighlight(`${prefix}${normalizeAnsiEscapes(record.text)}`);
  if (receiveHex.value) {
    startNewDisplayLineIfNeeded();
    writeLine(applySearchHighlight(`${prefix}${record.hex}`));
    lastDisplayDirection = record.direction;
    return;
  }

  if (shouldRenderRecordOnOwnLine()) {
    startNewDisplayLineIfNeeded();
    if (record.direction === "RX") {
      writeLine(displayText);
      lastDisplayDirection = "RX";
      return;
    }

    writeLine(displayText);
    lastDisplayDirection = "TX";
    return;
  }

  if (record.direction === "RX") {
    writeRaw(displayText);
    lastDisplayDirection = "RX";
    return;
  }

  startNewDisplayLineIfNeeded();
  writeRaw(displayText);
  lastDisplayDirection = "TX";
};

const appendHistoryRecord = (record: SerialHistoryRecord) => {
  historyRecords.value.push(record);
  if (historyRecords.value.length > 5000) {
    historyRecords.value.shift();
  }
};

const replayHistory = () => {
  if (!terminal) {
    return;
  }
  terminal.clear();
  logEntries.value = [];
  isDisplayLineOpen = false;
  lastDisplayDirection = null;
  pendingRxEspColor = null;
  renderSearchOccurrenceCursor = 0;
  historyRecords.value.forEach((record) => {
    renderHistoryRecord(record);
  });
  syncCurrentSearchSelection();
};

const appendLineEnding = (data: number[]) => {
  if (!sendNewline.value) {
    return data;
  }
  return [...data, 0x0d, 0x0a];
};

const refreshPorts = async () => {
  const list = await getSerialPortList();
  serialPortOptions.value = list.map((item: string) => ({ label: item, value: item }));

  if (!selectedPort.value || !list.includes(selectedPort.value)) {
    selectedPort.value = list[0];
  }

  if (selectedPort.value) {
    localStorage.setItem("port", selectedPort.value);
  }
};

const stopPeriodicSend = () => {
  periodicRunId += 1;
  if (periodicTimer) {
    clearTimeout(periodicTimer);
    periodicTimer = null;
  }
  periodicRunning.value = false;
};

const scheduleNextPeriodicSend = (runId: number) => {
  if (!periodicRunning.value || runId !== periodicRunId || !periodicSend.value || !connected.value) {
    return;
  }

  const interval = Math.max(10, Number(periodicInterval.value) || 1000);
  periodicTimer = setTimeout(async () => {
    if (runId !== periodicRunId || !periodicRunning.value) {
      return;
    }

    await sendData(true);

    if (runId !== periodicRunId || !periodicRunning.value) {
      return;
    }
    scheduleNextPeriodicSend(runId);
  }, interval);
};

const startPeriodicSend = () => {
  stopPeriodicSend();
  if (!periodicSend.value || !connected.value) {
    return;
  }
  const runId = periodicRunId;
  periodicRunning.value = true;
  void (async () => {
    await sendData(true);
    if (runId !== periodicRunId || !periodicRunning.value) {
      return;
    }
    scheduleNextPeriodicSend(runId);
  })();
};

const applySignals = async (silent = true) => {
  if (!connected.value) {
    return;
  }
  try {
    await serialAssistantSetSignals(rts.value, dtr.value);
  } catch (error) {
    if (!silent) {
      message.error(String(error));
    }
  }
};

const sendData = async (fromTimer = false) => {
  if (sendInFlight.value) {
    return;
  }

  sendInFlight.value = true;
  try {
    if (!connected.value) {
      if (!fromTimer) {
        message.warning(i18n.global.t("serial.notConnected"));
      }
      return;
    }

    let payload: number[];
    if (sendHex.value) {
      payload = parseHexInput(sendInput.value);
    } else {
      payload = Array.from(new TextEncoder().encode(sendInput.value));
    }
    payload = appendLineEnding(payload);

    if (payload.length === 0) {
      if (!fromTimer) {
        message.warning(i18n.global.t("serial.emptyPayload"));
      }
      return;
    }

    const txRecord: SerialHistoryRecord = {
      kind: "data",
      direction: "TX",
      text: new TextDecoder().decode(Uint8Array.from(payload)),
      hex: bytesToHex(payload),
      timestamp: Date.now(),
    };
    appendHistoryRecord(txRecord);
    renderHistoryRecord(txRecord);
    syncCurrentSearchSelection();

    await serialAssistantSend(payload);
  } catch (error) {
    if (fromTimer) {
      stopPeriodicSend();
    }
    if (!fromTimer) {
      message.error(String(error));
    }
  } finally {
    sendInFlight.value = false;
  }
};

const onSendInputKeydown = (event: KeyboardEvent) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    void onSendButtonClick();
  }
};

const onSendButtonClick = async () => {
  if (!connected.value) {
    message.warning(i18n.global.t("serial.notConnected"));
    return;
  }

  if (!periodicSend.value) {
    await sendData();
    return;
  }

  if (periodicRunning.value) {
    stopPeriodicSend();
  } else {
    startPeriodicSend();
  }
};

const findInLogs = () => {
  if (!searchBarVisible.value) {
    openSearchBar();
  }
  const keyword = findKeyword.value.trim();
  if (!keyword) {
    message.warning(i18n.global.t("serial.findKeywordRequired"));
    return;
  }
  updateSearchResults(keyword);
  const matches = searchMatches.value;
  if (matches.length === 0) {
    message.info(i18n.global.t("serial.findNotFound"));
    return;
  }
  jumpToNextSearchMatch();
  message.success(i18n.global.t("serial.findResult", { count: matches.length }));
};

const exportLogs = async () => {
  if (historyRecords.value.length === 0) {
    message.warning(i18n.global.t("serial.exportEmpty"));
    return;
  }

  const portName = (selectedPort.value ?? "serial")
    .replace(/[\\/:"*?<>|]+/g, "_")
    .replace(/\s+/g, "_");
  const defaultPath = `${portName}-${moment().format("YYYYMMDD-HHmmss")}.log`;

  try {
    const filePath = await saveTextFileDialog(defaultPath);
    if (!filePath) {
      return;
    }
    await writeAllText(filePath, buildExportContent());
    message.success(i18n.global.t("serial.exportSuccess"));
    exportPopoverOpen.value = false;
  } catch (error) {
    message.error(String(error));
  }
};

const syncExportOptionsFromDisplay = () => {
  exportReceiveHex.value = receiveHex.value;
  exportShowTimestamp.value = showTimestamp.value;
  exportShowTxRx.value = showTxRx.value;
};

const onExportPopoverOpenChange = (open: boolean) => {
  exportPopoverOpen.value = open;
  if (open) {
    syncExportOptionsFromDisplay();
  }
};

const confirmExportLogs = async () => {
  await exportLogs();
};

const clearTerminal = () => {
  terminal?.clear();
  logEntries.value = [];
  historyRecords.value = [];
  activeSearchKeyword.value = "";
  currentSearchMatchIndex.value = -1;
  searchMatches.value = [];
  isDisplayLineOpen = false;
  lastDisplayDirection = null;
  pendingRxEspColor = null;
};

const handleSerialEvent = (payload: SerialAssistantEventPayload) => {
  if (!payload) {
    return;
  }

  if (payload.kind === "data") {
    const rxText = payload.text ?? "";
    const rxHex =
      payload.hex && payload.hex.length > 0
        ? payload.hex
        : bytesToHex(Array.from(new TextEncoder().encode(rxText)));
    const rxRecord: SerialHistoryRecord = {
      kind: "data",
      direction: "RX",
      text: rxText,
      hex: rxHex,
      timestamp: Date.now(),
    };
    appendHistoryRecord(rxRecord);
    renderHistoryRecord(rxRecord);
    syncCurrentSearchSelection();
    return;
  }

  if (payload.kind === "error") {
    const errorRecord: SerialHistoryRecord = {
      kind: "error",
      text: payload.text ?? "",
      hex: "",
      timestamp: Date.now(),
    };
    appendHistoryRecord(errorRecord);
    renderHistoryRecord(errorRecord);
    syncCurrentSearchSelection();
    connected.value = false;
    stopPeriodicSend();
    return;
  }
  const infoRecord: SerialHistoryRecord = {
    kind: "info",
    text: payload.text ?? "",
    hex: "",
    timestamp: Date.now(),
  };
  appendHistoryRecord(infoRecord);
  renderHistoryRecord(infoRecord);
  syncCurrentSearchSelection();
};

const toggleConnection = async () => {
  try {
    if (connected.value) {
      connected.value = false;
      await serialAssistantClose();
      stopPeriodicSend();
      return;
    }

    if (!selectedPort.value) {
      message.warning(i18n.global.t("serial.selectPort"));
      return;
    }

    await serialAssistantOpen({
      port: selectedPort.value,
      baudRate: Number(selectedBaudRate.value),
      dataBits: Number(selectedDataBits.value),
      stopBits: Number(selectedStopBits.value),
      parity: selectedParity.value as "none" | "odd" | "even",
      flowControl: selectedFlowControl.value as "none" | "software" | "hardware",
    });

    connected.value = true;
    localStorage.setItem("port", selectedPort.value);
    await applySignals();
  } catch (error) {
    connected.value = false;
    stopPeriodicSend();
    message.error(String(error));
  }
};

watch([rts, dtr], () => {
  void applySignals();
});

watch(sendNewline, (value) => {
  localStorage.setItem("serial.sendNewline", value ? "1" : "0");
});

watch(findKeyword, (value) => {
  if (!searchBarVisible.value) {
    return;
  }
  updateSearchResults(value, true);
});

watch([periodicSend, connected], () => {
  if (!periodicSend.value || !connected.value) {
    stopPeriodicSend();
  }
});

watch(periodicInterval, () => {
  if (periodicRunning.value) {
    startPeriodicSend();
  }
});

watch(sendHex, (isHexMode, wasHexMode) => {
  if (isHexMode === wasHexMode) {
    return;
  }

  const currentInput = sendInput.value;
  if (!currentInput) {
    return;
  }

  try {
    if (isHexMode) {
      const data = Array.from(new TextEncoder().encode(currentInput));
      sendInput.value = bytesToHex(data);
      return;
    }

    const data = parseHexInput(currentInput);
    sendInput.value = new TextDecoder().decode(Uint8Array.from(data));
  } catch (error) {
    message.warning(String(error));
  }
});

watch([receiveHex, showTxRx, showTimestamp], () => {
  replayHistory();
});

watch(
  () => preferenceStore.resolvedTheme,
  (themeMode) => {
    terminal?.setOption("theme", getTerminalTheme(themeMode));
  }
);

onMounted(async () => {
  syncExportOptionsFromDisplay();
  terminal = new Terminal({
    fontSize: 14,
    convertEol: true,
    cursorBlink: true,
    scrollback: 5000,
    theme: getTerminalTheme(preferenceStore.resolvedTheme),
  });

  if (terminalContainer.value) {
    terminal.loadAddon(fitAddon);
    terminal.open(terminalContainer.value);
    fitTerminal();
    terminalResizeObserver = new ResizeObserver(() => {
      fitTerminal();
    });
    terminalResizeObserver.observe(terminalContainer.value);
  }

  resizeHandler = () => fitTerminal();
  window.addEventListener("resize", resizeHandler);
  window.addEventListener("keydown", onWindowKeydown);

  await refreshPorts();
  connected.value = await serialAssistantIsOpen();

  unlistenSerial = await listen<SerialAssistantEventPayload>("serial_assistant_event", (event) => {
    handleSerialEvent(event.payload);
  });
});

onBeforeUnmount(async () => {
  stopPeriodicSend();
  isDisplayLineOpen = false;
  lastDisplayDirection = null;
  pendingRxEspColor = null;
  if (pendingSearchScrollFrame !== null) {
    cancelAnimationFrame(pendingSearchScrollFrame);
    pendingSearchScrollFrame = null;
  }
  if (unlistenSerial) {
    await unlistenSerial();
  }
  if (resizeHandler) {
    window.removeEventListener("resize", resizeHandler);
  }
  window.removeEventListener("keydown", onWindowKeydown);
  terminalResizeObserver?.disconnect();
  terminalResizeObserver = null;
  if (connected.value) {
    try {
      await serialAssistantClose();
    } finally {
      connected.value = false;
    }
  }
  terminal?.dispose();
});
</script>

<style scoped>
.serial-page {
  height: 100%;
  min-height: 0;
  max-height: 100%;
  padding: 4px 4px 4px 0;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: var(--serial-page-bg, linear-gradient(180deg, #12151d 0%, #0f1218 100%));
  overflow: hidden;
}

.serial-page.is-light {
  --serial-page-bg: linear-gradient(180deg, #f7f9fd 0%, #edf2f8 100%);
  --serial-terminal-wrap-bg: linear-gradient(90deg, #eff4fb 0%, #f7faff 100%);
  --serial-terminal-bg: #f5f7fb;
  --serial-terminal-scroll-thumb: #aeb8c8;
  --serial-terminal-scroll-track: #dde4ef;
  --serial-card-bg: #f8fafc;
  --serial-panel-border: #d7dfeb;
  --serial-label-color: #334155;
  --serial-card-title: #1f2937;
  --serial-muted-text: #4b5563;
  --serial-input-bg: #ffffff;
  --serial-input-text: #1f2937;
  --serial-input-placeholder: #94a3b8;
  --serial-input-border: #d4dce8;
  --serial-input-border-active: #91acd8;
  --serial-bottom-bar-bg: #f1f5fb;
  --serial-search-input-bg: #ffffff;
  --serial-search-input-hover-bg: #f9fbff;
}

.serial-main {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 180px;
  gap: 12px;
  overflow: hidden;
}

.left-panel {
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
  overflow: hidden;
}

.terminal-wrap {
  position: relative;
  min-width: 0;
  min-height: 280px;
  flex: 1;
  border: 1px solid var(--serial-panel-border, rgba(255, 255, 255, 0.05));
  background: var(--serial-terminal-wrap-bg, linear-gradient(90deg, #1a1d25 0%, #20232d 100%));
  border-radius: 0;
  overflow: hidden;
}

.terminal-search {
  position: absolute;
  top: 10px;
  right: 12px;
  z-index: 8;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px;
  border: 1px solid var(--serial-panel-border, rgba(255, 255, 255, 0.08));
  border-radius: 9px;
  background: color-mix(in srgb, var(--serial-card-bg, #141824) 94%, transparent);
  backdrop-filter: blur(10px);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
}

.terminal-search :deep(.ant-input) {
  width: 180px;
  min-height: 28px;
  padding-top: 3px;
  padding-bottom: 3px;
}

.search-close {
  width: 24px;
  height: 24px;
  border: none;
  border-radius: 7px;
  background: transparent;
  color: var(--serial-muted-text, #cfd5e4);
  cursor: pointer;
  font-size: 14px;
  line-height: 1;
}

.search-close:hover {
  background: rgba(148, 163, 184, 0.14);
}

.terminal-view {
  width: 100%;
  height: 100%;
  padding-bottom: 6px;
  box-sizing: border-box;
  overflow: hidden;
}

.terminal-view :deep(.xterm),
.terminal-view :deep(.xterm-viewport) {
  background: var(--serial-terminal-bg, #1d1f27) !important;
}

.terminal-view :deep(.xterm) {
  width: 100% !important;
  height: 100% !important;
}

.terminal-view :deep(.xterm-viewport) {
  overflow-y: auto !important;
  overflow-x: hidden !important;
}

.terminal-view :deep(.xterm-viewport)::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

.terminal-view :deep(.xterm-viewport)::-webkit-scrollbar-thumb {
  background: var(--serial-terminal-scroll-thumb, #3a4157);
  border-radius: 8px;
  border: 2px solid var(--serial-terminal-scroll-track, #131722);
}

.terminal-view :deep(.xterm-viewport)::-webkit-scrollbar-track {
  background: var(--serial-terminal-scroll-track, #131722);
}

.right-panel {
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
  overflow: hidden;
  padding-right: 2px;
}

.right-panel > * {
  flex: 0 0 auto;
}

.settings-card {
  border: 1px solid var(--serial-panel-border, rgba(255, 255, 255, 0.05));
  border-radius: 8px;
  padding: 8px;
  background: var(--serial-card-bg, #141824);
}

.setting-row {
  display: grid;
  grid-template-columns: 44px 1fr;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
  min-width: 0;
}

.setting-row:last-child {
  margin-bottom: 0;
}

.setting-row label {
  color: var(--serial-label-color, #d8deed);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.2px;
}

.setting-row :deep(.ant-select),
.setting-row :deep(.ant-input),
.setting-row :deep(.ant-input-number) {
  width: 100%;
  min-width: 0;
}

.setting-row :deep(.ant-select-selection-item),
.setting-row :deep(.ant-select-selection-placeholder) {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.open-btn {
  height: 40px;
  font-size: 14px;
  font-weight: 600;
  border-radius: 8px;
}

.panel-card {
  background: var(--serial-card-bg, #141824);
  border: 1px solid var(--serial-panel-border, rgba(255, 255, 255, 0.05)) !important;
  border-radius: 8px;
}

.panel-card :deep(.ant-card-head) {
  min-height: 40px;
  border-bottom: none;
  color: var(--serial-card-title, #eaf0ff);
  font-size: 14px;
}

.panel-card :deep(.ant-card-body) {
  padding: 8px 10px;
}

.check-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}

.check-row:last-child {
  margin-bottom: 0;
}

.check-row span {
  color: var(--serial-muted-text, #cfd5e4);
  font-size: 12px;
}

.search-toggle {
  width: 34px;
  min-width: 34px;
  height: 28px;
  padding: 0;
  font-size: 12px;
}

.search-count {
  min-width: 38px;
  color: var(--serial-muted-text, #cfd5e4);
  font-size: 11px;
  text-align: right;
  white-space: nowrap;
}

.search-nav {
  width: 28px;
  min-width: 28px;
  height: 28px;
  padding: 0;
  font-size: 14px;
  line-height: 1;
}

.export-popover {
  min-width: 160px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.export-popover-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--serial-card-title, #eaf0ff);
}

.bottom-bar {
  display: flex;
  gap: 12px;
  align-items: stretch;
  padding: 10px;
  background: var(--serial-bottom-bar-bg, #0d1016);
  border: 1px solid var(--serial-panel-border, rgba(255, 255, 255, 0.05));
  border-radius: 8px;
}

.send-input {
  flex: 1;
  min-width: 0;
}

.send-input :deep(textarea) {
  min-height: 96px;
}

.bottom-actions {
  display: flex;
  flex-direction: column;
  gap: 4px;
  justify-content: flex-start;
}

.bottom-actions .ant-btn {
  width: 96px;
  min-height: 36px;
  border-radius: 8px;
}

.serial-page :deep(.ant-select-selector),
.serial-page :deep(.ant-input),
.serial-page :deep(.ant-input-number),
.serial-page :deep(.ant-input-number input) {
  background: var(--serial-input-bg, #10141d) !important;
  color: var(--serial-input-text, #edf2ff) !important;
  border: 1px solid var(--serial-input-border, transparent) !important;
  box-shadow: none !important;
}

.serial-page :deep(.ant-input),
.serial-page :deep(.ant-select-selector) {
  border-radius: 6px !important;
  min-height: 30px;
}

.serial-page :deep(.ant-select-selector:hover),
.serial-page :deep(.ant-input:hover),
.serial-page :deep(.ant-input:focus),
.serial-page :deep(.ant-input-number:hover),
.serial-page :deep(.ant-input-number-focused),
.serial-page :deep(.ant-select-focused .ant-select-selector) {
  border: 1px solid var(--serial-input-border-active, transparent) !important;
  box-shadow: 0 0 0 2px var(--accent-soft) !important;
}

.serial-page :deep(.ant-btn) {
  box-shadow: none !important;
}

.serial-page :deep(.ant-checkbox-wrapper),
.serial-page :deep(.ant-checkbox-wrapper span) {
  color: var(--serial-muted-text, #cfd5e4);
  font-size: 12px;
}

.serial-page :deep(.ant-card-head-title) {
  padding: 0;
}

.serial-page :deep(.xterm .xterm-viewport) {
  scrollbar-color: var(--serial-terminal-scroll-thumb, #3a4157)
    var(--serial-terminal-scroll-track, #131722);
}

@media (max-width: 1200px) {
  .serial-main {
    grid-template-columns: minmax(0, 1fr) 170px;
  }
}

@media (max-width: 900px) {
  .serial-main {
    grid-template-columns: minmax(0, 1fr) 160px;
  }

  .setting-row {
    grid-template-columns: 42px 1fr;
    gap: 6px;
  }

  .setting-row label {
    font-size: 13px;
  }
}

@media (max-width: 560px) {
  .serial-page {
    min-height: auto;
  }

  .serial-main {
    grid-template-columns: 1fr;
  }

  .right-panel {
    overflow: visible;
    padding-right: 0;
  }

  .terminal-wrap {
    min-height: 320px;
  }

  .bottom-bar {
    flex-direction: column;
    align-items: stretch;
  }

  .bottom-actions {
    flex-direction: row;
    justify-content: flex-end;
  }
}
</style>

<style>
.serial-port-dropdown {
  min-width: max-content !important;
}

.serial-port-dropdown .ant-select-item-option-content {
  white-space: nowrap !important;
  overflow: visible !important;
  text-overflow: clip !important;
}

.serial-page .xterm .xterm-viewport {
  background: var(--serial-terminal-bg, #1d1f27) !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
}

.serial-page .xterm .xterm-viewport::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

.serial-page .xterm .xterm-viewport::-webkit-scrollbar-thumb {
  background: var(--serial-terminal-scroll-thumb, #3a4157);
  border-radius: 8px;
  border: 2px solid var(--serial-terminal-scroll-track, #131722);
}

.serial-page .xterm .xterm-viewport::-webkit-scrollbar-track {
  background: var(--serial-terminal-scroll-track, #131722);
}
</style>
