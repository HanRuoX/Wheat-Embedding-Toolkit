<template>
  <div class="serial-page">
    <div class="serial-main">
      <div class="left-panel">
        <div class="terminal-wrap">
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
              dropdown-class-name="serial-port-dropdown"
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

        <div class="search-row">
          <a-input v-model:value="findKeyword" @pressEnter="findInLogs" />
          <a-button type="primary" @click="findInLogs">{{ $t("serial.find") }}</a-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { message } from "ant-design-vue";
import { Terminal } from "xterm";
import { FitAddon } from "xterm-addon-fit";
import moment from "moment";
import "xterm/css/xterm.css";
import { getSerialPortList } from "@/utils/common";
import {
  SerialAssistantEventPayload,
  serialAssistantClose,
  serialAssistantIsOpen,
  serialAssistantOpen,
  serialAssistantSend,
  serialAssistantSetSignals,
} from "@/utils/serial";
import i18n from "@/locales/i18n";

type SelectOption = { label: string; value: string };
type SerialHistoryRecord = {
  kind: "data" | "info" | "error";
  direction?: "TX" | "RX";
  text: string;
  hex: string;
  timestamp: number;
};

const terminalContainer = ref<HTMLElement | null>(null);
const serialPortOptions = ref<SelectOption[]>([]);
const selectedPort = ref<string | undefined>(localStorage.getItem("port") ?? undefined);
const selectedBaudRate = ref("115200");
const selectedParity = ref("none");
const selectedDataBits = ref("8");
const selectedStopBits = ref("1");
const selectedFlowControl = ref("none");

const connected = ref(false);
const sendHex = ref(false);
const sendNewline = ref(false);
const periodicSend = ref(false);
const periodicRunning = ref(false);
const periodicInterval = ref(1000);
const rts = ref(false);
const dtr = ref(false);
const receiveHex = ref(false);
const showTimestamp = ref(false);
const showTxRx = ref(false);
const findKeyword = ref("");
const activeSearchKeyword = ref("");
const sendInput = ref("");
const logEntries = ref<string[]>([]);
const historyRecords = ref<SerialHistoryRecord[]>([]);

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
let periodicTimer: ReturnType<typeof setInterval> | null = null;
let unlistenSerial: UnlistenFn | null = null;
let resizeHandler: (() => void) | null = null;
let rxLineBuffer = "";

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
};
const ansiRegex = /\x1b\[[0-9;]*m/g;

const normalizeAnsiEscapes = (text: string) =>
  text.replace(/\\033\[/g, "\x1b[").replace(/\\x1b\[/gi, "\x1b[");

const bytesToHex = (data: number[]) =>
  data.map((item) => item.toString(16).padStart(2, "0")).join(" ").toUpperCase();

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const applySearchHighlight = (text: string) => {
  const keyword = activeSearchKeyword.value.trim();
  if (!keyword) {
    return text;
  }

  const matcher = new RegExp(escapeRegExp(keyword), "gi");
  return text
    .split(/(\x1b\[[0-9;]*m)/g)
    .map((part) => {
      if (/^\x1b\[[0-9;]*m$/.test(part)) {
        return part;
      }
      return part.replace(matcher, (matched) => `${ansi.searchStart}${matched}${ansi.searchEnd}`);
    })
    .join("");
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

const writeRaw = (text: string) => {
  if (!terminal || !text) {
    return;
  }
  terminal.write(normalizeAnsiEscapes(text));
  terminal.scrollToBottom();
  splitAndSave(text);
};

const writeLine = (text: string) => {
  if (!terminal) {
    return;
  }
  terminal.writeln(normalizeAnsiEscapes(text));
  terminal.scrollToBottom();
  splitAndSave(text);
};

const colorizeEspLogLine = (line: string) => {
  const match = line.match(/^([IWEVD]) \((\d+)\)\s+([^:]+):\s?(.*)$/);
  if (!match) {
    return line;
  }

  const [, level, ms, tag, messageText] = match;
  const levelColor =
    level === "I"
      ? ansi.info
      : level === "W"
        ? ansi.warn
        : level === "E"
          ? ansi.error
          : level === "D"
            ? ansi.debug
            : ansi.verbose;

  return `${levelColor}${level}${ansi.reset} (${ansi.dim}${ms}${ansi.reset}) ${ansi.tag}${tag}${ansi.reset}: ${messageText}`;
};

const formatDisplayLine = (direction: "TX" | "RX", line: string, timestamp: number) => {
  const prefix = formatPrefix(direction, timestamp);
  const coloredLine = colorizeEspLogLine(line);
  if (!prefix) {
    return coloredLine;
  }
  return `${prefix}${coloredLine}`;
};

const writeParsedRxText = (text: string, timestamp: number) => {
  if (!text) {
    return;
  }
  const normalized = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const merged = rxLineBuffer + normalized;
  const lines = merged.split("\n");
  rxLineBuffer = lines.pop() ?? "";

  lines.forEach((line) => {
    writeLine(applySearchHighlight(formatDisplayLine("RX", line, timestamp)));
  });
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

  if (record.direction === "TX" && !showTxRx.value && !showTimestamp.value && !receiveHex.value) {
    return;
  }

  const prefix = formatPrefix(record.direction, record.timestamp);
  if (receiveHex.value) {
    writeLine(applySearchHighlight(`${prefix}${record.hex}`));
    return;
  }

  if (record.direction === "RX") {
    writeParsedRxText(record.text, record.timestamp);
    return;
  }

  record.text
    .replace(/\r\n/g, "\n")
    .split("\n")
    .filter((line) => line.length > 0)
    .forEach((line) => writeLine(applySearchHighlight(`${prefix}${line}`)));
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
  rxLineBuffer = "";
  historyRecords.value.forEach((record) => {
    renderHistoryRecord(record);
  });
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
  if (periodicTimer) {
    clearInterval(periodicTimer);
    periodicTimer = null;
  }
  periodicRunning.value = false;
};

const startPeriodicSend = () => {
  stopPeriodicSend();
  if (!periodicSend.value || !connected.value) {
    return;
  }
  const interval = Math.max(10, Number(periodicInterval.value) || 1000);
  periodicRunning.value = true;
  void sendData(true);
  periodicTimer = setInterval(() => {
    void sendData(true);
  }, interval);
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

    await serialAssistantSend(payload);

    const txRecord: SerialHistoryRecord = {
      kind: "data",
      direction: "TX",
      text: new TextDecoder().decode(Uint8Array.from(payload)),
      hex: bytesToHex(payload),
      timestamp: Date.now(),
    };
    appendHistoryRecord(txRecord);
    renderHistoryRecord(txRecord);
  } catch (error) {
    if (!fromTimer) {
      message.error(String(error));
    }
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
  const keyword = findKeyword.value.trim();
  if (!keyword) {
    message.warning(i18n.global.t("serial.findKeywordRequired"));
    return;
  }
  const keywordLower = keyword.toLowerCase();
  const count = logEntries.value.filter((line) => line.toLowerCase().includes(keywordLower)).length;
  if (count === 0) {
    activeSearchKeyword.value = "";
    replayHistory();
    message.info(i18n.global.t("serial.findNotFound"));
    return;
  }
  activeSearchKeyword.value = keyword;
  replayHistory();
  message.success(i18n.global.t("serial.findResult", { count }));
};

const clearTerminal = () => {
  terminal?.clear();
  logEntries.value = [];
  historyRecords.value = [];
  activeSearchKeyword.value = "";
  rxLineBuffer = "";
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
};

const toggleConnection = async () => {
  try {
    if (connected.value) {
      await serialAssistantClose();
      connected.value = false;
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

watch([receiveHex, showTxRx, showTimestamp], () => {
  replayHistory();
});

onMounted(async () => {
  terminal = new Terminal({
    fontSize: 14,
    convertEol: true,
    cursorBlink: true,
    theme: {
      background: "#1d1f27",
      foreground: "#f2f2f2",
      cursor: "#3a8bff",
    },
  });

  if (terminalContainer.value) {
    terminal.loadAddon(fitAddon);
    terminal.open(terminalContainer.value);
    fitAddon.fit();
  }

  resizeHandler = () => fitAddon.fit();
  window.addEventListener("resize", resizeHandler);

  await refreshPorts();
  connected.value = await serialAssistantIsOpen();

  unlistenSerial = await listen<SerialAssistantEventPayload>("serial_assistant_event", (event) => {
    handleSerialEvent(event.payload);
  });
});

onBeforeUnmount(async () => {
  rxLineBuffer = "";
  stopPeriodicSend();
  if (unlistenSerial) {
    await unlistenSerial();
  }
  if (resizeHandler) {
    window.removeEventListener("resize", resizeHandler);
  }
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
  height: 100vh;
  min-height: 0;
  max-height: 100vh;
  padding: 4px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: linear-gradient(180deg, #12151d 0%, #0f1218 100%);
  overflow: hidden;
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
  min-width: 0;
  min-height: 280px;
  flex: 1;
  border: none;
  background: linear-gradient(90deg, #1a1d25 0%, #20232d 100%);
  border-radius: 8px;
  overflow: hidden;
}

.terminal-view {
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.terminal-view :deep(.xterm),
.terminal-view :deep(.xterm-screen),
.terminal-view :deep(.xterm-viewport) {
  width: 100% !important;
  max-width: 100% !important;
}

.terminal-view :deep(.xterm-screen) {
  left: 0 !important;
}

.terminal-view :deep(.xterm-viewport)::-webkit-scrollbar {
  width: 0 !important;
  height: 0 !important;
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
  border: none;
  border-radius: 8px;
  padding: 8px;
  background: #141824;
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
  color: #d8deed;
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
  background: #141824;
  border: none !important;
  border-radius: 8px;
}

.panel-card :deep(.ant-card-head) {
  min-height: 40px;
  border-bottom: none;
  color: #eaf0ff;
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
  color: #cfd5e4;
  font-size: 12px;
}

.search-row {
  display: flex;
  gap: 6px;
}

.search-row :deep(.ant-input) {
  background: #0b0f17 !important;
  color: #eef3ff !important;
}

.search-row :deep(.ant-input::placeholder) {
  color: #9ca6bd !important;
}

.search-row :deep(.ant-input:hover),
.search-row :deep(.ant-input:focus) {
  background: #090d15 !important;
}

.search-row .ant-btn {
  width: 64px;
}

.bottom-bar {
  display: flex;
  gap: 12px;
  align-items: stretch;
  padding: 10px;
  background: #0d1016;
  border: none;
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
  background: #10141d !important;
  color: #edf2ff !important;
  border: none !important;
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
  border: none !important;
  box-shadow: none !important;
}

.serial-page :deep(.ant-btn) {
  box-shadow: none !important;
}

.serial-page :deep(.ant-checkbox-wrapper),
.serial-page :deep(.ant-checkbox-wrapper span) {
  color: #d2d8e6;
  font-size: 12px;
}

.serial-page :deep(.ant-card-head-title) {
  padding: 0;
}

.serial-page :deep(.xterm .xterm-viewport) {
  scrollbar-color: #3a4157 #131722;
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
</style>
