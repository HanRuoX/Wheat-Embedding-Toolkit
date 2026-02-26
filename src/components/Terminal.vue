<template>
  <div style="width: 100%">
    <div
      ref="terminalContainer"
      class="terminal-host xterm"
      :class="{ 'is-light': preferenceStore.resolvedTheme === 'light' }"
      style="height: 160px"
      @mousedown="focusTerminal"
    />
    <a-progress
      :percent="progress.value"
      v-if="progress.visible"
      :status="progress.status"
      :show-info="false"
    />
  </div>
</template>
<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import "xterm/css/xterm.css";
import "xterm/lib/xterm.js";
import { Terminal } from "xterm";
import { FitAddon } from "xterm-addon-fit";
import bus from "@/bus/terminal";
import cli, { execute } from "@/utils/cli";
import kleur from "kleur";
import { usePreferenceStore, type ResolvedTheme } from "@/stores/Preference";

const progress = ref({
  value: 0,
  visible: false,
  status: "active",
});

const fitAddon = new FitAddon();
const preferenceStore = usePreferenceStore();
const terminalContainer = ref<HTMLElement | null>(null);

const promptSymbol = navigator.userAgent.toLowerCase().includes("windows") ? ">" : "$";
const PROMPT = `${promptSymbol} `;
const SUPPORTED_COMMANDS = new Set(["esptool.py", "esptool"]);

const commandHistory: string[] = [];
let historyIndex = -1;
let currentInput = "";
let runningCommand = false;
let inputDisposable: { dispose: () => void } | null = null;
let resizeHandler: (() => void) | null = null;
let pageShowHandler: (() => void) | null = null;
const TERMINAL_FONT_FAMILY = [
  '"Cascadia Code"',
  '"JetBrains Mono"',
  '"SF Mono"',
  '"SFMono-Regular"',
  '"Consolas"',
  '"Menlo"',
  '"Monaco"',
  '"Courier New"',
  "monospace",
].join(", ");

const getTerminalTheme = (themeMode: ResolvedTheme) =>
  themeMode === "light"
    ? {
        background: "#ffffff",
        foreground: "#333333",
        cursor: "#333333",
        cursorAccent: "#ffffff",
        selectionBackground: "#add6ff",
        black: "#000000",
        red: "#cd3131",
        green: "#00bc00",
        yellow: "#949800",
        blue: "#0451a5",
        magenta: "#bc05bc",
        cyan: "#0598bc",
        white: "#555555",
        brightBlack: "#666666",
        brightRed: "#cd3131",
        brightGreen: "#14ce14",
        brightYellow: "#b5ba00",
        brightBlue: "#0451a5",
        brightMagenta: "#bc05bc",
        brightCyan: "#0598bc",
        brightWhite: "#a5a5a5",
      }
    : {
        background: "#1e1e1e",
        foreground: "#cccccc",
        cursor: "#aeafad",
        cursorAccent: "#1e1e1e",
        selectionBackground: "#264f78",
        black: "#000000",
        red: "#cd3131",
        green: "#0dbc79",
        yellow: "#e5e510",
        blue: "#2472c8",
        magenta: "#bc3fbc",
        cyan: "#11a8cd",
        white: "#e5e5e5",
        brightBlack: "#666666",
        brightRed: "#f14c4c",
        brightGreen: "#23d18b",
        brightYellow: "#f5f543",
        brightBlue: "#3b8eea",
        brightMagenta: "#d670d6",
        brightCyan: "#29b8db",
        brightWhite: "#e5e5e5",
      };

const terminal = new Terminal({
  fontSize: 12,
  fontFamily: TERMINAL_FONT_FAMILY,
  fontWeight: "400",
  fontWeightBold: "600",
  lineHeight: 1,
  letterSpacing: 0,
  minimumContrastRatio: 1,
  allowProposedApi: true,
  cursorStyle: "block",
  cursorBlink: true,
  theme: getTerminalTheme(preferenceStore.resolvedTheme),
});

terminal.attachCustomKeyEventHandler((arg) => {
  if (arg.ctrlKey && arg.code === "KeyC" && arg.type === "keydown") {
    const selection = terminal.getSelection();
    if (selection) {
      navigator.clipboard.writeText(selection);
      return false;
    }
  }
  return true;
});

const onBusWrite = (data: unknown) => {
  terminal.write(data as string);
};

const writeTerminalOutput = (data: unknown) => {
  const text = String(data ?? "");
  const normalized = text.replace(/\r?\n/g, "\r\n");
  if (normalized.length === 0) {
    terminal.write("\r\n");
    return;
  }
  if (/\r\n$/.test(normalized)) {
    terminal.write(normalized);
    return;
  }
  terminal.write(`${normalized}\r\n`);
};

const onBusWriteln = (data: unknown) => {
  writeTerminalOutput(data);
};

bus.on("write", onBusWrite);
bus.on("writeln", onBusWriteln);

watch(
  () => preferenceStore.resolvedTheme,
  (themeMode) => {
    terminal.setOption("theme", getTerminalTheme(themeMode));
  }
);

const focusTerminal = () => {
  terminal.focus();
};

const clearCurrentInput = () => {
  if (!currentInput) {
    return;
  }
  terminal.write("\b \b".repeat(currentInput.length));
  currentInput = "";
};

const replaceCurrentInput = (next: string) => {
  clearCurrentInput();
  if (!next) {
    return;
  }
  currentInput = next;
  terminal.write(next);
};

const writePrompt = (newLine = false) => {
  if (newLine) {
    terminal.write("\r\n");
  }
  terminal.write(PROMPT);
  currentInput = "";
  historyIndex = commandHistory.length;
};

const parseCommandLine = (input: string): string[] => {
  const tokens: string[] = [];
  let token = "";
  let quote: "'" | '"' | null = null;
  let escaped = false;

  for (const char of input) {
    if (escaped) {
      token += char;
      escaped = false;
      continue;
    }

    if (char === "\\") {
      escaped = true;
      continue;
    }

    if (quote) {
      if (char === quote) {
        quote = null;
      } else {
        token += char;
      }
      continue;
    }

    if (char === "'" || char === '"') {
      quote = char;
      continue;
    }

    if (/\s/.test(char)) {
      if (token.length > 0) {
        tokens.push(token);
        token = "";
      }
      continue;
    }

    token += char;
  }

  if (quote) {
    throw new Error("引号未闭合");
  }

  if (escaped) {
    token += "\\";
  }

  if (token.length > 0) {
    tokens.push(token);
  }

  return tokens;
};

const showHelp = () => {
  terminal.writeln("可用命令:");
  terminal.writeln("  help               查看帮助");
  terminal.writeln("  clear              清空终端");
  terminal.writeln("  esptool.py <args>  执行 esptool");
  terminal.writeln("  esptool <args>     等同于 esptool.py");
  terminal.writeln("示例:");
  terminal.writeln("  esptool.py --port COM3 chip_id");
  terminal.writeln("  esptool --port /dev/ttyUSB0 erase_flash");
};

const runCommand = () => {
  terminal.write("\r\n");
  const line = currentInput.trim();

  if (!line) {
    writePrompt();
    return;
  }

  if (line === "clear") {
    terminal.clear();
    writePrompt();
    return;
  }

  if (line === "help") {
    showHelp();
    writePrompt();
    return;
  }

  if (commandHistory[commandHistory.length - 1] !== line) {
    commandHistory.push(line);
  }
  historyIndex = commandHistory.length;

  let parts: string[] = [];
  try {
    parts = parseCommandLine(line);
  } catch (error) {
    terminal.writeln(`${kleur.red("[ERROR]")} ${String(error)}`);
    writePrompt();
    return;
  }

  if (parts.length === 0) {
    writePrompt();
    return;
  }

  const [command, ...args] = parts;
  if (!SUPPORTED_COMMANDS.has(command)) {
    terminal.writeln(`${kleur.yellow("[WARN]")} 仅支持 esptool 相关命令`);
    writePrompt();
    return;
  }

  if (runningCommand) {
    terminal.writeln(`${kleur.yellow("[WARN]")} 请等待当前命令执行完成`);
    writePrompt();
    return;
  }

  runningCommand = true;
  let finished = false;

  const finish = (errorText?: string) => {
    if (finished) {
      return;
    }
    finished = true;
    runningCommand = false;
    cli.off("close", onClose);
    cli.off("error", onError);
    if (errorText) {
      terminal.writeln(`${kleur.red("[ERROR]")} ${errorText}`);
    }
    writePrompt(true);
  };

  const onClose = () => {
    finish();
  };

  const onError = (error: unknown) => {
    finish(String(error));
  };

  cli.on("close", onClose);
  cli.on("error", onError);

  try {
    execute("esptool.py", args);
  } catch (error) {
    finish(String(error));
  }
};

const handleTerminalInput = (data: string) => {
  if (data === "\r") {
    runCommand();
    return;
  }

  if (data === "\u001b[A") {
    if (commandHistory.length === 0) {
      return;
    }
    historyIndex = Math.max(0, historyIndex - 1);
    replaceCurrentInput(commandHistory[historyIndex] ?? "");
    return;
  }

  if (data === "\u001b[B") {
    if (commandHistory.length === 0) {
      return;
    }
    historyIndex = Math.min(commandHistory.length, historyIndex + 1);
    if (historyIndex >= commandHistory.length) {
      replaceCurrentInput("");
    } else {
      replaceCurrentInput(commandHistory[historyIndex] ?? "");
    }
    return;
  }

  if (data === "\u007f") {
    if (!currentInput) {
      return;
    }
    currentInput = currentInput.slice(0, -1);
    terminal.write("\b \b");
    return;
  }

  if (data === "\u0003") {
    if (runningCommand) {
      terminal.writeln(`${kleur.yellow("[WARN]")} 当前不支持中断正在执行的命令`);
      return;
    }
    if (currentInput) {
      terminal.write("^C");
      writePrompt(true);
    }
    return;
  }

  if (runningCommand) {
    return;
  }

  if (/^[^\x00-\x1F\x7F]+$/.test(data)) {
    currentInput += data;
    terminal.write(data);
  }
};

// emitter.on("terminalWriteLine", (data) => {
//   terminal.writeln(data as string);

//   let regex = /Writing at\s(0x[0-9a-fA-F]+)\.\.\.\s\((\d+)\s%\)/;
//   let match = (data as string).match(regex);
//   if (match) {
//     const percentage = parseInt(match[2]);
//     progress.value.visible = true;
//     progress.value.value = percentage;
//     progress.value.status = "active";

//     if (percentage == 100) {
//       progress.value.status = "normal";
//     }
//   }

//   regex = /Detected flash size: (\d+)MB/;
//   match = (data as string).match(regex);
//   if (match) {
//     message.info(`${match[1]}MB`);
//   }
// });

onMounted(() => {
  terminal.loadAddon(fitAddon);
  if (!terminalContainer.value) {
    return;
  }
  terminal.open(terminalContainer.value);
  fitAddon.fit();
  focusTerminal();

  resizeHandler = () => fitAddon.fit();
  pageShowHandler = () => fitAddon.fit();
  window.addEventListener("resize", resizeHandler);
  window.addEventListener("pageshow", pageShowHandler);

  inputDisposable = terminal.onData(handleTerminalInput);
  terminal.writeln(`${kleur.bold().cyan("ESPTool Terminal")}`);
  terminal.writeln(`${kleur.dim("输入 help 查看命令示例，输入 clear 清空终端。")}`);
  writePrompt();
});

onBeforeUnmount(() => {
  inputDisposable?.dispose();
  if (resizeHandler) {
    window.removeEventListener("resize", resizeHandler);
  }
  if (pageShowHandler) {
    window.removeEventListener("pageshow", pageShowHandler);
  }
  bus.off("write", onBusWrite);
  bus.off("writeln", onBusWriteln);
});
</script>

<style scoped>
.terminal-host {
  padding: 0;
}

.terminal-host :deep(.terminal.xterm) {
  padding: 6px 8px;
  box-sizing: border-box;
  background: #1e1e1e;
}

.terminal-host.is-light :deep(.terminal.xterm) {
  background: #ffffff;
}

.terminal-host :deep(.xterm),
.terminal-host :deep(.xterm-rows) {
  font-variant-ligatures: none;
  text-rendering: geometricPrecision;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.terminal-host :deep(.xterm-viewport)::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

.terminal-host :deep(.xterm-viewport)::-webkit-scrollbar-thumb {
  background: rgba(121, 121, 121, 0.5);
  border-radius: 0;
}

.terminal-host :deep(.xterm-viewport)::-webkit-scrollbar-track {
  background: transparent;
}

.terminal-host.is-light :deep(.xterm-viewport)::-webkit-scrollbar-thumb {
  background: rgba(100, 100, 100, 0.35);
}
</style>
