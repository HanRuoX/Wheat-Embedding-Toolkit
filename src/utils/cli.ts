import { Command } from "@tauri-apps/plugin-shell";
import { write, writeln } from "@/bus/terminal";

import mitt from "mitt";
const emitter = mitt();
export default emitter;

const isWindows = navigator.userAgent.toLowerCase().includes("windows");
const isMacOS = navigator.userAgent.toLowerCase().includes("mac os");

const getCommandCandidates = (name: string): string[] => {
  if (name === "esptool.py") {
    if (isWindows) {
      return ["esptool.py", "esptool-windows-bundled"];
    }
    if (isMacOS) {
      return ["esptool.py", "esptool-homebrew-arm", "esptool-homebrew-intel"];
    }
  }

  if (name === "gen_esp32part.py" && isWindows) {
    return ["gen_esp32part.py", "gen_esp32part-windows-bundled"];
  }

  return [name];
};

export function execute(name:string,cmd: string[]) {
  const args = cmd.filter((x: string) => x != "");
  const candidates = getCommandCandidates(name);

  const spawnAt = (index: number) => {
    const command = new Command(candidates[index], args as string[]);
    command.on("close", (data) => {
      emitter.emit("close", data);
    });
    command.on("error", (error) => {
      emitter.emit("error", error);
    });
    command.stdout.on("data", (line) => {
      writeln(line);
      emitter.emit("stdout", line);
    });
    command.stderr.on("data", (line) => {
      writeln(line);
      emitter.emit("stderr", line);
    });
    void command.spawn().catch((error) => {
      if (index + 1 < candidates.length) {
        spawnAt(index + 1);
        return;
      }
      writeln(String(error));
      emitter.emit("error", error);
    });
  };

  spawnAt(0);
}
