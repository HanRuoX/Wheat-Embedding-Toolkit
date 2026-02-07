import { Command } from "@tauri-apps/plugin-shell";
import { write, writeln } from "@/bus/terminal";

import mitt from "mitt";
const emitter = mitt();
export default emitter;

const COMMAND_FALLBACKS: Record<string, string[]> = {
  "esptool.py": ["esptool.py", "esptool-homebrew-arm", "esptool-homebrew-intel"],
};

export function execute(name:string,cmd: string[]) {
  const args = cmd.filter((x: string) => x != "");
  const candidates = COMMAND_FALLBACKS[name] ?? [name];

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
