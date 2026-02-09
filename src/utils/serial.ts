import { invoke } from "@tauri-apps/api/core";

export type SerialParity = "none" | "odd" | "even";
export type SerialFlowControl = "none" | "software" | "hardware";

export interface SerialOpenOptions {
  port: string;
  baudRate: number;
  dataBits: number;
  stopBits: number;
  parity: SerialParity;
  flowControl: SerialFlowControl;
}

export interface SerialAssistantEventPayload {
  kind: "status" | "error" | "data";
  text: string;
  hex: string;
}

export async function serialAssistantOpen(options: SerialOpenOptions) {
  return invoke("serial_assistant_open", {
    port: options.port,
    baudRate: options.baudRate,
    dataBits: options.dataBits,
    stopBits: options.stopBits,
    parity: options.parity,
    flowControl: options.flowControl,
  });
}

export async function serialAssistantSend(data: number[]) {
  return (await invoke("serial_assistant_send", { data })) as number;
}

export async function serialAssistantClose() {
  return invoke("serial_assistant_close");
}

export async function serialAssistantIsOpen() {
  return (await invoke("serial_assistant_is_open")) as boolean;
}

export async function serialAssistantSetSignals(rts: boolean, dtr: boolean) {
  return invoke("serial_assistant_set_signals", { rts, dtr });
}
