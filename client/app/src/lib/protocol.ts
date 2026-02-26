import { decode } from "@msgpack/msgpack";

export interface MetaMessage {
  type: "meta";
  sr: number;
  ch: string[];
}

export interface DataMessage {
  type: "data";
  t: number[];
  d: number[][]; // each element is [ch0, ch1, ..., ch7]
}

export type ServerMessage = MetaMessage | DataMessage;

export function decodeMessage(data: ArrayBuffer): ServerMessage {
  return decode(new Uint8Array(data)) as ServerMessage;
}
