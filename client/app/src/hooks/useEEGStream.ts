import { useEffect, useRef, useState, useCallback } from "react";
import { decodeMessage } from "../lib/protocol";

export interface StreamMeta {
  samplingRate: number;
  channelNames: string[];
}

export interface StreamState {
  connected: boolean;
  meta: StreamMeta | null;
  error: string | null;
}

export interface RingBuffer {
  /** One Float32Array per channel */
  channels: Float32Array[];
  /** Current write position (wraps around) */
  writeIndex: number;
  /** Total capacity in samples */
  capacity: number;
  /** Total samples written (monotonically increasing, for change detection) */
totalWritten: number;
}

const BUFFER_SECONDS = 5;
const RECONNECT_DELAY_MS = 2000;

export function useEEGStream(serverUrl: string) {
  const [state, setState] = useState<StreamState>({
    connected: false,
    meta: null,
    error: null,
  });

  const bufferRef = useRef<RingBuffer | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmounted = useRef(false);

  const initBuffer = useCallback((numChannels: number, samplingRate: number) => {
    const capacity = samplingRate * BUFFER_SECONDS;
    const channels: Float32Array[] = [];
    for (let i = 0; i < numChannels; i++) {
      channels.push(new Float32Array(capacity));
    }
    bufferRef.current = { channels, writeIndex: 0, capacity, totalWritten: 0 };
  }, []);

  const connect = useCallback(() => {
    if (unmounted.current) return;

    const ws = new WebSocket(serverUrl);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => {
      if (!unmounted.current) {
        setState((s) => ({ ...s, connected: true, error: null }));
      }
    };

    ws.onmessage = (event: MessageEvent) => {
      const msg = decodeMessage(event.data as ArrayBuffer);

      if (msg.type === "meta") {
        initBuffer(msg.ch.length, msg.sr);
        if (!unmounted.current) {
          setState((s) => ({
            ...s,
            meta: { samplingRate: msg.sr, channelNames: msg.ch },
          }));
        }
        return;
      }

      // Data message — write directly into ring buffer, no React state
      const buf = bufferRef.current;
      if (!buf || msg.d.length === 0) return;

      for (const sample of msg.d) {
        for (let ch = 0; ch < sample.length && ch < buf.channels.length; ch++) {
          buf.channels[ch][buf.writeIndex] = sample[ch];
        }
        buf.writeIndex = (buf.writeIndex + 1) % buf.capacity;
        buf.totalWritten++;
      }
    };

    ws.onclose = () => {
      if (!unmounted.current) {
        setState((s) => ({ ...s, connected: false }));
        // Reconnect after delay
        reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
      }
    };

    ws.onerror = () => {
      if (!unmounted.current) {
        setState((s) => ({ ...s, error: "WebSocket error" }));
      }
      ws.close();
    };
  }, [serverUrl, initBuffer]);

  useEffect(() => {
    unmounted.current = false;
    connect();

    return () => {
      unmounted.current = true;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        wsRef.current.onclose = null; // prevent reconnect on unmount
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { state, bufferRef };
}
