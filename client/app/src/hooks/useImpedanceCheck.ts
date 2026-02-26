import { useState, useCallback, useRef } from "react";
import type {
  ChannelImpedance,
  ImpedanceStatus,
  ImpedanceThresholds,
} from "../lib/impedance";

const API_BASE = "http://localhost:8765";

export function useImpedanceCheck() {
  const [status, setStatus] = useState<ImpedanceStatus>("idle");
  const [results, setResults] = useState<ChannelImpedance[]>([]);
  const [currentChannel, setCurrentChannel] = useState<string | null>(null);
  const [thresholds, setThresholds] = useState<ImpedanceThresholds | null>(
    null
  );
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const startCheck = useCallback(async () => {
    // Abort any previous check
    abortRef.current?.abort();
    const abort = new AbortController();
    abortRef.current = abort;

    setStatus("running");
    setResults([]);
    setCurrentChannel(null);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/api/impedance/start`, {
        method: "POST",
        signal: abort.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const json = line.slice(6);
          if (!json) continue;

          const event = JSON.parse(json);

          switch (event.type) {
            case "start":
              setThresholds(event.thresholds);
              if (event.channels?.length > 0) {
                setCurrentChannel(event.channels[0]);
              }
              break;

            case "channel":
              setResults((prev) => [
                ...prev,
                {
                  index: event.index,
                  name: event.name,
                  impedance: event.impedance,
                },
              ]);
              // Set next channel as current (or null if this was the last)
              setCurrentChannel(null);
              break;

            case "status":
              // Progress update, could show in UI
              break;

            case "done":
              setStatus("done");
              setCurrentChannel(null);
              break;

            case "error":
              setStatus("error");
              setError(event.message);
              break;
          }
        }
      }

      // If we didn't get a done event, mark done anyway
      setStatus((s) => (s === "running" ? "done" : s));
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setStatus("error");
        setError((e as Error).message);
      }
    }
  }, []);

  return { status, results, currentChannel, thresholds, error, startCheck };
}
