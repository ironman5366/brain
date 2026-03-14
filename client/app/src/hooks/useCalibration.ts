import { useState, useEffect, useRef, useCallback } from "react";
import type { ChannelCalibration, CalibrationState } from "../lib/calibration";

const API_BASE = "http://localhost:8765";
const SIGNAL_POLL_MS = 1200;

const INITIAL_STATE: CalibrationState = {
  channels: [],
  messages: [],
  impedanceChecked: false,
  signalChecked: false,
  allGood: false,
};

export function useCalibration() {
  const [state, setState] = useState<CalibrationState>(INITIAL_STATE);
  const [impedanceRunning, setImpedanceRunning] = useState(false);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // --- SSE: subscribe to calibration events from the agent ---
  useEffect(() => {
    const es = new EventSource(`${API_BASE}/api/calibration/events`);

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        handleSSEEvent(data);
      } catch {
        // ignore parse errors
      }
    };

    es.onerror = () => {
      // EventSource auto-reconnects
    };

    return () => es.close();
  }, []);

  function handleSSEEvent(data: Record<string, unknown>) {
    const type = data.type as string;

    switch (type) {
      case "state":
        setState((prev) => {
          const next = { ...prev };
          if (data.impedance) {
            next.impedanceChecked = true;
            mergeImpedance(next, data.impedance as Record<string, unknown>);
          }
          if (data.signal_quality) {
            next.signalChecked = true;
            mergeSignal(
              next,
              (data.signal_quality as Record<string, unknown>[]) ?? []
            );
          }
          if (data.messages) {
            next.messages = data.messages as string[];
          }
          return next;
        });
        break;

      case "impedance":
        setImpedanceRunning(false);
        setState((prev) => {
          const next = { ...prev, impedanceChecked: true };
          mergeImpedance(next, data as Record<string, unknown>);
          return next;
        });
        break;

      case "signal_quality": {
        const channels = (data as Record<string, unknown>)
          .channels as Record<string, unknown>[];
        setState((prev) => {
          const next = { ...prev, signalChecked: true };
          mergeSignal(next, channels);
          return next;
        });
        break;
      }

      case "message":
        setState((prev) => ({
          ...prev,
          messages: [...prev.messages, data.text as string],
        }));
        break;
    }
  }

  // --- Polling: live signal quality updates ---
  const fetchSignal = useCallback(async () => {
    if (impedanceRunning) return; // don't poll during impedance check
    try {
      const resp = await fetch(
        `${API_BASE}/api/calibration/check-signal?duration_sec=2.0`
      );
      if (!resp.ok) return;
      const json = await resp.json();
      if (json.error) return;

      setState((prev) => {
        const next = { ...prev, signalChecked: true };
        mergeSignal(next, json.channels);
        next.allGood =
          next.impedanceChecked &&
          json.all_good &&
          next.channels.every(
            (c) => !c.impedance_rating || c.impedance_rating === "good"
          );
        return next;
      });
    } catch {
      // ignore fetch errors
    }
  }, [impedanceRunning]);

  useEffect(() => {
    // Start polling
    fetchSignal();
    pollTimerRef.current = setInterval(fetchSignal, SIGNAL_POLL_MS);
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, [fetchSignal]);

  return { state, impedanceRunning };
}

// --- Merge helpers ---

function mergeImpedance(
  state: CalibrationState,
  imp: Record<string, unknown>
) {
  const channels = imp.channels as Record<string, unknown>[];
  if (!channels) return;
  for (const ch of channels) {
    const name = ch.name as string;
    const existing = state.channels.find((c) => c.name === name);
    if (existing) {
      existing.impedance_ohms = ch.impedance_ohms as number;
      existing.impedance_kohms = ch.impedance_kohms as number;
      existing.impedance_rating = ch.rating as "good" | "ok" | "bad";
      existing.wire_color = ch.wire_color as string;
      existing.pin = ch.pin as string;
    } else {
      state.channels.push({
        name,
        wire_color: (ch.wire_color as string) ?? "unknown",
        pin: (ch.pin as string) ?? "unknown",
        impedance_ohms: ch.impedance_ohms as number,
        impedance_kohms: ch.impedance_kohms as number,
        impedance_rating: ch.rating as "good" | "ok" | "bad",
      });
    }
  }
  state.allGood = (imp.all_good as boolean) ?? false;
}

function mergeSignal(
  state: CalibrationState,
  channels: Record<string, unknown>[]
) {
  if (!channels) return;
  for (const ch of channels) {
    const name = ch.name as string;
    const existing = state.channels.find((c) => c.name === name);
    const signalData: Partial<ChannelCalibration> = {
      rms_uv: ch.rms_uv as number,
      line_noise_db: ch.line_noise_db as number,
      dc_drift_uv: ch.dc_drift_uv as number,
      has_alpha: ch.has_alpha as boolean,
      alpha_power_ratio: ch.alpha_power_ratio as number,
      signal_rating: ch.rating as "good" | "ok" | "bad",
      issues: ch.issues as string[],
      psd_frequencies: ch.psd_frequencies as number[],
      psd_db: ch.psd_db as number[],
    };

    if (existing) {
      Object.assign(existing, signalData);
      if (!existing.wire_color || existing.wire_color === "unknown") {
        existing.wire_color = (ch.wire_color as string) ?? "unknown";
      }
      if (!existing.pin || existing.pin === "unknown") {
        existing.pin = (ch.pin as string) ?? "unknown";
      }
    } else {
      state.channels.push({
        name,
        wire_color: (ch.wire_color as string) ?? "unknown",
        pin: (ch.pin as string) ?? "unknown",
        ...signalData,
      });
    }
  }
}
