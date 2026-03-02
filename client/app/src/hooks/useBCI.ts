/**
 * BCI hook — subscribes to server SSE events and executes flash sequences.
 * Claude controls the session via MCP tools; this hook renders the UI.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import { buildBCIFlashSchedule, P300_MATRIX } from "../lib/p300";
import type { BCIFlashSchedule, FlashEvent } from "../lib/p300";
import { MarkerSender } from "../lib/markers";

const API_BASE = "http://localhost:8765";

export type BCIMode =
  | "idle"
  | "ready"
  | "flashing"
  | "proposing"
  | "message";

export interface BCIState {
  mode: BCIMode;
  sessionId: string | null;
  spelled: string;
  // Flashing state
  highlightedRow: number | null;
  highlightedCol: number | null;
  flashProgress: number; // 0-1
  // Proposal state
  proposedLetter: string | null;
  proposalMessage: string | null;
  // Message state
  messageText: string | null;
}

const INITIAL_STATE: BCIState = {
  mode: "idle",
  sessionId: null,
  spelled: "",
  highlightedRow: null,
  highlightedCol: null,
  flashProgress: 0,
  proposedLetter: null,
  proposalMessage: null,
  messageText: null,
};

export function useBCI() {
  const [state, setState] = useState<BCIState>(INITIAL_STATE);
  const markerRef = useRef<MarkerSender | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const abortRef = useRef(false);

  // Audio context for sounds
  const audioCtxRef = useRef<AudioContext | null>(null);
  function getAudioCtx() {
    if (!audioCtxRef.current) audioCtxRef.current = new AudioContext();
    return audioCtxRef.current;
  }

  // Connect to SSE on mount
  useEffect(() => {
    const es = new EventSource(`${API_BASE}/api/bci/events`);
    eventSourceRef.current = es;

    es.onmessage = (e) => {
      const data = JSON.parse(e.data);
      handleEvent(data);
    };

    es.onerror = () => {
      // EventSource will auto-reconnect
    };

    return () => {
      es.close();
      eventSourceRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function handleEvent(data: Record<string, unknown>) {
    const type = data.type as string;

    switch (type) {
      case "state":
        // Initial state on connect
        setState((prev) => ({
          ...prev,
          mode: (data.state as BCIMode) || "idle",
          sessionId: (data.session_id as string) || null,
          spelled: (data.spelled as string) || "",
        }));
        break;

      case "started":
        setState((prev) => ({
          ...prev,
          mode: "ready",
          sessionId: data.session_id as string,
          spelled: "",
          messageText: null,
        }));
        // Create marker sender for this session
        markerRef.current = new MarkerSender(data.session_id as string);
        break;

      case "flash":
        runFlashSequences(data.sequences as number);
        break;

      case "propose":
        setState((prev) => ({
          ...prev,
          mode: "proposing",
          proposedLetter: data.letter as string,
          proposalMessage: (data.message as string) || null,
        }));
        break;

      case "message":
        setState((prev) => ({
          ...prev,
          mode: "message",
          messageText: data.text as string,
        }));
        break;

      case "play_sound":
        playSound(data);
        break;

      case "stopped":
        markerRef.current?.stop();
        markerRef.current = null;
        setState(INITIAL_STATE);
        break;
    }
  }

  async function runFlashSequences(sequences: number) {
    abortRef.current = false;

    const schedule = buildBCIFlashSchedule(sequences);
    let markerCount = 0;

    setState((prev) => ({
      ...prev,
      mode: "flashing",
      highlightedRow: null,
      highlightedCol: null,
      flashProgress: 0,
    }));

    // Ensure AudioContext is resumed
    const ctx = getAudioCtx();
    if (ctx.state === "suspended") await ctx.resume();

    await new Promise<void>((resolve) => {
      const startTime = performance.now();
      const { flashes, totalMs, flashDurationMs } = schedule;
      let lastFlashIdx = -1;

      const update = () => {
        if (abortRef.current) { resolve(); return; }

        const elapsed = performance.now() - startTime;
        const progress = Math.min(elapsed / totalMs, 1);

        let highlightedRow: number | null = null;
        let highlightedCol: number | null = null;

        // Find current flash
        for (let fi = flashes.length - 1; fi >= 0; fi--) {
          const f = flashes[fi];
          if (elapsed >= f.timeOffsetMs) {
            const flashAge = elapsed - f.timeOffsetMs;
            if (flashAge < flashDurationMs) {
              if (f.type === "row") highlightedRow = f.index;
              else highlightedCol = f.index;

              // Send marker on first frame
              if (fi !== lastFlashIdx) {
                lastFlashIdx = fi;
                markerCount++;
                markerRef.current?.send({
                  code: "p300_flash",
                  timestamp: performance.now(),
                  metadata: {
                    flash_type: f.type,
                    flash_index: f.index,
                    sequence_num: f.sequenceNum,
                  },
                });
              }
            }
            break;
          }
        }

        setState((prev) => ({
          ...prev,
          highlightedRow,
          highlightedCol,
          flashProgress: progress,
        }));

        if (elapsed < totalMs) {
          requestAnimationFrame(update);
        } else {
          resolve();
        }
      };

      requestAnimationFrame(update);
    });

    // Flush markers before reporting done
    await markerRef.current?.flush();

    // Report done to server
    try {
      await fetch(`${API_BASE}/api/bci/flash-done`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ marker_count: markerCount }),
      });
    } catch {
      // best effort
    }

    setState((prev) => ({
      ...prev,
      mode: "ready",
      highlightedRow: null,
      highlightedCol: null,
      flashProgress: 0,
    }));
  }

  function playSound(data: Record<string, unknown>) {
    const ctx = getAudioCtx();
    const frequency = (data.frequency as number) || 440;
    const durationMs = (data.duration_ms as number) || 200;
    const novel = data.novel as boolean;

    if (novel) {
      // Simple novel sound: noise burst
      const dur = durationMs / 1000;
      const buf = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * dur), ctx.sampleRate);
      const d = buf.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
      const src = ctx.createBufferSource();
      src.buffer = buf;
      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + dur);
      src.connect(gain);
      gain.connect(ctx.destination);
      src.start();
      src.stop(ctx.currentTime + dur);
    } else {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.frequency.value = frequency;
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + durationMs / 1000);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + durationMs / 1000);
    }
  }

  const submitFeedback = useCallback(async (accepted: boolean) => {
    try {
      await fetch(`${API_BASE}/api/bci/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ accepted }),
      });
    } catch {
      // best effort
    }

    setState((prev) => ({
      ...prev,
      mode: "ready",
      proposedLetter: null,
      proposalMessage: null,
      spelled: accepted && prev.proposedLetter
        ? prev.spelled + prev.proposedLetter
        : prev.spelled,
    }));
  }, []);

  return { state, submitFeedback, matrix: P300_MATRIX };
}
