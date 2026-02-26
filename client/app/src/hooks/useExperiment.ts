import { useState, useCallback, useRef } from "react";
import type {
  Protocol,
  ExperimentPhase,
  StimulusDef,
  TrialDef,
  SSVEPGenerator,
  SSVEPFrequency,
} from "../lib/experiment.types";
import { MarkerSender } from "../lib/markers";

const API_BASE = "http://localhost:8765";

export interface ExperimentProgress {
  block: number;
  totalBlocks: number;
  trial: number;
  totalTrials: number;
}

export function useExperiment() {
  const [phase, setPhase] = useState<ExperimentPhase>({ type: "idle" });
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [progress, setProgress] = useState<ExperimentProgress | null>(null);

  const markerRef = useRef<MarkerSender | null>(null);
  const abortRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Resolve to wait for "ready" click
  const readyResolveRef = useRef<(() => void) | null>(null);

  const cleanup = useCallback(() => {
    abortRef.current = true;
    if (timerRef.current) clearTimeout(timerRef.current);
    markerRef.current?.stop();
    markerRef.current = null;
  }, []);

  const start = useCallback(
    async (protocol: Protocol) => {
      abortRef.current = false;

      try {
        // Start server-side recording
        const resp = await fetch(`${API_BASE}/api/session/start`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            protocol_id: protocol.id,
            protocol_version: protocol.version,
          }),
        });
        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.detail || `HTTP ${resp.status}`);
        }

        const { session_id } = await resp.json();
        setSessionId(session_id);

        // Create marker sender
        const markers = new MarkerSender(session_id);
        markerRef.current = markers;

        // Send experiment_start marker
        markers.send({
          code: "experiment_start",
          timestamp: performance.now(),
          metadata: { protocol_id: protocol.id },
        });

        // Run each block
        for (
          let blockIdx = 0;
          blockIdx < protocol.blocks.length;
          blockIdx++
        ) {
          if (abortRef.current) break;

          const block = protocol.blocks[blockIdx];

          // Show block instruction and wait for ready
          if (block.instruction) {
            setPhase({
              type: "blockInstruction",
              blockIndex: blockIdx,
              instruction: block.instruction,
            });

            // Wait for ready() to be called
            await new Promise<void>((resolve) => {
              readyResolveRef.current = resolve;
            });
            readyResolveRef.current = null;
          }

          if (abortRef.current) break;

          // Send block_start marker
          markers.send({
            code: "block_start",
            timestamp: performance.now(),
            block_id: block.id,
          });

          if (block.trialGenerator.type === "ssvep") {
            // SSVEP: single continuous stimulation block
            const gen = block.trialGenerator;

            setProgress({
              block: blockIdx,
              totalBlocks: protocol.blocks.length,
              trial: 0,
              totalTrials: 1,
            });

            markers.send({
              code: "ssvep_start",
              timestamp: performance.now(),
              block_id: block.id,
              metadata: {
                frequencies: gen.frequencies.map((f) => f.hz),
                target_frequency: gen.targetFrequencyHz,
                duration_ms: gen.durationMs,
              },
            });

            await runSSVEPTrial(gen, blockIdx, setPhase);

            markers.send({
              code: "ssvep_end",
              timestamp: performance.now(),
              block_id: block.id,
            });
          } else {
            // Standard trial-based blocks
            const trials = generateTrials(block.trialGenerator);

            setProgress({
              block: blockIdx,
              totalBlocks: protocol.blocks.length,
              trial: 0,
              totalTrials: trials.length,
            });

            for (let trialIdx = 0; trialIdx < trials.length; trialIdx++) {
              if (abortRef.current) break;

              const trial = trials[trialIdx];

              setProgress((prev) =>
                prev ? { ...prev, trial: trialIdx } : null
              );

              markers.send({
                code: trial.markerCode,
                timestamp: performance.now(),
                block_id: block.id,
                trial_index: trialIdx,
              });

              await runTrial(trial, blockIdx, setPhase);
            }
          }

          // Send block_end marker
          markers.send({
            code: "block_end",
            timestamp: performance.now(),
            block_id: block.id,
          });

          // Audio cue to signal block end
          if (!abortRef.current) playTone();

          // Rest period between blocks
          if (block.restAfterMs > 0 && !abortRef.current) {
            await runRest(block.restAfterMs, blockIdx, setPhase);
          }
        }

        // Send experiment_end marker
        markers.send({
          code: "experiment_end",
          timestamp: performance.now(),
        });

        // Flush markers and stop
        await markers.stop();
        markerRef.current = null;

        // Stop server-side recording
        await fetch(`${API_BASE}/api/session/stop`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id }),
        });

        if (!abortRef.current) {
          playTone(880, 200);
          setPhase({ type: "complete", sessionId: session_id });
        }
      } catch (e) {
        cleanup();
        setPhase({
          type: "error",
          message: (e as Error).message,
        });
      }
    },
    [cleanup]
  );

  const ready = useCallback(() => {
    readyResolveRef.current?.();
  }, []);

  const abort = useCallback(async () => {
    abortRef.current = true;
    readyResolveRef.current?.(); // unblock if waiting for ready

    // Stop marker sender
    if (markerRef.current) {
      markerRef.current.send({
        code: "experiment_abort",
        timestamp: performance.now(),
      });
      await markerRef.current.stop();
      markerRef.current = null;
    }

    // Stop server-side recording
    if (sessionId) {
      try {
        await fetch(`${API_BASE}/api/session/stop`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId }),
        });
      } catch {
        // best effort
      }
    }

    setPhase({ type: "idle" });
    setProgress(null);
  }, [sessionId]);

  const reset = useCallback(() => {
    setPhase({ type: "idle" });
    setSessionId(null);
    setProgress(null);
  }, []);

  return { phase, sessionId, progress, start, ready, abort, reset };
}

// --- Audio ---

let audioCtx: AudioContext | null = null;

function playTone(frequency = 660, durationMs = 150) {
  if (!audioCtx) audioCtx = new AudioContext();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = "sine";
  osc.frequency.value = frequency;
  gain.gain.value = 0.3;
  // Fade out to avoid click
  gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + durationMs / 1000);
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + durationMs / 1000);
}

// --- Helpers ---

function generateTrials(
  generator: Protocol["blocks"][number]["trialGenerator"]
): TrialDef[] {
  switch (generator.type) {
    case "fixed":
      return generator.trials;
    default:
      // Other generators will be implemented as needed
      throw new Error(`Trial generator type "${generator.type}" not yet implemented`);
  }
}

function runTrial(
  trial: TrialDef,
  blockIndex: number,
  setPhase: React.Dispatch<React.SetStateAction<ExperimentPhase>>
): Promise<void> {
  return new Promise((resolve) => {
    const startTime = performance.now();
    const totalMs = trial.durationMs;

    // Update phase with countdown
    const update = () => {
      const elapsed = performance.now() - startTime;
      const remaining = Math.max(0, totalMs - elapsed);

      setPhase({
        type: "trial",
        blockIndex,
        trialIndex: 0,
        stimulus: trial.stimulus,
        remainingMs: remaining,
      });

      if (remaining > 0) {
        requestAnimationFrame(update);
      } else {
        resolve();
      }
    };

    update();
  });
}

function runSSVEPTrial(
  generator: SSVEPGenerator,
  blockIndex: number,
  setPhase: React.Dispatch<React.SetStateAction<ExperimentPhase>>
): Promise<void> {
  return new Promise((resolve) => {
    const startTime = performance.now();
    const totalMs = generator.durationMs;

    const update = () => {
      const elapsed = performance.now() - startTime;
      const remaining = Math.max(0, totalMs - elapsed);

      setPhase({
        type: "ssvepTrial",
        blockIndex,
        frequencies: generator.frequencies,
        targetFrequencyHz: generator.targetFrequencyHz,
        remainingMs: remaining,
      });

      if (remaining > 0) {
        requestAnimationFrame(update);
      } else {
        resolve();
      }
    };

    update();
  });
}

function runRest(
  durationMs: number,
  blockIndex: number,
  setPhase: React.Dispatch<React.SetStateAction<ExperimentPhase>>
): Promise<void> {
  return new Promise((resolve) => {
    const startTime = performance.now();

    const update = () => {
      const elapsed = performance.now() - startTime;
      const remaining = Math.max(0, durationMs - elapsed);

      setPhase({
        type: "blockRest",
        blockIndex,
        remainingMs: remaining,
      });

      if (remaining > 0) {
        requestAnimationFrame(update);
      } else {
        resolve();
      }
    };

    update();
  });
}
