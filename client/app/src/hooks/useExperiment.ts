import { useState, useCallback, useRef } from "react";
import type {
  Protocol,
  ExperimentPhase,
  TrialDef,
  SSVEPGenerator,
  P300Generator,
  OddballGenerator,
} from "../lib/experiment.types";
import { MarkerSender } from "../lib/markers";
import { buildP300Schedule } from "../lib/p300";

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

          // Ensure AudioContext is resumed (browsers require user gesture)
          if (audioCtx && audioCtx.state === "suspended") {
            await audioCtx.resume();
          }

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
          } else if (block.trialGenerator.type === "p300") {
            // P300: scheduled row/col flashes across multiple characters
            await runP300Block(
              block.trialGenerator,
              blockIdx,
              protocol.blocks.length,
              block.id,
              setPhase,
              setProgress,
              markers,
              abortRef,
            );
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

              // Play audio stimulus if applicable
              if (trial.stimulus.type === "audio") {
                if (trial.stimulus.novel) {
                  playNovelSound(trial.stimulus.durationMs);
                } else if (trial.stimulus.frequency) {
                  playTone(trial.stimulus.frequency, trial.stimulus.durationMs);
                }
              }

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

function playNovelSound(durationMs = 100) {
  if (!audioCtx) audioCtx = new AudioContext();
  const ctx = audioCtx;
  const t = ctx.currentTime;
  const dur = durationMs / 1000;

  // Output gain with fade-out envelope
  const out = ctx.createGain();
  out.gain.setValueAtTime(0.3, t);
  out.gain.exponentialRampToValueAtTime(0.001, t + dur);
  out.connect(ctx.destination);

  const variant = Math.floor(Math.random() * 4);

  if (variant === 0) {
    // White noise burst
    const buf = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * dur), ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(out);
    src.start(t);
    src.stop(t + dur);
  } else if (variant === 1) {
    // Frequency sweep (chirp)
    const osc = ctx.createOscillator();
    const startFreq = 200 + Math.random() * 800;
    const endFreq = 800 + Math.random() * 1200;
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(startFreq, t);
    osc.frequency.linearRampToValueAtTime(endFreq, t + dur);
    osc.connect(out);
    osc.start(t);
    osc.stop(t + dur);
  } else if (variant === 2) {
    // Complex tone — 3 oscillators at random frequencies
    const types: OscillatorType[] = ["sine", "square", "triangle", "sawtooth"];
    for (let i = 0; i < 3; i++) {
      const osc = ctx.createOscillator();
      const g = ctx.createGain();
      g.gain.value = 0.33;
      osc.type = types[Math.floor(Math.random() * types.length)];
      osc.frequency.value = 200 + Math.random() * 1800;
      osc.connect(g);
      g.connect(out);
      osc.start(t);
      osc.stop(t + dur);
    }
  } else {
    // AM noise — noise modulated by a low-frequency oscillator
    const buf = ctx.createBuffer(1, Math.ceil(ctx.sampleRate * dur), ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    const mod = ctx.createOscillator();
    const modGain = ctx.createGain();
    mod.frequency.value = 20 + Math.random() * 40;
    modGain.gain.value = 0.5;
    mod.connect(modGain);
    modGain.connect(out.gain);
    src.connect(out);
    mod.start(t);
    src.start(t);
    mod.stop(t + dur);
    src.stop(t + dur);
  }
}

// --- Helpers ---

function generateTrials(
  generator: Protocol["blocks"][number]["trialGenerator"]
): TrialDef[] {
  switch (generator.type) {
    case "fixed":
      return generator.trials;

    case "oddball": {
      const gen = generator as OddballGenerator;
      const nTarget = Math.round(gen.totalTrials * gen.targetRatio);
      const nDistractor = Math.round(gen.totalTrials * (gen.distractorRatio ?? 0));
      const nStandard = gen.totalTrials - nTarget - nDistractor;

      // Build sequence: "target" | "novel" | "standard"
      type TrialType = "target" | "novel" | "standard";
      const seq: TrialType[] = [
        ...Array<TrialType>(nTarget).fill("target"),
        ...Array<TrialType>(nDistractor).fill("novel"),
        ...Array<TrialType>(nStandard).fill("standard"),
      ];

      // Fisher-Yates shuffle
      for (let i = seq.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [seq[i], seq[j]] = [seq[j], seq[i]];
      }

      // Fix consecutive rare stimuli (targets and novels)
      const isRare = (t: TrialType) => t !== "standard";
      for (let i = 1; i < seq.length; i++) {
        if (isRare(seq[i]) && isRare(seq[i - 1])) {
          for (let j = i + 1; j < seq.length; j++) {
            if (seq[j] === "standard" && (j + 1 >= seq.length || !isRare(seq[j + 1]))) {
              [seq[i], seq[j]] = [seq[j], seq[i]];
              break;
            }
          }
        }
      }

      const distractor = gen.stimuli.distractors?.[0] ?? gen.stimuli.standard;

      return seq.map((trialType, idx) => {
        const jitter = gen.timing.isiJitterMs
          ? (Math.random() - 0.5) * 2 * gen.timing.isiJitterMs
          : 0;
        const stimulus =
          trialType === "target" ? gen.stimuli.target
          : trialType === "novel" ? distractor
          : gen.stimuli.standard;
        return {
          id: `oddball-${idx}`,
          stimulus,
          durationMs: gen.timing.stimulusDurationMs + gen.timing.isiMs + jitter,
          markerCode: `oddball_${trialType}`,
          captureResponse: gen.requiresResponse,
          responseWindowMs: gen.responseWindowMs,
        };
      });
    }

    default:
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

// --- P300 flash execution (uses schedule from ../lib/p300.ts) ---

async function runP300Block(
  gen: P300Generator,
  blockIndex: number,
  totalBlocks: number,
  blockId: string,
  setPhase: React.Dispatch<React.SetStateAction<ExperimentPhase>>,
  setProgress: React.Dispatch<React.SetStateAction<ExperimentProgress | null>>,
  markers: MarkerSender,
  abortRef: React.RefObject<boolean>,
): Promise<void> {
  const { chars, totalMs } = buildP300Schedule(gen);

  setProgress({
    block: blockIndex,
    totalBlocks,
    trial: 0,
    totalTrials: gen.targetLetters.length,
  });

  return new Promise((resolve) => {
    const startTime = performance.now();
    let currentCharIdx = -1;
    let lastFlashIdx = -1;
    let sentCharEnd = false;

    const update = () => {
      if (abortRef.current) { resolve(); return; }

      const now = performance.now();
      const elapsed = now - startTime;
      const remaining = Math.max(0, totalMs - elapsed);

      // Find which character we're in
      let charIdx = 0;
      for (let i = 0; i < chars.length; i++) {
        if (elapsed >= chars[i].preStartMs) charIdx = i;
      }
      const ch = chars[charIdx];

      // Character changed
      if (charIdx !== currentCharIdx) {
        // End previous character
        if (currentCharIdx >= 0 && !sentCharEnd) {
          markers.send({
            code: "p300_char_end",
            timestamp: now,
            block_id: blockId,
            metadata: { target_letter: chars[currentCharIdx].targetLetter, char_index: currentCharIdx },
          });
        }

        currentCharIdx = charIdx;
        lastFlashIdx = -1;
        sentCharEnd = false;

        setProgress((prev) => prev ? { ...prev, trial: charIdx } : null);

        markers.send({
          code: "p300_char_start",
          timestamp: now,
          block_id: blockId,
          metadata: { target_letter: ch.targetLetter, char_index: charIdx },
        });
      }

      // Determine charPhase and highlight state
      let charPhase: "pre" | "flashing" | "post" = "pre";
      let highlightedRow: number | null = null;
      let highlightedCol: number | null = null;

      const relativeMs = elapsed - ch.preStartMs;

      if (relativeMs < gen.preCharacterMs) {
        charPhase = "pre";
      } else if (elapsed >= ch.postStartMs) {
        charPhase = "post";
        if (!sentCharEnd && charIdx === chars.length - 1 && remaining <= 0) {
          markers.send({
            code: "p300_char_end",
            timestamp: now,
            block_id: blockId,
            metadata: { target_letter: ch.targetLetter, char_index: charIdx },
          });
          sentCharEnd = true;
        }
      } else {
        charPhase = "flashing";
        const flashElapsed = elapsed - ch.flashStartMs;

        // Find current flash
        for (let fi = ch.flashes.length - 1; fi >= 0; fi--) {
          const f = ch.flashes[fi];
          if (flashElapsed >= f.timeOffsetMs) {
            const flashAge = flashElapsed - f.timeOffsetMs;
            if (flashAge < gen.flashDurationMs) {
              // Flash is ON
              if (f.type === "row") highlightedRow = f.index;
              else highlightedCol = f.index;

              // Send marker on first frame of this flash
              if (fi !== lastFlashIdx) {
                lastFlashIdx = fi;
                markers.send({
                  code: "p300_flash",
                  timestamp: now,
                  block_id: blockId,
                  metadata: {
                    flash_type: f.type,
                    flash_index: f.index,
                    is_target: f.isTarget,
                    sequence_num: f.sequenceNum,
                    char_index: charIdx,
                    target_letter: ch.targetLetter,
                  },
                });
              }
            }
            // else: in ISI gap, highlight stays null
            break;
          }
        }
      }

      setPhase({
        type: "p300Trial",
        blockIndex,
        matrix: gen.matrix,
        targetLetter: ch.targetLetter,
        highlightedRow,
        highlightedCol,
        remainingMs: remaining,
        currentCharIndex: charIdx,
        totalChars: gen.targetLetters.length,
        charPhase,
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
