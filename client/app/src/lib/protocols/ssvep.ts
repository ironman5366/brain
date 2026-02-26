import type { Protocol, BlockDef } from "../experiment.types";

const STIM_DURATION_MS = 20_000;
const REST_MS = 3_000;

function baselineBlock(n: number): BlockDef {
  return {
    id: `baseline-${n}`,
    name: `Baseline (${n})`,
    instruction: "Fixate on the cross.\nKeep your eyes open and stay still.",
    trialGenerator: {
      type: "fixed",
      trials: [
        {
          stimulus: { type: "fixation", symbol: "+", size: "4rem" },
          durationMs: STIM_DURATION_MS,
          markerCode: "baseline",
        },
      ],
    },
    restAfterMs: REST_MS,
  };
}

function ssvepBlock(hz: number, n: number, isLast: boolean): BlockDef {
  return {
    id: `ssvep-${hz}hz-${n}`,
    name: `SSVEP ${hz} Hz (${n})`,
    instruction: `A flickering circle will appear at ${hz} Hz.\nFixate on its center. Try not to blink.`,
    trialGenerator: {
      type: "ssvep",
      frequencies: [
        {
          hz,
          position: "center",
          stimulus: { type: "shape", shape: "circle", size: 200, color: "#ffffff" },
        },
      ],
      durationMs: STIM_DURATION_MS,
      targetFrequencyHz: hz,
    },
    restAfterMs: isLast ? 0 : REST_MS,
  };
}

export const SSVEP_PROTOCOL: Protocol = {
  id: "ssvep-basic-v1",
  name: "SSVEP Basic",
  description:
    "Steady-state visual evoked potentials at 10 Hz and 15 Hz. " +
    "Expect spectral peaks at stimulation frequencies on O1/O2.",
  version: "1.0.0",
  blocks: [
    baselineBlock(1),
    ssvepBlock(10, 1, false),
    baselineBlock(2),
    ssvepBlock(15, 1, false),
    baselineBlock(3),
    ssvepBlock(10, 2, false),
    baselineBlock(4),
    ssvepBlock(15, 2, true),
  ],
};
