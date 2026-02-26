import type { Protocol } from "../experiment.types";

/**
 * Alpha eyes-open/closed protocol.
 *
 * Alpha power (8-13 Hz) should increase when eyes are closed,
 * especially at occipital channels (O1, O2). This is one of the
 * most reliable EEG effects and a great first validation test.
 *
 * Structure: 4 blocks x 15s + 3s rest between = ~1 min total.
 * Two repetitions of each condition for averaging.
 */
export const ALPHA_PROTOCOL: Protocol = {
  id: "alpha-eyes-open-closed-v1",
  name: "Alpha Eyes Open/Closed",
  description:
    "Measures alpha rhythm (8-13 Hz) with eyes open vs closed. " +
    "Alpha should increase with eyes closed, especially at O1/O2.",
  version: "1.0.0",
  blocks: [
    {
      id: "eyes-open-1",
      name: "Eyes Open (1)",
      instruction:
        "Sit quietly with your eyes OPEN.\nFixate on the cross in the center of the screen.",
      trialGenerator: {
        type: "fixed",
        trials: [
          {
            stimulus: { type: "fixation", symbol: "+", size: "4rem" },
            durationMs: 15_000,
            markerCode: "eyes_open",
          },
        ],
      },
      restAfterMs: 3_000,
    },
    {
      id: "eyes-closed-1",
      name: "Eyes Closed (1)",
      instruction:
        "CLOSE your eyes and sit quietly.\nKeep them closed until you hear the next instruction.",
      trialGenerator: {
        type: "fixed",
        trials: [
          {
            stimulus: { type: "blank" },
            durationMs: 15_000,
            markerCode: "eyes_closed",
          },
        ],
      },
      restAfterMs: 3_000,
    },
    {
      id: "eyes-open-2",
      name: "Eyes Open (2)",
      instruction:
        "OPEN your eyes.\nFixate on the cross in the center of the screen.",
      trialGenerator: {
        type: "fixed",
        trials: [
          {
            stimulus: { type: "fixation", symbol: "+", size: "4rem" },
            durationMs: 15_000,
            markerCode: "eyes_open",
          },
        ],
      },
      restAfterMs: 3_000,
    },
    {
      id: "eyes-closed-2",
      name: "Eyes Closed (2)",
      instruction:
        "CLOSE your eyes and sit quietly.\nKeep them closed until the experiment ends.",
      trialGenerator: {
        type: "fixed",
        trials: [
          {
            stimulus: { type: "blank" },
            durationMs: 15_000,
            markerCode: "eyes_closed",
          },
        ],
      },
      restAfterMs: 0,
    },
  ],
};
