import type { Protocol } from "../experiment.types";

export const ODDBALL_PROTOCOL: Protocol = {
  id: "auditory-oddball-v1",
  name: "Auditory Oddball",
  description:
    "Classic auditory P300. Count the rare high-pitched tones silently. ~2 min, 200 trials.",
  version: "1.0.0",
  blocks: [
    {
      id: "oddball-main",
      name: "Auditory Oddball",
      instruction:
        "You will hear two tones:\n" +
        "a frequent LOW tone and a rare HIGH tone.\n\n" +
        "Silently COUNT the high-pitched tones.\n" +
        "Keep your eyes on the fixation cross.\n" +
        "Try not to blink during the tones.",
      trialGenerator: {
        type: "oddball",
        totalTrials: 200,
        targetRatio: 0.2,
        stimuli: {
          standard: { type: "audio", frequency: 500, durationMs: 100 },
          target: { type: "audio", frequency: 1000, durationMs: 100 },
        },
        timing: {
          stimulusDurationMs: 100,
          isiMs: 600,
          isiJitterMs: 150,
        },
        requiresResponse: false,
      },
      restAfterMs: 0,
    },
  ],
};
