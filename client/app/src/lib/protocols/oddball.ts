import type { Protocol } from "../experiment.types";

export const ODDBALL_PROTOCOL: Protocol = {
  id: "auditory-oddball-v1",
  name: "Auditory Oddball",
  description:
    "Three-stimulus novelty oddball. Count the rare high beeps. ~2 min, 200 trials.",
  version: "1.1.0",
  blocks: [
    {
      id: "oddball-main",
      name: "Auditory Oddball",
      instruction:
        "You will hear three types of sounds:\n" +
        "- Frequent LOW beeps (most common)\n" +
        "- Rare HIGH beeps (count these!)\n" +
        "- Occasional weird sounds (ignore these)\n\n" +
        "Silently COUNT only the high-pitched beeps.\n" +
        "Keep your eyes on the fixation cross.",
      trialGenerator: {
        type: "oddball",
        totalTrials: 200,
        targetRatio: 0.1,
        distractorRatio: 0.1,
        stimuli: {
          standard: { type: "audio", frequency: 500, durationMs: 100 },
          target: { type: "audio", frequency: 1000, durationMs: 100 },
          distractors: [{ type: "audio", novel: true, durationMs: 100 }],
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
