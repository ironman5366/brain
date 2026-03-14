import type { Protocol } from "../experiment.types";

export const P300_MATRIX = [
  "A", "B", "C", "D", "E", "F",
  "G", "H", "I", "J", "K", "L",
  "M", "N", "O", "P", "Q", "R",
  "S", "T", "U", "V", "W", "X",
  "Y", "Z", "1", "2", "3", "4",
  "5", "6", "7", "8", "9", "_",
];

export const P300_PROTOCOL: Protocol = {
  id: "p300-speller-v1",
  name: "P300 Speller",
  description:
    "Copy-spelling P300 BCI calibration. Attend to the highlighted target letter " +
    "while rows and columns flash. 15 characters, ~8 min.",
  version: "1.0.0",
  blocks: [
    {
      id: "p300-copy-spell",
      name: "Copy Spelling",
      instruction:
        "A 6×6 character matrix will appear.\n" +
        "For each round, a target letter will be shown above the matrix.\n" +
        "Focus on that letter while rows and columns flash.\n" +
        "Try not to blink during flashing.",
      trialGenerator: {
        type: "p300",
        matrix: P300_MATRIX,
        targetLetters: ["B", "E", "H", "I", "L", "N", "O", "R", "S", "T", "U", "D", "G", "Y", "4"],
        flashDurationMs: 100,
        isiMs: 75,
        sequencesPerCharacter: 10,
        preCharacterMs: 3000,
        postCharacterMs: 2000,
      },
      restAfterMs: 0,
    },
  ],
};
