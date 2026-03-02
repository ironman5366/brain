/**
 * Shared P300 flash schedule logic — used by both useExperiment (copy spelling)
 * and useBCI (free spelling controlled by Claude).
 */

import type { P300Generator } from "./experiment.types";

export { P300_MATRIX } from "./protocols/p300";

// --- Types ---

export interface FlashEvent {
  timeOffsetMs: number;
  type: "row" | "col";
  index: number;
  isTarget: boolean;
  sequenceNum: number;
}

export interface P300CharSchedule {
  targetLetter: string;
  charIndex: number;
  preStartMs: number;
  flashStartMs: number;
  flashes: FlashEvent[];
  postStartMs: number;
  endMs: number;
}

// --- BCI flash schedule (no target letter) ---

export interface BCIFlashSchedule {
  flashes: FlashEvent[];
  totalMs: number;
  flashDurationMs: number;
  isiMs: number;
}

/**
 * Build a flash schedule for BCI free spelling.
 * No target letter — all rows and columns flash in random order.
 */
export function buildBCIFlashSchedule(
  sequences: number,
  flashDurationMs = 100,
  isiMs = 75,
): BCIFlashSchedule {
  const soaMs = flashDurationMs + isiMs;
  const flashesPerSeq = 12; // 6 rows + 6 cols
  const flashes: FlashEvent[] = [];
  let cursor = 0;

  for (let seq = 0; seq < sequences; seq++) {
    // Build shuffled order: rows 0-5 then cols 0-5
    const order: { type: "row" | "col"; index: number }[] = [];
    for (let i = 0; i < 6; i++) order.push({ type: "row", index: i });
    for (let i = 0; i < 6; i++) order.push({ type: "col", index: i });
    // Fisher-Yates shuffle
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }

    for (let fi = 0; fi < flashesPerSeq; fi++) {
      const flash = order[fi];
      flashes.push({
        timeOffsetMs: cursor + fi * soaMs,
        type: flash.type,
        index: flash.index,
        isTarget: false, // unknown in free spelling
        sequenceNum: seq,
      });
    }

    cursor += flashesPerSeq * soaMs;
  }

  return { flashes, totalMs: cursor, flashDurationMs, isiMs };
}

// --- Copy-spelling schedule (original, used by useExperiment) ---

export function buildP300Schedule(gen: P300Generator): {
  chars: P300CharSchedule[];
  totalMs: number;
} {
  const { matrix, targetLetters, flashDurationMs, isiMs, sequencesPerCharacter, preCharacterMs, postCharacterMs } = gen;
  const soaMs = flashDurationMs + isiMs;
  const flashesPerSeq = 12; // 6 rows + 6 cols
  const chars: P300CharSchedule[] = [];
  let cursor = 0;

  for (let ci = 0; ci < targetLetters.length; ci++) {
    const target = targetLetters[ci];
    const targetIdx = matrix.indexOf(target);
    const targetRow = Math.floor(targetIdx / 6);
    const targetCol = targetIdx % 6;

    const preStartMs = cursor;
    cursor += preCharacterMs;
    const flashStartMs = cursor;

    const flashes: FlashEvent[] = [];

    for (let seq = 0; seq < sequencesPerCharacter; seq++) {
      const order: { type: "row" | "col"; index: number }[] = [];
      for (let i = 0; i < 6; i++) order.push({ type: "row", index: i });
      for (let i = 0; i < 6; i++) order.push({ type: "col", index: i });
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }

      for (let fi = 0; fi < flashesPerSeq; fi++) {
        const flash = order[fi];
        const isTarget =
          (flash.type === "row" && flash.index === targetRow) ||
          (flash.type === "col" && flash.index === targetCol);

        flashes.push({
          timeOffsetMs: cursor - flashStartMs + fi * soaMs,
          type: flash.type,
          index: flash.index,
          isTarget,
          sequenceNum: seq,
        });
      }

      cursor += flashesPerSeq * soaMs;
    }

    const postStartMs = cursor;
    cursor += postCharacterMs;

    chars.push({
      targetLetter: target,
      charIndex: ci,
      preStartMs,
      flashStartMs,
      flashes,
      postStartMs,
      endMs: cursor,
    });
  }

  return { chars, totalMs: cursor };
}
