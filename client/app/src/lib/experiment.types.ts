// ---- Stimulus Types ----

export interface InstructionStimulus {
  type: "instruction";
  text: string;
  fontSize?: string;
}

export interface FixationStimulus {
  type: "fixation";
  symbol?: string; // default "+"
  size?: string;
}

export interface BlankStimulus {
  type: "blank";
}

export interface TextStimulus {
  type: "text";
  content: string;
  fontSize?: string;
  color?: string;
}

export interface ImageStimulus {
  type: "image";
  src: string;
  width?: number;
  height?: number;
}

export interface ShapeStimulus {
  type: "shape";
  shape: "circle" | "square" | "checkerboard";
  size: number;
  color?: string;
  gridSize?: number;
}

export interface AudioStimulus {
  type: "audio";
  frequency?: number;
  src?: string;
  durationMs: number;
}

export type StimulusDef =
  | InstructionStimulus
  | FixationStimulus
  | BlankStimulus
  | TextStimulus
  | ImageStimulus
  | ShapeStimulus
  | AudioStimulus;

// ---- Trial ----

export interface TrialDef {
  id?: string;
  stimulus: StimulusDef;
  durationMs: number;
  /** Marker code sent to server when this trial's stimulus appears */
  markerCode: string;
  /** Whether to capture a user response for this trial */
  captureResponse?: boolean;
  responseWindowMs?: number;
}

// ---- Trial Generators ----

export interface FixedSequenceGenerator {
  type: "fixed";
  trials: TrialDef[];
}

export interface OddballGenerator {
  type: "oddball";
  totalTrials: number;
  targetRatio: number;
  stimuli: {
    standard: StimulusDef;
    target: StimulusDef;
    distractors?: StimulusDef[];
  };
  timing: {
    stimulusDurationMs: number;
    isiMs: number;
    isiJitterMs?: number;
  };
  requiresResponse: boolean;
  responseWindowMs?: number;
}

export interface RSVPGenerator {
  type: "rsvp";
  items: StimulusDef[];
  durationPerItemMs: number;
  targetIndices?: number[];
}

export interface SSVEPFrequency {
  hz: number;
  position: "center" | "left" | "right" | { x: number; y: number };
  stimulus: StimulusDef;
}

export interface SSVEPGenerator {
  type: "ssvep";
  frequencies: SSVEPFrequency[];
  durationMs: number;
  targetFrequencyHz?: number;
}

export interface P300Generator {
  type: "p300";
  /** 36 characters, row-major (6×6 Farwell-Donchin matrix) */
  matrix: string[];
  /** Target letters for copy-spelling, one per character attempt */
  targetLetters: string[];
  flashDurationMs: number;
  isiMs: number;
  sequencesPerCharacter: number;
  preCharacterMs: number;
  postCharacterMs: number;
}

export interface SpatialCueGenerator {
  type: "spatial-cue";
  totalTrials: number;
  validCueRatio: number;
  cue: StimulusDef;
  target: StimulusDef;
  timing: {
    fixationMs: number;
    cueDurationMs: number;
    soaMs: number;
    targetDurationMs: number;
    responseWindowMs: number;
  };
}

export type TrialGeneratorDef =
  | FixedSequenceGenerator
  | OddballGenerator
  | RSVPGenerator
  | SSVEPGenerator
  | P300Generator
  | SpatialCueGenerator;

// ---- Block & Protocol ----

export interface BlockDef {
  id: string;
  name: string;
  /** Instruction shown before block starts (participant clicks "Ready") */
  instruction?: string;
  trialGenerator: TrialGeneratorDef;
  /** Rest period in ms shown after this block completes (0 = no rest) */
  restAfterMs: number;
}

export interface Protocol {
  id: string;
  name: string;
  description: string;
  version: string;
  blocks: BlockDef[];
}

// ---- Experiment Engine State ----

export type ExperimentPhase =
  | { type: "idle" }
  | { type: "blockInstruction"; blockIndex: number; instruction: string }
  | { type: "trial"; blockIndex: number; trialIndex: number; stimulus: StimulusDef; remainingMs: number }
  | { type: "ssvepTrial"; blockIndex: number; frequencies: SSVEPFrequency[]; targetFrequencyHz?: number; remainingMs: number }
  | { type: "p300Trial"; blockIndex: number; matrix: string[]; targetLetter: string; highlightedRow: number | null; highlightedCol: number | null; remainingMs: number; currentCharIndex: number; totalChars: number; charPhase: "pre" | "flashing" | "post" }
  | { type: "blockRest"; blockIndex: number; remainingMs: number }
  | { type: "complete"; sessionId: string }
  | { type: "error"; message: string };

// ---- API Types ----

export interface StartSessionResponse {
  session_id: string;
  started_at: number;
}

export interface StopSessionResponse {
  session_id: string;
  duration_sec: number;
  total_markers: number;
  total_responses: number;
}

export interface EventMarker {
  code: string;
  timestamp: number; // performance.now() from frontend
  block_id?: string;
  trial_index?: number;
  metadata?: Record<string, unknown>;
}

export interface SessionSummary {
  session_id: string;
  protocol_id: string;
  status: string;
  duration_sec: number;
  total_markers: number;
  started_at: number;
}
