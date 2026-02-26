import { useExperiment } from "../../hooks/useExperiment";
import type { Protocol } from "../../lib/experiment.types";
import { ALPHA_PROTOCOL } from "../../lib/protocols/alpha";
import { SSVEP_PROTOCOL } from "../../lib/protocols/ssvep";
import { StimulusRenderer } from "./StimulusRenderer";
import { SSVEPRenderer } from "./SSVEPRenderer";

const PROTOCOLS: Protocol[] = [ALPHA_PROTOCOL, SSVEP_PROTOCOL];

export function ExperimentApp() {
  const experiment = useExperiment();
  const { phase } = experiment;

  // Idle — show protocol picker
  if (phase.type === "idle") {
    return (
      <ProtocolPicker
        protocols={PROTOCOLS}
        onStart={(protocol) => experiment.start(protocol)}
      />
    );
  }

  // Block instruction — show instruction + ready button
  if (phase.type === "blockInstruction") {
    return (
      <BlockInstruction
        instruction={phase.instruction}
        onReady={() => experiment.ready()}
        onAbort={() => experiment.abort()}
        progress={experiment.progress}
      />
    );
  }

  // Running trial — show stimulus + countdown
  if (phase.type === "trial") {
    return (
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <StimulusRenderer stimulus={phase.stimulus} />
        <TrialFooter
          remainingMs={phase.remainingMs}
          progress={experiment.progress}
          onAbort={() => experiment.abort()}
        />
      </div>
    );
  }

  // SSVEP trial — flickering stimulus
  if (phase.type === "ssvepTrial") {
    return (
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
        <SSVEPRenderer
          frequencies={phase.frequencies}
          targetFrequencyHz={phase.targetFrequencyHz}
        />
        <TrialFooter
          remainingMs={phase.remainingMs}
          progress={experiment.progress}
          onAbort={() => experiment.abort()}
        />
      </div>
    );
  }

  // Rest between blocks
  if (phase.type === "blockRest") {
    return (
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "#141416",
          gap: "1.5rem",
        }}
      >
        <span
          style={{
            fontFamily: "monospace",
            fontSize: "1.5rem",
            color: "#888",
          }}
        >
          Rest
        </span>
        <span
          style={{
            fontFamily: "monospace",
            fontSize: "2.5rem",
            color: "#555",
          }}
        >
          {Math.ceil(phase.remainingMs / 1000)}s
        </span>
      </div>
    );
  }

  // Complete
  if (phase.type === "complete") {
    return (
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "1.5rem",
        }}
      >
        <span
          style={{
            fontFamily: "monospace",
            fontSize: "1.5rem",
            fontWeight: "bold",
            color: "#4caf50",
          }}
        >
          Experiment Complete
        </span>
        <span
          style={{ fontFamily: "monospace", fontSize: "0.85rem", color: "#888" }}
        >
          Session: {phase.sessionId}
        </span>
        <button
          onClick={() => experiment.reset()}
          style={{
            padding: "0.6rem 2rem",
            fontFamily: "monospace",
            fontSize: "0.9rem",
            border: "1px solid #555",
            borderRadius: 6,
            backgroundColor: "#1a1a2e",
            color: "#eee",
            cursor: "pointer",
            marginTop: "1rem",
          }}
        >
          Run Another
        </button>
      </div>
    );
  }

  // Error
  if (phase.type === "error") {
    return (
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "1.5rem",
        }}
      >
        <span
          style={{
            fontFamily: "monospace",
            fontSize: "1.2rem",
            color: "#f44",
          }}
        >
          Error: {phase.message}
        </span>
        <button
          onClick={() => experiment.reset()}
          style={{
            padding: "0.6rem 2rem",
            fontFamily: "monospace",
            fontSize: "0.9rem",
            border: "1px solid #555",
            borderRadius: 6,
            backgroundColor: "#1a1a2e",
            color: "#eee",
            cursor: "pointer",
          }}
        >
          Back
        </button>
      </div>
    );
  }

  return null;
}

// --- Sub-components ---

function ProtocolPicker({
  protocols,
  onStart,
}: {
  protocols: Protocol[];
  onStart: (protocol: Protocol) => void;
}) {
  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "2rem",
        gap: "2rem",
      }}
    >
      <h2
        style={{
          fontFamily: "monospace",
          color: "#eee",
          margin: 0,
          fontSize: "1.1rem",
        }}
      >
        Select Experiment
      </h2>

      {protocols.map((p) => {
        const totalSec = p.blocks.reduce((sum, b) => {
          let trialMs = 0;
          if (b.trialGenerator.type === "fixed") {
            trialMs = b.trialGenerator.trials.reduce((s, t) => s + t.durationMs, 0);
          } else if (b.trialGenerator.type === "ssvep") {
            trialMs = b.trialGenerator.durationMs;
          }
          return sum + trialMs / 1000 + b.restAfterMs / 1000;
        }, 0);

        return (
          <button
            key={p.id}
            onClick={() => onStart(p)}
            style={{
              padding: "1.5rem 2rem",
              maxWidth: 500,
              width: "100%",
              backgroundColor: "#1a1a2e",
              border: "1px solid #333",
              borderRadius: 12,
              cursor: "pointer",
              color: "#eee",
              textAlign: "left",
              transition: "border-color 0.15s, background-color 0.15s",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = "#555";
              e.currentTarget.style.backgroundColor = "#1e1e38";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = "#333";
              e.currentTarget.style.backgroundColor = "#1a1a2e";
            }}
          >
            <div
              style={{
                fontFamily: "monospace",
                fontSize: "1.1rem",
                fontWeight: "bold",
              }}
            >
              {p.name}
            </div>
            <div
              style={{
                fontFamily: "monospace",
                fontSize: "0.8rem",
                color: "#888",
                marginTop: "0.5rem",
              }}
            >
              {p.description}
            </div>
            <div
              style={{
                fontFamily: "monospace",
                fontSize: "0.75rem",
                color: "#555",
                marginTop: "0.25rem",
              }}
            >
              {p.blocks.length} blocks / ~{Math.round(totalSec)}s total
            </div>
          </button>
        );
      })}
    </div>
  );
}

function BlockInstruction({
  instruction,
  onReady,
  onAbort,
  progress,
}: {
  instruction: string;
  onReady: () => void;
  onAbort: () => void;
  progress: { block: number; totalBlocks: number } | null;
}) {
  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "2rem",
        gap: "2rem",
      }}
    >
      {progress && (
        <span
          style={{
            fontFamily: "monospace",
            fontSize: "0.8rem",
            color: "#555",
          }}
        >
          Block {progress.block + 1} of {progress.totalBlocks}
        </span>
      )}

      <div
        style={{
          fontFamily: "monospace",
          fontSize: "1.5rem",
          color: "#eee",
          textAlign: "center",
          lineHeight: 1.6,
          maxWidth: 600,
          whiteSpace: "pre-line",
        }}
      >
        {instruction}
      </div>

      <button
        onClick={onReady}
        style={{
          padding: "0.75rem 3rem",
          fontSize: "1.1rem",
          fontFamily: "monospace",
          fontWeight: "bold",
          border: "1px solid #4caf50",
          borderRadius: 8,
          backgroundColor: "#1a2e1a",
          color: "#4caf50",
          cursor: "pointer",
          marginTop: "1rem",
        }}
      >
        Ready
      </button>

      <button
        onClick={onAbort}
        style={{
          padding: "0.4rem 1.5rem",
          fontFamily: "monospace",
          fontSize: "0.8rem",
          border: "1px solid #444",
          borderRadius: 6,
          backgroundColor: "transparent",
          color: "#666",
          cursor: "pointer",
        }}
      >
        Abort
      </button>
    </div>
  );
}

function TrialFooter({
  remainingMs,
  progress,
  onAbort,
}: {
  remainingMs: number;
  progress: { block: number; totalBlocks: number } | null;
  onAbort: () => void;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0.5rem 1rem",
        borderTop: "1px solid #222",
        backgroundColor: "#141416",
      }}
    >
      <span
        style={{ fontFamily: "monospace", fontSize: "0.75rem", color: "#444" }}
      >
        {progress
          ? `Block ${progress.block + 1}/${progress.totalBlocks}`
          : ""}
      </span>
      <span
        style={{ fontFamily: "monospace", fontSize: "0.85rem", color: "#555" }}
      >
        {Math.ceil(remainingMs / 1000)}s
      </span>
      <button
        onClick={onAbort}
        style={{
          padding: "0.2rem 0.8rem",
          fontFamily: "monospace",
          fontSize: "0.7rem",
          border: "1px solid #333",
          borderRadius: 4,
          backgroundColor: "transparent",
          color: "#555",
          cursor: "pointer",
        }}
      >
        Abort
      </button>
    </div>
  );
}
