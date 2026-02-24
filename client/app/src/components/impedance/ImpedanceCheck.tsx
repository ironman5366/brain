import { useImpedanceCheck } from "../../hooks/useImpedanceCheck";
import { ImpedanceBar } from "./ImpedanceBar";

interface Props {
  channelNames: string[];
}

export function ImpedanceCheck({ channelNames }: Props) {
  const { status, results, thresholds, error, startCheck } =
    useImpedanceCheck();

  const isRunning = status === "running";
  const measuredCount = results.length;
  const totalChannels = channelNames.length;

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        padding: "2rem",
        maxWidth: 640,
        margin: "0 auto",
        width: "100%",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          marginBottom: "1.5rem",
        }}
      >
        <button
          onClick={startCheck}
          disabled={isRunning}
          style={{
            padding: "0.6rem 1.5rem",
            fontSize: "0.9rem",
            fontFamily: "monospace",
            fontWeight: "bold",
            border: "1px solid #555",
            borderRadius: 6,
            backgroundColor: isRunning ? "#333" : "#1a1a2e",
            color: isRunning ? "#666" : "#eee",
            cursor: isRunning ? "not-allowed" : "pointer",
          }}
        >
          {isRunning ? "Measuring..." : "Check Impedance"}
        </button>

        {isRunning && (
          <span
            style={{
              fontFamily: "monospace",
              fontSize: "0.85rem",
              color: "#888",
            }}
          >
            {measuredCount}/{totalChannels} channels
          </span>
        )}

        {status === "done" && (
          <span
            style={{
              fontFamily: "monospace",
              fontSize: "0.85rem",
              color: "#4caf50",
            }}
          >
            Complete
          </span>
        )}
      </div>

      {error && (
        <div
          style={{
            fontFamily: "monospace",
            fontSize: "0.85rem",
            color: "#f44",
            marginBottom: "1rem",
          }}
        >
          Error: {error}
        </div>
      )}

      {thresholds && results.length > 0 && (
        <div>
          {results.map((r) => (
            <ImpedanceBar
              key={r.name}
              name={r.name}
              impedance={r.impedance}
              thresholds={thresholds}
            />
          ))}
        </div>
      )}

      {isRunning && measuredCount < totalChannels && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            padding: "0.5rem 0",
            fontFamily: "monospace",
            fontSize: "0.85rem",
            color: "#666",
          }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              backgroundColor: "#ff9800",
              display: "inline-block",
              animation: "pulse 1s ease-in-out infinite",
            }}
          />
          Measuring {channelNames[measuredCount]}...
        </div>
      )}

      {status === "idle" && (
        <div
          style={{
            fontFamily: "monospace",
            fontSize: "0.85rem",
            color: "#555",
            marginTop: "1rem",
            lineHeight: 1.6,
          }}
        >
          Checks electrode-skin contact quality by measuring impedance on each
          channel. The EEG stream will pause briefly during the check.
        </div>
      )}
    </div>
  );
}
