import type { StimulusDef } from "../../lib/experiment.types";

interface Props {
  stimulus: StimulusDef;
}

export function StimulusRenderer({ stimulus }: Props) {
  switch (stimulus.type) {
    case "fixation":
      return (
        <div
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: "#141416",
          }}
        >
          <span
            style={{
              fontFamily: "monospace",
              fontSize: stimulus.size ?? "4rem",
              color: "#888",
              userSelect: "none",
            }}
          >
            {stimulus.symbol ?? "+"}
          </span>
        </div>
      );

    case "blank":
      return (
        <div style={{ flex: 1, backgroundColor: "#141416" }} />
      );

    case "instruction":
      return (
        <div
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: "#141416",
            padding: "2rem",
          }}
        >
          <span
            style={{
              fontFamily: "monospace",
              fontSize: stimulus.fontSize ?? "1.5rem",
              color: "#eee",
              textAlign: "center",
              lineHeight: 1.6,
              whiteSpace: "pre-line",
            }}
          >
            {stimulus.text}
          </span>
        </div>
      );

    case "audio":
      return (
        <div
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: "#141416",
          }}
        >
          <span
            style={{
              fontFamily: "monospace",
              fontSize: "4rem",
              color: "#888",
              userSelect: "none",
            }}
          >
            +
          </span>
        </div>
      );

    default:
      return (
        <div
          style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: "#141416",
          }}
        >
          <span
            style={{
              fontFamily: "monospace",
              fontSize: "1rem",
              color: "#555",
            }}
          >
            Stimulus type "{(stimulus as { type: string }).type}" not yet
            implemented
          </span>
        </div>
      );
  }
}
