import { useState } from "react";
import { useSessions } from "../hooks/useSessions";
import { useNarrow } from "../hooks/useMediaQuery";
import { SessionList } from "./sessions/SessionList";
import { ReportView } from "./sessions/ReportView";

export type AppId = "eeg" | "impedance" | "bandpower" | "fft" | "experiment" | "bci" | "calibration" | "ball";

interface AppCard {
  id: AppId;
  title: string;
  description: string;
}

const APPS: AppCard[] = [
  {
    id: "eeg",
    title: "Visualizer",
    description: "Real-time EEG trace display",
  },
  {
    id: "impedance",
    title: "Impedance Check",
    description: "Electrode contact quality",
  },
  {
    id: "bandpower",
    title: "Band Power",
    description: "EEG frequency band analysis",
  },
  {
    id: "fft",
    title: "FFT Spectrum",
    description: "Frequency spectrum analysis",
  },
  {
    id: "experiment",
    title: "Experiments",
    description: "Run EEG recording protocols",
  },
  {
    id: "bci",
    title: "BCI Speller",
    description: "P300 speller controlled by Claude",
  },
  {
    id: "calibration",
    title: "Calibration",
    description: "Claude-guided headset setup",
  },
  {
    id: "ball",
    title: "Brain Ball",
    description: "Move a ball with your mind",
  },
];

interface Props {
  onSelectApp: (id: AppId) => void;
}

export function Dashboard({ onSelectApp }: Props) {
  const { sessions, selectedId, report, loadingReport, selectSession } =
    useSessions();
  const narrow = useNarrow();
  const [showSessions, setShowSessions] = useState(false);

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: narrow ? "column" : "row", overflow: "hidden" }}>
      {/* Session sidebar — collapsible on narrow screens */}
      {narrow ? (
        <button
          onClick={() => setShowSessions((v) => !v)}
          style={{
            padding: "0.4rem 1rem",
            background: "none",
            border: "none",
            borderBottom: "1px solid #333",
            color: "#888",
            fontFamily: "monospace",
            fontSize: "0.75rem",
            cursor: "pointer",
            textAlign: "left",
          }}
        >
          Sessions ({sessions.length}) {showSessions ? "▲" : "▼"}
        </button>
      ) : null}

      {(narrow ? showSessions : true) && (
        <SessionList
          sessions={sessions}
          selectedId={selectedId}
          onSelect={(id) => {
            selectSession(id);
            if (narrow) setShowSessions(false);
          }}
        />
      )}

      {/* Right: Tool grid or report */}
      {selectedId ? (
        <ReportView
          report={report}
          loading={loadingReport}
          sessionId={selectedId}
        />
      ) : (
        <div
          style={{
            flex: 1,
            overflow: "auto",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: narrow ? "1rem" : "2rem",
            minWidth: 0,
          }}
        >
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))",
              gap: "0.75rem",
              width: "100%",
              maxWidth: 600,
            }}
          >
            {APPS.map((app) => (
              <button
                key={app.id}
                onClick={() => onSelectApp(app.id)}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: "0.5rem",
                  padding: narrow ? "1rem 0.75rem" : "1.25rem 1rem",
                  backgroundColor: "#1a1a2e",
                  border: "1px solid #333",
                  borderRadius: 12,
                  cursor: "pointer",
                  color: "#eee",
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
                <span
                  style={{
                    fontFamily: "monospace",
                    fontSize: "0.9rem",
                    fontWeight: "bold",
                    textAlign: "center",
                    wordBreak: "break-word",
                  }}
                >
                  {app.title}
                </span>
                <span
                  style={{
                    fontFamily: "monospace",
                    fontSize: "0.7rem",
                    color: "#888",
                    textAlign: "center",
                    wordBreak: "break-word",
                  }}
                >
                  {app.description}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
