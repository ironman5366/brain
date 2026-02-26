export type AppId = "eeg" | "impedance" | "bandpower" | "fft";

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
];

interface Props {
  onSelectApp: (id: AppId) => void;
}

export function Dashboard({ onSelectApp }: Props) {
  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "2rem",
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, 200px)",
          gap: "1.5rem",
          justifyContent: "center",
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
              gap: "0.75rem",
              padding: "2rem 1.5rem",
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
                fontSize: "1.1rem",
                fontWeight: "bold",
              }}
            >
              {app.title}
            </span>
            <span
              style={{
                fontFamily: "monospace",
                fontSize: "0.8rem",
                color: "#888",
              }}
            >
              {app.description}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
