import { useEffect, useRef } from "react";
import type { AppId } from "../components/Dashboard";

const API_BASE = "http://localhost:8765";

type View = "dashboard" | AppId;

export function useNavigation(onNavigate: (view: View) => void) {
  const callbackRef = useRef(onNavigate);
  callbackRef.current = onNavigate;

  useEffect(() => {
    const es = new EventSource(`${API_BASE}/api/nav/events`);

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "navigate") {
          callbackRef.current(data.view as View);
        }
      } catch {
        // ignore
      }
    };

    return () => es.close();
  }, []);
}
