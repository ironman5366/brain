import { useCallback, useEffect, useState } from "react";

const API_BASE = "http://localhost:8765";

export interface SessionSummary {
  session_id: string;
  protocol_id: string;
  status: string;
  duration_sec: number;
  total_markers: number;
  started_at: number;
  has_report: boolean;
}

export function useSessions() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [loadingReport, setLoadingReport] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const resp = await fetch(`${API_BASE}/api/sessions`);
      if (resp.ok) setSessions(await resp.json());
    } catch {
      // server not up yet
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Fetch report when selection changes
  useEffect(() => {
    if (!selectedId) {
      setReport(null);
      return;
    }

    setLoadingReport(true);
    fetch(`${API_BASE}/api/sessions/${selectedId}/report`)
      .then((resp) => (resp.ok ? resp.json() : null))
      .then((data) => setReport(data?.content ?? null))
      .catch(() => setReport(null))
      .finally(() => setLoadingReport(false));
  }, [selectedId]);

  const selectSession = useCallback(
    (id: string) => {
      setSelectedId((prev) => (prev === id ? null : id));
    },
    [],
  );

  return { sessions, selectedId, report, loadingReport, selectSession, refresh };
}
