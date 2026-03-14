import type { EventMarker } from "./experiment.types";

const API_BASE = "http://localhost:8765";
const FLUSH_INTERVAL_MS = 500;
const FLUSH_BATCH_SIZE = 10;

/**
 * Buffers event markers and sends them to the server in batches.
 * Fire-and-forget: if a send fails, markers are kept for next flush.
 */
export class MarkerSender {
  private sessionId: string;
  private buffer: EventMarker[] = [];
  private flushTimer: ReturnType<typeof setInterval> | null = null;

  constructor(sessionId: string) {
    this.sessionId = sessionId;
    this.flushTimer = setInterval(() => this.flush(), FLUSH_INTERVAL_MS);
  }

  /** Record a marker. Safe to call from rAF callbacks. */
  send(marker: EventMarker): void {
    this.buffer.push({
      ...marker,
      client_time_ms:
        marker.client_time_ms ?? performance.timeOrigin + marker.timestamp,
    });
    if (this.buffer.length >= FLUSH_BATCH_SIZE) {
      this.flush();
    }
  }

  /** Flush buffered markers to server. */
  async flush(): Promise<void> {
    if (this.buffer.length === 0) return;
    const batch = this.buffer.splice(0);
    try {
      await fetch(`${API_BASE}/api/session/marker`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: this.sessionId,
          markers: batch,
        }),
      });
    } catch {
      // Put markers back for retry
      this.buffer.unshift(...batch);
    }
  }

  /** Stop flushing and send any remaining markers. */
  async stop(): Promise<void> {
    if (this.flushTimer) clearInterval(this.flushTimer);
    this.flushTimer = null;
    await this.flush();
  }
}
