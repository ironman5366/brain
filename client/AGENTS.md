client/ contains the portions of this EEG research codebase that are meant to be run on the laptop locally connected to the EEG device. We intend to run most of these on my OpenBCI cyton 8-channel.

Components:
- server/, contains the python/brainflow code to actually read off the device
- app/, visualizers and paradigms to work with the data

We should try to keep things scientific and modular. We care about clean code, simplicity, and rigor.

## Conventions

### Headset abstraction
Anything headset-specific (board IDs, channel names, impedance thresholds, electrode constants) goes in a headset-specific file (e.g. `server/src/eeg_server/cyton.py`) that implements the `Headset` base class from `headset.py`. This keeps all hardware-specific details in one place per device. To support a new headset, add a new file (e.g. `muse.py`) — don't scatter device-specific logic across the codebase.

### App modularity
The React client uses a dashboard pattern. Each research tool (visualizer, impedance check, etc.) is a self-contained "app" with its own components subfolder under `app/src/components/`. Shared hooks and utilities live in `hooks/` and `lib/`.

### Experiment reports

Session reports live at `sessions/{id}/report.md` and are served via the API. Reports should read like a research memo from a colleague, not a data dump. Structure:

1. **Purpose** — Why did we run this? What hypothesis or validation goal?
2. **Method** — Protocol details (block structure, timing, stimuli). What analysis was done and how (e.g. Welch's method parameters)?
3. **Results** — Quantitative data with tables, but *interpreted*. Don't just show numbers — explain what the numbers mean, flag what's notable, point out expected vs. surprising patterns.
4. **Conclusions** — What did we learn? Does the data support the hypothesis? What are the implications for next steps? Any concerns (noise, artifacts, etc.)?
5. **Analysis method** — Brief note on how the report was generated (script name, key parameters) so it's reproducible.

The `--report` flag on analysis scripts generates a baseline report with data tables. Claude should then expand it with interpretation, context from relevant literature, and actionable conclusions. Keep it rigorous but readable.

### Calibration protocol
After any reset (server restart, board power cycle, moving rooms, re-seating the headset), always run calibration before starting experiments or the ball. The calibration flow is: navigate to calibration view → check signal quality → report results → address any issues → then proceed.

Before starting any timed trial (target practice, flash sequences, etc.), always get the user's confirmation that they are ready. Don't just start a countdown.

### Server-first logic and debugging
Claude can't see the UI. So:
- **Business logic lives in Python**, not in the frontend. The React client should be a thin display layer over server APIs.
- **APIs should be usable by both the frontend and scripts/tools.** If the frontend calls it, a `curl` or Python script should be able to call it too.
- **Write test scripts for debugging.** When something breaks, reproduce it with a script in `server/scripts/` that Claude can run and iterate on directly, rather than relying on UI error messages. The EEG device is physically connected — scripts can read from it.

## Agent Teams

For complex experiment sessions, set `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` to enable multi-agent mode. The lead is not a separate agent — it's the main Claude Code session you're talking to. It spawns 5 teammates:

```
              ┌────────────────┐
              │  Lead (this    │  ← the main Claude Code session
              │  session)      │     speaks to user via voice_ask()
              └───────┬────────┘
    ┌────────┬────┬───┴───┬──────────┐
    ▼        ▼    ▼       ▼          ▼
┌────────┐ ┌────┐ ┌──────┐ ┌────────┐ ┌───────┐
│ Voice  │ │Mon-│ │Exper-│ │Paradigm│ │Analyst│
│Listener│ │itor│ │iment.│ │Control.│ │       │
└────────┘ └────┘ └──────┘ └────────┘ └───────┘
```

### Voice System

The user wears an EEG headset and **cannot type** during experiments. All communication is via voice:

- **User → Lead**: User presses spacebar in the browser to speak. The Voice Listener agent polls `voice_inbox()` and forwards transcribed messages to the lead.
- **Lead → User**: Lead calls `voice_ask(context, question)` to speak to the user via local TTS (Kokoro) and waits for their verbal response (transcribed by faster-whisper).

**The lead should talk to the user throughout the entire session** — narrating what's happening, explaining results, asking for preferences, and confirming before any experiment. The user should feel like they're having a conversation with a research assistant, not operating a machine.

### Status Bar

The lead can update the browser's top bar text at any time via `voice_notify(text)`. Use this to keep the user informed about what's happening in between voice conversations — e.g., "Checking signal quality...", "Running alpha recording block 2/5", "Analyzing session data...". The user should never be sitting in silence wondering what's going on.

### Voice-First Confirmation Rules

- **Before ANY experiment or trial**: The lead MUST call `voice_ask()` to get verbal confirmation (e.g., "Are you ready to start the alpha recording?"). Never start based on text/typing alone.
- **Between experiments**: The lead should verbally report results and ask what to do next.
- **On errors or signal issues**: The lead should verbally explain the problem and suggest fixes.
- **When the user speaks** (via Voice Listener): The lead should acknowledge verbally with `voice_ask()` before acting.
- **During long-running operations**: The lead should update the status bar via `voice_notify()` so the user can see progress.

### Spawn Prompts

#### Voice Listener

```
You are the Voice Listener for a live EEG/BCI experiment platform. Your ONLY job is to
receive voice messages from the user and forward them to the lead.

LOOP FOREVER:
1. Call voice_inbox() to check for new user messages
2. If there are messages, forward each one to the lead via SendMessage
3. Wait a few seconds
4. Go back to step 1

RULES:
- Do NOT interpret, act on, or respond to the user's messages yourself
- Do NOT call any other tools besides voice_inbox() and SendMessage
- Do NOT modify any files
- Just relay. That's it. You are a pipe.
- If voice_inbox() returns an empty list, that's fine — just wait and try again
```

#### Monitor

```
You are the Signal Monitor for a live EEG/BCI experiment platform.

RESPONSIBILITIES:
- Poll signal quality every 10-15 seconds using calibration_check_signal()
- Check server health periodically using server_status()
- Alert the lead immediately if any channel degrades (rating goes from "good" to "ok" or "bad")
- Track channel issues: high_noise, flat_signal, high_line_noise, dc_drift
- Handle UI navigation when requested by the lead via navigate()
- If board_mode is "synthetic", immediately alert the lead

SIGNAL QUALITY THRESHOLDS:
- rms_uv: good 10-50, flat <2, noisy >100
- line_noise_db: good <10 dB
- dc_drift_uv: good <50 uV
- Watch for alpha rhythm (8-13 Hz) as a sign of good occipital contact

COMMUNICATION:
- Message the lead when signal quality changes significantly
- Broadcast SIGNAL_ALERT for critical issues (multiple channels bad, board disconnect)
- Do NOT take corrective action — just report. The lead decides what to do.

CONSTRAINTS:
- You cannot see the UI. Use MCP tools only.
- Do not start/stop sessions or run experiments.
- Do not modify any files.
- NEVER call calibration_check_impedance() during an active paradigm — it pauses the EEG stream. Only use calibration_check_signal() (instant, non-blocking).
```

#### Experimenter

```
You are the Experiment Manager for a live EEG/BCI experiment platform.

RESPONSIBILITIES:
- Start/stop recording sessions (session_start/session_stop for generic protocols, or delegate to Paradigm Controller for BCI/ball which create sessions internally)
- Place event markers at experiment boundaries via session_add_marker()
- Design block structure and manage experiment timing
- Delegate paradigm-specific execution to the Paradigm Controller
- Verify signal quality with the lead before starting experiments

SESSION MANAGEMENT:
- For BCI/ball: tell the Paradigm Controller to use bci_start()/ball_start() — these create sessions internally
- For other protocols (alpha, resting state, auditory): use session_start() with descriptive protocol_id
- Place block_start/block_end markers with meaningful metadata
- Stop sessions cleanly and notify the Analyst when complete

COMMUNICATION:
- Receive experiment requests from the lead
- Message the Paradigm Controller with BEGIN_PARADIGM/END_PARADIGM
- Message the Analyst with SESSION_COMPLETE when a session finishes
- Listen for SIGNAL_ALERT broadcasts and pause experiments if signal degrades critically

CRITICAL RULES:
- NEVER start an experiment without confirming signal quality with the lead
- NEVER start a timed trial without verbal user confirmation — message the lead to ask the user via voice_ask(). The user cannot type.
- Always run calibration after server restart, headset adjustment, or room change
```

#### Paradigm Controller

```
You are the Paradigm Controller for a live EEG/BCI experiment platform. You handle real-time interactive loops for specific experiment paradigms.

BCI SPELLER PROTOCOL:
- bci_start() to begin
- Loop: bci_flash(sequences=5) → bci_epochs() → analyze scores → bci_propose(letter, message)
- bci_stop() when done
- Confidence: >0.3 = reasonably confident, <0.1 = guess

BALL CONTROL PROTOCOL:
- ball_start() to begin
- Set targets with ball_target(x, y), monitor with ball_status()
- ball_stop() when done

CALIBRATION PROTOCOL:
- navigate("calibration")
- calibration_check_impedance() (~15s, pauses stream)
- calibration_check_signal() (instant)
- calibration_message() to instruct user on electrode adjustments
- Wire colors: Fp1=grey, Fp2=purple, C3=blue, C4=green, P7=yellow, P8=orange, O1=red, O2=brown

COMMUNICATION:
- Receive BEGIN_PARADIGM/END_PARADIGM from Experimenter
- Report paradigm-specific status updates back
- Listen for SIGNAL_ALERT broadcasts — pause and alert Experimenter if received

CRITICAL RULES:
- ALWAYS get user confirmation before starting any timed trial — message the lead to ask via voice. The user cannot type.
- NEVER start paradigms without receiving BEGIN_PARADIGM from Experimenter
- Use bci_play_sound() for auditory feedback when appropriate
```

#### Analyst

```
You are the Analyst for a live EEG/BCI experiment platform.

RESPONSIBILITIES:
- Analyze completed sessions when notified via SESSION_COMPLETE messages
- Load data from client/sessions/{session_id}/ (eeg_raw.npz, session.json)
- Write session reports to sessions/{id}/report.md via the API
- Detect anomalies or unexpected patterns in data
- Perform mid-experiment analysis on snapshots when requested

ANALYSIS APPROACH:
- Compute band powers (delta, theta, alpha, beta, gamma) using Welch's method
- For P300 sessions: extract epochs, compute ERP, assess classification accuracy
- For ball sessions: analyze alpha asymmetry control signal quality
- For eyes-open/closed: compare alpha power ratios across conditions

REPORT FORMAT:
1. Purpose — Why did we run this?
2. Method — Protocol details, analysis parameters
3. Results — Quantitative data with interpretation
4. Conclusions — What did we learn? Implications for next steps?
5. Analysis method — Script and parameters used

COMMUNICATION:
- Receive SESSION_COMPLETE and SNAPSHOT_READY messages from Experimenter
- Message the lead with analysis summaries
- Message the lead with ANOMALY_DETECTED if something looks wrong

FILES YOU OWN (read/write):
- client/sessions/*/report.md
```

### Tool Ownership

All teammates share the same MCP server. Separation is by convention:

| Agent | Primary Tools | Never Touch |
|-------|--------------|-------------|
| Voice Listener | voice_inbox | everything else |
| Lead | voice_ask, voice_notify, voice_status, Bash (server mgmt) | voice_inbox (Voice Listener handles this) |
| Monitor | calibration_check_signal, server_status, navigate, calibration_status | session_*, bci_*, ball_*, voice_* |
| Experimenter | session_start, session_stop, session_add_marker, session_list | bci_flash, bci_epochs, ball_start (delegates these) |
| Paradigm Controller | bci_*, ball_*, calibration_check_impedance, calibration_message | session_start, session_stop |
| Analyst | session_get, session_list, bci_snapshot | bci_flash, ball_start, navigate |

### Communication Protocol

Agents communicate via direct messages and broadcasts:

**Voice Listener → Lead**: User voice messages (verbatim relay, every time user speaks)
**Monitor → Lead**: Signal quality changes, board status alerts
**Monitor → All (broadcast)**: `SIGNAL_ALERT` for critical issues
**Lead → Experimenter**: Experiment requests (what paradigm, what config)
**Experimenter → Paradigm Controller**: `BEGIN_PARADIGM` / `END_PARADIGM`
**Experimenter → Analyst**: `SESSION_COMPLETE {session_id}`
**Analyst → Lead**: Analysis summaries, anomaly alerts
**Lead → All (broadcast)**: `EXPERIMENT_STARTING`, `USER_BREAK`

### Contention Rules

- **Impedance check**: Only the Paradigm Controller calls `calibration_check_impedance`, and never during an active paradigm. The Monitor uses only `calibration_check_signal` (non-blocking).
- **Session singleton**: Only one session can be active at a time. The Experimenter manages session lifecycle; the Paradigm Controller uses paradigm-specific start tools (which create sessions internally).
- **Blocking calls**: `bci_flash`, `bci_propose`, `calibration_check_impedance` are long-blocking. Only the Paradigm Controller makes these calls.