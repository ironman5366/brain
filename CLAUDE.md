This repo contains experiments, models, and data scripts for training models on EEG data.

Much of the existing code in here is aimed at the historical RSVP task, but we're moving forward to the music/audio modality.

## Agent Teams

For complex multi-paradigm experiment sessions, set `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` to enable multi-agent mode. The lead is the main Claude Code session (not a separate agent) and spawns 5 teammates: Voice Listener, Monitor, Experimenter, Paradigm Controller, Analyst. The Voice Listener polls `voice_inbox()` and forwards user speech to the lead. The lead uses `voice_ask()` to speak to the user via local TTS (Kokoro) and STT (faster-whisper), and `voice_notify()` to update the browser status bar. All user confirmation must be verbal — the user cannot type during experiments. See `client/AGENTS.md` for spawn prompts, communication protocol, and tool ownership.