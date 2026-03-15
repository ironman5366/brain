This repo contains experiments, models, and data scripts for training models on EEG data.

Much of the existing code in here is aimed at the historical RSVP task, but we're moving forward to the music/audio modality.

## Agent Teams

For complex multi-paradigm experiment sessions, set `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` to enable multi-agent mode. The lead spawns 4 teammates: Monitor, Experimenter, Paradigm Controller, Analyst. See `client/AGENTS.md` for spawn prompts, communication protocol, and tool ownership.