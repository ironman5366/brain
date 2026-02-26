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

### Server-first logic and debugging
Claude can't see the UI. So:
- **Business logic lives in Python**, not in the frontend. The React client should be a thin display layer over server APIs.
- **APIs should be usable by both the frontend and scripts/tools.** If the frontend calls it, a `curl` or Python script should be able to call it too.
- **Write test scripts for debugging.** When something breaks, reproduce it with a script in `server/scripts/` that Claude can run and iterate on directly, rather than relying on UI error messages. The EEG device is physically connected — scripts can read from it.