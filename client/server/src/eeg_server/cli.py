import logging
import signal
import sys
from pathlib import Path

import click
import uvicorn
from dotenv import load_dotenv

from .board import EEGStream
from .config import BoardMode, ServerConfig, find_serial_port
from .cyton import CytonHeadset
from .web import create_app

logger = logging.getLogger(__name__)


def _run_server(mode: str, serial_port: str, port: int, host: str) -> None:
    """Core server startup — extracted so watchfiles can restart it."""
    # Load .env (walks up from cwd to find it)
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Auto-detect serial port for cyton mode if not specified
    if mode == "cyton" and not serial_port:
        serial_port = find_serial_port()
        if not serial_port:
            logger.error(
                "No USB serial device found. Plug in the Cyton dongle "
                "or pass --serial-port explicitly."
            )
            sys.exit(1)

    config = ServerConfig(
        board_mode=BoardMode(mode),
        serial_port=serial_port,
        host=host,
        port=port,
    )

    # Select headset (only Cyton for now; synthetic mode uses no headset)
    headset = CytonHeadset() if config.board_mode == BoardMode.CYTON else None

    # Start board acquisition + LSL stream
    eeg_stream = EEGStream(config, headset=headset)

    def shutdown(signum, frame):
        logger.info("Shutting down...")
        eeg_stream.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        eeg_stream.start()
    except Exception as e:
        logger.error("Failed to start board: %s", e)
        sys.exit(1)

    # Create and run FastAPI app
    app = create_app(config, eeg_stream)

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        eeg_stream.stop()


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["cyton", "synthetic"]),
    default="synthetic",
    help="Board mode: 'cyton' for real hardware, 'synthetic' for simulated data",
)
@click.option(
    "--serial-port",
    default="",
    help="Serial port for Cyton board. Auto-detected if omitted.",
)
@click.option("--port", default=8765, type=int, help="Server port")
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--reload", is_flag=True, default=False, help="Auto-restart on code changes")
def main(mode: str, serial_port: str, port: int, host: str, reload: bool) -> None:
    """Start the EEG streaming server."""
    if reload:
        from watchfiles import run_process

        src_dir = str(Path(__file__).resolve().parent)
        logger.info("Watching %s for changes...", src_dir)
        run_process(
            src_dir,
            target=_run_server,
            args=(mode, serial_port, port, host),
        )
    else:
        _run_server(mode, serial_port, port, host)
