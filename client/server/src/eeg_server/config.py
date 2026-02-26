import glob
import logging
from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


def find_serial_port() -> str:
    """Auto-detect an OpenBCI USB serial device on macOS."""
    matches = glob.glob("/dev/cu.usbserial-*")
    if len(matches) == 1:
        logger.info("Auto-detected serial port: %s", matches[0])
        return matches[0]
    if len(matches) > 1:
        logger.warning(
            "Multiple USB serial devices found: %s — pass --serial-port explicitly",
            matches,
        )
    return ""


class BoardMode(str, Enum):
    CYTON = "cyton"
    SYNTHETIC = "synthetic"


class ServerConfig(BaseSettings):
    model_config = {"env_prefix": "EEG_"}

    # Board settings
    board_mode: BoardMode = BoardMode.SYNTHETIC
    serial_port: str = ""

    # LSL settings
    lsl_stream_name: str = "BrainFlow_EEG"
    lsl_stream_type: str = "EEG"

    # WebSocket server settings
    host: str = "0.0.0.0"
    port: int = 8765
    ws_buffer_ms: int = 50
    cors_origins: list[str] = Field(default=["http://localhost:5173"])
