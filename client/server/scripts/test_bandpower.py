"""
Test band power computation against the live board via the REST API.

Usage:
    cd server/
    uv run python scripts/test_bandpower.py
"""

import json
import logging
import urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

API_BASE = "http://localhost:8765"


def test_via_api():
    """Test the REST endpoint against a running server."""
    logger.info("=== Testing GET /api/bandpower ===")

    req = urllib.request.Request(f"{API_BASE}/api/bandpower")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())

    if "error" in data:
        logger.error("Error: %s", data["error"])
        return

    logger.info(
        "Window: %d samples at %d Hz (%d channels)",
        data["window_samples"],
        data["sampling_rate"],
        data["num_channels"],
    )
    logger.info("Total power: %.4f", data["total_power"])
    logger.info("")

    rel_sum = 0.0
    for band in data["bands"]:
        rel_sum += band["relative"]
        logger.info(
            "  %-6s %5.1f-%5.1f Hz: power=%.6f  relative=%5.1f%%  stddev=%.6f  | %s",
            band["name"],
            band["low"],
            band["high"],
            band["power"],
            band["relative"] * 100,
            band["stddev"],
            band["description"],
        )

    logger.info("")
    logger.info("Relative powers sum: %.4f (should be ~1.0)", rel_sum)

    # Sanity checks
    assert abs(rel_sum - 1.0) < 0.01, f"Relative powers don't sum to 1: {rel_sum}"
    assert len(data["bands"]) == 5, f"Expected 5 bands, got {len(data['bands'])}"
    assert data["total_power"] > 0, "Total power should be positive"
    logger.info("All checks passed.")


if __name__ == "__main__":
    test_via_api()
