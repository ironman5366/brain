# Builtin imports
from pathlib import Path

# Location you've downloaded https://huggingface.co/datasets/Alljoined/Alljoined-1.6M
ALLJOINED_BASE_DIR = Path("/kreka/research/willy/side/alljoined")


def fetch_alljoined():
    raw_dir = ALLJOINED_BASE_DIR / "raw_eeg"

    for subj_dir in raw_dir.iterdir():
        # Like "alljoined::sub-01"
        subj_id = f"alljoined::{subj_dir.name}"

        for session_dir in subj_dir.iterdir():
            session_id = f"{subj_id}::{session_dir.name}"

            for block_dir in session_dir.iterdir():
                block_id = f"{session_id}::{block_dir.name}"

                metadata_files = list(block_dir.glob("*.json"))
                edf_files = list(block_dir.glob("*.edf"))

                # Make sure we're not missing something
                assert len(edf_files) == 1, f"More than 1 edf file in {block_dir}"

                metadata_file = None
                if metadata_files:
                    assert len(metadata_files) == 1, (
                        f"More than 1 metadata file in {block_dir}"
                    )
                    metadata_file = metadata_files[0]

                yield {
                    "file": str(edf_files[0]),
                    "subject": subj_id,
                    "session": session_id,
                    "block": block_id,
                    "metadata": metadata_file
                }
