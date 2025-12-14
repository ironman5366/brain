# Builtin imports
import os

# External imports
import dotenv

dotenv.load_dotenv()

WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
