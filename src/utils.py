import os
import pandas as pd
from datasets import Dataset
import logging
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig
from gymnasium.envs.toy_text.utils import categorical_sample

LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]  # These are the R, G, Y, B locations

LOG_FILE = "logs/grpo_taxi.log"


def step(env, state, a):
    """step function adapted from taxi.py"""
    env.s = state
    transitions = env.unwrapped.P[env.s][a]
    i = categorical_sample([t[0] for t in transitions], env.np_random)
    p, s, r, t = transitions[i]
    env.s = s
    env.lastaction = a
    return int(s), r, t, False, None


def decode(i):
    """decode state from integer (taken from taxi.py)"""
    out = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5
    return list(reversed(out))


def load_dataset(file_path):
    """Load the dataset"""
    df = pd.read_csv(file_path)
    dataset = Dataset.from_pandas(df)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    return train_dataset, test_dataset


def set_up_logger():

    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set logging level
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        handlers=[
            logging.FileHandler(LOG_FILE, mode="a",
                                encoding="utf-8"),  # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )

    logger = logging.getLogger("grpo-taxi")

    return logger


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.resume_from_checkpoint:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint
