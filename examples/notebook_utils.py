from __future__ import annotations

from typing import Dict, Any

import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# Import the training entrypoint
from train_dqn_simple import train_dqn_simple


def run_dqn_quick(num_episodes: int = 5,
                  max_steps: int = 100,
                  device: str = "cpu",
                  checkpoint_dir: str = "experiments/checkpoints",
                  minitask_type: str = "development_build_city",
                  minitask_level: str = "easy",
                  buffer_capacity: int = 10000) -> Dict[str, Any]:
    """Run a short DQN training and return metrics as a dict with a DataFrame.
    
    Args:
        num_episodes: Number of episodes to train
        max_steps: Max steps per episode
        device: Device ('cpu' or 'cuda')
        checkpoint_dir: Directory to save checkpoints
        minitask_type: CivRealm minitask type
        minitask_level: Difficulty level
        buffer_capacity: Replay buffer capacity
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    result = train_dqn_simple(
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps,
        checkpoint_dir=checkpoint_dir,
        device=device,
        batch_size=32,
        train_freq=4,
        warmup_steps=max(30, max_steps // 3),
        minitask_type=minitask_type,
        minitask_level=minitask_level,
        buffer_capacity=buffer_capacity,
        checkpoint_freq=0,  # No intermediate checkpoints for quick runs
    )
    return result


def plot_dqn_curves(logs_df: pd.DataFrame) -> None:
    """Plot reward and average loss per episode from the logs DataFrame."""
    if not isinstance(logs_df, pd.DataFrame) or logs_df.empty:
        print("No logs to plot.")
        return

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Reward per episode
    ax[0].plot(logs_df["episode"], logs_df["reward"], marker="o")
    ax[0].set_title("Episode Reward")
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Reward")

    # Average loss per episode
    ax[1].plot(logs_df["episode"], logs_df["avg_loss"], color="tab:red", marker="o")
    ax[1].set_title("Average Loss")
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Loss")

    plt.tight_layout()
    plt.show()
