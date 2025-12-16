"""
Long-duration DQN training script for 300-episode runs.
Optimized for compatibility with teammate's battle_ancient_era training approach.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path to import train_dqn_simple
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_dqn_simple import train_dqn_simple


def train_dqn_long(
    num_episodes: int = 300,
    max_steps_per_episode: int = 100,
    minitask_type: str = "battle_ancient_era",
    minitask_level: str = "easy",
    checkpoint_dir: str = "experiments/checkpoints_long",
    device: str = "cpu",
    buffer_capacity: int = 50000,
    checkpoint_freq: int = 50,
    resume_from: str = None,
    start_episode: int = 0,
) -> None:
    """
    Run long DQN training with incremental logging and checkpoints.
    
    Args:
        num_episodes: Total episodes (default 300 for consistency with teammate)
        max_steps_per_episode: Max steps per episode
        minitask_type: CivRealm minitask type
        minitask_level: Difficulty level
        checkpoint_dir: Directory to save checkpoints
        device: Device ('cpu' or 'cuda')
        buffer_capacity: Replay buffer capacity (50k for long training)
        checkpoint_freq: Save checkpoint every N episodes
        resume_from: Path to checkpoint to resume from
        start_episode: Episode number to start from (when resuming)
    """
    
    print("="*80)
    if resume_from:
        print(f" RESUMING DQN TRAINING FROM EPISODE {start_episode}")
    else:
        print(" LONG DQN TRAINING - 300 EPISODES")
    print("="*80)
    print(f" Minitask:      {minitask_type} ({minitask_level})")
    print(f" Episodes:      {start_episode+1}-{num_episodes}")
    print(f" Max Steps:     {max_steps_per_episode}")
    print(f" Buffer:        {buffer_capacity}")
    print(f" Device:        {device}")
    print(f" Checkpoints:   Every {checkpoint_freq} episodes")
    if resume_from:
        print(f" Resume from:   {resume_from}")
    print("="*80)
    print()
    
    # Run training
    result = train_dqn_simple(
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        checkpoint_dir=checkpoint_dir,
        device=device,
        batch_size=32,
        train_freq=4,
        warmup_steps=100,
        minitask_type=minitask_type,
        minitask_level=minitask_level,
        buffer_capacity=buffer_capacity,
        checkpoint_freq=checkpoint_freq,
        resume_from=resume_from,
        start_episode=start_episode,
    )
    
    # Save full training logs (support both keys from trainer)
    logs_df = result.get("logs_df") or result.get("episode_logs")
    csv_path = Path(checkpoint_dir) / "training_logs.csv"
    logs_df.to_csv(csv_path, index=False)
    print(f"\n✓ Training logs saved to: {csv_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print(" TRAINING SUMMARY")
    print("="*80)
    print(f" Total episodes:       {len(logs_df)}")
    print(f" Total steps:          {result['total_steps']}")
    print(f" Training steps:       {result['train_steps']}")
    print(f" Final checkpoint:     {result['checkpoint_path']}")
    print(f" Mean reward:          {logs_df['reward'].mean():.2f}")
    print(f" Best reward:          {logs_df['reward'].max():.2f}")
    print(f" Final epsilon:        {logs_df['epsilon'].iloc[-1]:.3f}")
    print(f" Valid actions:        {result['valid_actions']}/{result['total_steps']} ({result['valid_actions']/result['total_steps']*100:.1f}%)")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Long DQN training (300 episodes)")
    parser.add_argument("--num-episodes", type=int, default=300, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--minitask-type", type=str, default="battle_ancient_era", help="Minitask type")
    parser.add_argument("--minitask-level", type=str, default="easy", help="Minitask level")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/checkpoints_long")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--buffer-capacity", type=int, default=50000, help="Replay buffer capacity")
    parser.add_argument("--checkpoint-freq", type=int, default=50, help="Checkpoint frequency")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--start-episode", type=int, default=0, help="Episode to start from (when resuming)")
    
    args = parser.parse_args()
    
    train_dqn_long(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        minitask_type=args.minitask_type,
        minitask_level=args.minitask_level,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        buffer_capacity=args.buffer_capacity,
        checkpoint_freq=args.checkpoint_freq,
        resume_from=args.resume_from,
        start_episode=args.start_episode,
    )


if __name__ == "__main__":
    main()
