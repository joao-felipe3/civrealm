"""Training script for DQN agent on CivRealm minigames."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any
import civrealm  # Required to register CivRealm environments

import gymnasium as gym
import numpy as np
import torch

from agents import DQNAgent, ReplayBuffer
from networks import CivFeatureExtractor, CivDQNNetwork
from utils.action_utils import get_action_dims, extract_masks, extract_obs_features


def train_dqn(
    env_name: str = "civrealm/FreecivTensorMinitask-v0",
    minitask_type: str = "development_build_city",
    minitask_level: str = "easy",
    num_episodes: int = 1000,
    max_steps_per_episode: int = 500,
    batch_size: int = 32,
    buffer_capacity: int = 10000,
    learning_starts: int = 1000,
    train_freq: int = 4,
    save_freq: int = 100,
    log_freq: int = 10,
    checkpoint_dir: str = "experiments/checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Train DQN agent on CivRealm minigame."""
    
    print("=" * 70)
    print("DQN TRAINING ON CIVREALM")
    print("=" * 70)
    print(f"Environment: {env_name}")
    print(f"Minitask: {minitask_type} ({minitask_level})")
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes}")
    print("=" * 70)
    
    # Create environment
    print("\n[1] Creating environment...")
    env = gym.make(
        env_name,
        minitask_pattern={"type": minitask_type, "level": minitask_level}
    )
    print("    ✓ Environment created")
    
    # Get action dimensions
    action_dims = get_action_dims(env.action_space)
    print(f"\n[2] Action space dimensions:")
    for key, dim in action_dims.items():
        print(f"    {key}: {dim}")
    
    # Create network and agent
    print("\n[3] Creating DQN network and agent...")
    feature_extractor = CivFeatureExtractor()
    q_network = CivDQNNetwork(
        feature_extractor=feature_extractor,
        action_dims=action_dims,
    )
    
    agent = DQNAgent(
        network=q_network,
        action_dims=action_dims,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=num_episodes * max_steps_per_episode // 2,
        target_update_freq=1000,
        device=device,
    )
    print(f"    ✓ Agent created with {sum(p.numel() for p in q_network.parameters())} parameters")
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    print(f"    ✓ Replay buffer created (capacity: {buffer_capacity})")
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n[4] Starting training loop...")
    print("=" * 70)
    
    global_step = 0
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs_features = extract_obs_features(obs)
        episode_reward = 0.0
        episode_steps = 0
        
        for step in range(max_steps_per_episode):
            # Extract masks
            masks = extract_masks(obs)
            
            # Select action
            action = agent.select_action(obs_features, masks)
            
            # Environment step
            try:
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except Exception as e:
                # Handle invalid actions gracefully
                print(f"    Warning: Invalid action at episode {episode}, step {step}: {e}")
                next_obs = obs
                reward = -0.1  # Small penalty
                done = True
            
            next_obs_features = extract_obs_features(next_obs)
            next_masks = extract_masks(next_obs)
            
            # Store transition
            replay_buffer.push(obs_features, action, reward, next_obs_features, done, next_masks)
            
            # Train
            if global_step >= learning_starts and global_step % train_freq == 0:
                if len(replay_buffer) >= batch_size:
                    batch = replay_buffer.sample(batch_size)
                    metrics = agent.train_step(*batch)
            
            # Update state
            obs = next_obs
            obs_features = next_obs_features
            episode_reward += reward
            episode_steps += 1
            global_step += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Logging
        if (episode + 1) % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Steps: {episode_steps} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward (last {log_freq}): {avg_reward:.2f} | "
                  f"Buffer: {len(replay_buffer)}")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"dqn_episode_{episode + 1}.pt")
            agent.save(checkpoint_path)
            print(f"    ✓ Checkpoint saved: {checkpoint_path}")
    
    # Final save
    final_path = os.path.join(checkpoint_dir, "dqn_final.pt")
    agent.save(final_path)
    print(f"\n[5] Training complete!")
    print(f"    ✓ Final model saved: {final_path}")
    print(f"    ✓ Average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.2f}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on CivRealm")
    parser.add_argument("--minitask-type", type=str, default="development_build_city",
                        help="Minitask type")
    parser.add_argument("--minitask-level", type=str, default="easy",
                        help="Minitask difficulty level")
    parser.add_argument("--num-episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--buffer-capacity", type=int, default=10000,
                        help="Replay buffer capacity")
    parser.add_argument("--learning-starts", type=int, default=1000,
                        help="Steps before training starts")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/checkpoints",
                        help="Directory for saving checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    train_dqn(
        minitask_type=args.minitask_type,
        minitask_level=args.minitask_level,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        learning_starts=args.learning_starts,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
