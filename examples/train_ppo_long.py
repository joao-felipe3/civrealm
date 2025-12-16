"""
Simplified PPO training script for CivRealm mini-games.
Uses the existing civtensor PPO implementation with a simpler interface.
"""

import argparse
import sys
from pathlib import Path
import time

import torch
import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def create_env(minitask_type: str, minitask_level: str = "easy"):
    """Create CivRealm environment with retries."""
    from civrealm.envs import FreecivMinitaskEnv
    
    for attempt in range(3):
        try:
            env = FreecivMinitaskEnv(minitask_pattern=f"{minitask_type} {minitask_level}")
            env.reset()
            return env
        except Exception as e:
            print(f"[WARN] Env creation failed (attempt {attempt+1}/3): {type(e).__name__}")
            if attempt == 2:
                raise
            time.sleep(2)


def train_ppo(
    num_episodes: int = 300,
    max_steps_per_episode: int = 100,
    minitask_type: str = "development_build_city",
    minitask_level: str = "easy",
    checkpoint_dir: str = "experiments/checkpoints_ppo",
    device: str = "cpu",
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    ppo_epochs: int = 4,
    checkpoint_freq: int = 50,
):
    """
    Train PPO agent using simplified approach.
    
    Note: This is a simplified educational version. For production use,
    consider the full civtensor/algorithms/ppo.py implementation.
    """
    
    print("="*80)
    print(" PPO TRAINING (Simplified)")
    print("="*80)
    print(f" Minitask:      {minitask_type} ({minitask_level})")
    print(f" Episodes:      {num_episodes}")
    print(f" Max Steps:     {max_steps_per_episode}")
    print(f" Learning Rate: {lr}")
    print(f" Gamma:         {gamma}")
    print(f" GAE Lambda:    {gae_lambda}")
    print(f" Clip Epsilon:  {clip_epsilon}")
    print(f" PPO Epochs:    {ppo_epochs}")
    print(f" Device:        {device}")
    print("="*80)
    print()
    
    # Import PPO agent implementation
    try:
        from agents.ppo_agent_simple import SimplePPOAgent
    except ImportError:
        print("[ERROR] SimplePPOAgent not found. Creating basic implementation...")
        print("[INFO] For full PPO, use civtensor/algorithms/ppo.py or run examples/train.py")
        return
    
    # Create environment
    print("[1] Creating environment...")
    env = create_env(minitask_type, minitask_level)
    print("    ✓ Environment created")
    
    # Extract action space info
    print("\n[2] Analyzing action space...")
    action_sizes = {}
    for key, space in env.action_space.spaces.items():
        action_sizes[key] = space.n
        print(f"    {key}: {space.n}")
    
    # Create feature extractor and agent
    print("\n[3] Creating PPO agent...")
    from networks.feature_extractor import CivFeatureExtractor
    
    feature_extractor = CivFeatureExtractor(env.observation_space)
    feature_dim = feature_extractor.get_feature_dim()
    
    agent = SimplePPOAgent(
        feature_dim=feature_dim,
        action_sizes=action_sizes,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        ppo_epochs=ppo_epochs,
        device=device
    )
    print(f"    ✓ Agent created with {agent.get_num_params():,} parameters")
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("\n[4] Starting training...")
    print("="*80)
    
    episode_logs = []
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Rollout episode
        for step in range(max_steps_per_episode):
            # Extract features
            state_tensor = feature_extractor.extract(obs).to(agent.device)
            
            # Select action from policy
            action, log_prob, value = agent.select_action(state_tensor)
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store transition
            agent.store_transition(state_tensor, action, log_prob, reward, value, done or truncated)
            
            episode_reward += reward
            episode_steps += 1
            obs = next_obs
            
            if done or truncated:
                break
        
        # Update policy (PPO update with multiple epochs)
        policy_loss, value_loss, entropy = agent.update()
        
        # Log episode
        episode_logs.append({
            'episode': episode,
            'reward': episode_reward,
            'steps': episode_steps,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
        })
        
        # Print progress
        print(f"Ep {episode:3d}/{num_episodes} | "
              f"Steps: {episode_steps:3d} | "
              f"Reward: {episode_reward:7.2f} | "
              f"P_Loss: {policy_loss:8.4f} | "
              f"V_Loss: {value_loss:8.4f}")
        
        # Save checkpoint
        if episode % checkpoint_freq == 0:
            ckpt_path = Path(checkpoint_dir) / f"ppo_ep{episode}.pt"
            agent.save(str(ckpt_path))
            print(f"    ✓ Checkpoint saved: {ckpt_path}")
    
    # Save final model
    final_path = Path(checkpoint_dir) / "ppo_model.pt"
    agent.save(str(final_path))
    
    # Save logs
    df = pd.DataFrame(episode_logs)
    csv_path = Path(checkpoint_dir) / "training_logs.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*80)
    print(" TRAINING COMPLETE")
    print("="*80)
    print(f" Total episodes:  {num_episodes}")
    print(f" Mean reward:     {df['reward'].mean():.2f}")
    print(f" Best reward:     {df['reward'].max():.2f}")
    print(f" Final model:     {final_path}")
    print(f" Training logs:   {csv_path}")
    print("="*80)
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="PPO training (simplified)")
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--minitask-type", type=str, default="development_build_city")
    parser.add_argument("--minitask-level", type=str, default="easy")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/checkpoints_ppo")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--checkpoint-freq", type=int, default=50)
    
    args = parser.parse_args()
    
    train_ppo(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        minitask_type=args.minitask_type,
        minitask_level=args.minitask_level,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs,
        checkpoint_freq=args.checkpoint_freq,
    )


if __name__ == "__main__":
    main()
