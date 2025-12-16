"""Simplified training script with robust environment handling."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Any
import time
import socket
import warnings

# Disable CivRealm development server checks
os.environ["CIVREALM_DISABLE_DEV_CHECK"] = "1"

# Suppress asyncio event loop warnings (non-critical in daemon threads)
warnings.filterwarnings("ignore", message=".*no current event loop.*")

import civrealm
import gymnasium as gym
from gymnasium import logger as gym_logger
import numpy as np
import torch

from agents import DQNAgent, ReplayBuffer
from networks import CivFeatureExtractor, CivDQNNetwork
from utils.action_utils import get_action_dims, extract_masks, extract_obs_features

# Silence Gym warnings about observation space mismatches
gym_logger.set_level(gym_logger.ERROR)


def _safe_make_env(minitask_type: str, minitask_level: str, attempts: int = 3, sleep_base: float = 1.25):
    """Create CivRealm env, retrying if constructor triggers a reset-timeout inside __init__.

    Some CivRealm envs invoke reset() during construction. If that reset raises
    BeginTurnTimeoutException, gym.make() itself will fail. This helper retries
    a few times with backoff and returns the created env or None if it cannot be created.
    """
    for a in range(attempts):
        try:
            env = gym.make(
                "civrealm/FreecivTensorMinitask-v0",
                minitask_pattern={"type": minitask_type, "level": minitask_level},
            )
            return env
        except Exception as exc:
            name = type(exc).__name__
            print(f"[WARN] Env creation failed (attempt {a+1}/{attempts}): {name}")
            time.sleep(sleep_base * (a + 1))
    return None


def _make_noop(action_dims: Dict[str, int]) -> Dict[str, int]:
    """Return a safe no-op action (actor_type=5 => skip)."""
    action = {"actor_type": 5}
    for head_name in sorted(action_dims.keys()):
        if head_name == "actor_type":
            continue
        action[head_name] = 0
    return action


def _get_valid_heads_for_actor(actor_type: int) -> list:
    """Return list of valid head names for given actor_type index."""
    mapping = {
        0: ['unit_id', 'unit_action_type'],           # unit
        1: ['city_id', 'city_action_type'],           # city
        2: ['dipl_id', 'dipl_action_type'],           # diplomacy
        3: ['gov_action_type'],                        # government
        4: ['tech_action_type'],                       # technology
        5: [],                                         # skip/none
    }
    return mapping.get(actor_type, [])


def _select_conditional_action(obs_features, masks, agent, action_dims):
    """Select action with actor_type consistency: first pick actor_type, then only fill relevant heads."""
    agent.q_network.eval()
    with torch.no_grad():
        obs_tensor = {k: torch.from_numpy(v).to(agent.device) for k, v in obs_features.items()}
        q_values = agent.q_network(obs_tensor)
    
    # Step 1: Select actor_type (apply mask if available)
    actor_q = q_values['actor_type'].squeeze(0).cpu().numpy()
    actor_mask = masks.get('actor_type_mask', np.ones(action_dims['actor_type']))
    # Ensure mask is 1D
    actor_mask = np.asarray(actor_mask).flatten()
    if actor_mask.shape[0] != action_dims['actor_type']:
        actor_mask = DQNAgent._align_mask_numpy(actor_mask, action_dims['actor_type'])

    # If no valid actor is available, return no-op
    valid_actor_idx = np.where(actor_mask > 0)[0]
    if valid_actor_idx.size == 0:
        return _make_noop(action_dims)

    actor_q = actor_q.copy()
    actor_q[actor_mask == 0] = -np.inf
    actor_type = int(np.argmax(actor_q))
    
    # Step 2: Get valid heads for this actor_type
    valid_heads = _get_valid_heads_for_actor(actor_type)
    
    # Step 3: Build action dict
    action = {'actor_type': actor_type}
    for head_name in sorted(action_dims.keys()):
        if head_name == 'actor_type':
            continue
        
        dim = action_dims[head_name]
        
        if head_name in valid_heads:
            # Select this head with its mask
            q = q_values[head_name].squeeze(0).cpu().numpy()
            mask_key = f"{head_name}_mask"
            head_mask = masks.get(mask_key, np.ones(dim))
            # Ensure mask is 1D and matches dim
            head_mask = np.asarray(head_mask).flatten()
            if head_mask.shape[0] != dim:
                head_mask = DQNAgent._align_mask_numpy(head_mask, dim)

            valid_idx = np.where(head_mask > 0)[0]
            if valid_idx.size == 0:
                # No valid option for this head => bail out to no-op to avoid invalid action
                return _make_noop(action_dims)

            q = q.copy()
            q[head_mask == 0] = -np.inf
            action[head_name] = int(np.argmax(q))
        else:
            # Not valid for this actor; default to 0
            action[head_name] = 0
    
    return action


def train_dqn_simple(
    num_episodes: int = 10,
    max_steps_per_episode: int = 100,
    checkpoint_dir: str = "experiments/checkpoints",
    device: str = "cpu",
    batch_size: int = 32,
    train_freq: int = 4,
    warmup_steps: int = 100,
    minitask_type: str = "development_build_city",
    minitask_level: str = "easy",
    buffer_capacity: int = 10000,
    checkpoint_freq: int = 50,
    resume_from: str = None,
    start_episode: int = 0,
) -> Dict[str, Any]:
    """Simplified training with actual DQN learning enabled.
    
    Args:
        num_episodes: Number of episodes to train
        max_steps_per_episode: Max steps per episode
        checkpoint_dir: Directory to save checkpoints
        device: Device (cpu/cuda)
        batch_size: Training batch size
        train_freq: Train every N steps
        warmup_steps: Steps before training starts
        minitask_type: CivRealm minitask type (e.g., 'battle_ancient_era', 'development_build_city')
        minitask_level: Difficulty level ('easy', 'normal', 'hard')
        buffer_capacity: Replay buffer capacity
        checkpoint_freq: Save checkpoint every N episodes
        resume_from: Path to checkpoint to resume from (optional)
        start_episode: Episode number to start from (for resuming)
    
    Returns:
        Dict with episode logs and training metrics
    """
    
    print("=" * 70)
    print("DQN TRAINING - SIMPLIFIED VERSION")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Device: {device}")
    print("=" * 70)
    
    env = None
    try:
        # Create environment once
        print("\n[1] Creating environment...")
        print(f"    Minitask: {minitask_type} ({minitask_level})")
        env = _safe_make_env(minitask_type, minitask_level, attempts=3)
        if env is None:
            raise RuntimeError("Failed to create CivRealm environment after 3 attempts")
        print("    ✓ Environment created")
        # Give CivRealm server a brief time to stabilize before first reset
        time.sleep(1.5)
        
        # Get action dimensions
        action_dims = get_action_dims(env.action_space)
        print(f"\n[2] Action space has {len(action_dims)} heads:")
        for key, dim in sorted(action_dims.items()):
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
            epsilon_end=0.1,
            epsilon_decay_steps=num_episodes * max_steps_per_episode // 2,  # Decay over first half
            target_update_freq=100,
            device=device,
        )
        
        # Load checkpoint if resuming
        if resume_from:
            print(f"    ✓ Loading checkpoint: {resume_from}")
            try:
                agent.load(resume_from, strict=False)
                print(f"    ✓ Resuming from episode {start_episode}")
            except Exception as e:
                print(f"    ⚠ Failed to load checkpoint: {e}")
                print(f"    ⚠ Starting fresh training")
        
        print(f"    ✓ Agent created with {sum(p.numel() for p in q_network.parameters())} parameters")
        print(f"    ✓ Training: batch_size={batch_size}, train_freq={train_freq}, warmup={warmup_steps}")
        
        # Create replay buffer
        replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        print(f"    ✓ Replay buffer created (capacity={buffer_capacity})")
        
        # Create checkpoint directory
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Training loop
        print("\n[4] Starting training...")
        print("=" * 70)
        
        global_step = 0
        episode_rewards = []
        valid_actions_count = 0
        invalid_actions_count = 0
        total_loss = 0.0
        train_steps = 0
        
        episode_logs = []

        for episode in range(start_episode, num_episodes):
            # Robust reset with timeout and up to 7 attempts
            obs = None
            info = None
            for reset_try in range(7):
                result_holder = {}

                def _do_reset():
                    try:
                        # Some CivRealm internals expect an event loop; set one per thread to avoid warnings/errors
                        import asyncio
                        asyncio.set_event_loop(asyncio.new_event_loop())
                        o, i = env.reset()
                        result_holder["obs"] = o
                        result_holder["info"] = i
                    except Exception as exc:
                        result_holder["exc"] = exc

                import threading
                t = threading.Thread(target=_do_reset, daemon=True)
                t.start()
                t.join(timeout=15.0)

                if t.is_alive():
                    # Timeout - first retry without recreate, then recreate
                    if reset_try < 2:
                        time.sleep(1 * (reset_try + 1))
                        continue
                    try:
                        env.close()
                    except Exception:
                        pass
                    time.sleep(1 * (reset_try + 1))
                    env = _safe_make_env(minitask_type, minitask_level, attempts=2)
                    if env is None:
                        # Back off and retry outer loop
                        time.sleep(1.5)
                        continue
                    time.sleep(1)
                    continue

                if "exc" in result_holder:
                    # Specific handling for CivRealm begin_turn timeouts
                    exc = result_holder["exc"]
                    if isinstance(exc, Exception) and "BeginTurnTimeoutException" in type(exc).__name__:
                        print(f"[WARN] Reset begin_turn timeout (attempt {reset_try+1}/7): {type(exc).__name__}")
                    else:
                        print(f"[WARN] Reset failed (attempt {reset_try+1}/7): {type(exc).__name__}")
                    # First retries: same env, just wait
                    if reset_try < 2:
                        time.sleep(1 * (reset_try + 1))
                        continue
                    try:
                        env.close()
                    except Exception:
                        pass
                    # Exponential-ish backoff to allow server to recover
                    time.sleep(1 * (reset_try + 1))
                    env = _safe_make_env(minitask_type, minitask_level, attempts=2)
                    if env is None:
                        time.sleep(1.5)
                        continue
                    # Small delay to let server initialize
                    time.sleep(1.5)
                    continue

                # success
                obs = result_holder.get("obs")
                info = result_holder.get("info", {})
                break

            if obs is None:
                print(f"[STOP] Failed to reset at episode {episode}: giving up after 7 attempts")
                break
            
            obs_features = extract_obs_features(obs)
            episode_reward = 0.0
            episode_steps = 0
            
            for step in range(max_steps_per_episode):
                # Extract masks
                masks = extract_masks(obs)

                # Keep agent step counter in sync even if we bypass agent.select_action()
                agent.steps = global_step

                # Epsilon schedule (based on global_step)
                progress = min(global_step / max(1, agent.epsilon_decay_steps), 1.0)
                eps_val = agent.epsilon_start + (agent.epsilon_end - agent.epsilon_start) * progress

                # Prefer environment-provided valid actions
                # If missing, use conditional per-head selection by actor_type
                avail = obs.get("available_actions")
                if isinstance(avail, list) and len(avail) > 0:
                    # Epsilon-greedy over environment's valid actions
                    if np.random.rand() < eps_val:
                        action = avail[np.random.randint(0, len(avail))]
                    else:
                        # Greedy: pick first valid action as heuristic
                        action = avail[0]
                else:
                    # Fallback: conditional per-head selection by actor_type (prevents invalids)
                    action = _select_conditional_action(obs_features, masks, agent, action_dims)
                
                # Environment step
                try:
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    valid_actions_count += 1
                except Exception as e:
                    # Invalid action - apply penalty and terminate episode
                    invalid_actions_count += 1
                    next_obs = obs
                    reward = -1.0  # Stronger penalty for invalid actions
                    done = True
                
                next_obs_features = extract_obs_features(next_obs)
                next_masks = extract_masks(next_obs)
                
                # Store transition
                replay_buffer.push(obs_features, action, reward, next_obs_features, done, next_masks)
                
                # Train agent (after warmup and at train_freq intervals)
                if global_step >= warmup_steps and global_step % train_freq == 0:
                    if len(replay_buffer) >= batch_size:
                        obs_b, act_b, rew_b, next_obs_b, done_b, masks_b = replay_buffer.sample(batch_size)
                        metrics = agent.train_step(obs_b, act_b, rew_b, next_obs_b, done_b, masks_b)
                        total_loss += float(metrics["loss"])
                        train_steps += 1
                
                # Update state
                obs = next_obs
                obs_features = next_obs_features
                episode_reward += reward
                episode_steps += 1
                global_step += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)

            avg_loss = (total_loss / train_steps) if train_steps > 0 else 0.0
            # record episode log for notebook use
            episode_logs.append({
                "episode": episode + 1,
                "steps": episode_steps,
                "reward": float(episode_reward),
                "epsilon": float(eps_val),
                "avg_loss": float(avg_loss),
                "buffer": int(len(replay_buffer)),
            })
            print(f"Ep {episode + 1:3d}/{num_episodes} | "
                f"Steps: {episode_steps:3d} | "
                f"Reward: {episode_reward:6.2f} | "
                f"ε: {eps_val:.3f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Buf: {len(replay_buffer):4d}")
            
            # Save intermediate checkpoint
            if checkpoint_freq > 0 and (episode + 1) % checkpoint_freq == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"dqn_ep{episode+1}.pt")
                agent.save(ckpt_path)
                print(f"    ✓ Checkpoint saved: {ckpt_path}")
        
        # Final stats
        print("\n" + "=" * 70)
        print("[5] Training complete!")
        print("=" * 70)
        print(f"    Total steps: {global_step}")
        print(f"    Training steps: {train_steps}")
        print(f"    Valid actions: {valid_actions_count} ({100 * valid_actions_count / max(1, valid_actions_count + invalid_actions_count):.1f}%)")
        print(f"    Invalid actions: {invalid_actions_count}")
        print(f"    Avg reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"    Best reward: {np.max(episode_rewards):.2f}")
        # Final epsilon from schedule
        final_progress = min(global_step / max(1, agent.epsilon_decay_steps), 1.0)
        final_eps = agent.epsilon_start + (agent.epsilon_end - agent.epsilon_start) * final_progress
        print(f"    Final epsilon: {final_eps:.3f}")
        
        # Save model
        save_path = os.path.join(checkpoint_dir, "dqn_model.pt")
        agent.save(save_path)
        print(f"    ✓ Model saved to {save_path}")
        
    finally:
        # Always close environment (with timeout to avoid hanging)
        if env is not None:
            print("\n[6] Closing environment...")
            try:
                # Create a simple timeout mechanism using threading
                import threading
                
                def close_with_timeout():
                    try:
                        env.close()
                    except:
                        pass
                
                close_thread = threading.Thread(target=close_with_timeout, daemon=True)
                close_thread.start()
                close_thread.join(timeout=4.0)  # Wait max 4 seconds
                
                if close_thread.is_alive():
                    print("    ⚠ Environment close timed out (4s) - forcing exit")
                else:
                    print("    ✓ Environment closed")
            except Exception as e:
                print(f"    ⚠ Close failed: {type(e).__name__}")
            
            # Brief pause before exit
            time.sleep(0.5)

    # Prepare return payload for programmatic use (e.g., notebooks)
    try:
        import pandas as pd  # optional
        logs_df = pd.DataFrame(episode_logs)
    except Exception:
        logs_df = episode_logs

    result = {
        # Provide both keys for compatibility with callers/notebooks
        "episode_logs": logs_df,
        "logs_df": logs_df,
        "total_steps": global_step,
        "train_steps": train_steps,
        "valid_actions": valid_actions_count,
        "invalid_actions": invalid_actions_count,
        "avg_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "best_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "final_epsilon": float(final_eps),
        "checkpoint_path": save_path,
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified DQN training on CivRealm")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--train-freq", type=int, default=4, help="Train every N steps")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps before training")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--minitask-type", type=str, default="development_build_city", help="Minitask type")
    parser.add_argument("--minitask-level", type=str, default="easy", help="Minitask level")
    parser.add_argument("--buffer-capacity", type=int, default=10000, help="Replay buffer capacity")
    parser.add_argument("--checkpoint-freq", type=int, default=50, help="Save checkpoint every N episodes")
    
    args = parser.parse_args()
    
    train_dqn_simple(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        warmup_steps=args.warmup_steps,
        minitask_type=args.minitask_type,
        minitask_level=args.minitask_level,
        buffer_capacity=args.buffer_capacity,
        checkpoint_freq=args.checkpoint_freq,
    )
