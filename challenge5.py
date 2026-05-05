import os
import random
import time
from typing import List, Tuple

import flappy_bird_gymnasium  # noqa: F401  # Needed to register FlappyBird-v0
import gymnasium as gym
import imageio
import matplotlib
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Global configuration (no CLI args, edit values directly)
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV_ID = "FlappyBird-v0"
USE_LIDAR = True

LEARNING_RATE = 5e-4
GAMMA = 0.99

NUM_TRAIN_EPISODES = 400
MAX_STEPS_PER_EPISODE = 250
PRINT_EVERY = 25
NUM_TRAJECTORIES_PER_UPDATE = 8
ENTROPY_COEF = 0.01
LR_STAGE_1_THRESHOLD = 15.0
LR_STAGE_2_THRESHOLD = 25.0
LR_STAGE_1 = 2e-4
LR_STAGE_2 = 1e-4

EVAL_EVERY = 200
NUM_EVAL_EPISODES = 10
SKIP_EVAL = False

UPDATE_PLOT = True
PLOT_PATH = "reinforce_mean_reward2.png"

SAVE_GIF_AFTER_TRAIN = True
GIF_PATH = "flappy_bird2"
GIF_FRAMES = 1000
GIF_FPS = 30

# Reuse one trained policy across different seeds in run()
REUSE_POLICY_ACROSS_SEEDS = True
POLICY_CHECKPOINT_PATH = "challenge5_policy2.pt"

os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.init()


class TensorWrapper(gym.ObservationWrapper):
    """Convert environment observations to float32 tensors."""

    def observation(self, obs):
        return torch.tensor(obs, dtype=torch.float32)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64 ),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear( 64 , action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class REINFORCEAgent(nn.Module):
    """Simple REINFORCE policy-gradient agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = LEARNING_RATE,
        gamma: float = GAMMA,
    ):
        super().__init__()
        self.policy = PolicyNetwork(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.action_dim = action_dim

    def act(self, state, epsilon: float = 0.0, greedy: bool = False):
        """
        Select action from policy.
        - `greedy=True` for deterministic evaluation.        
        """
        state_tensor = state.to(DEVICE, dtype=torch.float32).unsqueeze(0)
        action_probs = self.policy(state_tensor)

        if greedy:
            action = torch.argmax(action_probs, dim=-1).item()
            log_prob = torch.log(action_probs[0, action].clamp(min=1e-8))
            zero_entropy = torch.zeros((), dtype=torch.float32, device=DEVICE)
            return action, log_prob, zero_entropy

        distribution = Categorical(action_probs)
        action_tensor = distribution.sample()
        action = action_tensor.item()
        log_prob = distribution.log_prob(action_tensor)
        entropy = distribution.entropy().squeeze(0)
        return action, log_prob, entropy

    def save_policy(self, checkpoint_path: str = POLICY_CHECKPOINT_PATH) -> None:
        torch.save(self.policy.state_dict(), checkpoint_path)
        print(f"Saved policy checkpoint: {checkpoint_path}")

    def load_policy(self, checkpoint_path: str = POLICY_CHECKPOINT_PATH) -> bool:
        if not os.path.exists(checkpoint_path):
            return False
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        self.policy.load_state_dict(state_dict)
        self.policy.to(DEVICE)
        self.policy.eval()
        print(f"Loaded policy checkpoint: {checkpoint_path}")
        return True


def discount_rewards(rewards: List[float], gamma: float) -> torch.Tensor:
    discounted_returns: List[float] = []
    running_return = 0.0
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        discounted_returns.insert(0, running_return)
    return torch.as_tensor(discounted_returns, dtype=torch.float32, device=DEVICE)


def policy_loss(log_probs : List[torch.Tensor], discounted_rewards: torch.Tensor):
    log_probs = torch.stack(log_probs).reshape(-1)
    return -(log_probs * discounted_rewards.reshape(-1)).sum()


def update_policy(
    agent: REINFORCEAgent,
    log_probs: List[torch.Tensor],
    normalized_returns: torch.Tensor,
    entropies: List[torch.Tensor],
) -> None:
    base_loss = policy_loss(log_probs, normalized_returns)
    entropy_bonus = torch.stack(entropies).mean()
    loss = base_loss - ENTROPY_COEF * entropy_bonus
    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
    agent.optimizer.step()


def run_episode(
    env,
    agent: REINFORCEAgent,
    training: bool = True,
) -> Tuple[float, List[torch.Tensor], List[float], List[torch.Tensor], float]:
    """
    Run one episode.
    Returns:
      total_reward, log_probs, rewards, entropies, score
    where score approximates pipes passed by counting +1 rewards.
    """
    state, _ = env.reset()
    done = False
    rewards: List[float] = []
    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    total_reward = 0.0
    score = 0.0
    steps = 0

    while not done and steps < MAX_STEPS_PER_EPISODE:
        action, log_prob, entropy = agent.act(state, greedy=not training)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        rewards.append(float(reward))
        log_probs.append(log_prob)
        entropies.append(entropy)
        total_reward += float(reward)
        if reward >= 0.99:
            score += 1.0

        state = next_state
        steps += 1

    if training:
        return float(total_reward), log_probs, rewards, entropies, float(score)
    return float(total_reward), [], [], [], float(score)


def update_reinforce_plot(training_rewards: List[float], eval_points: List[Tuple[int, float]]) -> None:
    """Save one plot containing training rewards and eval mean rewards."""
    plt.figure("REINFORCE Mean Reward", figsize=(9, 5))
    plt.clf()

    train_x = np.arange(1, len(training_rewards) + 1)
    plt.plot(
        train_x,
        training_rewards,
        color="tab:blue",
        linewidth=1.0,
        alpha=0.8,
        label="Training reward (per episode)",
    )

    if eval_points:
        eval_x = [episode for episode, _ in eval_points]
        eval_y = [mean_reward for _, mean_reward in eval_points]
        plt.plot(
            eval_x,
            eval_y,
            color="tab:orange",
            marker="o",
            linewidth=1.6,
            label="Evaluation mean reward",
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("FlappyBird REINFORCE: training and evaluation rewards")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH+str(CURRENT_SEED)+".png")
    if matplotlib.get_backend().lower() != "agg":
        plt.pause(0.001)


def generate_policy_gif(agent: REINFORCEAgent, seed: int = 42) -> None:
    """Create a GIF of greedy policy behavior after training."""
    gif_env = gym.make(
        ENV_ID,
        use_lidar=USE_LIDAR,
        render_mode="rgb_array",
        disable_env_checker=True,
        pipe_gap=140,
    )
    gif_env = apply_wrappers(gif_env)
    state, _ = gif_env.reset(seed=seed)

    frames = []
    episode_reward = 0.0
    for _ in range(GIF_FRAMES):
        action, _, _ = agent.act(state, greedy=True)
        next_state, reward, terminated, truncated, _ = gif_env.step(action)
        frame = gif_env.render()
        if frame is not None:
            frames.append(frame)
        episode_reward += float(reward)
        state = next_state

        if terminated or truncated:
            print(f"[gif] episode reward: {episode_reward:.2f}")
            episode_reward = 0.0
            state, _ = gif_env.reset()

    gif_env.close()
    if not frames:
        print("[gif] no frames rendered; skipping gif save")
        return

    imageio.mimsave(GIF_PATH+str(CURRENT_SEED)+".gif", frames, fps=GIF_FPS)
    print(f"[gif] saved: {GIF_PATH+str(CURRENT_SEED)}")


def apply_wrappers(env):
    """
    Challenge-required wrapper hook.
    Keep signature unchanged.
    """
    # env = NoWindowWrapper(env)
    env = TensorWrapper(env)
    return env


def init_model(train_env):
    """
    Challenge-required model initialization hook.
    Keep signature unchanged.
    """
    obs_dim = int(np.prod(train_env.observation_space.shape))
    action_dim = int(train_env.action_space.n)
    return REINFORCEAgent(state_dim=obs_dim, action_dim=action_dim)


def train_model(agent, train_env):
    """
    Challenge-required training hook.
    Keep signature unchanged.
    """
    train_env = apply_wrappers(train_env)
    if REUSE_POLICY_ACROSS_SEEDS:
        # Resume from checkpoint if available; harmless if file doesn't exist.
        agent.load_policy(POLICY_CHECKPOINT_PATH)
    agent.train()

    episode_rewards: List[float] = []
    eval_points: List[Tuple[int, float]] = []
    episode_buffer: List[Tuple[List[torch.Tensor], List[float], List[torch.Tensor]]] = []

    if matplotlib.get_backend().lower() != "agg":
        plt.ion()

    for episode in range(1, NUM_TRAIN_EPISODES + 1):
        total_reward, log_probs, rewards, entropies, _ = run_episode(train_env, agent, training=True)
        episode_rewards.append(total_reward)
        episode_buffer.append((log_probs, rewards, entropies))

        # LR curriculum based on moving average reward.
        window = min(100, len(episode_rewards))
        avg_reward = float(np.mean(episode_rewards[-window:]))
        target_lr = None
        if avg_reward > LR_STAGE_2_THRESHOLD:
            target_lr = LR_STAGE_2
        elif avg_reward > LR_STAGE_1_THRESHOLD:
            target_lr = LR_STAGE_1

        if target_lr is not None:
            current_lr = agent.optimizer.param_groups[0]["lr"]
            if target_lr < current_lr:
                for group in agent.optimizer.param_groups:
                    group["lr"] = target_lr
                print(f"[lr] ep={episode:4d} avg_reward(100)={avg_reward:8.2f} -> lr={target_lr:.1e}")

        if len(episode_buffer) == NUM_TRAJECTORIES_PER_UPDATE or episode == NUM_TRAIN_EPISODES:
            all_log_probs: List[torch.Tensor] = []
            all_returns: List[torch.Tensor] = []
            all_entropies: List[torch.Tensor] = []

            for trajectory_log_probs, trajectory_rewards, trajectory_entropies in episode_buffer:
                returns = discount_rewards(trajectory_rewards, agent.gamma)
                all_log_probs.extend(trajectory_log_probs)
                all_returns.append(returns)
                all_entropies.extend(trajectory_entropies)

            if all_log_probs and all_returns and all_entropies:
                batch_returns = torch.cat(all_returns)
                batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std() + 1e-8)
                update_policy(agent, all_log_probs, batch_returns, all_entropies)

            episode_buffer = []

        if (not SKIP_EVAL) and episode % EVAL_EVERY == 0:
            eval_rewards = []
            for _ in range(NUM_EVAL_EPISODES):
                eval_reward, _, _, _, _ = run_episode(train_env, agent, training=False)
                eval_rewards.append(eval_reward)
            mean_eval_reward = float(np.mean(eval_rewards))
            eval_points.append((episode, mean_eval_reward))

            if UPDATE_PLOT:
                update_reinforce_plot(episode_rewards, eval_points)

            print(f"[eval] ep={episode:4d} mean_eval_reward={mean_eval_reward:8.2f}")

        if episode % PRINT_EVERY == 0:
            print(f"[train] ep={episode:4d} avg_reward(100)={avg_reward:8.2f}")

    agent.eval()

    if SAVE_GIF_AFTER_TRAIN:
        generate_policy_gif(agent)

    if REUSE_POLICY_ACROSS_SEEDS:
        agent.save_policy(POLICY_CHECKPOINT_PATH)

    return agent


# ============================================================
# Online-evaluation-compatible structure
# ============================================================

CURRENT_SEED = 42

def set_seed(seed: int) -> None:
    """
    Local fallback version.
    In online evaluation, the harness may provide its own set_seed.
    """
    global CURRENT_SEED
    CURRENT_SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_envs(seed: int):
    """
    Local fallback version.
    Returns: train_env, valid_envs
    """
    train_env = gym.make(
        ENV_ID, 
        use_lidar=USE_LIDAR, 
        disable_env_checker=True,
        pipe_gap = 140)
    train_env.reset(seed=seed)

    valid_envs = []
    for eval_seed in [seed + 100, seed + 101, seed + 102]:
        env = gym.make(ENV_ID, use_lidar=USE_LIDAR, disable_env_checker=True, pipe_gap=140)
        env.reset(seed=eval_seed)
        valid_envs.append(env)

    return train_env, valid_envs


def evaluate_model(agent, valid_envs):
    """
    Local fallback version.
    Computes mean score (pipes passed proxy) over validation envs.
    """
    scores = []
    for env_idx, env in enumerate(valid_envs):
        wrapped_env = apply_wrappers(env)
        for ep in range(5):
            obs, _ = wrapped_env.reset(seed=1000 + env_idx * 10 + ep)
            done = False
            score = 0.0
            steps = 0

            while not done and steps < MAX_STEPS_PER_EPISODE:
                action, _, _ = agent.act(obs, greedy=True)
                obs, reward, terminated, truncated, _ = wrapped_env.step(action)
                done = terminated or truncated
                if reward >= 0.99:
                    score += 1.0
                steps += 1

            scores.append(score)
        wrapped_env.close()

    return float(np.mean(scores)) if scores else 0.0


def run():
    scores = []
    for i, seed in enumerate([42, 43, 44]):
        print(f"Run #{i}")
        set_seed(seed)

        start = time.time()

        train_env, valid_envs = create_envs(seed)

        # Initialize the model using student's init_model function
        agent = init_model(train_env)

        # Train the model using student's train_model function
        agent = train_model(agent, train_env)

        if SKIP_EVAL:
            score = 0.0
            print("Evaluation skipped (SKIP_EVAL=True).")
        else:
            score = evaluate_model(agent, valid_envs)
        scores.append(score)
        print(f"Score={score:.3f} | elapsed={time.time() - start:.1f}s")

        train_env.close()

    return float(np.mean(scores))


if __name__ == "__main__":
    mean_score = run()
    print(f"Average score across runs: {mean_score:.3f}")