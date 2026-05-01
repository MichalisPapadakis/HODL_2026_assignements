from __future__ import annotations
import argparse
import os

import gymnasium as gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Plotting related
UPDATE_PLOT = True

import imageio
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

def update_mean_reward_plot(
    training_rewards,
    test_rewards,
    plot_path: str = "reinforce_mean_reward.png",
) -> None:
    """Refresh plot with per-episode train rewards and eval mean rewards."""
    plt.figure("REINFORCE Mean Reward", figsize=(8, 5))
    plt.clf()
    train_episodes = np.arange(1, len(training_rewards) + 1)
    plt.plot(
        train_episodes,
        training_rewards,
        color="tab:blue",
        linewidth=1.0,
        alpha=0.8,
        label="Training reward (per episode)",
    )
    if test_rewards:
        test_episodes = [episode for episode, _ in test_rewards]
        test_means = [mean_reward for _, mean_reward in test_rewards]
        plt.plot(
            test_episodes,
            test_means,
            color="tab:orange",
            marker="o",
            linewidth=1.5,
            label="Evaluation mean reward",
        )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("REINFORCE training and evaluation rewards")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    if matplotlib.get_backend().lower() != "agg":
        plt.pause(0.001)

def generate_gif(
    env_id: str,
    agent: REINFORCE,
    gif_path: str = "reinforce_policy.gif",
    n_frames: int = 500,
    fps: int = 30,
    seed: int = 42,
) -> None:
    """Run greedy policy and save an animation GIF."""
    gif_env = gym.make(env_id, render_mode="rgb_array")
    gif_env = NoWindowWrapper(gif_env)
    state, _ = gif_env.reset(seed=seed)

    frames = []
    episode_reward = 0.0

    for _ in range(n_frames):
        action, _ = agent.act(state, greedy=True)
        next_state, reward, terminated, truncated, _ = gif_env.step(action)
        frame = gif_env.render()
        frames.append(frame)
        episode_reward += reward
        state = next_state

        if terminated or truncated:
            print(f"[GIF] Episode reward: {episode_reward:.2f}")
            episode_reward = 0.0
            state, _ = gif_env.reset()

    if not frames:
        print("[GIF] No frames generated, skipping save.")
        gif_env.close()
        return

    imageio.mimsave(gif_path, frames, fps=fps)
    gif_env.close()
    print(f"[GIF] Saved policy animation to: {gif_path}")


## REINFORCE implementation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoWindowWrapper(gym.Wrapper):
    """Prevents pygame windows from opening in headless runs."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class REINFORCE:
    """REINFORCE (Monte Carlo Policy Gradient) algorithm implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def act(self, state: np.ndarray, greedy: bool = False):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action_probs = self.policy(state_tensor)

        if greedy:
            action = torch.argmax(action_probs, dim=-1).item()
            log_prob = torch.log(action_probs[0, action].clamp(min=1e-8))
            return action, log_prob

        dist = Categorical(action_probs)
        action_tensor = dist.sample()
        action = action_tensor.item()
        log_prob = dist.log_prob(action_tensor)
        return action, log_prob


def run_episode(env: gym.Env, agent: REINFORCE, training: bool = True):
    state, _ = env.reset()
    rewards = []
    log_probs = []
    done = False

    while not done:
        action, log_prob = agent.act(state, greedy=not training)
        next_state, reward, terminated, truncated, _ = env.step(action)

        rewards.append(reward)
        log_probs.append(log_prob)

        state = next_state
        done = terminated or truncated

    if training:
        return float(np.sum(rewards)), log_probs, rewards
    return float(np.sum(rewards))


def discount_rewards(rewards, gamma: float) -> torch.Tensor:
    discounted = []
    r_acc = 0.0
    for reward in reversed(rewards):
        r_acc = reward + gamma * r_acc
        discounted.insert(0, r_acc)
    return torch.as_tensor(discounted, dtype=torch.float32, device=device)


def policy_loss(log_probs, discounted_rewards: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for log_prob, ret in zip(log_probs, discounted_rewards):
        loss = loss + (-log_prob * ret)
    return loss


def update_policy(agent: REINFORCE, log_probs, rewards) -> None:
    returns = discount_rewards(rewards, agent.gamma)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = policy_loss(log_probs, returns)

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()


def train_agent(
    env: gym.Env,
    agent: REINFORCE,
    num_episodes: int = 1500,
    print_every: int = 50,
    test_every: int = 200,
    num_tests: int = 10,
    plot_path: str = "reinforce_mean_reward.png",
):
    episode_rewards = []
    test_rewards = []

    if matplotlib.get_backend().lower() != "agg":
        plt.ion()

    for episode in range(1, num_episodes + 1):
        total_reward, log_probs, rewards = run_episode(env, agent, training=True)
        update_policy(agent, log_probs, rewards)
        episode_rewards.append(total_reward)

        if episode % test_every == 0:
            eval_rewards = [run_episode(env, agent, training=False) for _ in range(num_tests)]
            mean_eval_reward = float(np.mean(eval_rewards))
            test_rewards.append((episode, mean_eval_reward))
            if UPDATE_PLOT:
                update_mean_reward_plot(episode_rewards, test_rewards, plot_path=plot_path)
            print(f"[Eval] Episode {episode}: eval mean {mean_eval_reward:.2f}")

        if episode % print_every == 0:
            window = min(100, len(episode_rewards))
            avg_reward = float(np.mean(episode_rewards[-window:]))
            print(f"Episode {episode}/{num_episodes} | Moving avg reward: {avg_reward:.2f}")

    final_eval = float(np.mean([run_episode(env, agent, training=False) for _ in range(num_tests)]))
    print(f"\nTraining completed. Final mean reward over {num_tests} eval episodes: {final_eval:.2f}")
    return episode_rewards, test_rewards


def main():
    parser = argparse.ArgumentParser(description="Train a REINFORCE agent on LunarLander.")
    parser.add_argument("--env-id", default="LunarLander-v3", help="Gymnasium environment id.")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of training episodes.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--print-every", type=int, default=50, help="Progress print interval.")
    parser.add_argument("--test-every", type=int, default=200, help="Evaluation interval.")
    parser.add_argument("--num-tests", type=int, default=10, help="Episodes per evaluation.")
    parser.add_argument(
        "--plot-path",
        default="reinforce_mean_reward.png",
        help="Output path for the mean-reward plot.",
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Generate a GIF of the trained greedy policy after training.",
    )
    parser.add_argument(
        "--gif-path",
        default="reinforce_policy.gif",
        help="Output path for policy GIF.",
    )
    parser.add_argument(
        "--gif-frames",
        type=int,
        default=500,
        help="Number of frames to render for GIF.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=30,
        help="Frames per second for GIF.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env_id)
    env = NoWindowWrapper(env)
    env.reset(seed=args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Using device: {device}")
    print(f"Environment: {args.env_id} | state_dim={state_dim}, action_dim={action_dim}")

    agent = REINFORCE(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
    )

    train_agent(
        env=env,
        agent=agent,
        num_episodes=args.episodes,
        print_every=args.print_every,
        test_every=args.test_every,
        num_tests=args.num_tests,
        plot_path=args.plot_path,
    )

    env.close()

    if args.save_gif:
        generate_gif(
            env_id=args.env_id,
            agent=agent,
            gif_path=args.gif_path,
            n_frames=args.gif_frames,
            fps=args.gif_fps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
