from pathlib import Path
import yaml
import torch
import sys


# Add parent directory to path for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from networks.q_network import QNet
from envs.environment import GridWorld
from trainers.dqn_trainer import train
from evaluation.utils import save_gif

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_eval(episodes=5, n_gems=3, render=True, network=None):
    """
    Run evaluation episodes on the GridWorld environment using a trained Q-network.

    Args:
        episodes (int): Number of evaluation episodes.
        n_gems (int): Number of gems in the environment.
        render (bool): If True, frames will be rendered and saved as GIFs.
        network (torch.nn.Module): Trained Q-network for action selection.
    """
    with open(Path("config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    eval_env = GridWorld(
        grid_size=cfg["grid_size"],
        n_gems=n_gems,
        start=tuple(cfg["start"]),
        max_steps=cfg["max_steps"],
        reward_step=cfg["reward_step"],
        reward_gem=cfg["reward_gem"],
        reward_goal=cfg["reward_goal"],
        reward_invalid=cfg["reward_invalid"],
        actions=[(-1, 0), (1, 0), (0, -1), (0, 1)],
    )

    for e in range(episodes):
        frames = []
        state = eval_env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < cfg["max_steps"]:
            if render:
                eval_env.render(
                    title=f"Eval ep {e+1} | Gems={n_gems} | R={total_reward:.1f}",
                    save_gif=True,
                    frames_list=frames,
                )

            with torch.no_grad():
                q_values = network(
                    torch.tensor(state[None, ...], dtype=torch.float32, device=device)
                )
                action = int(q_values.argmax(dim=1).item())

            state, reward, done, _ = eval_env.step(action)
            total_reward += reward
            steps += 1

        print(
            f"[EVAL] Episode {e+1}: Total reward={total_reward:.1f}, "
            f"steps={steps}, gems collected={eval_env.collected}/{eval_env.n_gems}"
        )

        if render and frames:
            save_gif(frames, episode_num=e)


if __name__ == "__main__":
    model_path = Path("saved_models") / "online.pth"

    # Load config
    with open(Path("config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    # Load or train model
    if model_path.exists():
        online = QNet(cfg["grid_size"], 4).to(device)
        online.load_state_dict(torch.load(model_path, map_location=device))
        online.eval()
        print("Model loaded from saved_models/online.pth")
    else:
        print("Model not found, training from scratch...")
        online, _ = train(Path("config.yaml"), save_model=True)

    # Run evaluation
    run_eval(episodes=10, render=True, network=online)
