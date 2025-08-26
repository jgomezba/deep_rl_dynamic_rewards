import os, sys, yaml, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from networks.q_network import QNet
from buffers.buffer import ReplayBuffer
from envs.environment import GridWorld
from utils import select_action, soft_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config_path: str, save_model:bool = False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    env = GridWorld(grid_size=cfg["grid_size"],
                    n_gems=cfg["n_gems"],
                    start=tuple(cfg["start"]),
                    max_steps=cfg["max_steps"],
                    reward_step=cfg["reward_step"],
                    reward_gem=cfg["reward_gem"],
                    reward_goal=cfg["reward_goal"],
                    reward_invalid=cfg["reward_invalid"],
                    actions=[(-1,0),(1,0),(0,-1),(0,1)])

    online = QNet(cfg["grid_size"], 4).to(device)
    target = QNet(cfg["grid_size"], 4).to(device)
    target.load_state_dict(online.state_dict())

    opt = optim.Adam(online.parameters(), lr=cfg["lr"])
    replay = ReplayBuffer(cfg["replay_capacity"])

    epsilon = cfg["eps_start"]
    global_steps = 0
    returns = []

    for ep in range(cfg["episodes"]):
        state = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            a = select_action(online, state, epsilon)
            next_state, r, done, _ = env.step(a)
            replay.push(state, a, r, next_state, float(done))
            ep_return += r
            state = next_state
            global_steps += 1

            if len(replay) >= cfg["batch_size"]:
                s, a_b, r_b, s2, d_b = replay.sample(cfg["batch_size"])

                q = online(s).gather(1, a_b.view(-1,1)).squeeze(1)
                with torch.no_grad():
                    max_next = target(s2).max(dim=1)[0]
                    y = r_b + (1.0 - d_b) * cfg["gamma"] * max_next

                loss = nn.MSELoss()(q, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
                opt.step()

                if global_steps % cfg["target_update_every"] == 0:
                    soft_update(target, online, tau=0.05)

        epsilon = max(cfg["eps_end"], epsilon * cfg["eps_decay"])
        returns.append(ep_return)

        if (ep+1) % 20 == 0:
            print(f"Ep {ep+1}/{cfg['episodes']} | Return: {np.mean(returns[-50:]):.2f} | ε={epsilon:.3f}")

    print("Training completed!")

    if save_model:
        model_path = Path("saved_models")
        model_path.mkdir(exist_ok=True)
        torch.save(online.state_dict(), model_path / "online.pth")

        print("Saved models")

    return online, env


if __name__ == "__main__":
    train(Path("config.yaml"),True)