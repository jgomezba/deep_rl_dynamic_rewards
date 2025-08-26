from pathlib import Path
import yaml
import torch
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from networks.q_network import QNet
from envs.environment import GridWorld
from trainers.dqn_trainer import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_eval(episodes=5, n_gems=3, render=True, network=None):
    with open(Path("config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    eval_env = GridWorld(grid_size=cfg["grid_size"],
                         n_gems=n_gems,
                         start=tuple(cfg["start"]),
                         max_steps=cfg["max_steps"],
                         reward_step=cfg["reward_step"],
                         reward_gem=cfg["reward_gem"],
                         reward_goal=cfg["reward_goal"],
                         reward_invalid=cfg["reward_invalid"],
                         actions=[(-1,0),(1,0),(0,-1),(0,1)])
    
    for e in range(episodes):
        s = eval_env.reset()
        done = False
        total_r = 0.0
        steps = 0
        while not done and steps < cfg["max_steps"]:
            if render:
                eval_env.render(title=f"Eval ep {e+1} | Gems={n_gems} | R={total_r:.1f}")
            with torch.no_grad():
                q = network(torch.tensor(s[None,...], dtype=torch.float32, device=device))
                a = int(q.argmax(dim=1).item())
            s, r, done, _ = eval_env.step(a)
            total_r += r
            steps += 1
        print(f"[EVAL] Episodio {e+1}: Recompensa total={total_r:.1f}, pasos={steps}, gemas recogidas={eval_env.collected}/{eval_env.n_gems}")


if __name__ == "__main__":
    model_path = Path(*["saved_models","online.pth"])
    
    with open(Path("config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)

    if model_path.exists():
        online = QNet(cfg["grid_size"], 4).to(device)
        online.load_state_dict(torch.load(model_path, map_location=device))
        online.eval()
        print("Modelo cargado desde saved_models/online.pth")
    else:
        print("No se encontró modelo, entrenando desde cero...")
        online, _ = train(Path("config.yaml"), save_model=True)
    
    run_eval(episodes=1, render=True, network=online)
