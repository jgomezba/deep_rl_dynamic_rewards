import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.q_network import QNet
from buffers.buffer import ReplayBuffer
from envs.environment import GridWorld



import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from utils import select_action, soft_update

# ---------- Configuración ----------
GRID_SIZE = 5
N_GEMS = 3
MAX_STEPS = GRID_SIZE**2 * 4

REWARD_GOAL = 20
REWARD_STEP = -1
REWARD_GEM = 5
REWARD_INVALID = -1 

START = (0,0)
GOAL = (GRID_SIZE-1, GRID_SIZE-1)
ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]  
N_ACTIONS = 4

# Entrenamiento
EPISODES = 1500
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
REPLAY_CAPACITY = 50_000
TARGET_UPDATE_EVERY = 500     
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.997

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

env = GridWorld(GRID_SIZE, N_GEMS)
online = QNet(GRID_SIZE, N_ACTIONS).to(device)
target = QNet(GRID_SIZE, N_ACTIONS).to(device)
target.load_state_dict(online.state_dict())

opt = optim.Adam(online.parameters(), lr=LR)
replay = ReplayBuffer(REPLAY_CAPACITY)

epsilon = EPS_START
global_steps = 0

returns = []

for ep in range(EPISODES):
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


        if len(replay) >= BATCH_SIZE:
            s, a_b, r_b, s2, d_b = replay.sample(BATCH_SIZE)

            q = online(s).gather(1, a_b.view(-1,1)).squeeze(1)
            with torch.no_grad():

                max_next = target(s2).max(dim=1)[0]
                y = r_b + (1.0 - d_b) * GAMMA * max_next

            loss = nn.MSELoss()(q, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            opt.step()

            if global_steps % TARGET_UPDATE_EVERY == 0:
                soft_update(target, online, tau=0.05)

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    returns.append(ep_return)

    if (ep+1) % 20 == 0:
        print(f"Ep {ep+1}/{EPISODES} | Return: {np.mean(returns[-50:]):.2f} | ε={epsilon:.3f}")