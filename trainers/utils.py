import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ACTIONS = 4

def select_action(net, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)
    with torch.no_grad():
        s = torch.tensor(state[None, ...], dtype=torch.float32, device=device)
        q = net(s)
        return int(q.argmax(dim=1).item())

def soft_update(target, online, tau=0.01):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.mul_(1 - tau).add_(p.data * tau)