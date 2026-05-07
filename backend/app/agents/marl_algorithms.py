"""
PropOS MARL Algorithm Suite — QMIX, IQL, SAC
==============================================
Completes Table 2: all five negotiation algorithms.
  - QMIX: Value-based mixing for joint value estimation
  - IQL: Independent Q-Learning for routine deals
  - SAC: Entropy-regularized off-policy for complex multi-deal scenarios
  (MAPPO and MARLIN are in negotiation_engine.py)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("propos.marl")
OBS_DIM = 18
NUM_ACTIONS = 7


class QNetwork(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=NUM_ACTIONS, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, obs): return self.net(obs)


# ── QMIX ──────────────────────────────────────────────────────────────

class QMIXMixer(nn.Module):
    """Monotonic mixing network: agent Q-values → Q_tot via hypernetworks."""
    def __init__(self, n_agents=2, state_dim=OBS_DIM*2, mix_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.mix_dim = mix_dim
        self.hyper_w1 = nn.Sequential(nn.Linear(state_dim, mix_dim), nn.ReLU(), nn.Linear(mix_dim, n_agents * mix_dim))
        self.hyper_b1 = nn.Linear(state_dim, mix_dim)
        self.hyper_w2 = nn.Sequential(nn.Linear(state_dim, mix_dim), nn.ReLU(), nn.Linear(mix_dim, mix_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, mix_dim), nn.ReLU(), nn.Linear(mix_dim, 1))

    def forward(self, agent_qs, state):
        B = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.mix_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.mix_dim)
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.mix_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        h = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        return (torch.bmm(h, w2) + b2).squeeze(-1).squeeze(-1)


class QMIXAgent:
    def __init__(self, lr=5e-4, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=5000, device="cpu"):
        self.device = torch.device(device)
        self.gamma, self.eps, self.eps_end, self.eps_decay = gamma, eps_start, eps_end, eps_decay
        self.step_count = 0
        self.q_nets = [QNetwork().to(self.device) for _ in range(2)]
        self.target_q_nets = [QNetwork().to(self.device) for _ in range(2)]
        self.mixer = QMIXMixer().to(self.device)
        self.target_mixer = QMIXMixer().to(self.device)
        for i in range(2): self.target_q_nets[i].load_state_dict(self.q_nets[i].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        params = list(self.mixer.parameters()) + [p for q in self.q_nets for p in q.parameters()]
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.replay, self.max_buf = [], 10000

    def select_action(self, obs, agent_idx):
        self.step_count += 1
        self.eps = max(self.eps_end, self.eps - (1.0 - self.eps_end) / self.eps_decay)
        if np.random.random() < self.eps: return np.random.randint(NUM_ACTIONS)
        with torch.no_grad(): return self.q_nets[agent_idx](torch.FloatTensor(obs).unsqueeze(0).to(self.device)).argmax(1).item()

    def store(self, t):
        if len(self.replay) >= self.max_buf: self.replay.pop(0)
        self.replay.append(t)

    def sync_targets(self, tau=0.005):
        for i in range(2):
            for p, tp in zip(self.q_nets[i].parameters(), self.target_q_nets[i].parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for p, tp in zip(self.mixer.parameters(), self.target_mixer.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


# ── IQL ───────────────────────────────────────────────────────────────

class IQLAgent:
    """Independent Q-Learning — low overhead for routine bilateral deals."""
    def __init__(self, role, lr=1e-3, gamma=0.99, epsilon=0.1, device="cpu"):
        self.role, self.device, self.gamma, self.epsilon = role, torch.device(device), gamma, epsilon
        self.q_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay, self.max_buf = [], 5000

    def select_action(self, obs):
        if np.random.random() < self.epsilon: return np.random.randint(NUM_ACTIONS)
        with torch.no_grad(): return self.q_net(torch.FloatTensor(obs).unsqueeze(0).to(self.device)).argmax(1).item()

    def store(self, obs, action, reward, next_obs, done):
        if len(self.replay) >= self.max_buf: self.replay.pop(0)
        self.replay.append((obs, action, reward, next_obs, done))

    def update(self, batch_size=32):
        if len(self.replay) < batch_size: return None
        idx = np.random.choice(len(self.replay), batch_size, replace=False)
        b = [self.replay[i] for i in idx]
        obs = torch.FloatTensor(np.array([x[0] for x in b])).to(self.device)
        act = torch.LongTensor([x[1] for x in b]).to(self.device)
        rew = torch.FloatTensor([x[2] for x in b]).to(self.device)
        nobs = torch.FloatTensor(np.array([x[3] for x in b])).to(self.device)
        done = torch.FloatTensor([x[4] for x in b]).to(self.device)
        q = self.q_net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad(): tgt = rew + self.gamma * (1-done) * self.target_net(nobs).max(1).values
        loss = F.mse_loss(q, tgt)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        return loss.item()

    def sync_target(self): self.target_net.load_state_dict(self.q_net.state_dict())


# ── SAC ───────────────────────────────────────────────────────────────

class SACActorContinuous(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=1, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
    def forward(self, obs):
        h = self.shared(obs); return self.mean(h), self.log_std(h).clamp(-20, 2)
    def sample(self, obs):
        mu, ls = self.forward(obs); std = ls.exp()
        d = Normal(mu, std); x = d.rsample(); a = torch.tanh(x)
        lp = d.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)
        return a, lp.sum(-1)

class SACCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=1, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(obs_dim+act_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(nn.Linear(obs_dim+act_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, obs, act):
        x = torch.cat([obs, act], -1); return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

class SACAgent:
    """Entropy-regularized off-policy — handles memory explosion in multi-deal scenarios."""
    def __init__(self, role, lr=3e-4, gamma=0.99, alpha=0.2, tau=0.005, device="cpu"):
        self.role, self.device, self.gamma, self.tau = role, torch.device(device), gamma, tau
        self.actor = SACActorContinuous().to(self.device)
        self.critic = SACCritic().to(self.device)
        self.target_critic = SACCritic().to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -1.0
        self.replay, self.max_buf = [], 50000

    @property
    def alpha(self): return self.log_alpha.exp()

    def select_action(self, obs, deterministic=False):
        o = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic: m, _ = self.actor(o); return torch.tanh(m).cpu().numpy().flatten()[0]
            a, _ = self.actor.sample(o); return a.cpu().numpy().flatten()[0]

    def action_to_discrete(self, ca): return max(0, min(NUM_ACTIONS-1, int((ca+1)/2*(NUM_ACTIONS-1))))

    def store(self, o, a, r, no, d):
        if len(self.replay) >= self.max_buf: self.replay.pop(0)
        self.replay.append((o, a, r, no, d))

    def update(self, bs=256):
        if len(self.replay) < bs: return None
        idx = np.random.choice(len(self.replay), bs, replace=False)
        b = [self.replay[i] for i in idx]
        obs = torch.FloatTensor(np.array([x[0] for x in b])).to(self.device)
        act = torch.FloatTensor(np.array([x[1] for x in b])).unsqueeze(-1).to(self.device)
        rew = torch.FloatTensor([x[2] for x in b]).to(self.device)
        nobs = torch.FloatTensor(np.array([x[3] for x in b])).to(self.device)
        done = torch.FloatTensor([x[4] for x in b]).to(self.device)
        with torch.no_grad():
            na, nlp = self.actor.sample(nobs); tq1, tq2 = self.target_critic(nobs, na)
            tgt = rew + self.gamma * (1-done) * (torch.min(tq1, tq2) - self.alpha * nlp)
        q1, q2 = self.critic(obs, act)
        cl = F.mse_loss(q1, tgt) + F.mse_loss(q2, tgt)
        self.critic_opt.zero_grad(); cl.backward(); self.critic_opt.step()
        na2, lp2 = self.actor.sample(obs); q1n, q2n = self.critic(obs, na2)
        al = (self.alpha.detach() * lp2 - torch.min(q1n, q2n)).mean()
        self.actor_opt.zero_grad(); al.backward(); self.actor_opt.step()
        alpha_loss = -(self.log_alpha * (lp2 + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1-self.tau) * tp.data)
        return {"critic_loss": cl.item(), "actor_loss": al.item(), "alpha": self.alpha.item()}
