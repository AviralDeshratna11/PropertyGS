"""
PropOS MARL Negotiation Engine
===============================
Multi-Agent Reinforcement Learning framework using MAPPO
(Multi-Agent Proximal Policy Optimization) for fiduciary
bilateral real estate bargaining.

Architecture:
  - Stochastic Markov Game formulation
  - Centralized Training, Decentralized Execution (CTDE)
  - Fiduciary reward reshaping with cooperative resilience
  - Nash equilibrium convergence tracking

Reference: PropOS Core Logic Layer specification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import IntEnum
import logging
import uuid

logger = logging.getLogger("propos.marl")


# ══════════════════════════════════════════════════════════════════════
# §1  ACTION & OBSERVATION SPACES
# ══════════════════════════════════════════════════════════════════════

class NegotiationAction(IntEnum):
    """Discrete action space for each agent."""
    HOLD = 0           # Maintain current position
    CONCEDE_SMALL = 1  # 1-2% concession
    CONCEDE_MEDIUM = 2 # 3-5% concession
    CONCEDE_LARGE = 3  # 6-10% concession
    COUNTER = 4        # Counter-offer at current position + strategic adjustment
    ACCEPT = 5         # Accept opponent's latest offer
    WALK_AWAY = 6      # Terminate negotiation


# Concession percentages mapped to each action
CONCESSION_MAP = {
    NegotiationAction.HOLD: 0.0,
    NegotiationAction.CONCEDE_SMALL: 0.015,
    NegotiationAction.CONCEDE_MEDIUM: 0.04,
    NegotiationAction.CONCEDE_LARGE: 0.08,
    NegotiationAction.COUNTER: 0.0,
    NegotiationAction.ACCEPT: 0.0,
    NegotiationAction.WALK_AWAY: 0.0,
}

# Observation vector dimensions
OBS_DIM = 18
NUM_ACTIONS = len(NegotiationAction)


@dataclass
class MarketContext:
    """External market signals fed into the state space."""
    property_fair_value_usd: int        # Estimated FMV from comps
    days_on_market: int
    market_temperature: float           # 0=cold, 1=hot
    comparable_sold_prices: List[int] = field(default_factory=list)
    interest_rate_pct: float = 5.5
    inventory_months: float = 4.0       # Months of supply


@dataclass
class AgentConfig:
    """Per-agent configuration (fiduciary boundaries)."""
    agent_id: str
    role: str                           # "buyer" | "seller"
    reserve_price_usd: int              # Walk-away price
    target_price_usd: int               # Ideal outcome
    urgency: float = 0.5                # 0=patient, 1=desperate
    risk_tolerance: float = 0.5


# ══════════════════════════════════════════════════════════════════════
# §2  NEGOTIATION ENVIRONMENT (Stochastic Markov Game)
# ══════════════════════════════════════════════════════════════════════

class NegotiationEnvironment:
    """
    Stochastic Markov Game: (N, S, A, P, R, γ)

    N = {buyer_agent, seller_agent}
    S = property attributes ⊕ market context ⊕ bid history
    A = A_buyer × A_seller (joint action space)
    P = deterministic transitions (sequential bargaining)
    R = fiduciary reward with cooperative resilience shaping
    γ = discount factor
    """

    def __init__(
        self,
        buyer_config: AgentConfig,
        seller_config: AgentConfig,
        market: MarketContext,
        max_rounds: int = 20,
        gamma: float = 0.99,
    ):
        self.buyer = buyer_config
        self.seller = seller_config
        self.market = market
        self.max_rounds = max_rounds
        self.gamma = gamma

        # State tracking
        self.round = 0
        self.current_bid = 0
        self.current_ask = market.property_fair_value_usd
        self.bid_history: List[Tuple[int, int]] = []
        self.done = False
        self.agreed_price: Optional[int] = None

        # Initialize positions
        self._initial_bid = int(buyer_config.target_price_usd * 0.92)
        self._initial_ask = int(seller_config.target_price_usd * 1.05)
        self.current_bid = self._initial_bid
        self.current_ask = self._initial_ask

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset and return initial observations for [buyer, seller]."""
        self.round = 0
        self.current_bid = self._initial_bid
        self.current_ask = self._initial_ask
        self.bid_history = []
        self.done = False
        self.agreed_price = None
        return self._get_obs("buyer"), self._get_obs("seller")

    def _get_obs(self, role: str) -> np.ndarray:
        """
        Build observation vector (dim=18) for an agent.

        Components:
          [0]  normalized_own_price       — current bid or ask / FMV
          [1]  normalized_opponent_price   — opponent's last offer / FMV
          [2]  spread_ratio               — (ask - bid) / FMV
          [3]  round_progress             — round / max_rounds
          [4]  own_urgency
          [5]  own_risk_tolerance
          [6]  market_temperature
          [7]  days_on_market_norm        — days / 365
          [8]  interest_rate_norm         — rate / 10
          [9]  inventory_months_norm      — months / 12
          [10] distance_to_reserve_norm   — |current - reserve| / FMV
          [11] distance_to_target_norm    — |current - target| / FMV
          [12] bid_momentum               — change in last 3 bids
          [13] ask_momentum               — change in last 3 asks
          [14] concession_rate_self       — avg concession %
          [15] concession_rate_opponent
          [16] rounds_remaining_norm
          [17] fmv_ratio                  — own price / FMV
        """
        fmv = max(self.market.property_fair_value_usd, 1)
        cfg = self.buyer if role == "buyer" else self.seller

        own_price = self.current_bid if role == "buyer" else self.current_ask
        opp_price = self.current_ask if role == "buyer" else self.current_bid

        spread = self.current_ask - self.current_bid

        # Momentum: average price change over last 3 rounds
        bid_mom, ask_mom = 0.0, 0.0
        if len(self.bid_history) >= 2:
            recent = self.bid_history[-3:]
            bid_changes = [recent[i+1][0] - recent[i][0] for i in range(len(recent)-1)]
            ask_changes = [recent[i][1] - recent[i+1][1] for i in range(len(recent)-1)]
            bid_mom = np.mean(bid_changes) / fmv if bid_changes else 0.0
            ask_mom = np.mean(ask_changes) / fmv if ask_changes else 0.0

        # Concession rates
        conc_self, conc_opp = 0.0, 0.0
        if len(self.bid_history) >= 2:
            idx = 0 if role == "buyer" else 1
            opp_idx = 1 - idx
            prices_self = [h[idx] for h in self.bid_history]
            prices_opp = [h[opp_idx] for h in self.bid_history]
            conc_self = abs(prices_self[-1] - prices_self[0]) / max(fmv, 1)
            conc_opp = abs(prices_opp[-1] - prices_opp[0]) / max(fmv, 1)

        obs = np.array([
            own_price / fmv,
            opp_price / fmv,
            spread / fmv,
            self.round / self.max_rounds,
            cfg.urgency,
            cfg.risk_tolerance,
            self.market.market_temperature,
            min(self.market.days_on_market / 365.0, 1.0),
            self.market.interest_rate_pct / 10.0,
            self.market.inventory_months / 12.0,
            abs(own_price - cfg.reserve_price_usd) / fmv,
            abs(own_price - cfg.target_price_usd) / fmv,
            bid_mom,
            ask_mom,
            conc_self,
            conc_opp,
            (self.max_rounds - self.round) / self.max_rounds,
            own_price / fmv,
        ], dtype=np.float32)

        return obs

    def step(
        self, buyer_action: int, seller_action: int
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool, dict]:
        """
        Execute one round of bilateral bargaining.

        Returns:
            (buyer_obs, seller_obs), (buyer_reward, seller_reward), done, info
        """
        self.round += 1
        info = {"round": self.round, "buyer_action": int(buyer_action),
                "seller_action": int(seller_action)}

        ba = NegotiationAction(buyer_action)
        sa = NegotiationAction(seller_action)

        fmv = self.market.property_fair_value_usd

        # ── Apply buyer action ────────────────────────────────────────
        if ba == NegotiationAction.ACCEPT:
            self.agreed_price = self.current_ask
            self.done = True
        elif ba == NegotiationAction.WALK_AWAY:
            self.done = True
        elif ba in (NegotiationAction.CONCEDE_SMALL, NegotiationAction.CONCEDE_MEDIUM,
                    NegotiationAction.CONCEDE_LARGE):
            delta = int(fmv * CONCESSION_MAP[ba])
            self.current_bid = min(self.current_bid + delta, self.buyer.reserve_price_usd)
        elif ba == NegotiationAction.COUNTER:
            midpoint = (self.current_bid + self.current_ask) // 2
            self.current_bid = min(
                int(self.current_bid + (midpoint - self.current_bid) * 0.3),
                self.buyer.reserve_price_usd
            )

        # ── Apply seller action ───────────────────────────────────────
        if not self.done:
            if sa == NegotiationAction.ACCEPT:
                self.agreed_price = self.current_bid
                self.done = True
            elif sa == NegotiationAction.WALK_AWAY:
                self.done = True
            elif sa in (NegotiationAction.CONCEDE_SMALL, NegotiationAction.CONCEDE_MEDIUM,
                        NegotiationAction.CONCEDE_LARGE):
                delta = int(fmv * CONCESSION_MAP[sa])
                self.current_ask = max(self.current_ask - delta, self.seller.reserve_price_usd)
            elif sa == NegotiationAction.COUNTER:
                midpoint = (self.current_bid + self.current_ask) // 2
                self.current_ask = max(
                    int(self.current_ask - (self.current_ask - midpoint) * 0.3),
                    self.seller.reserve_price_usd
                )

        # ── Check convergence ─────────────────────────────────────────
        if not self.done and self.current_bid >= self.current_ask:
            self.agreed_price = (self.current_bid + self.current_ask) // 2
            self.done = True

        # ── Timeout ───────────────────────────────────────────────────
        if not self.done and self.round >= self.max_rounds:
            self.done = True

        # Record history
        self.bid_history.append((self.current_bid, self.current_ask))

        # ── Compute rewards ───────────────────────────────────────────
        buyer_reward = self._compute_reward("buyer", ba, sa)
        seller_reward = self._compute_reward("seller", sa, ba)

        info.update({
            "current_bid": self.current_bid,
            "current_ask": self.current_ask,
            "spread": self.current_ask - self.current_bid,
            "agreed_price": self.agreed_price,
            "buyer_reward": float(buyer_reward),
            "seller_reward": float(seller_reward),
            "cooperative_resilience": self._cooperative_resilience(),
            "nash_distance": self._nash_distance(),
        })

        return (
            (self._get_obs("buyer"), self._get_obs("seller")),
            (buyer_reward, seller_reward),
            self.done,
            info,
        )

    # ══════════════════════════════════════════════════════════════════
    # §3  FIDUCIARY REWARD SHAPING
    # ══════════════════════════════════════════════════════════════════

    def _compute_reward(
        self, role: str, own_action: NegotiationAction, opp_action: NegotiationAction
    ) -> float:
        """
        Hybrid fiduciary reward = α·surplus + β·progress + δ·cooperative_resilience + ε·info_gain

        This is "Approval Reward" design — not pure behaviorist — to prevent
        scheming and treacherous turns per the Fiduciary AI framework.
        """
        cfg = self.buyer if role == "buyer" else self.seller
        fmv = max(self.market.property_fair_value_usd, 1)

        reward = 0.0

        # ─── (1) Terminal surplus reward ──────────────────────────────
        if self.done and self.agreed_price is not None:
            if role == "buyer":
                surplus = (cfg.reserve_price_usd - self.agreed_price) / fmv
            else:
                surplus = (self.agreed_price - cfg.reserve_price_usd) / fmv
            reward += 10.0 * max(surplus, 0)  # Scale for learning signal

            # Bonus for deals near FMV (Nash-adjacent)
            fmv_deviation = abs(self.agreed_price - fmv) / fmv
            reward += 3.0 * max(0, 1.0 - fmv_deviation * 5)

        # ─── (2) Deal failure penalty ─────────────────────────────────
        elif self.done and self.agreed_price is None:
            reward -= 5.0  # Both agents penalized for failed negotiation
            if own_action == NegotiationAction.WALK_AWAY:
                reward -= 2.0  # Extra penalty for walkaway initiator

        # ─── (3) Progress shaping (anti-sparse-reward) ────────────────
        if not self.done:
            spread = self.current_ask - self.current_bid
            prev_spread = (
                self.bid_history[-2][1] - self.bid_history[-2][0]
                if len(self.bid_history) >= 2 else spread
            )
            spread_reduction = (prev_spread - spread) / fmv
            reward += 2.0 * spread_reduction  # Reward narrowing the gap

        # ─── (4) Cooperative resilience bonus ─────────────────────────
        coop = self._cooperative_resilience()
        reward += 0.5 * coop

        # ─── (5) Information gain (reward process quality) ────────────
        if own_action in (NegotiationAction.CONCEDE_SMALL, NegotiationAction.COUNTER):
            reward += 0.1  # Small bonus for constructive moves
        if own_action == NegotiationAction.HOLD and self.round > 3:
            reward -= 0.05  # Mild penalty for stalling after round 3

        # ─── (6) Time pressure (urgency-weighted) ─────────────────────
        time_penalty = -0.1 * cfg.urgency * (self.round / self.max_rounds)
        reward += time_penalty

        return reward

    def _cooperative_resilience(self) -> float:
        """
        Joint surplus metric: how much total value is created for both parties.
        Higher = more win-win; incentivizes Pareto-optimal outcomes.
        """
        if self.agreed_price is not None:
            buyer_surplus = max(0, self.buyer.reserve_price_usd - self.agreed_price)
            seller_surplus = max(0, self.agreed_price - self.seller.reserve_price_usd)
            fmv = max(self.market.property_fair_value_usd, 1)
            return (buyer_surplus + seller_surplus) / fmv
        # During negotiation: potential surplus based on current spread
        zone = max(0, self.buyer.reserve_price_usd - self.seller.reserve_price_usd)
        fmv = max(self.market.property_fair_value_usd, 1)
        return zone / fmv

    def _nash_distance(self) -> float:
        """Distance of current midpoint from Nash bargaining solution (FMV)."""
        midpoint = (self.current_bid + self.current_ask) / 2
        fmv = max(self.market.property_fair_value_usd, 1)
        return abs(midpoint - fmv) / fmv


# ══════════════════════════════════════════════════════════════════════
# §4  NEURAL NETWORK ARCHITECTURE (Actor-Critic for MAPPO)
# ══════════════════════════════════════════════════════════════════════

class ActorNetwork(nn.Module):
    """Policy network: observation → action distribution."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = NUM_ACTIONS, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)


class CriticNetwork(nn.Module):
    """
    Centralized value function: global_state → V(s).

    In CTDE (Centralized Training, Decentralized Execution), the critic
    sees the full joint observation during training but is not used at inference.
    """

    def __init__(self, global_obs_dim: int = OBS_DIM * 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════
# §5  MAPPO AGENT (Proximal Policy Optimization for Multi-Agent)
# ══════════════════════════════════════════════════════════════════════

class MAPPOAgent:
    """
    Multi-Agent PPO agent with clipped surrogate objective.

    Each party (buyer/seller) has its own actor but shares a centralized critic
    during training. At deployment, only the actor is needed.
    """

    def __init__(
        self,
        role: str,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        device: str = "cpu",
    ):
        self.role = role
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.device = torch.device(device)

        self.actor = ActorNetwork().to(self.device)
        self.critic = CriticNetwork().to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Episode buffer
        self._buffer: Dict[str, list] = self._new_buffer()

    def _new_buffer(self) -> Dict[str, list]:
        return {
            "obs": [], "global_obs": [], "actions": [], "log_probs": [],
            "rewards": [], "dones": [], "values": [],
        }

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[int, float]:
        """Select action using current policy (decentralized execution)."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        dist = self.actor(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def store_transition(
        self, obs: np.ndarray, global_obs: np.ndarray,
        action: int, log_prob: float, reward: float, done: bool, value: float
    ):
        self._buffer["obs"].append(obs)
        self._buffer["global_obs"].append(global_obs)
        self._buffer["actions"].append(action)
        self._buffer["log_probs"].append(log_prob)
        self._buffer["rewards"].append(reward)
        self._buffer["dones"].append(done)
        self._buffer["values"].append(value)

    @torch.no_grad()
    def get_value(self, global_obs: np.ndarray) -> float:
        obs_t = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)
        return self.critic(obs_t).item()

    def compute_gae(self, next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalized Advantage Estimation (GAE-λ)."""
        rewards = self._buffer["rewards"]
        values = self._buffer["values"]
        dones = self._buffer["dones"]

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, next_value: float, epochs: int = 4, batch_size: int = 64) -> dict:
        """
        PPO update with clipped surrogate objective.

        L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]
        """
        advantages, returns = self.compute_gae(next_value)

        obs = torch.FloatTensor(np.array(self._buffer["obs"])).to(self.device)
        global_obs = torch.FloatTensor(np.array(self._buffer["global_obs"])).to(self.device)
        actions = torch.LongTensor(self._buffer["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(self._buffer["log_probs"]).to(self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        n = len(self._buffer["obs"])

        for _ in range(epochs):
            indices = torch.randperm(n)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                idx = indices[start:end]

                # Actor update
                dist = self.actor(obs[idx])
                new_log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                policy_loss -= self.entropy_coeff * entropy

                self.actor_optim.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optim.step()

                # Critic update
                values_pred = self.critic(global_obs[idx])
                value_loss = self.value_coeff * F.mse_loss(values_pred, returns[idx])

                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        self._buffer = self._new_buffer()

        batches = max(1, (n * epochs) // batch_size)
        return {
            "policy_loss": total_policy_loss / batches,
            "value_loss": total_value_loss / batches,
            "entropy": total_entropy / batches,
        }

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


# ══════════════════════════════════════════════════════════════════════
# §6  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

class MARLTrainer:
    """Orchestrates CTDE training of buyer and seller MAPPO agents."""

    def __init__(self, config: dict):
        self.buyer_agent = MAPPOAgent(
            role="buyer",
            lr=config.get("lr", 3e-4),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_eps=config.get("clip_eps", 0.2),
        )
        self.seller_agent = MAPPOAgent(
            role="seller",
            lr=config.get("lr", 3e-4),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_eps=config.get("clip_eps", 0.2),
        )
        self.config = config

    def train(
        self,
        buyer_config: AgentConfig,
        seller_config: AgentConfig,
        market: MarketContext,
        episodes: int = 10000,
    ) -> dict:
        """Full training loop with domain randomization."""
        metrics = {"episode_rewards": [], "deal_rates": [], "nash_distances": []}
        deal_count = 0

        for ep in range(episodes):
            # Domain randomization: jitter market conditions
            jittered_market = MarketContext(
                property_fair_value_usd=int(market.property_fair_value_usd * np.random.uniform(0.95, 1.05)),
                days_on_market=max(1, market.days_on_market + np.random.randint(-10, 10)),
                market_temperature=np.clip(market.market_temperature + np.random.normal(0, 0.05), 0, 1),
                comparable_sold_prices=market.comparable_sold_prices,
                interest_rate_pct=market.interest_rate_pct + np.random.normal(0, 0.2),
                inventory_months=max(0.5, market.inventory_months + np.random.normal(0, 0.3)),
            )

            env = NegotiationEnvironment(buyer_config, seller_config, jittered_market)
            buyer_obs, seller_obs = env.reset()

            ep_buyer_reward = 0.0
            ep_seller_reward = 0.0

            while not env.done:
                global_obs = np.concatenate([buyer_obs, seller_obs])

                # Centralized value estimation
                buyer_value = self.buyer_agent.get_value(global_obs)
                seller_value = self.seller_agent.get_value(global_obs)

                # Decentralized action selection
                buyer_action, buyer_lp = self.buyer_agent.select_action(buyer_obs)
                seller_action, seller_lp = self.seller_agent.select_action(seller_obs)

                # Environment step
                (new_buyer_obs, new_seller_obs), (br, sr), done, info = env.step(buyer_action, seller_action)

                # Store transitions
                self.buyer_agent.store_transition(buyer_obs, global_obs, buyer_action, buyer_lp, br, done, buyer_value)
                self.seller_agent.store_transition(seller_obs, global_obs, seller_action, seller_lp, sr, done, seller_value)

                buyer_obs, seller_obs = new_buyer_obs, new_seller_obs
                ep_buyer_reward += br
                ep_seller_reward += sr

            # End-of-episode update
            final_global = np.concatenate([buyer_obs, seller_obs])
            buyer_metrics = self.buyer_agent.update(self.buyer_agent.get_value(final_global))
            seller_metrics = self.seller_agent.update(self.seller_agent.get_value(final_global))

            if env.agreed_price is not None:
                deal_count += 1

            metrics["episode_rewards"].append((ep_buyer_reward, ep_seller_reward))
            metrics["nash_distances"].append(env._nash_distance())

            if (ep + 1) % 500 == 0:
                recent_deals = sum(1 for i in range(max(0, ep-499), ep+1)
                                   if metrics["episode_rewards"][i][0] > 0) / min(ep+1, 500)
                avg_nash = np.mean(metrics["nash_distances"][-500:])
                logger.info(
                    f"Episode {ep+1}/{episodes} | "
                    f"Deal Rate: {recent_deals:.1%} | "
                    f"Avg Nash Dist: {avg_nash:.4f} | "
                    f"Buyer PL: {buyer_metrics['policy_loss']:.4f} | "
                    f"Seller PL: {seller_metrics['policy_loss']:.4f}"
                )

        metrics["total_deal_rate"] = deal_count / episodes
        return metrics


# ══════════════════════════════════════════════════════════════════════
# §7  INFERENCE ENGINE (for API serving)
# ══════════════════════════════════════════════════════════════════════

class NegotiationEngine:
    """Production inference wrapper — runs a single negotiation session."""

    def __init__(self, buyer_checkpoint: str, seller_checkpoint: str, device: str = "cpu"):
        self.buyer_agent = MAPPOAgent(role="buyer", device=device)
        self.seller_agent = MAPPOAgent(role="seller", device=device)
        self.buyer_agent.load(buyer_checkpoint)
        self.seller_agent.load(seller_checkpoint)
        self._sessions: Dict[str, NegotiationEnvironment] = {}

    def create_session(
        self, buyer_config: AgentConfig, seller_config: AgentConfig, market: MarketContext
    ) -> str:
        session_id = str(uuid.uuid4())[:12]
        self._sessions[session_id] = NegotiationEnvironment(buyer_config, seller_config, market)
        self._sessions[session_id].reset()
        return session_id

    def run_round(self, session_id: str) -> dict:
        env = self._sessions[session_id]
        if env.done:
            return {"status": "complete", "agreed_price": env.agreed_price}

        buyer_obs = env._get_obs("buyer")
        seller_obs = env._get_obs("seller")

        buyer_action, _ = self.buyer_agent.select_action(buyer_obs)
        seller_action, _ = self.seller_agent.select_action(seller_obs)

        _, (br, sr), done, info = env.step(buyer_action, seller_action)

        return {
            "status": "accepted" if done and env.agreed_price else ("failed" if done else "in_progress"),
            **info,
        }

    def run_full_negotiation(
        self, buyer_config: AgentConfig, seller_config: AgentConfig, market: MarketContext
    ) -> dict:
        session_id = self.create_session(buyer_config, seller_config, market)
        rounds = []
        while not self._sessions[session_id].done:
            result = self.run_round(session_id)
            rounds.append(result)
        env = self._sessions[session_id]
        del self._sessions[session_id]
        return {
            "session_id": session_id,
            "rounds": rounds,
            "agreed_price": env.agreed_price,
            "total_rounds": env.round,
            "cooperative_resilience": env._cooperative_resilience(),
            "nash_distance": env._nash_distance(),
        }
