"""Create an identity-residual actor and random critic using base replay only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch

from rl_posttrain.actors import DeterministicActor
from rl_posttrain.critics import DoubleQCritic
from rl_posttrain.h_summary import HSummaryConfig
from rl_posttrain.online_td3bc import replay_scenes
from rl_posttrain.replay_buffer import OfflineReplayBuffer
from rl_posttrain.td3bc_ref import PreparedReplay, TD3BCConfig


def initialize(replay: OfflineReplayBuffer, seed: int = 0,
               config: TD3BCConfig | None = None) -> dict:
    scenes = replay_scenes(replay)
    allowed = {f"room{r}_table{t}" for r in (1, 2) for t in (1, 2)}
    if not scenes or not scenes.issubset(allowed):
        raise ValueError(f"Initialization replay must contain only RL training scenes: {scenes}")
    cfg = config or TD3BCConfig(h_summary=HSummaryConfig(mode="h_proj128"))
    prepared = PreparedReplay(replay, cfg)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    actor = DeterministicActor(
        prepared.actor_obs_dim, prepared.action_dim, cfg.actor_hidden_dims,
        h_summary=cfg.h_summary, parameterization="reference_residual",
        actor_obs_normalizer=prepared.actor_obs_normalizer.state_dict(),
    )
    critic = DoubleQCritic(
        prepared.critic_obs_dim, prepared.action_dim, cfg.critic_hidden_dims,
        h_summary=cfg.h_summary, actor_obs_dim=prepared.actor_obs_dim,
    )
    with torch.no_grad():
        errors = []
        for start in range(0, prepared.size, 512):
            obs = torch.from_numpy(prepared.arrays["actor_obs"][start:start + 512])
            ref = torch.from_numpy(prepared.arrays["bc_target_norm"][start:start + 512])
            errors.append(float((actor(obs) - ref).abs().max()))
    max_error = max(errors)
    if max_error > 1e-6:
        raise AssertionError(f"Initial actor differs from reference: {max_error}")
    return {
        "format": "td3bc_ref_actor_v1",
        "actor_parameterization": "reference_residual",
        "initialization": {"mode": "base_replay_identity_random_critic", "seed": seed,
                           "offline_weights_loaded": False, "base_transitions": prepared.size,
                           "base_scenes": sorted(scenes), "max_reference_error": max_error},
        "config": asdict(cfg),
        "actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(),
        "actor_obs_dim": prepared.actor_obs_dim, "critic_obs_dim": prepared.critic_obs_dim,
        "action_dim": prepared.action_dim, "actor_hidden_dims": cfg.actor_hidden_dims,
        "critic_hidden_dims": cfg.critic_hidden_dims, "h_summary": cfg.h_summary.state_dict(),
        "action_normalizer": prepared.action_normalizer.state_dict(),
        "actor_obs_normalizer": prepared.actor_obs_normalizer.state_dict(),
        "critic_obs_normalizer": prepared.critic_obs_normalizer.state_dict(),
        "action_spec": prepared.action_spec,
        "online_episode": 0, "global_update": 0, "actor_updates": 0,
        "critic_updates": 0, "env_steps": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_replay", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    replay = OfflineReplayBuffer.load(args.base_replay, replay_filter="base_only")
    payload = initialize(replay, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(payload["initialization"], flush=True)
    print(f"Saved initialization to {output}", flush=True)


if __name__ == "__main__":
    main()
