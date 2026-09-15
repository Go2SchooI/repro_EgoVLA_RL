"""Independent actor/critic initialization for the SAC B/C comparison."""
import copy

import torch

from rl_posttrain.actors import DeterministicActor
from rl_posttrain.critics import DoubleQCritic
from rl_posttrain.h_summary import HSummaryConfig


def initialize(source, variant, seed):
    if variant not in ("B", "C"):
        raise ValueError("Expected B (offline actor) or C (EgoVLA reference actor)")
    if source.get("format") != "td3bc_ref_actor_v1" or source.get("online_episode", 0) != 0:
        raise ValueError("Expected the original offline checkpoint")
    if source.get("actor_parameterization", "direct_tanh") != "direct_tanh":
        raise ValueError("Expected the original direct-tanh actor")
    # Whitelist metadata so no optimizer, learned critic or training RNG leaks in.
    keys = ["config", "actor_obs_dim", "critic_obs_dim", "action_dim",
            "actor_hidden_dims", "critic_hidden_dims", "h_summary",
            "action_normalizer", "actor_obs_normalizer", "critic_obs_normalizer", "action_spec"]
    out = {k: copy.deepcopy(source[k]) for k in keys if k in source}
    hs = HSummaryConfig.from_state_dict(out.get("h_summary"))
    # Independent streams guarantee identical fresh critics for paired B/C seeds.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(400000 + seed)
        critic = DoubleQCritic(out["critic_obs_dim"], out["action_dim"], out["critic_hidden_dims"],
                              h_summary=hs, actor_obs_dim=out["actor_obs_dim"])
    if variant == "B":
        out["actor_parameterization"] = "direct_tanh"
        out["actor_state_dict"] = copy.deepcopy(source["actor_state_dict"])
    else:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(300000 + seed)
            actor = DeterministicActor(out["actor_obs_dim"], out["action_dim"], out["actor_hidden_dims"],
                h_summary=hs, parameterization="reference_logit_residual",
                actor_obs_normalizer=out["actor_obs_normalizer"])
        out["actor_parameterization"] = "reference_logit_residual"
        out["actor_state_dict"] = actor.state_dict()
    out.update(format="td3bc_ref_actor_v1", critic_state_dict=critic.state_dict(),
               critic_target_state_dict=copy.deepcopy(critic.state_dict()),
               online_episode=0, env_steps=0, global_update=0, critic_updates=0, actor_updates=0)
    out["sac_initialization"] = dict(variant=variant, seed=seed,
        mode="offline_actor_fresh_critic" if variant == "B" else "egovla_reference_fresh_critic",
        offline_actor_weights_loaded=variant == "B", offline_critic_weights_loaded=False,
        normalization_source="same base-replay metadata as A", prior_replay_retained=True,
        actor_seed=300000 + seed if variant == "C" else None, critic_seed=400000 + seed,
        reference_clip_epsilon=1e-6 if variant == "C" else None)
    return out
