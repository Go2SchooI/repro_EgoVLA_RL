"""Online SAC on frozen EgoVLA features, warm-started from an offline actor."""
import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from rl_posttrain.online_td3bc import OnlineTD3BCAgent, _torch_load, _soft_update
from rl_posttrain.sac_actor import SquashedGaussianActor


class OnlineSACAgent(OnlineTD3BCAgent):
    def __init__(self, checkpoint_path, cfg, device="cuda", resume_state=None):
        source = dict(resume_state) if resume_state is not None else _torch_load(checkpoint_path, "cpu")
        if source.get("format") not in ("td3bc_ref_actor_v1", "sac_actor_v1", "online_sac_checkpoint_v1"):
            raise ValueError("Unsupported SAC source checkpoint format")
        if float(cfg["sac"]["initial_temperature"]) <= 0:
            raise ValueError("SAC temperature must be positive")
        is_sac = source.get("algorithm") == "sac"
        if source.get("actor_parameterization", "direct_tanh") not in ("direct_tanh", "reference_logit_residual"):
            raise ValueError("Unsupported SAC actor parameterization")
        if not is_sac and int(source.get("online_episode", 0)) != 0:
            raise ValueError("SAC warm-start must be the offline checkpoint, not an online milestone")
        compatible = copy.deepcopy(source)
        compatible["format"] = "td3bc_ref_actor_v1"
        for key in ["actor_state_dict", "actor_target_state_dict"]:
            if key in compatible:
                compatible[key] = {k: v for k, v in compatible[key].items()
                                   if not k.startswith("log_std_head.")}
        compatible.pop("actor_optimizer_state_dict", None)
        compatible.pop("critic_optimizer_state_dict", None)
        super().__init__(checkpoint_path, cfg, device, resume_state=compatible)
        opts = source.get("sac_policy", {}) if is_sac else {
            k: cfg["sac"][k] for k in ["init_log_std", "log_std_min", "log_std_max"]}
        self.actor = SquashedGaussianActor(self.actor_obs_dim, self.action_dim,
            self.actor_hidden_dims, h_summary=self.h_summary,
            parameterization=self.actor_parameterization,
            actor_obs_normalizer=source.get("actor_obs_normalizer"), **opts).to(self.device)
        loaded = self.actor.load_state_dict(source["actor_state_dict"], strict=False)
        expected = [] if is_sac else ["log_std_head.weight", "log_std_head.bias"]
        if sorted(loaded.missing_keys) != sorted(expected) or loaded.unexpected_keys:
            raise ValueError(f"Unexpected actor state transfer: {loaded}")
        self.actor_target = copy.deepcopy(self.actor)  # Export compatibility only; SAC never uses it.
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(cfg["optimization"]["actor_lr"]))
        self.log_alpha = torch.tensor(np.log(float(cfg["sac"]["initial_temperature"])),
                                      dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(cfg["sac"]["temperature_lr"]))
        self.target_entropy = float(cfg["sac"].get("target_entropy", -self.action_dim))
        self.mean_anchor_weight = float(cfg["sac"].get("mean_anchor_weight", 0.0))
        if not np.isfinite(self.mean_anchor_weight) or self.mean_anchor_weight < 0:
            raise ValueError("Mean anchor weight must be finite and nonnegative")
        if is_sac and float(source.get("mean_anchor_weight", 0.0)) != self.mean_anchor_weight:
            raise ValueError("Cannot change mean anchor weight while resuming SAC")
        self.mean_anchor = None
        if self.mean_anchor_weight > 0:
            if is_sac and "mean_anchor_state_dict" not in source:
                raise ValueError("Anchored SAC checkpoint is missing its frozen initial actor")
            self.mean_anchor = copy.deepcopy(self.actor)
            if is_sac:
                self.mean_anchor.load_state_dict(source["mean_anchor_state_dict"])
            self.mean_anchor.eval().requires_grad_(False)
        if is_sac:
            for key in ["init_log_std", "log_std_min", "log_std_max"]:
                if float(cfg["sac"][key]) != float(source["sac_policy"][key]):
                    raise ValueError("Cannot change SAC sampling configuration on resume")
            if float(source["target_entropy"]) != self.target_entropy:
                raise ValueError("Cannot change SAC target entropy on resume")
            with torch.no_grad(): self.log_alpha.copy_(source["log_alpha"])
            for key, opt in [("actor_optimizer_state_dict", self.actor_opt),
                             ("critic_optimizer_state_dict", self.critic_opt),
                             ("temperature_optimizer_state_dict", self.alpha_opt)]:
                if key in source: opt.load_state_dict(source[key])
        else:
            self.global_update = self.critic_updates = self.actor_updates = 0
            self.online_episode = self.env_steps = 0
            self.saved_rng_state = None
            self.replay_manifest = None
            self.initialization = copy.deepcopy(source.get("sac_initialization", dict(
                mode="offline_actor_and_critic", source=str(checkpoint_path),
                deterministic_mean_preserved=True, temperature_initialized=True)))

    def train_step(self, batch, cfg, *, update_actor, update_actor_target):
        self.global_update += 1
        gamma, tau = float(cfg["td3"]["gamma"]), float(cfg["td3"]["tau"])
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(batch["next_actor_obs"])
            q1, q2 = self.critic_target(batch["next_critic_obs"], next_action)
            target_q = torch.minimum(q1, q2) - alpha * next_logp
            target = batch["reward"] + gamma * (1 - batch["done"]) * target_q
        q1, q2 = self.critic(batch["critic_obs"], batch["action_norm"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        if not torch.isfinite(critic_loss): raise FloatingPointError("Nonfinite SAC critic loss")
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.critic_updates += 1
        _soft_update(self.critic, self.critic_target, tau)
        logs = dict(critic_loss=float(critic_loss.detach()), actor_loss=0.0,
                    q1_mean=float(q1.detach().mean()), q2_mean=float(q2.detach().mean()),
                    target_q_mean=float(target_q.mean()), td_target_mean=float(target.mean()),
                    **{"sac/temperature": float(alpha), "sac/actor_updated": 0.0})
        # SAC updates its actor each critic update after the common warm-up.
        if update_actor:
            for p in self.critic.parameters(): p.requires_grad_(False)
            try:
                action, logp, mean_action = self.actor.sample(batch["actor_obs"])
                aq1, aq2 = self.critic(batch["critic_obs"], action)
                sac_actor_loss = (alpha * logp - torch.minimum(aq1, aq2)).mean()
                actor_loss = sac_actor_loss
                anchor_loss = mean_action.new_zeros(())
                if self.mean_anchor is not None:
                    with torch.no_grad(): reference_mean = self.mean_anchor(batch["actor_obs"])
                    anchor_loss = F.mse_loss(mean_action, reference_mean)
                    actor_loss = actor_loss + self.mean_anchor_weight * anchor_loss
                if not torch.isfinite(actor_loss): raise FloatingPointError("Nonfinite SAC actor loss")
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                self.actor_opt.step()
            finally:
                for p in self.critic.parameters(): p.requires_grad_(True)
            self.actor_updates += 1
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                _, log_std = self.actor.distribution_parameters(batch["actor_obs"])
            logs.update(actor_loss=float(actor_loss.detach()), **{
                "sac/actor_updated": 1.0, "sac/entropy": float(-logp.detach().mean()),
                "sac/unregularized_actor_loss": float(sac_actor_loss.detach()),
                "sac/mean_anchor_loss": float(anchor_loss.detach()),
                "sac/weighted_mean_anchor_loss": float(self.mean_anchor_weight * anchor_loss.detach()),
                "sac/temperature_loss": float(alpha_loss.detach()),
                "sac/std_mean": float(log_std.exp().mean()),
                "sac/temperature": float(self.log_alpha.exp().detach())})
        return logs

    def actor_checkpoint_payload(self, cfg):
        payload = super().actor_checkpoint_payload(cfg)
        payload.update(format="sac_actor_v1", online_format="sac_actor_export_v1",
                       algorithm="sac", sac_policy=self.actor.policy_config(),
                       log_alpha=self.log_alpha.detach().clone(), target_entropy=self.target_entropy)
        if self.mean_anchor is not None:
            payload.update(mean_anchor_weight=self.mean_anchor_weight,
                           mean_anchor_state_dict=self.mean_anchor.state_dict())
        return payload

    def save_training_checkpoint(self, path, cfg):
        payload = self.actor_checkpoint_payload(cfg)
        payload.update(format="online_sac_checkpoint_v1",
            actor_optimizer_state_dict=self.actor_opt.state_dict(),
            critic_optimizer_state_dict=self.critic_opt.state_dict(),
            temperature_optimizer_state_dict=self.alpha_opt.state_dict())
        if self.replay_rng is not None:
            payload["rng_state"] = dict(replay_generator=copy.deepcopy(self.replay_rng.bit_generator.state),
                python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [])
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path
