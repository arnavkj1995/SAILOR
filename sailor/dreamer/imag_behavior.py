import copy

import einops
import torch
import torch.nn as nn

from sailor.classes.actor_net import ActorNet
from sailor.dreamer import tools
from sailor.dreamer.networks import MLPEnsemble

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model, base_policy):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        self.base_policy = base_policy
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        if config.use_residual_actor:
            self.actor = ActorNet(config=config, base_policy=base_policy)
            self._actor_opt = tools.Optimizer(
                "residual_actor",
                self.actor.parameters(),
                config.residual_actor["lr"],
                config.residual_actor["eps"],
                config.residual_actor["grad_clip"],
                **kw,
            )
            print(
                f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
            )

        self.value = MLPEnsemble(
            config.critic["num_models"],
            config.critic["num_subsample"],
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config.device))
            self.reward_ema = RewardEMA(device=self._config.device)

    def get_annealing_weight(self, training_step):
        # Not used currently
        start_weight = self._config.residual_training["l2_anneal_start_weight"]
        end_weight = self._config.residual_training["l2_anneal_end_weight"]
        anneal_steps = self._config.residual_training["l2_anneal_steps"]

        return start_weight - (start_weight - end_weight) * min(
            1.0, training_step / anneal_steps
        )

    def compute_bc_loss(self, input_data, imag_action_dict):
        # Compute BC loss for expert buffer
        # Unroll IMAG_ACTION keys into dim0, BS, BL, ...
        for key in imag_action_dict.keys():
            # Use einops to rearrange
            imag_action_dict[key] = einops.rearrange(
                imag_action_dict[key],
                "t (bs bl) ... -> t bs bl ...",
                bs=self._config.batch_size,
                bl=self._config.batch_length,
            )

        # Get actions for first timestep and first half of bs (BS//2 x BL)
        actor_residual_action = imag_action_dict["residual_action"][
            0,
            : self._config.batch_size // 2,
            : self._config.batch_length,
        ]
        actor_base_action = imag_action_dict["base_action"][
            0,
            : self._config.batch_size // 2,
            : self._config.batch_length,
        ]

        # Add actions (straight through gradient)
        actor_action = self.get_action_sum(
            base_actions=actor_base_action,
            residual_actions=actor_residual_action,
        )  # Shape (BS//2, BL, action_dim)

        # Get expert actions for these
        expert_action = input_data["obs_orig"]["action"][
            : self._config.batch_size // 2,
            : self._config.batch_length,
            :,
            0,
        ]  # Shape (BS//2, BL, action_dim)

        # Get the bc_loss
        bc_loss = torch.nn.functional.mse_loss(actor_action, expert_action)
        return bc_loss

    def imagine_with_actor(self, input_data, objective, update_actor=False):
        metrics = {}
        actor_loss = None
        with tools.RequiresGrad(self.actor):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                # Get imagined rollout of policy in the current world model
                imag_feat, imag_state, imag_action_dict = self._imagine(
                    input_data,
                    horizon=self._config.imag_horizon,
                    mode="default" if update_actor else "base_only",
                    policy=self.actor,
                )
                # imag_feat: 15 x (BS*BL) x feat_size => 15=imag_horizon
                # Everything else is 15 x (BS*BL) x ...

                # Get predicted reward for the horizon steps using imag_state (check objective in dreamer.py)
                reward = objective(imag_feat, imag_state, None)

                # Actor(imag_feat) returns a distribution
                # Entropy: computes entropy of actor policy E_{x~p(x)}[-log(p(x))]
                # Higher the entropy, more spread out the policy is
                # actor_ent: 15 x (BS*BL)
                actor_ent = self.actor.get_entropy(
                    feat=imag_feat, base_action=imag_action_dict["base_action"]
                )

                # Compute the target values using bootstrapping and terminal value fn
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )

                # Compute the actor loss, use only the residual action for the loss
                # As the update rule is
                # pi(a_t | s_t) = pi_B(a_b | s_t) + pi_R(a_r | s_t, a_b)
                # E_{tau ~ pi} [ sum_{t=0}^{T} [ ∇ log(pi_R(a_r | s_t, a_b)) * Advantage ] ]
                # Note: Advantage is computed using the full action
                # NOTE: currently detatching the residual action as done in standard policy gradients
                if update_actor:
                    actor_loss, mets = self._compute_actor_loss(
                        imag_feat=imag_feat,
                        base_policy_actions=imag_action_dict["base_action"],
                        imag_action=imag_action_dict["residual_action"],
                        target=target,
                        weights=weights,
                        base=base,
                    )

                    actor_loss -= (
                        self._config.residual_actor["entropy"]
                        * actor_ent[:-1, ..., None]
                    )
                    actor_loss = torch.mean(actor_loss)
                    mets["actor_orig_loss"] = actor_loss.item()

                    # Get BC loss for expert buffer
                    bc_loss = self.compute_bc_loss(input_data, imag_action_dict)
                    mets["actor_bc_loss"] = bc_loss.item()

                    # L2 norm loss
                    res_l2_norm = torch.norm(
                        imag_action_dict["residual_action"], dim=-1
                    ).mean()

                    if self._config.residual_training["action_l2_reg_weight"] != 0:
                        # Scale factor = (curr_l2_norm / target_l2_norm) * weight
                        scale_factor = float(
                            (
                                res_l2_norm.item()
                                / self._config.residual_training[
                                    "action_l2_norm_target"
                                ]
                            )
                            * self._config.residual_training["action_l2_reg_weight"]
                        )
                        metrics["actor_res_l2_norm_scale_factor"] = scale_factor

                        actor_loss += scale_factor * res_l2_norm

                    metrics["actor_res_l2_norm"] = res_l2_norm.item()
                    metrics.update(mets)

        return (
            metrics,
            reward,
            imag_feat,
            imag_state,
            imag_action_dict,
            weights,
            target,
            actor_loss,
        )

    def _train(
        self,
        input_data,
        objective,
        training_step,
        is_warmstart,
    ):
        """
        Train the actor and value networks
        """
        self._update_slow_target()
        metrics = {}

        # Get imagined rollout of the base policy in the current world model
        if self._config.use_residual_actor:
            (
                metrics_actor,
                reward,
                imag_feat,
                imag_state,
                imag_action_dict,
                weights,
                target,
                actor_loss,
            ) = self.imagine_with_actor(
                input_data, objective, update_actor=not is_warmstart
            )
            metrics.update(metrics_actor)
        else:
            imag_feat, imag_state, imag_action_dict = self._imagine(
                input_data,
                self._config.imag_horizon,
                mode="base_only",
            )

            # Get predicted reward for the horizon steps using imag_state and base policy actions
            reward = objective(imag_feat, imag_state, imag_action_dict["base_action"])

            # Compute the target values using bootstrapping and terminal value fn
            target, weights, _ = self._compute_target(imag_feat, imag_state, reward)

        # Compute the critic losses
        with tools.RequiresGrad(self.value):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                value = self.value(imag_feat[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = self.value.update(imag_feat[:-1].detach(), target)
                if self._config.critic["slow_target"]:
                    slow_target = self._slow_value(imag_feat[:-1].detach())
                    slow_loss = self.value.update(
                        imag_feat[:-1].detach(), slow_target.mode().detach()
                    )
                    metrics.update(tools.tensorstats(value_loss, "orig_critic_loss"))
                    metrics.update(tools.tensorstats(slow_loss, "slow_critic_loss"))
                    value_loss += slow_loss
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        metrics.update(
            tools.tensorstats(imag_action_dict["base_action"], "imag_base_action")
        )
        metrics.update(self.value.get_stats(features=imag_feat[:-1].detach()))

        # Train the value network
        with tools.RequiresGrad(self):
            if not is_warmstart:
                metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))

        return imag_feat, imag_state, None, weights, metrics

    def _imagine(self, input_data, horizon, mode="base_only", policy: ActorNet = None):
        """
        MODE:
        - base_only: Use only base actions for imagination [used during training]
        - residual_buffer: Use residual actions from buffer [used during value estimation]
        - default: Use residual policy from dreamer
        """
        assert mode in ["default", "residual_buffer", "base_only"]
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in input_data["post"].items()}
        start["ID"] = torch.tensor(0)  # To keep track of imag_timestep

        base_action = flatten(input_data["obs_orig"]["base_action"])[
            ..., :horizon
        ]  # (BS*BL) x action_dim x horizon
        residual_action = flatten(input_data["obs_orig"]["residual_action"])[
            ..., :horizon
        ]  # (BS*BL) x action_dim x horizon

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            base_policy_action = base_action[..., state["ID"]]
            if mode == "base_only":
                residual_policy_action = torch.zeros_like(base_policy_action)
            elif mode == "residual_buffer":
                residual_policy_action = residual_action[..., state["ID"]]
            else:
                residual_policy_action = policy.get_residual_dist(
                    feat=feat.detach(), base_action=base_policy_action
                ).sample()
            action_dict = {
                "base_action": base_policy_action,
                "residual_action": residual_policy_action,
            }
            action_sum = self.get_action_sum(
                base_actions=base_policy_action, residual_actions=residual_policy_action
            )
            # Use the total_action to simulate dynamics for next state
            succ = dynamics.img_step(state, action_sum)
            succ["ID"] = state["ID"] + 1
            return succ, feat, action_dict

        succ, feats, actions_dict = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions_dict

    def get_action_sum(self, base_actions, residual_actions):
        # Assert same shape
        assert (
            base_actions.shape == residual_actions.shape
        ), f"{base_actions.shape} != {residual_actions.shape}"
        return (
            (base_actions + residual_actions).clamp(-1, 1).detach()
            + residual_actions
            - residual_actions.detach()
        )

    def reset(self):
        self.base_policy.reset()

    def get_action(self, obs, feat, latent, weighting_in_base=True):
        if self._config.use_residual_actor:
            action_dict = self.actor.get_action(obs, feat, weighting_in_base)
            action_dict["residual_action"] = action_dict["residual_action"].mode()
            return action_dict

        with torch.no_grad():
            base_action = self.base_policy.get_action(
                obs, weighting=weighting_in_base, get_full_action=True
            )
            base_action = torch.tensor(
                base_action, device=feat.device, dtype=feat.dtype
            )

            # Call the MPPI planner and get the best planned actions
            mppi_action = self.mppi_actions(latent, base_action=base_action)

            # Pass the 0th actions for execution in the environment
            action_dict = {
                "base_action": base_action[:, :, 0],
                "residual_action": mppi_action[:, :, 0],
            }

            if self._config.generate_highres_eval:
                action_dict["reward_output"] = torch.cat(
                    [
                        self._world_model.reward_discrim(feat).mode(),
                        self.value.get_all_critic_mean(feat).mode(),
                    ],
                    dim=-1,
                )

        return action_dict

    def mppi_actions(self, latent, base_action):
        """
        latent: Dict containing stoch, deter, logit, each of shape N_envs x ...
        base_action: N_envs x pred_horizon x action_dim

        Output:
        Action shape: N_envs x action_dim
        """
        num_samples = self._config.mppi["num_samples"]
        horizon = self._config.mppi["horizon"]
        action_dim = self._config.num_actions

        _BS = latent["stoch"].shape[0]
        _state = {k: v.unsqueeze(1) for k, v in latent.items()}  # _BS x 1 x 32 x 32
        _state = {
            k: v.repeat(*[1 if i != 1 else num_samples for i in range(len(v.shape))])
            for k, v in _state.items()
        }  # _BS x num_samples x 32 x 32

        # Convert base_action to shape (num_envs, num_samples, pred_horizon, action_dim)
        base_action = base_action.unsqueeze(1).expand(-1, num_samples, -1, -1)

        mean = torch.zeros(_BS, action_dim, horizon, device=self._config.device)
        std = (
            torch.ones(_BS, action_dim, horizon, device=self._config.device)
            * self._config.mppi["init_std"]
        )

        # Perform CEM iterations to find optimal action sequence
        for _ in range(self._config.mppi["iterations"]):
            # MPPI actions: (_BS, action_dim, num_samples, horizon)
            mppi_actions = (
                mean.unsqueeze(2)
                + std.unsqueeze(2)
                * torch.randn(
                    _BS, action_dim, num_samples, horizon, device=self._config.device
                )
            ).clamp(
                -self._config.mppi["abs_residual"], self._config.mppi["abs_residual"]
            )
            input_data = {
                "post": _state,
                "obs_orig": {
                    # Actions: BS x num_samples x action_dim x pred_horizon
                    "base_action": base_action,
                    "residual_action": mppi_actions.permute(0, 2, 1, 3),
                },
            }

            imag_feat, imag_state, imag_action = self._imagine(
                input_data, horizon, mode="residual_buffer"
            )

            # # Estimate n-step TD value
            with torch.no_grad():
                if self._config.train_dp_mppi_params["use_discrim"]:
                    if self._config.train_dp_mppi_params["discrim_state_only"]:
                        get_reward = lambda f, a: self._world_model.get_reward(f).mode()
                    else:
                        get_reward = lambda f, a: self._world_model.get_reward(
                            torch.cat([f, a], dim=-1)
                        ).mode()
                else:
                    get_reward = lambda f, a: self._world_model.get_reward(f).mode()

                get_cont = lambda f: self._world_model.heads["cont"](f).mean
                G, discount = 0, 1
                for t in range(horizon - 1):
                    total_action = torch.clamp(
                        imag_action["base_action"][t]
                        + imag_action["residual_action"][t],
                        -1,
                        1,
                    )
                    reward = get_reward(
                        imag_feat[t], total_action
                    )  # (BS*num_samples, 1)
                    cont = get_cont(imag_feat[t])  # (BS*num_samples, 1)
                    G += discount * reward

                    # Uncertainty cost
                    if self._config.mppi["uncertainty_cost"] > 0:
                        q_std = self.value.get_std(imag_feat[t])  # (BS*num_samples, 1)
                        G -= discount * self._config.mppi["uncertainty_cost"] * q_std

                    # Action L2 cost
                    if self._config.mppi["action_l2_cost"] > 0:
                        act_norm = (
                            torch.norm(total_action, dim=-1)[:, None] / action_dim
                        )  # (BS*num_samples, 1)
                        G -= discount * self._config.mppi["action_l2_cost"] * act_norm

                    discount *= self._config.mppi["discount"] * cont

                final_value = (
                    G + discount * self.value(imag_feat[-1]).mode()
                )  # Shape: (BS*num_samples, 1)
                if self._config.mppi["uncertainty_cost"] > 0:
                    q_std = self.value.get_std(imag_feat[-1])
                    final_value -= (
                        discount * self._config.mppi["uncertainty_cost"] * q_std
                    )

            # Estimate lambda return
            # reward = self._world_model.heads["reward"](imag_feat).mode()
            # target, _, _ = self._compute_target(
            #     imag_feat, imag_state, reward
            # )  # (horizon, BS*num_samples, 1)
            # final_value = torch.stack(target, dim=1).sum(dim=0)  # (BS*num_samples, 1)

            values = final_value.squeeze(-1).reshape(
                _BS, num_samples
            )  # (_BS, num_samples)

            elite_idxs = torch.topk(
                values, self._config.mppi["num_elites"], dim=1
            ).indices  # (_BS, num_elites)
            elite_actions = torch.gather(
                mppi_actions.permute(
                    0, 3, 2, 1
                ),  # (_BS, action_dim, num_samples, horizon) -> [BS, horizon, num_samples, action_dim]
                2,
                elite_idxs.unsqueeze(1)
                .unsqueeze(-1)
                .expand(-1, horizon, -1, action_dim),
            )  # Shape: [BS, horizon, num_elites, action_dim]
            elite_value = torch.gather(values, 1, elite_idxs)  # (_BS, num_elites)

            # update the mean and std
            max_value = elite_value.max(1)[0]  # (_BS,) max value across elite actions
            score = torch.exp(
                self._config.mppi["temperature"]
                * (elite_value - max_value.unsqueeze(1))
            )  # (_BS, num_elites)
            score /= score.sum(1, keepdim=True)  # Normalize score across elites
            score = (
                score.unsqueeze(1).expand(-1, horizon, -1).unsqueeze(-1)
            )  # (_BS, horizon, num_elites, 1)
            mean = torch.sum(score * elite_actions, dim=2) / (
                score.sum(2, keepdim=True) + 1e-9
            ).squeeze(
                -1
            )  # (_BS, horizon, action_dim)

            std = torch.sqrt(
                torch.sum(score * (elite_actions - mean.unsqueeze(2)) ** 2, dim=2)
                / (score.sum(2, keepdim=True) + 1e-9).squeeze(-1)
            )  # (_BS, horizon, action_dim)
            std = std.clamp_(
                self._config.mppi["min_std"], self._config.mppi["max_std"]
            )  # Clamp the standard deviation
            mean = mean.permute(0, 2, 1)
            std = std.permute(0, 2, 1)

        # Select final action
        max_value_idx = values.argmax(dim=1)  # Shape: (_BS,)

        # Expand max_value_idx to match the shape of mppi_actions (shape: (_BS, action_dim, num_samples, pred_horizon))
        expanded_idx = (
            max_value_idx.unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, mppi_actions.size(1), -1, mppi_actions.size(3))
        )
        max_mppi_actions = torch.gather(mppi_actions, 2, expanded_idx).squeeze(
            2
        )  # Shape: (_BS, action_dim, pred_horizon)

        return max_mppi_actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[:-1],
            value[:-1],
            discount[:-1],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        base_policy_actions,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        # Note: policy here is only the residual policy action as required to compute reinforce update for Residual Policy
        policy = self.actor.get_residual_dist(feat=inp, base_action=base_policy_actions)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = target
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target

        metrics.update(tools.tensorstats(actor_target, "actor_target"))
        metrics.update(tools.tensorstats(weights[:-1], "weights"))
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1
