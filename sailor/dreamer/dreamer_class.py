import numpy as np
import torch
import torch.nn as nn
from termcolor import cprint

from sailor.classes.rollout_utils import select_latest_obs
from sailor.dreamer import tools
from sailor.dreamer.imag_behavior import ImagBehavior
from sailor.dreamer.wm import WorldModel

to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(
        self,
        obs_space,
        base_policy,
        config,
        logger,
        dataset,
        expert_dataset=None,
    ):
        super(Dreamer, self).__init__()

        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)

        self._metrics = {}
        self._step = logger.step // config.action_repeat if logger is not None else 0
        self._dataset = dataset
        self._expert_dataset = expert_dataset
        self._base_policy = base_policy

        # Create Models
        self._wm = WorldModel(
            obs_space=obs_space,
            step=self._step,
            config=config,
        )
        self._task_behavior = ImagBehavior(config, self._wm, base_policy=base_policy)

        # Compile if necessary
        if config.compile:
            cprint("Compiling models", "green")
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)

    def get_action(self, obs_orig, state):
        """
        Called during evaluation
        """
        if state is None:
            latent = action = None
        else:
            latent, action = state

        # Create obs_dreamer = BS x BL x ...
        obs_dreamer = {k: np.expand_dims(v, axis=1) for k, v in obs_orig.items()}

        obs_dreamer = self._wm.preprocess(obs_dreamer)
        embed = self._wm.encoder(obs_dreamer)  # BS x BL x (1024 + encoding_dim)
        embed = embed.squeeze(1)  # Remove BL dim

        # Add action
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs_dreamer["is_first"]
        )
        feat = self._wm.dynamics.get_feat(latent)

        action_dict = self._task_behavior.get_action(
            obs=obs_orig, feat=feat, latent=latent
        )

        latent = {k: v.detach() for k, v in latent.items()}
        action_dict = {k: v.detach() for k, v in action_dict.items()}
        action_sum = self._task_behavior.get_action_sum(
            action_dict["base_action"], action_dict["residual_action"]
        )
        state = (latent, action_sum)
        return action_dict, state

    def reset(self):
        self._task_behavior.reset()

    def _train(self, data, training_step, is_warmstart):
        # Obs shape BS x BL x ... x stack_dim
        metrics = {}
        data_wm = select_latest_obs(data)  # Select only last obs and remove stacking
        post, context, mets = self._wm._train(data_wm)
        metrics.update(mets)
        start = {"obs_orig": data, "post": post}

        if self._config.train_dp_mppi_params["use_discrim"]:
            if self._config.train_dp_mppi_params["discrim_state_only"]:
                reward = lambda f, s, a: self._wm.get_reward(
                    self._wm.dynamics.get_feat(s)
                ).mode()
            else:
                reward = lambda f, s, a: self._wm.get_reward(
                    torch.cat([f, a], dim=-1)
                ).mode()
        else:
            reward = lambda f, s, a: self._wm.get_reward(
                self._wm.dynamics.get_feat(s)
            ).mode()

        metrics.update(
            self._task_behavior._train(
                start,
                reward,
                training_step,
                is_warmstart=is_warmstart,
            )[-1]
        )
        return metrics

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)
        cprint("Saved dreamer checkpoint to {}".format(path), "green")

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
        cprint("Loaded dreamer checkpoint from {}".format(path), "green")
