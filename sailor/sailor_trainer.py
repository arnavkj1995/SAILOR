import collections
import copy
import os
import pickle as pkl
import time

import numpy as np
import torch
from termcolor import cprint

from sailor.classes.preprocess import Preprocessor
from sailor.classes.resnet_encoder import ResNetEncoder
from sailor.classes.rollout_utils import collect_onpolicy_trajs, mixed_sample
from sailor.dreamer import tools
from sailor.dreamer.dreamer_class import Dreamer
from sailor.policies.diffusion_base_policy import (DiffusionBasePolicy,
                                                   DiffusionPolicyAgent)
from sailor.policies.residual_policy import ResidualPolicy
from sailor.trainer_utils import (count_n_transitions, label_expert_eps,
                                  make_retrain_dp_dataset)


class SAILORTrainer:
    def __init__(
        self,
        config,
        expert_eps,
        state_dim,
        action_dim,
        train_env,
        eval_envs,
        expert_val_eps,
        train_eps,
        init_step,
        logger: tools.Logger = None,
    ):
        self.config = config
        self.expert_eps = expert_eps
        self.train_env = train_env
        self.eval_envs = eval_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.expert_val_eps = expert_val_eps
        self.train_eps = train_eps
        self.logger = logger

        self.num_expert_transitions = count_n_transitions(self.expert_eps)
        self.expert_datset = tools.make_dataset(
            self.expert_eps,
            batch_length=self.config.batch_length,
            batch_size=self.config.batch_size,
        )

        self.replay_buffer = collections.OrderedDict()
        self._step = init_step
        cprint(f"Initializing SAILORTrainer with init_step: {self._step}")
        self._env_step = 0
        self.base_policy = self.init_dp(load_dp_weights=True)
        self.dreamer_class: Dreamer = Dreamer(
            obs_space=self.eval_envs.observation_space,
            base_policy=self.base_policy,
            config=self.config,
            logger=None,
            dataset=None,
            expert_dataset=self.expert_datset,
        ).to(self.config.device)

        self.residual_policy = ResidualPolicy(
            config=self.config,
            dreamer_class=self.dreamer_class,
            expert_eps=self.expert_eps,
            train_eps=self.train_eps,
            train_env=self.train_env,
            eval_envs=self.eval_envs,
            logger=self.logger,
        )

    def init_dp(self, load_dp_weights=False, set_in_dreamer=True):
        # Initialize the models
        if self.config.state_only:
            encoder = None
        else:
            encoder = ResNetEncoder(num_cams=self.config.dp["num_cams"])
        preprocessor = Preprocessor(
            config=self.config,
        )
        base_policy = DiffusionBasePolicy(
            preprocessor,
            encoder=encoder,
            config=self.config,
            device=self.config.device,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            name="DP_Distilled",
            logger=self.logger,
        )

        if load_dp_weights:
            if self.config.dp["pretrained_ckpt"] is None:
                raise ValueError("No pretrained checkpoint provided for DP.")

            full_ckpt_path = os.path.join(
                self.config.scratch_dir, self.config.dp["pretrained_ckpt"]
            )
            base_policy.trainer.load_checkpoint(full_ckpt_path)

        if hasattr(self, "dreamer_class") and set_in_dreamer:
            del self.dreamer_class._base_policy
            torch.cuda.empty_cache()
            self.dreamer_class._base_policy = base_policy

        return base_policy

    def train_wm_critic(self, itrs):
        num_buffer_transitions = count_n_transitions(self.replay_buffer)
        print("Number of expert transitions: ", self.num_expert_transitions)
        print("Number of buffer transitions: ", num_buffer_transitions)
        print(f"Mixed training with 50% expert data for {itrs} iterations")

        expert_dataset = tools.make_dataset(
            self.expert_eps,
            batch_length=self.config.batch_length,
            batch_size=self.config.batch_size // 2,
        )
        train_dataset = tools.make_dataset(
            self.replay_buffer,
            batch_length=self.config.batch_length,
            batch_size=self.config.batch_size // 2,
        )
        for n_wm_itr in range(itrs):
            batch = mixed_sample(
                batch_size=self.config.batch_size,
                expert_dataset=expert_dataset,
                train_dataset=train_dataset,
                device=self.config.device,
                remove_obs_stack=False,
                sqil_discriminator=self.config.train_dp_mppi_params["use_discrim"],
            )

            metrics = self.dreamer_class._train(
                data=batch,
                training_step=self._step,
            )
            self._step += 1

            if self._step % self.config.log_every == 0:
                print(
                    f"[WM + Critic Training] Itr: {n_wm_itr}/{itrs}, Value Loss: {metrics['value_loss']}"
                )

                # Add metrics to logger and log it with prefix warmup
                if self.logger is not None:
                    for key, value in metrics.items():
                        if not isinstance(value, float):
                            value = np.mean(value)
                        self.logger.scalar(f"wm_critic_train/{key}", value)

                    self.logger.scalar(f"wm_critic_train/step", self._step)
                    self.logger.scalar(f"wm_critic_train/itr", n_wm_itr)
                    self.logger.write(step=self._step, fps=True)

    def relabel_with_mppi_post(
        self, num_trajs_to_relabel, batch_size=32, select_from_end=True
    ):
        # Relabel all transitions in replay buffer with MPPI
        num_replay_trajs = len(self.replay_buffer.keys())
        if num_trajs_to_relabel > num_replay_trajs:
            print("num_trajs_to_relabel > num_replay_trajs, relabelling all trajs")
            to_relable_keys = list(self.replay_buffer.keys())
        elif select_from_end:
            print(
                f"Num Trajs in Replay Buffer: {num_replay_trajs}. Relabling last {num_trajs_to_relabel} trajectories"
            )
            all_keys = sorted(list(self.replay_buffer.keys()))
            to_relable_keys = all_keys[-num_trajs_to_relabel:]
        else:
            print(
                f"Num Trajs in Replay Buffer: {num_replay_trajs}. Relabelling {num_trajs_to_relabel} trajectories"
            )
            to_relable_keys = np.random.choice(
                list(self.replay_buffer.keys()), num_trajs_to_relabel, replace=False
            )

        start_time = time.time()
        relabelled_buffer = collections.OrderedDict()

        # Group keys into batches
        batched_trajkeys = [
            to_relable_keys[i : i + batch_size]
            for i in range(0, len(to_relable_keys), batch_size)
        ]

        cprint(
            f"Number of Batches: {len(batched_trajkeys)}, Batch Size: {batch_size}",
            "yellow",
        )

        for idx, trajkeys in enumerate(batched_trajkeys):
            # Copy as is to relabelled buffer
            data_keys = None
            for trajkey in trajkeys:
                data_traj_i = copy.deepcopy(self.replay_buffer[trajkey])
                data_traj_i["base_action"] = np.stack(data_traj_i["base_action"])
                relabelled_buffer[trajkey] = data_traj_i
                if data_keys is None:
                    data_keys = data_traj_i.keys()

            batch_traj_lens = [
                len(self.replay_buffer[key]["state"]) for key in trajkeys
            ]

            all_mppi_actions = {key: [] for key in trajkeys}
            latent = None
            action = None
            for i in range(max(batch_traj_lens)):
                # If traj is over select last index as dummy
                idx_to_select = []
                dones = []
                for key_idx, key in enumerate(trajkeys):
                    traj_len = batch_traj_lens[key_idx]
                    if i < traj_len:
                        idx_to_select.append(i)
                        dones.append(False)
                    else:
                        idx_to_select.append(traj_len - 1)
                        dones.append(True)

                # Construct stacked batch
                obs_dreamer = {}
                for data_key in data_keys:
                    data = []
                    for key_idx, trajkey in enumerate(trajkeys):
                        data.append(
                            relabelled_buffer[trajkey][data_key][idx_to_select[key_idx]]
                        )
                    obs_dreamer[data_key] = np.expand_dims(np.stack(data), axis=1)
                # Obs dreamer data of shape batch_n_trajs x 1 x ...

                # WM pass to get latents
                obs_dreamer = self.dreamer_class._wm.preprocess(obs_dreamer)
                embed = self.dreamer_class._wm.encoder(
                    obs_dreamer
                )  # BS x BL x (1024 + encoding_dim)
                embed = embed.squeeze(1)  # Remove BL dim
                latent, _ = self.dreamer_class._wm.dynamics.obs_step(
                    latent, action, embed, obs_dreamer["is_first"]
                )

                # MPPI actions
                base_action = []
                for key_idx, key in enumerate(trajkeys):
                    base_action.append(
                        relabelled_buffer[key]["base_action"][idx_to_select[key_idx]][
                            ..., : self.config.pred_horizon
                        ]
                    )
                base_action = np.stack(base_action)
                with torch.no_grad():
                    mppi_actions = self.dreamer_class._task_behavior.mppi_actions(
                        latent=latent,
                        base_action=torch.tensor(
                            base_action,
                            dtype=torch.float32,
                            device=self.config.device,
                        ),
                    )
                    mppi_actions = mppi_actions.cpu().numpy()

                # Append if traj is not done
                for key_idx, key in enumerate(trajkeys):
                    if not dones[key_idx]:
                        all_mppi_actions[key].append(mppi_actions[key_idx])

                # Force the rollout to be whatever it was in the data
                latent = {k: v.detach() for k, v in latent.items()}
                action = []
                for key_idx, key in enumerate(trajkeys):
                    action.append(
                        torch.tensor(
                            relabelled_buffer[key]["action"][idx_to_select[key_idx]],
                            dtype=torch.float32,
                            device=self.config.device,
                        )
                    )
                action = torch.stack(action)

            # Insert into relabelled buffer
            for key in trajkeys:
                all_mppi_actions[key] = np.stack(all_mppi_actions[key])
                relabelled_buffer[key]["residual_action"] = all_mppi_actions[key]
                summed_action = np.clip(
                    relabelled_buffer[key]["base_action"][
                        ..., : self.config.pred_horizon
                    ]
                    + all_mppi_actions[key],
                    -1,
                    1,
                )
                relabelled_buffer[key]["action"] = summed_action

            print(
                f"[{idx+1}/{len(batched_trajkeys)}] Relabelled Batch of Trajectories, max_traj_len: {max(batch_traj_lens)}"
            )

        print("Finished Relabelling with MPPI in ", time.time() - start_time)
        return relabelled_buffer

    def get_dp_training_buffer(self, num_trajs_to_keep):
        """
        Drop drop_ratio fraction of trajectories from the replay buffer
        """
        cprint("\nDropping Trajectories from Replay Buffer", "green")
        total_trajectories = len(self.replay_buffer.keys())

        # If no trajectories to drop, return
        if total_trajectories <= num_trajs_to_keep:
            print(
                f"Total Trajectories: {total_trajectories} <= num_trajs_to_keep: {num_trajs_to_keep}, not subsampling"
            )
            return

        print(
            f"Subsampling {num_trajs_to_keep} trajectories from {total_trajectories} trajectories"
        )

        keys_to_drop = list(self.replay_buffer.keys())[
            : total_trajectories - num_trajs_to_keep
        ]

        for key in keys_to_drop:
            self.replay_buffer.pop(key)

        print(
            f"After Subsampling Trajectories, Total Trajectories in Buffer: {len(self.replay_buffer.keys())}\n"
        )

    def eval_base_policy(self, prefix, round_id, base_policy):
        cprint("\nEvaluating Base Policy", "green")
        # Evaluate the base policy
        eval_policy_loss, sr, reward, episode_len = base_policy.eval_policy(
            eval_envs=self.eval_envs,
            expert_val_eps=self.expert_val_eps,
            step=prefix,
        )
        # Log metrics
        if self.logger is not None:
            num_buffer_transitions = count_n_transitions(self.replay_buffer)
            self.logger.scalar(f"eval/dp_l2_loss", eval_policy_loss)
            self.logger.scalar(f"eval/dp_success_rate", sr)
            self.logger.scalar(f"eval/dp_reward", reward)
            self.logger.scalar(f"eval/dp_episode_len", episode_len)
            self.logger.scalar(f"train/num_buffer_transitions", num_buffer_transitions)
            self.logger.scalar(f"train/env_step", self._env_step)
            self.logger.scalar(f"train/n_round", round_id)
            self.logger.write(step=self._step, fps=True)

    def eval_mppi_policy(self, prefix, round_id):
        cprint("\nEvaluating MPPI Policy", "green")
        self.residual_policy.evaluate_agent(step_name=prefix, step=self._step)

        if self.logger is not None:
            num_buffer_transitions = count_n_transitions(self.replay_buffer)
            self.logger.scalar(f"train/num_buffer_transitions", num_buffer_transitions)
            self.logger.scalar(f"train/n_round", round_id)
            self.logger.scalar(f"train/env_step", self._env_step)
            self.logger.write(step=self._step, fps=True)

    def trim_buffer(self, buffer):
        """
        Trim the buffer to the desired length
        """
        desired_transitions = self.config.num_buffer_transitions
        current_transitions = count_n_transitions(buffer)
        if current_transitions > desired_transitions:
            print(
                f"Trimming buffer from {current_transitions} to {desired_transitions}"
            )

        # Sort the keys by time collected
        all_keys = sorted(list(buffer.keys()))

        # Keep deleting keys till the desired transitions are reached
        to_delete = current_transitions - desired_transitions
        deleted_traj_count = 0
        while to_delete > 0:
            key = all_keys.pop(0)
            len_traj = len(buffer[key]["state"])
            del buffer[key]
            to_delete -= len_traj
            deleted_traj_count += 1

        print(
            f"Deleted {deleted_traj_count} trajectories from buffer, final transitions: {count_n_transitions(buffer)}"
        )

    def warm_start_wm(self):
        cprint(
            "\n-------------Warmstarting WM + Critic-------------",
            "green",
            attrs=["bold"],
        )
        num_steps_to_collect = int(
            self.config.train_dp_mppi_params["warmstart_percentage_env_steps"]
            * self.config.train_dp_mppi_params["n_env_steps"]
        )
        num_warmstart_itrs = int(
            num_steps_to_collect
            * self.config.train_dp_mppi_params["warmstart_train_ratio"]
        )
        cprint(
            f"Number of steps to collect for warmstart: {num_steps_to_collect}\
            \nNumber of warm start itrs: {num_warmstart_itrs}",
            "green",
        )

        if num_warmstart_itrs <= 0:
            cprint(f"Skipping warmstart as num_warmstart_itrs <= 0", "yellow")
            return

        cprint(
            "Collecting warm start trajectories...",
            "yellow",
        )
        collect_onpolicy_trajs(
            num_steps=num_steps_to_collect,
            max_traj_len=self.config.time_limit if not self.config.debug else 10,
            base_policy=DiffusionPolicyAgent(
                config=self.config,
                diffusion_policy=self.base_policy,
                noise_std=self.config.train_dp_mppi_params["data_collect_noise_std"],
            ),
            train_env=self.train_env,
            pred_horizon=self.config.pred_horizon,
            obs_horizon=self.config.obs_horizon,
            train_eps=self.replay_buffer,
            save_dir=None,
            state_only=self.config.state_only,
        )

        label_expert_eps(
            expert_eps=self.expert_eps,
            dreamer_class=self.dreamer_class,
        )
        self.train_wm_critic(itrs=num_warmstart_itrs)
        self._env_step += count_n_transitions(self.replay_buffer)

    def collect_trajs(self):
        init_transitions = count_n_transitions(self.replay_buffer)
        n_steps_collected = self.residual_policy.collect_residual_onpolicy_trajs(
            num_steps=self.config.train_dp_mppi_params["min_env_steps_per_round"],
            buffer=self.replay_buffer,
        )

        final_transitions = count_n_transitions(self.replay_buffer)
        self._env_step += final_transitions - init_transitions
        return n_steps_collected

    def train_dp_with_mppi(self):
        self.warm_start_wm()
        self.eval_base_policy(prefix="init", round_id=-1, base_policy=self.base_policy)
        relabelled_buffer = collections.OrderedDict()

        for round_id in range(10000000):
            cprint(
                f"\n-------------Starting Round: {round_id} | Num Env Steps: {self._env_step}-------------",
                "green",
                attrs=["bold"],
            )
            start_time = time.time()

            if self._env_step >= self.config.train_dp_mppi_params["n_env_steps"]:
                print(
                    f"Reached max env steps: {self.config.train_dp_mppi_params['n_env_steps']}. Stopping training."
                )

                # Eval MPPI policy
                self.eval_mppi_policy(prefix=f"round_{round_id}", round_id=round_id)

                # Eval Base Policy
                self.eval_base_policy(
                    prefix=f"round_{round_id}",
                    round_id=round_id,
                    base_policy=self.base_policy,
                )

                # Save Checkpoint
                save_dir = self.config.logdir / "latest_residual_checkpoint.pt"
                self.dreamer_class.save_checkpoint(path=save_dir)

                ckpt_file = self.config.logdir / f"latest_base_policy.pt"
                self.base_policy.trainer.save_checkpoint(
                    ckpt_file, global_step=self._step
                )

                break

            # Need to relabel with new base policy
            label_expert_eps(
                expert_eps=self.expert_eps,
                dreamer_class=self.dreamer_class,
            )

            # Collect data in the environment
            n_steps_collected = self.collect_trajs()

            # Trim the buffer to the desired length
            self.trim_buffer(self.replay_buffer)

            # Train WM + Critic
            cprint(f"\nStarting WM + Critic Training at Round: {round_id}", "green")
            self.train_wm_critic(
                itrs=int(
                    self.config.train_dp_mppi_params["rounds_train_ratio"]
                    * n_steps_collected
                )
            )

            if round_id % self.config.train_dp_mppi_params["eval_every_round"] == 0:
                # Evaluate MPPI Policy
                self.eval_mppi_policy(prefix=f"round_{round_id}", round_id=round_id)

                # Save Checkpoint
                save_dir = self.config.logdir / "latest_residual_checkpoint.pt"
                self.dreamer_class.save_checkpoint(path=save_dir)

                ckpt_file = self.config.logdir / f"latest_base_policy.pt"
                self.base_policy.trainer.save_checkpoint(
                    ckpt_file, global_step=self._step
                )

            # Update DP every update_dp_every rounds
            if (
                round_id % self.config.train_dp_mppi_params["update_dp_every"] == 0
                and round_id > 0
                and self._env_step
                <= int(0.95 * self.config.train_dp_mppi_params["n_env_steps"])
            ):
                # Relabel all transitions in replay buffer with MPPI
                cprint(f"\nBegin Relabelling with MPPI at Round: {round_id}", "green")
                relabelled_buffer_curr = self.relabel_with_mppi_post(
                    num_trajs_to_relabel=self.config.train_dp_mppi_params[
                        "n_traj_to_relabel_per_round"
                    ]
                )
                # Add to dp_train_buffer
                relabelled_buffer.update(relabelled_buffer_curr)

                # Keep only last n_dp_traj_buffer_size trajectories in the buffer
                sorted_keys = sorted(list(relabelled_buffer.keys()))
                if (
                    len(sorted_keys)
                    > self.config.train_dp_mppi_params["n_dp_traj_buffer_size"]
                ):
                    keys_to_keep = sorted_keys[
                        -self.config.train_dp_mppi_params["n_dp_traj_buffer_size"] :
                    ]
                    relabelled_buffer = {k: relabelled_buffer[k] for k in keys_to_keep}
                    print(
                        f"Trimmed dp_train_buffer to {len(relabelled_buffer)} trajectories"
                    )

                print(
                    "Num Trajectories in DP Train Buffer: ",
                    len(relabelled_buffer.keys()),
                )

                # Train DP (reinitialize)
                cprint(f"\nTraining DP with MPPI at Round: {round_id}", "green")
                dp_dataset = make_retrain_dp_dataset(
                    replay_buffer=relabelled_buffer,
                    expert_eps=self.expert_eps,
                    config=self.config,
                )
                self._step = self.base_policy.train_base_policy(
                    train_dataset=dp_dataset,
                    expert_val_eps=self.expert_val_eps,
                    eval_envs=self.eval_envs,
                    init_step=self._step,
                    train_steps=self.config.train_dp_mppi_params["n_dp_train_itrs"],
                    log_prefix="base_dp",
                    run_eval=False,
                )

            if round_id % self.config.train_dp_mppi_params["eval_every_round"] == 0:
                # Evaluate Base Policy
                self.eval_base_policy(
                    prefix=f"round_{round_id}",
                    round_id=round_id,
                    base_policy=self.base_policy,
                )

            print(f"Round {round_id} took {time.time() - start_time} seconds")
            if self.logger is not None:
                self.logger.scalar(f"round_time", time.time() - start_time)

        if self.config.train_dp_mppi_params["perform_final_policy_iteration"]:
            cprint("\nPerforming Final Policy Iteration", "green")
            self.final_policy_iteration()

    def final_policy_iteration(self):
        relabelled_buffer = self.relabel_with_mppi_post(
            num_trajs_to_relabel=self.config.train_dp_mppi_params[
                "final_policy_iteration_num_trajs"
            ],
            select_from_end=True,
        )

        # Reset diffusion policy trainer
        del self.base_policy.trainer
        self.base_policy.initialize_trainer()
        self.base_policy.name = "DP_Final_Policy_Iteration"

        cprint(
            f"\nStarting Final Policy Iteration with {len(relabelled_buffer.keys())} trajectories and "
            f"{self.config.train_dp_mppi_params['final_policy_iteration_num_steps']} steps",
            "green",
        )

        # Train DP with the relabelled buffer
        dp_dataset = make_retrain_dp_dataset(
            replay_buffer=relabelled_buffer,
            expert_eps=self.expert_eps,
            config=self.config,
        )
        self._step = self.base_policy.train_base_policy(
            train_dataset=dp_dataset,
            expert_val_eps=self.expert_val_eps,
            eval_envs=self.eval_envs,
            init_step=self._step,
            train_steps=self.config.train_dp_mppi_params[
                "final_policy_iteration_num_steps"
            ],
            log_prefix="final_policy_iteration",
            run_eval=True,
        )
