import torch
import torch.nn as nn

from sailor.dreamer.networks import MLP


def get_residual_actor_net(config):
    """
    Input: state (dyn and stoch) + action from base policy
    Ouptut: residual action
    """
    feat_size = config.num_actions  # action output size of base policy

    # Add state size
    if config.dyn_discrete:
        feat_size += config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
        feat_size += config.dyn_stoch + config.dyn_deter

    residual_actor = MLP(
        feat_size,
        (config.num_actions,),
        config.residual_actor["layers"],
        config.units,
        config.act,
        config.norm,
        config.residual_actor["dist"],
        config.residual_actor["std"],
        config.residual_actor["min_std"],
        config.residual_actor["max_std"],
        absmax=config.residual_training["abs_residual"],
        temp=config.residual_actor["temp"],
        unimix_ratio=config.residual_actor["unimix_ratio"],
        outscale=config.residual_actor["outscale"],
        name="ResidualActor",
    )

    # Set weight and bias of residual actor.mean_layer to 0
    nn.init.constant_(residual_actor.mean_layer.weight, 0)
    nn.init.constant_(residual_actor.mean_layer.bias, 0)
    return residual_actor


class ActorNet(nn.Module):
    def __init__(self, config, base_policy):
        super(ActorNet, self).__init__()
        self.base_policy = base_policy

        self.residual_actor = get_residual_actor_net(config)
        self.config = config

    def reinit_residual_actor(self):
        # Randomly reinitialize the residual actor
        new_net = get_residual_actor_net(self.config)

        # Copy over parameters in new_net to self.residual_actor
        for old_param, new_param in zip(
            self.residual_actor.parameters(), new_net.parameters()
        ):
            old_param.data.copy_(new_param.data)

        del new_net

    def get_base_action(self, obs, weighting_in_base=True, get_full_action=False):
        with torch.no_grad():
            base_action = self.base_policy.get_action(
                obs, weighting=weighting_in_base, get_full_action=get_full_action
            )
        return base_action

    def get_action(self, obs, feat, weighting_in_base=True):
        """
        Obs: raw observations for forward pass through the base policy
        feat: features from the dynamics model for forward pass through the residual actor

        Returns
        - base_action: torch tensor
        - residual_action: distribution
        """
        base_action = self.get_base_action(obs, weighting_in_base)
        base_action = torch.tensor(base_action, device=feat.device, dtype=feat.dtype)

        cat_input = torch.cat([feat, base_action], dim=-1)
        residual_action = self.residual_actor(cat_input)
        return {"base_action": base_action, "residual_action": residual_action}

    def get_residual_dist(self, feat, base_action):
        """
        Get the distribution of the residual actor
        feat: ... x feat_size
        base_action: ... x base_action_size
        """
        cat_input = torch.cat([feat, base_action], dim=-1)
        return self.residual_actor(cat_input)

    def reset(self):
        self.base_policy.reset()

    def get_entropy(self, feat, base_action):
        """
        Get the entropy of the residual actor
        feat: ... x feat_size
        base_action: ... x base_action_size
        """
        # Cat across the last dimension
        cat_input = torch.cat([feat, base_action], dim=-1)
        actor_dist = self.residual_actor(cat_input)
        return actor_dist.entropy()

    def parameters(self):
        """
        We only optimize the residual actor, so we only return its parameters
        """
        return list(self.residual_actor.parameters())
