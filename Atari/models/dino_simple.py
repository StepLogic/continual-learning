import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from .dino_encoder import DinoEncoder


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DinoSimpleAgent(nn.Module):
    def __init__(self, envs, dino_size, frame_stack, device):
        super().__init__()
        self.dino = DinoEncoder(dino_size=dino_size, device=device)
        self.middle = nn.Sequential(
            layer_init(nn.Linear(self.dino.embed_dim * frame_stack, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        h = self.middle(self.dino.encode(x * 255.0))  # denormalize
        return self.critic(h)

    def get_action_and_value(self, x, action=None, deterministic=False):
        hidden = self.middle(self.dino.encode(x * 255.0))  # denormalize
        logits = self.actor(hidden)
        if deterministic:
            action = logits.argmax(dim=-1)
            probs = Categorical(logits=logits)
        else:
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
        logprob = probs.log_prob(action)
        entropy = probs.entropy()
        return action, logprob, entropy, self.critic(hidden)

    def save(self, dirname, model_name="model"):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.middle, f"{dirname}/{model_name}_middle.pt")
        torch.save(self.actor, f"{dirname}/{model_name}_actor.pt")
        torch.save(self.critic, f"{dirname}/{model_name}_critic.pt")

    @classmethod
    def load(cls, dirname, envs, dino_size="s", frame_stack=4, device="cpu", model_name="model"):
        model = cls(envs=envs, dino_size=dino_size, frame_stack=frame_stack, device=device)
        model.middle = torch.load(f"{dirname}/{model_name}_middle.pt", weights_only=False)
        model.actor = torch.load(f"{dirname}/{model_name}_actor.pt", weights_only=False)
        model.critic = torch.load(f"{dirname}/{model_name}_critic.pt", weights_only=False)
        return model
