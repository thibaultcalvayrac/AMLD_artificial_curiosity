import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import lfilter
from collections import deque

class ObservationEncoder(nn.Module):
    """ A convolutional neural network with 5 convolutions to encode observations from the environment """

    def __init__(self, input_channels, activation_function=F.elu):
        
        super(ObservationEncoder, self).__init__()
        self.c1 = nn.Conv2d(input_channels, 32, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.c4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.c5 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.activation_function = activation_function

    def forward(self, observation):
        observation = torch.Tensor([observation])
        features = F.elu(self.c1(observation))
        features = F.elu(self.c2(features))
        features = F.elu(self.c3(features))
        features = F.elu(self.c4(features))
        features = self.c5(features)
        if self.activation_function is not None:
            features = self.activation_function(features)
        features = features.view(-1, 32*3*3)
        return features


class ActorCritic(nn.Module):
    """ A recurrent neural network with two heads: an actor that infers the policy and a critic that estimates the value function """

    def __init__(self, num_actions, input_channels=4, num_features=288, internal_state_dim=256, gamma=0.99):
        
        super(ActorCritic, self).__init__()
        self.internal_state_dim = internal_state_dim
        self.gamma = gamma

        self.encoder = ObservationEncoder(input_channels)
        self.lstm = nn.LSTMCell(num_features, internal_state_dim)
        self.actor = nn.Linear(internal_state_dim, num_actions)
        self.critic = nn.Linear(internal_state_dim, 1)

        def zero_weight(shape):
            return torch.zeros(shape, requires_grad=True)
        self.actor.weight.data = zero_weight(self.actor.weight.size())
        self.actor.bias.data = zero_weight(self.actor.bias.size())

        self.internal_state = (torch.zeros(1, self.internal_state_dim), torch.zeros(1, self.internal_state_dim))

    def forward(self, observation):
        features = self.encoder(observation)
        self.internal_state = self.lstm(features, self.internal_state)

        policy = self.actor(self.internal_state[0])
        value = self.critic(self.internal_state[0])

        log_policy = F.log_softmax(policy, dim=-1)
        action = torch.exp(log_policy).multinomial(num_samples=1).data[0]
        
        return value, log_policy, action

    def reset_internal_state(self):
        self.internal_state = (torch.zeros(1, self.internal_state_dim), torch.zeros(1, self.internal_state_dim))

    def detach_internal_state(self):
        self.internal_state = (self.internal_state[0].detach(), self.internal_state[1].detach())

    def loss(self, memory):

        values = memory.get_all('value')
        log_policies = memory.get_all('log_policy')
        actions = memory.get_all('action')
        rewards = memory.get_all('reward')

        factor = len(actions) / 20.0

        def discount(x, gamma): return lfilter([1], [1, -gamma], x[::-1])[::-1]

        np_values = values.view(-1).data.numpy()
        rewards[-1] += self.gamma * np_values[-1]
        discounted_r = discount(np.asarray(rewards), self.gamma)
        
        sampled_log_policy = log_policies.gather(1, actions.view(-1, 1))
        advantages = discounted_r - np_values[:-1]

        policy_loss = - factor * (sampled_log_policy.view(-1) * torch.Tensor(advantages.copy()).float()).sum()
        value_loss = 0.5 * factor * F.mse_loss(torch.Tensor(discounted_r.copy()).float(), values[:-1, 0])
        entropy_loss = - 5e-4 * (-log_policies * torch.exp(log_policies)).sum()
        loss = policy_loss + value_loss + entropy_loss
        return loss


class IntrinsicCuriosityModule(nn.Module):
    """ A module made of two neural networks that produces intrinsic motivation:
    - an inverse model that predicts the action performed between two consecutive observations
    - a forward model that predicts the next observation features given the last observation and the last action """

    def __init__(self, num_actions, input_channels=4, num_features=288, normalize_curiosity=True):

        super(IntrinsicCuriosityModule, self).__init__()
        self.observation_encoder = ObservationEncoder(input_channels, activation_function=None)

        n_hidden = 256
        self.forward_model_1 = nn.Linear(num_features + num_actions, n_hidden)
        self.forward_model_ = nn.Linear(n_hidden, n_hidden)
        self.forward_model_2 = nn.Linear(n_hidden, num_features)
        self.inverse_model_1 = nn.Linear(2 * num_features, n_hidden)
        self.inverse_model_ = nn.Linear(n_hidden, n_hidden)
        self.inverse_model_2 = nn.Linear(n_hidden, num_actions)
        self.num_features = num_features
        self.num_actions = num_actions
        self.normalize_curiosity = normalize_curiosity
        self.past_curiosity = deque(maxlen=100000)

    def encode_observation(self, observation):
        features = self.observation_encoder(observation)
        return features

    def forward_model(self, last_features, last_action):
        action_one_hot = [1 if i == last_action else 0 for i in range(self.num_actions)]
        action_one_hot = torch.Tensor(action_one_hot).view(-1, self.num_actions)

        x = torch.cat((last_features, action_one_hot), dim=1)
        x = F.elu(self.forward_model_1(x))
        x = F.elu(self.forward_model_(x))
        predicted_features = self.forward_model_2(x)

        return predicted_features

    def inverse_model(self, last_features, features):

        x = torch.cat((last_features, features), dim=1)
        x = F.elu(self.inverse_model_1(x))
        x = F.elu(self.inverse_model_(x))
        predicted_action = self.inverse_model_2(x)

        return predicted_action

    def curiosity(self, predicted_features, features):
        curiosity = F.mse_loss(predicted_features, features.detach()).detach().numpy()
        if self.normalize_curiosity:
            self.past_curiosity.append(curiosity)
            if np.std(self.past_curiosity) != 0:
                curiosity /= np.std(self.past_curiosity)
        return curiosity

    def loss(self, memory):

        predicted_features = memory.get_all('predicted_features')
        predicted_actions = memory.get_all('predicted_action')
        features = memory.get_all('features')[1:]
        actions = memory.get_all('action')

        factor = 10.0 * len(actions) / 20.0
        forward_loss = factor * F.mse_loss(predicted_features, features.detach())
        inverse_loss = factor * F.cross_entropy(predicted_actions, actions)
        loss = forward_loss + inverse_loss
        return loss

