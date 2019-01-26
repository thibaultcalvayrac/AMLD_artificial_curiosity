import numpy as np
import torch

from environments import make_environment
from networks import ActorCritic, IntrinsicCuriosityModule
from utils import Memory, load_checkpoint

class ActorCriticAgent:
    """ Advantage Actor Critic agent """

    def __init__(self, num_actions, checkpoint=None):
        self.network, self.trainable_parameters = self.init_network(num_actions)
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=1e-4)
        self.memory = Memory()
        if checkpoint is not None:
            load_checkpoint(self.network, self.optimizer, checkpoint)

    def init_network(self, num_actions):
        
        network = {'actor_critic': ActorCritic(num_actions)}
        trainable_parameters = list(network['actor_critic'].parameters())
        return network, trainable_parameters

    def play(self, environment, max_games=1, max_steps=500, train=False, verbose=False, recorder=None):
        
        n_steps = 0
        n_games = 0
        current_game_infos = {'game': n_games + 1, 'reward': 0, 'game_duration': 0}
        observation = environment.reset()
        if recorder is not None:
            recorder.reset()
            recorder.record(environment)

        while (n_steps < max_steps) and (n_games < max_games):
            
            self.init_rollout(observation)
            for rollout_step in range(20):
                
                value, log_policy, action = self.network['actor_critic'](observation)
                self.memory.append({'value': value, 'log_policy': log_policy, 'action': action})
                
                observation, extrinsic_reward, is_game_over, infos = environment.step(action.numpy()[0])
                if recorder is not None:
                    recorder.record(environment)

                reward = self.get_reward(observation, extrinsic_reward)
                self.memory.append({'reward': reward})
                
                current_game_infos['reward'] += reward
                current_game_infos['game_duration'] += 1
                n_steps += 1

                if is_game_over:
                    n_games += 1
                    print(current_game_infos)
                    current_game_infos = {'game': n_games + 1, 'reward': 0, 'game_duration': 0}
                    observation = environment.reset()
                    break
            
            self.end_rollout(observation, is_game_over)
            if verbose:
                print(current_game_infos)
            
            if train:
                loss = self.compute_loss()
                self.backpropagate(loss)

        if recorder is not None:
            recorder.stop()

    def init_rollout(self, observation):
        
        self.memory.reset()
        self.network['actor_critic'].detach_internal_state()

    def end_rollout(self, observation, is_game_over):
        
        if is_game_over:
            next_value = torch.Tensor([[0]])
            self.network['actor_critic'].reset_internal_state()
        else:
            next_value = self.network['actor_critic'](observation)[0].detach()
        self.memory.append({'value': next_value})

    def get_reward(self, observation, extrinsic_reward):
        
        return np.clip(extrinsic_reward, -1, 1)

    def compute_loss(self):
        
        loss = self.network['actor_critic'].loss(self.memory)
        return loss

    def backpropagate(self, loss, max_gradient_norm=40):
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.trainable_parameters, max_gradient_norm)
        self.optimizer.step()


class CuriousActorCriticAgent(ActorCriticAgent):
    """ Advantage Actor Critic Agent using intrinsic reward """

    def init_network(self, num_actions):
        network, trainable_parameters = super().init_network(num_actions)
        network['icm'] = IntrinsicCuriosityModule(num_actions)
        trainable_parameters += list(network['icm'].parameters())
        return network, trainable_parameters

    def init_rollout(self, observation):
        
        super().init_rollout(observation)
        
        features = self.network['icm'].observation_encoder(observation)
        self.memory.append({'features': features})

    def end_rollout(self, observation, is_game_over):
        next_value = self.network['actor_critic'](observation)[0].detach()
        self.memory.append({'value': next_value})

    def get_reward(self, observation, extrinsic_reward):
        
        last_features = self.memory.get_last('features')
        last_action = self.memory.get_last('action')
        predicted_features = self.network['icm'].forward_model(last_features, last_action)
        
        features = self.network['icm'].encode_observation(observation)
        predicted_action = self.network['icm'].inverse_model(last_features, features)

        self.memory.append({'predicted_features': predicted_features, 'features': features, 'predicted_action': predicted_action})
        
        intrinsic_reward = self.network['icm'].curiosity(predicted_features, features)
        
        return np.clip(intrinsic_reward, -1, 1)

    def compute_loss(self):
        loss = super().compute_loss()
        loss += self.network['icm'].loss(self.memory)
        return loss
