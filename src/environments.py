import cv2
import numpy as np
import gym
import gym_super_mario_bros
import gym.spaces
import warnings
warnings.filterwarnings("ignore")

from collections import deque
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv


NAME_MAPPER = {
    'Pong': 'PongNoFrameskip-v4',
    'Breakout': 'BreakoutNoFrameskip-v4',
    'SuperMarioBros level 1': 'SuperMarioBrosNoFrameskip-1-1-v0', 
    'SuperMarioBros level 2': 'SuperMarioBrosNoFrameskip-1-2-v0',
}

SUPPORTED_ENVIRONMENTS = list(NAME_MAPPER.keys())


def make_environment(name):
    """ A factory function to create wrapped environments """

    assert name in SUPPORTED_ENVIRONMENTS
    name = NAME_MAPPER[name]

    if 'Pong' in name:
        env = gym.make(name)
        env = CustomPongActionSpace(env)
    elif 'Breakout' in name:
        env = gym.make(name)
        env = CustomBreakoutActionSpace(env)
    elif 'SuperMarioBros' in name:
        env = gym_super_mario_bros.make(name)
        env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
        env = CustomMarioReward(env)

    env = PreprocessFrames(env, num_skip=4, size=84)
    env = StackFrames(env, num_stack=4)
    env = NormalizeObservation(env)
    
    return env


class CustomPongActionSpace(gym.ActionWrapper):
    """ Reduces the Pong action space to 3 actions """

    def __init__(self, env):
        super(CustomPongActionSpace, self).__init__(env)
        self.action_space = gym.spaces.Discrete(n=3)

    def action(self, action):
        return action if action == 0 else action+1


class CustomBreakoutActionSpace(gym.ActionWrapper):
    """ Reduces the Breakout action space to 3 actions """

    def __init__(self, env):
        super(CustomBreakoutActionSpace, self).__init__(env)
        self.action_space = gym.spaces.Discrete(n=3)

    def action(self, action):
        return action+1


class CustomMarioReward(gym.RewardWrapper):
    """ Reduces the Mario reward to be between 0 ad 1 """

    def reward(self, reward):
        return reward / 15.0


class PreprocessFrames(gym.Wrapper):
    """ Skip, normalize and resize original frames from the environment """

    def __init__(self, env, num_skip=4, size=84):
        super(PreprocessFrames, self).__init__(env)
        self.num_skip = num_skip
        self.size = size
        self.unwrapped.original_size = self.env.observation_space.shape

    def preprocess_frame(self, frame):
        frame = np.mean(frame, axis=2)
        frame = cv2.resize(frame, (self.size, self.size))
        frame = frame.astype(np.float32) / 255.0
        frame = frame.reshape(1, self.size, self.size)
        return frame

    def step(self, action):
        _observation, reward, done, info = self.env.step(action)
        actual_steps = 1
        while actual_steps < self.num_skip and not done:
            _observation, _reward, done, info = self.env.step(action)
            reward += _reward
            actual_steps += 1
        self.unwrapped.original_observation = _observation.copy()
        observation = self.preprocess_frame(_observation.copy())
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.unwrapped.original_observation = observation
        return self.preprocess_frame(observation)


class StackFrames(gym.Wrapper):
    """ Stack consecutive frames together """

    def __init__(self, env, num_stack=4):
        super(StackFrames, self).__init__(env)
        self.num_stack = num_stack
        self.stack = deque(maxlen=num_stack)

    def step(self, action):
        _observation, reward, done, info = self.env.step(action)
        self.stack.append(_observation)
        observation = np.concatenate(self.stack, axis=0)
        return observation, reward, done, info

    def reset(self):
        _observation = self.env.reset()
        for i in range(self.stack.maxlen):
            self.stack.append(_observation)
        observation = np.concatenate(self.stack, axis=0)
        return observation


class NormalizeObservation(gym.ObservationWrapper):
    """ Normalize observations through time """

    def __init__(self, env):
        super(NormalizeObservation, self).__init__(env)
        self.mean = 0
        self.std = 0
        self.num_steps = 0
        self.alpha = 0.9999

    def observation(self, observation):
        self.num_steps += 1
        self.mean = self.mean * self.alpha + observation.mean() * (1 - self.alpha)
        self.std = self.std * self.alpha + observation.std() * (1 - self.alpha)
        mean = self.mean / (1 - pow(self.alpha, self.num_steps))
        std = self.std / (1 - pow(self.alpha, self.num_steps))
        observation = (observation - mean) / std
        return observation

