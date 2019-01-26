import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from matplotlib.ticker import FormatStrFormatter
from collections import deque

class Memory:
    """ A simple data structure to store agent experiences """

    def __init__(self):
        self.memory = {}

    def reset(self):
        self.memory = {}

    def append(self, update):
        for key in update:
            if key not in self.memory:
                self.memory[key] = []
            self.memory[key].append(update[key])

    def get_last(self, key):
        return self.memory[key][-1]

    def get_all(self, key):
        try:
            return torch.cat(self.memory[key])
        except TypeError:
            return np.asarray(self.memory[key])


class Recorder():
    """ A class to display frames from the environment as an animation """

    def __init__(self):
        self.frames = []
        self.animation = None

    def reset(self):
        self.frames = []
        self.animation = None

    def record(self, environment):
        self.frames.append(environment.unwrapped.original_observation)

    def stop(self):
        print("Preparing the animation...")
        patch = plt.imshow(self.frames[0])
        plt.axis('off')

        def animation_step(i):
            patch.set_data(self.frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animation_step, frames=len(self.frames), interval=100)
        self.animation = HTML(anim.to_jshtml())

    def replay(self):
        return self.animation


def load_checkpoint(network, optimizer, checkpoint):
    """ Util function to load pretrained models """
    
    assert os.path.exists(checkpoint)
    checkpoint = torch.load(checkpoint)
    for key in network:
        network[key].load_state_dict(checkpoint[key])
    optimizer.load_state_dict(checkpoint['optimizer'])

