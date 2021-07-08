import os
import time
import threading
import random
import numpy as np
import tensorflow as tf

from tensorflow.compat.v1.train import AdamOptimizer
from tensorflow.keras.layers import Conv2D, Flatten, Dense


# ActorCritic 인공신경망
class ActorCritic(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(ActorCritic, self).__init__()

        self.fc1 = Dense(128, activation='relu', input_shape=state_size)
        self.fc2 = Dense(64, activation= 'relu')
        self.fc3 = Dense(32, activation= 'relu')
        self.policy = Dense(action_size, activation='linear')
        self.value = Dense(1, activation='linear')

    # Enable class object to be callable
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        policy = self.policy(x)
        value = self.value(x)
        return policy, value

