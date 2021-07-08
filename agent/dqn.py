import os
import sys
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

from environment.New_Parallel_Machine import Forming
from environment.Steelplate import *


class DQN(tf.keras.Model):
    def __init__(self, action_size, state_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(128, activation='relu', input_shape=state_size)
        self.fc2 = Dense(64, activation='relu')
        self.fc3 = Dense(32, activation='relu')
        self.fc_out = Dense(action_size, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        q = self.fc_out(x)
        return q


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 5e-5
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.batch_size = 300
        self.train_start = 1000

        self.memory = deque(maxlen=3000)

        self.model = DQN(action_size, state_size)
        self.target_model = DQN(action_size, state_size)
        self.optimizer = Adam(lr=self.learning_rate)

        self.update_target_model()


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            return np.argmax(q_value[0])


    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array([sample[2] for sample in mini_batch])
        next_states = np.array([sample[3] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action*predicts, axis=1)

            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

        return loss

if __name__ == "__main__":
    model_path = os.path.join(os.getcwd(), 'save_model', 'model')

    writer = tf.summary.create_file_writer('./summary/dqn/train')

    df, index, process_all, process_list, m_type_dict, machine_num = import_steel_plate_schedule(
        '../environment/data/forming_data.csv')
    steel_plate = SteelPlate(df, index, process_list, m_type_dict)
    env = Forming(parts=steel_plate.parts, process_all=process_all, machine_num=machine_num)

    state_dim = 64
    state_size = (state_dim,)
    action_size = 4
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    score_avg = 0

    num_episode = 50000

    for e in range(num_episode):
        action_list = []
        done = False
        score = 0

        state = env.reset()
        state = np.reshape(state, [1, state_dim])

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_dim])

            score += reward
            agent.append_sample(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                loss = agent.train_model()

            state = next_state

            action_list.append(action)

            if done:
                agent.update_target_model()

                score_avg = 0.9*score_avg + 0.1*score if score_avg != 0 else score
                print("episode: {:4d} | score_avg: {:5.4f} | memory_length: {:4d} | epsilon: {:.4f}".format(
                    e, score_avg, len(agent.memory), agent.epsilon
                ))

                print(action_list)

                scores.append(score)
                episodes.append(e)

                # Performance index
                mean_weighted_tardiness = env.mean_weighted_tardiness
                make_span = env.make_span

                with writer.as_default():
                    tf.summary.scalar("Perf/Reward", score, step=e)
                    tf.summary.scalar("Perf/Mean weighted tardiness", mean_weighted_tardiness, step=e)
                    tf.summary.scalar("Perf/Make span", make_span, step=e)
                    tf.summary.scalar("Loss/Total loss", loss, step=e)

                if e % 250 == 0:
                    agent.model.save_weights("./save_model/model", save_format="tf")


