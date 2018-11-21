import random
import numpy as np
import matplotlib.pyplot as plt
import gym
from collections import deque 
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers import Activation, Input
from keras.layers.merge import Add
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras import backend as K
import keras
import tensorflow as tf
from tqdm import tqdm
from itertools import count
from prioritized_memory import SumTree, Memory


class DQN:

    def __init__(self, state_size, action_size):
        self.wt = 1
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return self.wt * K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        inputs = Input(shape=(self.state_size, ))
        net = Dense(24, input_dim=self.state_size, activation='relu')(inputs)
        net = Dense(24, activation='relu')(net)
        advt = Dense(24, activation='relu')(net)
        advt = Dense(self.action_size)(advt)
        value = Dense(24, activation='relu')(net)
        value = Dense(1)(value)
        advt = Lambda(lambda advt: advt - tf.reduce_mean(
            advt, axis=-1, keep_dims=True))(advt)
        value = Lambda(lambda value: tf.tile(value, [1, self.action_size]))(value)
        final = Add()([value, advt])
        model = Model(
            inputs=inputs,
            outputs=final)
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, memory, batch_size):
        i = 0
        abs_err = []
        idx, batch, weights = memory.sample(batch_size)
        for state, action, reward, next_state, done in batch:
            self.wt = weights[i]
            i += 1
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                _t = self.target_model.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(_t)
            abs_err.append(np.abs(np.sum(self.model.predict(state) - target)))
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        memory.batch_update(idx, np.array(abs_err))

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == '__main__':
    memory_size = 100000
    pretrain_length = 100000
    memory = Memory(memory_size)
    env = gym.make('CartPole-v1')
    state = env.reset()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print('Building randomized priority tree', end='')
    for i in range(pretrain_length):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        state = np.reshape(state, [1, state_size])
        next_state = np.reshape(next_state, [1, state_size])
        memory.store((state, action, reward, next_state, done))
        state = next_state
        if done:
            env.reset()
    agent = DQN(state_size, action_size)
    done = False
    batch_size = 32
    EPISODES = 5000
    with tqdm(total=EPISODES) as pbar:
        for e in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            for time in range(500):
                env.render()
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                memory.store((state, action, reward, next_state, done))
                state = next_state
                if done:
                    agent.update_target_model()
                    pbar.set_description("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
                    break
                agent.replay(memory, batch_size)
            pbar.update(1)
