# -*- coding: utf-8 -*-
"""
Reinforcement Learning based Process Optimization and Strategy Development in
Conventional Tunneling; G.H. Erharter, T.F. Hansen, Z. Liu, T. Marcher
more publication info...

Geotechnician class is part of the environment and checks for stability
The DQNAgent is the actual reinforcement learning agent that "plays" the game
of tunnels. The DQNAgent - code is heavily based on the DQNAgent class of part
6 of the tutorial series "Reinforcement Learning w/ Python" by Sentdex
(Harrison Kinsley):
https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/
see also: https://www.youtube.com/sentdex

Created on Wed Jul  1 15:30:21 2020
code contributors: Georg H. Erharter, Tom F. Hansen
"""

#### disable warnings from tensorflow if desired
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
####

from collections import deque
from typing import Deque, Tuple

import numpy as np
import random

from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import RMSprop


class geotechnician:
    """calculations performed by the geotechnican"""

    def __init__(self):
        pass

    def face_pressure(self, tunnel_diameter: float, cutting_length: int,
                      rockmass_dict: dict) -> float:
        """ face pressure equation after Vermeer et al (2002) for open face
        tunnelling """
        unit_weight = rockmass_dict['spec. weight [N/m³]']
        cohesion = rockmass_dict['cohesion [Pa]']
        friction_angle = rockmass_dict['friction angle [°]']
        term1 = (2+3*(cutting_length/tunnel_diameter)**(6*np.tan(np.radians(friction_angle))))/(18*np.tan(np.radians(friction_angle)))-0.05
        term2 = cohesion / np.tan(np.radians(friction_angle))
        pf = unit_weight * tunnel_diameter * term1 - term2

        return pf

    def check_stability(self, sup_section: list, pos_excavation: float,
                        tunnel_diameter: float,
                        cutting_length: int, rockmass_dict: dict) -> float:
        """checks face stability, and returns action values"""
        # if excavation is within supported area pf = always negative
        if sup_section[pos_excavation] == 1:
            pf = -1
        else:
            pf = self.face_pressure(tunnel_diameter, cutting_length,
                                    rockmass_dict)
        return pf


class DQNAgent:
    """functionality to make, train and interact with the DQN agent"""

    def __init__(self, OBSERVATION_SPACE_VALUES: tuple, actions: list,
                 REPLAY_MEMORY_SIZE: int = 100_000,
                 MIN_REPLAY_MEMORY_SIZE: int = 1_000,
                 MINIBATCH_SIZE: int = 64, DISCOUNT: float = 0.99,
                 UPDATE_TARGET_EVERY: int = 10,  # 5
                 checkpoint=None):

        self.OBSERVATION_SPACE_VALUES = OBSERVATION_SPACE_VALUES
        self.actions = actions
        self.MIN_REPLAY_MEMORY_SIZE = MIN_REPLAY_MEMORY_SIZE
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.DISCOUNT = DISCOUNT
        self.UPDATE_TARGET_EVERY = UPDATE_TARGET_EVERY

        # Main model  -  gets trained every step
        if checkpoint is None:
            print('new model created')
            self.model = self.create_model()
        else:
            print('model loaded')
            self.model = load_model(checkpoint)

        # Target network  - what we predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An efficient container with stored experience(St,at,rt+1,st+1, done) from last n steps for training
        self.replay_memory: Deque = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self) -> Sequential:
        """
        creates the CNN-model.
        except input layer, network equal to original DQN network
        """
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(1, 16), strides=(1, 8),
                         input_shape=self.OBSERVATION_SPACE_VALUES))
        model.add(Activation('relu'))

        model.add(Conv2D(64, kernel_size=(1, 8), strides=(1, 4)))
        model.add(Activation('relu'))

        model.add(Conv2D(32, kernel_size=(1, 4), strides=(1, 2)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))

        model.add(Dense(len(self.actions), activation='linear'))
        model.compile(loss="mse", optimizer=RMSprop(lr=0.00025, momentum=0.95),
                      metrics=['accuracy'])
        model.summary()
        return model

    def update_replay_memory(self, transition: Tuple[np.array, int, int,
                                                     np.array, bool]) -> None:
        """transition is a tuple of experience-info at a certain timestep
         (st, at, rt+1, st+1, done)"""
        self.replay_memory.append(transition)

    def get_qs(self, state: np.array) -> np.array:
        """Queries main network for Q values given current observation space
        (environment state)"""
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train(self, terminal_state: bool, step: int) -> History:
        """Trains the ANN.
        Start training only if certain number of samples is already saved
        """
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table, like a
        # standard ANN minibatch is a list of tuples
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []
        # traversing all the experience-tuples in minibatch
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                # using the Bellmann equation
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        hist = self.model.fit(np.array(X), np.array(y),
                              batch_size=self.MINIBATCH_SIZE, verbose=0,
                              shuffle=False)

        if terminal_state:
            self.target_update_counter += 1

        # updateing to determine if we want to update target model
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        return hist

    def decay_epsilon(self, epsilon, MIN_EPSILON,
                      EPSILON_DECAY: float) -> float:
        """function that decays epsilon after every finished episode"""
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
        return epsilon

    def save(self, checkpoint) -> None:
        """saves a model from a certain episode"""
        self.model.save(checkpoint)
