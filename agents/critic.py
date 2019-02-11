import numpy as np
#import tensorflow as tf

#TF Imports
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import initializers

import math


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        l2_reg=1e-2
        init_val_state=1 / math.sqrt(self.state_size)
        init_val_action=1 / math.sqrt(self.action_size)
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=400, kernel_regularizer=regularizers.l2(l2_reg), kernel_initializer=initializers.RandomUniform(minval=-init_val_state, maxval=init_val_state))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.LeakyReLU(1e-2)(net_states)
        
        net_states = layers.Dense(units=300, kernel_regularizer=regularizers.l2(l2_reg), kernel_initializer=initializers.RandomUniform(minval=-1/20, maxval=1/20))(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.LeakyReLU(1e-2)(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=400, kernel_regularizer=regularizers.l2(l2_reg), kernel_initializer=initializers.RandomUniform(minval=-init_val_action, maxval=init_val_action))(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.LeakyReLU(1e-2)(net_actions)
        
        net_actions = layers.Dense(units=300, kernel_regularizer=regularizers.l2(l2_reg), kernel_initializer=initializers.RandomUniform(minval=-1/20, maxval=1/20))(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.LeakyReLU(1e-2)(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        #net = layers.Activation('relu')(net)
        net = layers.LeakyReLU(1e-2)(net)
        
        # Fully connected and batch normalization
        net = layers.Dense(units=200, kernel_regularizer=regularizers.l2(l2_reg))(net)
        net = layers.BatchNormalization()(net)
        net = layers.LeakyReLU(1e-2)(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3), name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=1e-3)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)