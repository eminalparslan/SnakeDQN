import numpy as np
from collections import deque
import random
import time

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import Callback

from snake_env import SnakeEnv

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class DQNAgent:
    MODEL_NAME = "128_256v2"
    LEARNING_RATE = 1e-3
    DISCOUNT = 0.99
    REPLAY_MEMORY_SIZE = 50_000
    MIN_MEMORY_SIZE = 1_000
    MINIBATCH_SIZE = 64
    UPDATE_TARGET_EVERY = 5

    def __init__(self):
        # Main model
        self.model = self.create_model()
        # Target model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        # Memory buffer
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        # Modified TensorBoard
        self.callback = CustomTensorBoard(log_dir=f"logs/{self.MODEL_NAME}-{time.time()}")
        # Counter for when to update target network
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(128, (3, 3), activation="relu", input_shape=SnakeEnv.OBSERVATION_SPACE_VALUES))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(SnakeEnv.ACTION_SPACE_SIZE, activation="linear"))
        
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.LEARNING_RATE), metrics=["accuracy"])
        return model

    def update_replay_memory(self, step_result):
        # stepResult = (obs, action, reward, newObs, done)
        self.replay_memory.append(step_result)

    def train(self, terminalState):
        if len(self.replay_memory) < self.MIN_MEMORY_SIZE:
            return

        # Randomly sampled minibatch from 
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        # Get current states from minibatch
        current_obs_mb = np.array([step_results[0] for step_results in minibatch])/255
        # Get model predictions from current states (predicted Q values)
        current_qs_mb = self.model.predict(current_obs_mb)

        # Get model predictions for future states
        future_obs_mb = np.array([step_results[3] for step_results in minibatch])/255
        future_qs_mb = self.target_model.predict(future_obs_mb)

        # Features and labels
        X = []
        y = []

        # Go through each step in minibatch
        for i, (obs, action, reward, newObs, done) in enumerate(minibatch):
            # Calculate new Q value
            if not done:
                max_future_q = np.max(future_qs_mb[i])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for current state
            current_qs = current_qs_mb[i]
            current_qs[action] = new_q

            # Add to training data
            X.append(obs)
            y.append(current_qs);

        # Fit on minibatch
        self.model.fit(np.array(X)/255, np.array(y), batch_size=self.MINIBATCH_SIZE, 
                        verbose=0, shuffle=False, callbacks=[self.callback] if terminalState else None)
        
        # Update target network counter every episode
        if terminalState:
            self.target_update_counter += 1

        # Update target network
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, obs):
        return self.model.predict(obs.reshape(-1, *SnakeEnv.OBSERVATION_SPACE_VALUES)/255)[0]

class CustomTensorBoard(Callback):
    def __init__(self, log_dir):
        self.step = 1
        self.writer = tf.summary.create_file_writer(log_dir)

    def update_stats(self, **logs):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, self.step)
        self.writer.flush()
