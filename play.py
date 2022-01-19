import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from snake_env import SnakeEnv

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

env = SnakeEnv()
model = load_model("models/128_256546.00max_36.62avg-197.00min1642487453.model")

while True:
    done = False
    obs = env.reset()
    while not done:
        # Get predicted q-values
        qs = model.predict(obs.reshape(-1, *env.OBSERVATION_SPACE_VALUES)/255)[0]
        # Choose action based on highest q-value
        action = np.argmax(qs)
        # Take a step
        newObs, reward, done = env.step(action)
        # Render
        env.render()
        obs = newObs
