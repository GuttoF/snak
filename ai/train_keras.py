import json

import numpy as np
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.utils import plot_model  # type: ignore

from game import SnakeGame
from plot_utils import plot_training_results

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.95  # Discount factor
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.01
EPSILON_DECAY = 0.995
TARGET_SCORE = 100
MAX_EPISODES = 5000
EARLY_STOP_EPISODES = 10


def build_model():
    model = Sequential(
        [
            Dense(24, input_shape=(7,), activation="relu"),
            Dense(24, activation="relu"),
            Dense(4, activation="linear"),  # 4 possible actions
        ]
    )
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model


def train_keras_model():
    model = build_model()
    game = SnakeGame()
    scores = []
    rewards = []
    epsilon = INITIAL_EPSILON
    best_score = 0

    for episode in range(1, MAX_EPISODES + 1):
        game.reset()
        state = game.get_state().reshape(1, -1)
        total_reward = 0

        while not game.game_over:
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, 4)  # Random action
            else:
                action = np.argmax(model.predict(state, verbose=0)[0])  # Best action

            reward = game.step(action)
            total_reward += reward
            next_state = game.get_state().reshape(1, -1)

            # Update Q-Value
            target = reward
            if not game.game_over:
                target += GAMMA * np.amax(model.predict(next_state, verbose=0)[0])

            target_f = model.predict(state, verbose=0)
            target_f[0][action] = target

            # Neural Network training
            model.fit(state, target_f, epochs=1, verbose=0)
            state = next_state

        scores.append(game.score)
        rewards.append(total_reward)

        if epsilon > FINAL_EPSILON:
            epsilon *= EPSILON_DECAY

        if episode % EARLY_STOP_EPISODES == 0:
            avg_score = np.mean(scores[-EARLY_STOP_EPISODES:])
            print(
                f"Episode {episode}, Average Score: {avg_score:.2f}, "
                f"Epsilon: {epsilon:.4f}"
            )
            if avg_score > best_score:
                best_score = avg_score
                model.save_weights("saved_models/best_keras_weights.weights.h5")
                print(f"New best model saved with average score: {best_score:.2f}")
            if avg_score >= TARGET_SCORE:
                print(
                    f"Stopping training. Target average score of "
                    f"{TARGET_SCORE} reached."
                )
                break

    model_json = model.to_json()
    with open("saved_models/keras_model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("saved_models/keras_weights.weights.h5")

    weights = model.get_weights()
    weights_as_lists = [w.tolist() for w in weights]
    with open("saved_models/keras_weights.json", "w") as json_weights_file:
        json.dump(weights_as_lists, json_weights_file)

    plot_training_results(scores, rewards, "plots/keras_training.png")
    plot_model(
        model,
        to_file="plots/keras_model_structure.png",
        show_shapes=True,
        show_layer_names=True,
    )


train_keras_model()
