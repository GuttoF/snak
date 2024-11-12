import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot # type: ignore

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


class SnakeNet(nn.Module):
    def __init__(self):
        super(SnakeNet, self).__init__()
        self.fc1 = nn.Linear(7, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # 4 possible actions


def train_pytorch_model():
    model = SnakeNet()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    game = SnakeGame()
    scores = []
    rewards = []
    epsilon = INITIAL_EPSILON
    best_score = 0

    for episode in range(1, MAX_EPISODES + 1):
        game.reset()
        state = torch.FloatTensor(game.get_state()).unsqueeze(0)
        total_reward = 0

        while not game.game_over:
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, 4)  # Random action
            else:
                with torch.no_grad():
                    action = torch.argmax(model(state)).item()  # Best action

            reward = game.step(action)
            total_reward += reward
            next_state = torch.FloatTensor(game.get_state()).unsqueeze(0)

            # Update Q-Value
            target = reward
            if not game.game_over:
                target += GAMMA * torch.max(model(next_state)).item()

            target_f = model(state)
            target_val = target_f.clone()
            target_val[0][action] = target

            # Neural Network training
            optimizer.zero_grad()
            loss = criterion(target_f, target_val)
            loss.backward()
            optimizer.step()
            state = next_state

        scores.append(game.score)
        rewards.append(total_reward)

        if epsilon > FINAL_EPSILON:
            epsilon *= EPSILON_DECAY

        if episode % EARLY_STOP_EPISODES == 0:
            avg_score = np.mean(scores[-EARLY_STOP_EPISODES:])
            print(
                f"Episódio {episode}, Pontuação Média: {avg_score:.2f}, "
                f"Epsilon: {epsilon:.4f}"
            )
            if avg_score > best_score:
                best_score = avg_score
                torch.save(model.state_dict(), "saved_models/best_pytorch_model.pth")
                print(f"Novo melhor modelo salvo com pontuação média: {best_score:.2f}")
            if avg_score >= TARGET_SCORE:
                print(
                    (
                        f"Parando o treinamento. Pontuação média alvo de "
                        f"{TARGET_SCORE} alcançada."
                    )
                )
                break

    with open("saved_models/pytorch_model.json", "w") as f:
        json.dump({"layers": [24, 24, 4]}, f)

    np.save(
        "saved_models/pytorch_weights.npy",
        [param.detach().numpy() for param in model.parameters()],
    )

    weights_as_lists = [param.detach().numpy().tolist() for param in model.parameters()]
    with open("saved_models/pytorch_weights.json", "w") as json_weights_file:
        json.dump(weights_as_lists, json_weights_file)

    plot_training_results(scores, rewards, "plots/pytorch_training.png")
    dummy_input = torch.randn(1, 7)
    model_output = model(dummy_input)
    make_dot(model_output, params=dict(model.named_parameters())).render(
        "plots/pytorch_model_structure", format="png"
    )


train_pytorch_model()
