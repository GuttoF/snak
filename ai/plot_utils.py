import numpy as np
from matplotlib import pyplot as plt


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_training_results(
    scores, rewards, filename="training_plot.png", window_size=10
):
    plt.figure(figsize=(12, 5))
    tableau_colors = plt.get_cmap("tab10")(range(10))

    plt.subplot(1, 2, 1)
    plt.plot(scores, label="Score", color=tableau_colors[0], alpha=0.5)
    plt.plot(
        moving_average(scores, window_size),
        label="Moving Avg (Score)",
        color=tableau_colors[0],
    )
    plt.axhline(y=max(scores), color="grey", linestyle="--", label="Best Score")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.title("Score per Episode")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rewards, label="Total Reward", color=tableau_colors[1], alpha=0.5)
    plt.plot(
        moving_average(rewards, window_size),
        label="Moving Avg (Reward)",
        color=tableau_colors[1],
    )
    plt.axhline(y=max(rewards), color="grey", linestyle="--", label="Best Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
