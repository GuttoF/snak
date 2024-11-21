import imageio  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pygame


def plot_training_evolution(scores, steps, rewards):
    epochs = list(range(len(scores)))

    cumulative_scores = np.cumsum(scores)
    average_scores_per_epoch = cumulative_scores / (np.arange(len(scores)) + 1)

    cumulative_steps = np.cumsum(steps)
    average_steps_per_epoch = cumulative_steps / (np.arange(len(steps)) + 1)

    cumulative_rewards = np.cumsum(rewards)
    average_rewards_per_epoch = cumulative_rewards / (np.arange(len(rewards)) + 1)

    fig, axs = plt.subplots(3, 1, figsize=(14, 12))

    axs[0].plot(
        epochs,
        scores,
        label="Score",
        marker="o",
        color="b",
        linestyle="-",
        markersize=3,
    )
    axs[0].plot(
        epochs,
        average_scores_per_epoch,
        label="Média",
        color="g",
        linestyle="-",
        linewidth=2,
    )
    axs[0].set_title("Evolução do Score por Episódio")
    axs[0].set_ylabel("Score")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(
        epochs, steps, label="Steps", marker="o", color="b", linestyle="-", markersize=3
    )
    axs[1].plot(
        epochs,
        average_steps_per_epoch,
        label="Média",
        color="g",
        linestyle="-",
        linewidth=2,
    )
    axs[1].set_title("Evolução dos Steps por Episódio")
    axs[1].set_ylabel("Steps")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(
        epochs,
        rewards,
        label="Reward",
        marker="o",
        color="b",
        linestyle="-",
        markersize=3,
    )
    axs[2].plot(
        epochs,
        average_rewards_per_epoch,
        label="Média",
        color="g",
        linestyle="-",
        linewidth=2,
    )
    axs[2].set_title("Evolução da Recompensa por Episódio")
    axs[2].set_xlabel("Episódios")
    axs[2].set_ylabel("Reward")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def record_game_as_gif(env, agent, save_path="gameplay.gif", fps=10):
    frames = []
    state = env.reset()

    for _ in range(500):  # Limite de passos
        frame = pygame.surfarray.array3d(env.render())
        frames.append(frame)

        action, _ = agent.select_action(state)
        state, _, done, _ = env.step(action)

        if done:
            break

    pygame.quit()
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Gameplay salva como {save_path}")
