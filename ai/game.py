import random

import numpy as np


class SnakeGame:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.snake = [(5, 5), (5, 4), (5, 3)]
        self.direction = 1  # 1=Right
        self.food = self.spawn_food()
        self.game_over = False
        self.score = 0

    def spawn_food(self):
        while True:
            food = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            )
            if food not in self.snake:
                return food

    def step(self, action):
        self.update_direction(action)
        new_head = self.get_new_head()

        if self.is_collision(new_head):
            self.game_over = True
            return -10  # Punishment for hitting wall or itself

        if new_head is not None:
            self.snake.insert(0, new_head)
        if new_head == self.food:
            self.food = self.spawn_food()
            self.score += 1
            return 10  # Reward for eating food
        else:
            self.snake.pop()
            return -0.1  # Small punishment for each step

    def update_direction(self, action):
        if action == 0:
            self.direction = 0  # Up
        elif action == 1:
            self.direction = 1  # Right
        elif action == 2:
            self.direction = 2  # Down
        elif action == 3:
            self.direction = 3  # Left

    def get_new_head(self):
        head = self.snake[0]
        if self.direction == 0:
            return (head[0] - 1, head[1])
        elif self.direction == 1:
            return (head[0], head[1] + 1)
        elif self.direction == 2:
            return (head[0] + 1, head[1])
        elif self.direction == 3:
            return (head[0], head[1] - 1)

    def is_collision(self, new_head):
        return (
            new_head[0] < 0
            or new_head[0] >= self.grid_size
            or new_head[1] < 0
            or new_head[1] >= self.grid_size
            or new_head in self.snake
        )

    def get_state(self):
        head = self.snake[0]
        return np.array(
            [
                self.direction,
                head[0] - self.food[0],
                head[1] - self.food[1],
                int((head[0] + 1, head[1]) in self.snake),  # Down obstacle
                int((head[0] - 1, head[1]) in self.snake),  # Up obstacle
                int((head[0], head[1] + 1) in self.snake),  # Right obstacle
                int((head[0], head[1] - 1) in self.snake),  # Left obstacle
            ]
        )
