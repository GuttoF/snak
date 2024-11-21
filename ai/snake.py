import os
import sys
import torch
import random
import pygame
import numpy as np # noqa
from PIL import Image # noqa
from Agent import AgentDiscretePPO
from torchvision import transforms # type: ignore # noqa
from pygame.locals import * # type: ignore # noqa

# Configuração dos rounds e recompensas
round = 3  # default mode: round 1
rewards = [
    {'eat': 2.0, 'hit': -0.5, 'bit': -0.8},  # round 1
    {'eat': 2.0, 'hit': -1.0, 'bit': -1.5},  # round 2
    {'eat': 2.0, 'hit': -1.5,  'bit': -2.0},  # round 3
    {'eat': 2.0, 'hit': -2.0,  'bit': -2.0}  # round 4
]

# Classe principal do jogo Snake
class Snake:
    def __init__(self):
        self.snake_speed = 100  # Velocidade da cobra
        self.windows_width = 600
        self.windows_height = 600  # Tamanho da janela do jogo
        self.cell_size = 50  # Tamanho do quadrado do corpo da cobra
        self.map_width = int(self.windows_width / self.cell_size)
        self.map_height = int(self.windows_height / self.cell_size)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.gray = (230, 230, 230)
        self.dark_gray = (40, 40, 40)
        self.DARKGreen = (0, 155, 0)
        self.Green = (0, 255, 0)
        self.Red = (255, 0, 0)
        self.blue = (0, 0, 255)
        self.dark_blue = (0, 0, 139)
        self.cyan = (0, 255, 255)
        self.yellow = (255, 255, 0)
        self.BG_COLOR = self.black  # Cor de fundo do jogo

        # Direções
        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3
        self.HEAD = 0  # Índice da cabeça da cobra

        pygame.init()
        self.snake_speed_clock = pygame.time.Clock()

        self.snake_coords = [
            {'x': self.map_width // 2, 'y': self.map_height // 2},
            {'x': self.map_width // 2 - 1, 'y': self.map_height // 2},
            {'x': self.map_width // 2 - 2, 'y': self.map_height // 2}
        ]
        self.direction = self.RIGHT
        self.food = self.get_random_location()
        self.state = self.getState()

    def reset(self):
        startx = random.randint(3, self.map_width - 8)
        starty = random.randint(3, self.map_height - 8)
        self.snake_coords = [
            {'x': startx, 'y': starty},
            {'x': startx - 1, 'y': starty},
            {'x': startx - 2, 'y': starty}
        ]
        self.direction = self.RIGHT
        self.food = self.get_random_location()
        return self.getState()

    def step(self, action):
        if action == self.LEFT and self.direction != self.RIGHT:
            self.direction = self.LEFT
        elif action == self.RIGHT and self.direction != self.LEFT:
            self.direction = self.RIGHT
        elif action == self.UP and self.direction != self.DOWN:
            self.direction = self.UP
        elif action == self.DOWN and self.direction != self.UP:
            self.direction = self.DOWN

        self.move_snake(self.direction, self.snake_coords)
        ret = self.snake_is_alive(self.snake_coords)
        d = (ret > 0)
        flag = self.snake_is_eat_food(self.snake_coords, self.food)
        reward = self.getReward(flag, ret)

        return [self.getState(), reward, d, None]

    def getReward(self, flag, ret):
        reward = 0
        if flag:
            reward += rewards[round - 1]['eat']
        if ret == 1:
            reward += rewards[round - 1]['hit']
        if ret == 2:
            reward += rewards[round - 1]['bit']
        return reward

    def render(self):
        self.screen = pygame.display.set_mode(
            (self.windows_width, self.windows_height)
        )
        self.screen.fill(self.BG_COLOR)
        self.draw_snake(self.screen, self.snake_coords)
        self.draw_food(self.screen, self.food)
        if self.snake_coords is not None:
            self.draw_score(self.screen, len(self.snake_coords) - 3)
        pygame.display.update()
        self.snake_speed_clock.tick(self.snake_speed)

    def getState(self):
        [xhead, yhead] = [
            self.snake_coords[self.HEAD]['x'],
            self.snake_coords[self.HEAD]['y']
        ]
        [xfood, yfood] = [self.food['x'], self.food['y']]
        deltax = (xfood - xhead) / self.map_width
        deltay = (yfood - yhead) / self.map_height
        checkPoint = [
            [xhead, yhead-1],
            [xhead-1, yhead],
            [xhead, yhead+1],
            [xhead+1, yhead]
        ]
        tem = [0, 0, 0, 0]
        for coord in self.snake_coords[1:]:
            if [coord['x'], coord['y']] in checkPoint:
                index = checkPoint.index([coord['x'], coord['y']])
                tem[index] = 1

        for i, point in enumerate(checkPoint):
            if (point[0] >= self.map_width or point[0] < 0 or
                point[1] >= self.map_height or point[1] < 0):
                tem[i] = 1

        state = [deltax, deltay]
        state.extend(tem)

        return state

    def draw_food(self, screen, food):
        x = food['x'] * self.cell_size
        y = food['y'] * self.cell_size
        appleRect = pygame.Rect(x, y, self.cell_size, self.cell_size)
        pygame.draw.rect(screen, self.Red, appleRect)

    def draw_snake(self, screen, snake_coords):
        for i, coord in enumerate(snake_coords):
            color = self.Green
            if i == 0:
                color = self.yellow
            if i == len(snake_coords) - 1:
                color = self.cyan
            x = coord['x'] * self.cell_size
            y = coord['y'] * self.cell_size
            wormSegmentRect = pygame.Rect(x, y, self.cell_size, self.cell_size)
            pygame.draw.rect(screen, self.dark_blue, wormSegmentRect)
            wormInnerSegmentRect = pygame.Rect(
                x + 4, y + 4, self.cell_size - 8, self.cell_size - 8
            )
            pygame.draw.rect(screen, color, wormInnerSegmentRect)

    def move_snake(self, direction, snake_coords):
        if direction == self.UP:
            newHead = {
                'x': snake_coords[self.HEAD]['x'], 'y': snake_coords[self.HEAD]['y'] - 1
            }
        elif direction == self.DOWN:
            newHead = {
                'x': snake_coords[self.HEAD]['x'],
                'y': snake_coords[self.HEAD]['y'] + 1
            }
        elif direction == self.LEFT:
            newHead = {
                'x': snake_coords[self.HEAD]['x'] - 1,
                'y': snake_coords[self.HEAD]['y']
            }
        elif direction == self.RIGHT:
            newHead = {
                'x': snake_coords[self.HEAD]['x'] + 1,
                'y': snake_coords[self.HEAD]['y']
            }
        else:
            raise Exception('Error for direction!')
        snake_coords.insert(0, newHead)

    def snake_is_alive(self, snake_coords):
        if (snake_coords[self.HEAD]['x'] == -1 or
            snake_coords[self.HEAD]['x'] == self.map_width or
            snake_coords[self.HEAD]['y'] == -1 or
            snake_coords[self.HEAD]['y'] == self.map_height):
            return 1
        for snake_body in snake_coords[1:]:
            if (snake_body['x'] == snake_coords[self.HEAD]['x'] and
                snake_body['y'] == snake_coords[self.HEAD]['y']):
                return 2
        return 0

    def snake_is_eat_food(self, snake_coords, food):
        if snake_coords[self.HEAD]['x'] == food['x'] and \
                snake_coords[self.HEAD]['y'] == food['y']:
            while True:
                food['x'] = random.randint(0, self.map_width - 1)
                food['y'] = random.randint(0, self.map_height - 1)
                if all(
                    coord['x'] != food['x'] or coord['y'] != food['y']
                    for coord in snake_coords
                ):
                    break
            return True
        else:
            del snake_coords[-1]
        return False

    def get_random_location(self):
        return {
            'x': random.randint(0, self.map_width - 1),
            'y': random.randint(0, self.map_height - 1)
        }

    def draw_score(self, screen, score):
        font = pygame.font.SysFont('Arial', 28)
        scoreSurf = font.render(f'Score: {score}', True, self.white)
        scoreRect = scoreSurf.get_rect()
        scoreRect.topleft = (self.windows_width - 130, 0)
        screen.blit(scoreSurf, scoreRect)

    @staticmethod
    def terminate():
        pygame.quit()
        sys.exit()

def filter_pkl(lists):
    return [file for file in lists if file.endswith('.pkl')]

def get_latest_weight(path):
    if not os.path.exists(path):
        print(f"Path: {path} does not exist")
        sys.exit()
    lists = filter_pkl(os.listdir(path))
    if not lists:
        print(f"No .pkl files found in {path}")
        sys.exit()
    lists.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
    return os.path.join(path, lists[-1])

if __name__ == "__main__":
    random.seed(100)
    env = Snake()
    env.snake_speed = 10
    agent = AgentDiscretePPO()
    agent.init(512, 6, 4)
    latest_weight = get_latest_weight('./history/')
    print(f"Loading model weights from: {latest_weight}")
    agent.act.load_state_dict(torch.load(latest_weight))
    for _ in range(15):
        o = env.reset()
        while True:
            env.render()
            for event in pygame.event.get():
                pass
            a, _ = agent.select_action(o)
            o2, r, d, _ = env.step(a)
            o = o2
            if d:
                break
