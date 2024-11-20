import random
import argparse
from snake import * # noqa
import torch
import pygame


parser = argparse.ArgumentParser(description='test')
parser.add_argument(
    '--weight',
    type=str,
    default='./model/act-weight_round3_472_82.5.pkl',
    help='Select saved model.'
)
args = parser.parse_args()

if __name__ == "__main__":
    random.seed(100)
    env = Snake() # noqa
    env.snake_speed = 20
    agent = AgentDiscretePPO() # noqa
    agent.init(512, 6, 4)
    device = "cpu"

    state_dict = torch.load(args.weight, map_location=device)
    agent.act.load_state_dict(state_dict)
    for _ in range(15):
        o = env.reset()
        while 1:
            env.render()
            for event in pygame.event.get():
                pass

            a, _ = agent.select_action(o)
            o2, r, d, _ = env.step(a)
            o = o2
            if d:
                break
