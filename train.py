import argparse
import sys

from absl import flags

import numpy as np

from runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--map_name', type=str, default='MoveToBeacon', help='Map name')
    parser.add_argument('--screen_size', type=int, default=64, help='Screen size')
    parser.add_argument('--step_mul', type=int, default=8, help='Frames per step')
    parser.add_argument('--max_steps', type=int, default=2500, help='Maximum steps per env')
    parser.add_argument('--generations', type=int, default=100, help='Generations count')

    # Hyperparameters
    parser.add_argument('--population_size', type=int, default=100, help='Population size')
    parser.add_argument('--parents_count', type=int, default=10, help='Parents count')
    parser.add_argument('--mutation_power', type=int, default=0.002, help='Mutation power')

    # Booleans
    parser.add_argument('--uniform', type=bool, default=True, help='Use uniform to choose parents')
    parser.add_argument('--crossover', type=bool, default=True, help='Whether crossover is needed')

    # Env parameters
    parser.add_argument('--envs_number', type=int, default=1, help='Number of envs to run in parallel')
    parser.add_argument('--save_dir', type=str, default='/media/Data/Diploma/saves/test/', help='Save dir')
    parser.add_argument('--eval', type=bool, default=False, help='Evaluate with random models')
    parser.add_argument('--render', type=bool, default=False, help='Visualize')

    # Model parameters
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--activations', type=bool, default=True, help='Activations after conv layers')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    FLAGS([sys.argv[0]])

    np.random.seed()

    args = parse_args()
    runner = Runner(args)
    runner.run()
