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
    parser.add_argument('--game_length', type=int, default=500, help='Maximum steps per game')
    parser.add_argument('--max_games', type=int, default=5, help='Maximum games per run')
    parser.add_argument('--generations', type=int, default=100, help='Generations count')

    # Hyperparameters
    parser.add_argument('--population_size', type=int, default=100, help='Population size')
    parser.add_argument('--parents_count', type=int, default=10, help='Parents count')
    parser.add_argument('--mutation_power', type=float, default=0.002, help='Mutation power')

    # Booleans
    parser.add_argument('--no-uniform', dest='uniform_feature', action='store_false', default=True,
                        help='Use uniform distribution to choose parents')
    parser.add_argument('--no-crossover', dest='crossover_feature', action='store_false', default=True)

    # Env parameters
    parser.add_argument('--envs_number', type=int, default=1, help='Number of envs to run in parallel')
    parser.add_argument('--save_dir', type=str, default='/media/Data/Diploma/saves/', help='Save dir')
    parser.add_argument('--eval', type=bool, default=False, help='Evaluate with random models')
    parser.add_argument('--render', type=bool, default=False, help='Visualize')

    # Model parameters
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')

    parser.add_argument('--load', action='store_true', help='Load model', default=False)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    FLAGS([sys.argv[0]])

    np.random.seed()

    args = parse_args()
    runner = Runner(args)
    runner.run()

