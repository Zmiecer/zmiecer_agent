import os

from shutil import copyfile
from multiprocessing import Process

import numpy as np

from environment import MyEnv
from genetics import Genetics

from pysc2.maps.lib import Map
from pysc2.maps.mini_games import mini_games


class Runner(object):
    def __init__(self, args):
        # TODO: По-хорошему, save/load должен как-то подхватывать last generation
        current_map = args.map_name
        if current_map not in mini_games:
            current_map = Map()
            current_map.directory="mini_games"
            current_map.players = 1
            current_map.score_index = 0
            current_map.game_steps_per_episode = 0
            current_map.step_mul = 8
            current_map.filename = args.map_name
        
        self.map_name = current_map
        self.game_length = args.game_length
        self.max_games = args.max_games
        self.step_mul = args.step_mul
        self.screen_size = args.screen_size
        self.visualize = args.render
        self.envs_number = args.envs_number
        self.max_generations = args.generations

        # Training parameters
        self.population_size = args.population_size
        self.parents_count = args.parents_count
        self.mutation_power = args.mutation_power

        # Booleans
        self.choose_uniformly = args.uniform_feature
        self.do_crossover = args.crossover_feature

        # Model parameters
        self.layers = args.layers

        self.load = args.load

        # Starting params
        self.generation = 0
        self.scores = np.zeros(self.population_size)

        # Save dir
        self.save_dir = args.save_dir + 'pop_{}_par_{}_mp_{}_c_{}_u_{}/'.format(
            self.population_size,
            self.parents_count,
            self.mutation_power,
            self.do_crossover,
            self.choose_uniformly
        )
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save_parent_models(self, parent_indices):
        for parent_number, parent_index in enumerate(parent_indices):
            copyfile(self.save_dir + 'model_{}'.format(parent_index) + '.h5',
                     self.save_dir + 'parent_{}'.format(parent_number) + '.h5')

    def log_generation(self, parent_indices):
        mean_score = np.sum(self.scores) / self.population_size

        parent_scores = self.scores[parent_indices]
        mean_parent_score = np.sum(parent_scores) / self.parents_count

        print('Generation: {},  best score: {}, mean score: {}'.format(
            self.generation, np.max(self.scores), mean_score))
        print('Parent models: {}, parent scores: {}, mean parent score: {}'.format(
            parent_indices, parent_scores, mean_parent_score))

        with open(self.save_dir + 'log.txt', mode='a') as log:
            print('Generation: {},  best score: {}, mean score: {}'.format(
                self.generation, np.max(self.scores), mean_score), file=log)
            print('Parent models: {}, parent scores: {}, mean parent score: {}'.format(
                parent_indices, parent_scores, mean_parent_score), file=log)

    def get_parent_indices(self):
        return np.argpartition(self.scores, -self.parents_count)[-self.parents_count:]

    def genetics(self):
        parent_indices = self.get_parent_indices()
        parents_scores = self.scores[parent_indices]

        # Logging
        self.log_generation(parent_indices)

        # Copy files to parents
        self.save_parent_models(parent_indices)

        # Genetics
        genetics = Genetics(
            population_size=self.population_size,
            parents_count=self.parents_count,
            choose_uniformly=self.choose_uniformly,
            mutation_power=self.mutation_power,
            do_crossover=self.do_crossover,
            parents_scores=parents_scores,
            save_dir=self.save_dir,
            envs_number=self.envs_number
        )
        genetics.generate_new_models()

    def run_envs(self):
        processes = []
        for pool_number in range(self.envs_number):
            p = Process(target=self.run_env, args=(pool_number,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def load_scores(self):
        for model_number in range(self.population_size):
            with open(self.save_dir + 'score_{}.txt'.format(model_number), mode='r') as score_file:
                score = int(score_file.readline())
                self.scores[model_number] = score

    def reset_scores(self):
        self.scores = np.zeros(self.population_size)

    def run(self):
        g = Genetics(
            population_size=self.population_size,
            parents_count=self.parents_count,
            choose_uniformly=self.choose_uniformly,
            mutation_power=self.mutation_power,
            do_crossover=self.do_crossover,
            parents_scores=np.zeros(self.parents_count),
            save_dir=self.save_dir,
            envs_number=self.envs_number
        )
        g.generate_new_models(load=self.load)

        while self.generation < self.max_generations:
            self.run_envs()

            self.load_scores()
            self.genetics()
            self.reset_scores()
            self.generation += 1

    def run_env(self, pool_number):
        env = MyEnv(
            map_name=self.map_name,
            step_mul=self.step_mul,
            screen_size=self.screen_size,
            minimap_size=self.screen_size,
            game_length=self.game_length,
            max_games=self.max_games,
            envs_number=self.envs_number,
            visualize=self.visualize,
            pool_number=pool_number,
            population_size=self.population_size,
            generation=self.generation,
            save_dir=self.save_dir
        )
        env.run()
