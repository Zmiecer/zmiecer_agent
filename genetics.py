from copy import deepcopy
from multiprocessing import Process

import numpy as np


class Genetics(object):
    def __init__(self,
                 population_size,
                 parents_count,
                 choose_uniformly,
                 mutation_power,
                 do_crossover,
                 parents_scores,
                 save_dir,
                 envs_number
                 ):

        # Training parameters
        self.population_size = population_size
        self.parents_count = parents_count
        self.mutation_power = mutation_power
        self.choose_uniformly = choose_uniformly
        self.do_crossover = do_crossover

        # Global parameters
        self.save_dir = save_dir
        self.envs_number = envs_number
        self.parents_scores = parents_scores

    @staticmethod
    def generate_parent_probs(best_scores):
        scores_for_probs = deepcopy(best_scores)
        scores_for_probs += 1
        # scores_for_probs[scores_for_probs == 0] = 1. / self.population_size
        parents_probs = scores_for_probs / np.sum(scores_for_probs)
        return parents_probs

    def mutate_weights(self, weights):
        new_weights = []
        for layer in weights:
            layer += self.mutation_power * np.random.normal(size=layer.shape)
            new_weights.append(layer)
        return new_weights

    def choose_parents(self):
        if self.choose_uniformly:
            current_parents = np.random.choice(self.parents_count, 2, replace=False)
        else:
            parent_scores = self.parents_scores
            parents_probs = self.generate_parent_probs(parent_scores)
            current_parents = np.random.choice(self.parents_count, 2, replace=False, p=parents_probs)
        return current_parents

    def crossover(self, first_parent_index, second_parent_index):
        from model import AtariModel
        first_parent = AtariModel()
        first_parent.load_weights(self.save_dir + 'parent_{}'.format(first_parent_index) + '.h5')
        first_weights = first_parent.get_weights()

        second_parent = AtariModel()
        second_parent.load_weights(self.save_dir + 'parent_{}'.format(second_parent_index) + '.h5')
        second_weights = second_parent.get_weights()

        new_weights = []
        for first_layer, second_layer in zip(first_weights, second_weights):
            new_layer = []
            for first_column, second_column in zip(first_layer, second_layer):
                layer_number = np.random.randint(2)
                if layer_number == 0:
                    new_layer.append(first_column)
                else:
                    new_layer.append(second_column)
            new_weights.append(np.array(new_layer))

        return new_weights

    def generate_model(self, index, load=True):
        from model import AtariModel
        model = AtariModel()
        if load:
            if index == self.population_size - 1:
                best_parent_index = np.argmax(self.parents_scores)
                model.load_weights(self.save_dir + 'parent_{}'.format(best_parent_index) + '.h5')
            else:

                first_parent_index, second_parent_index = self.choose_parents()
                if self.do_crossover:
                    weights = self.crossover(first_parent_index, second_parent_index)
                else:
                    model.load_weights(self.save_dir + 'parent_{}'.format(first_parent_index) + '.h5')
                    weights = model.get_weights()
                new_weights = self.mutate_weights(weights)
                model.set_weights(new_weights)
        model.save_weights(self.save_dir + 'model_{}'.format(index) + '.h5')
        print('Model {} generated'.format(index))

    def generate_new_models(self, load=True):
        processes = []

        for pool_first in range(0, self.population_size, self.envs_number):
            pool_last = min(pool_first + self.envs_number, self.population_size)
            for index in range(pool_first, pool_last):
                p = Process(target=self.generate_model, args=(index, load,))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
