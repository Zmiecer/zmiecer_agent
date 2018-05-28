import time
import numpy as np

from pysc2.env.environment import StepType
from pysc2.env.sc2_env import SC2Env
from pysc2.lib.actions import FUNCTIONS, FunctionCall

from utils import OUTPUT_SHAPES
from utils import mask_unavailable_actions, reshape_channels


class MyEnv(SC2Env):
    def __init__(self,
                 map_name,
                 step_mul,
                 screen_size,
                 minimap_size,
                 game_length,
                 max_games,
                 envs_number,
                 visualize,
                 pool_number,
                 population_size,
                 generation,
                 save_dir):
        super(MyEnv, self).__init__(map_name=map_name,
                                    step_mul=step_mul,
                                    screen_size_px=(screen_size, screen_size),
                                    minimap_size_px=(minimap_size, minimap_size),
                                    game_steps_per_episode=0,
                                    visualize=visualize)
        self.game_length = game_length
        self.max_games = max_games
        self.pool_number = pool_number
        self.envs_number = envs_number
        self.population_size = population_size
        self.generation = generation
        self.save_dir = save_dir

    @property
    def episode_count(self):
        return self._episode_count

    @property
    def state(self):
        return self._state

    @staticmethod
    def translate_observations(observations):
        screen = observations['screen']
        minimap = observations['minimap']
        screen = reshape_channels(screen)
        minimap = reshape_channels(minimap)
        return screen, minimap

    @staticmethod
    def result_to_dict(result):
        res_dict = {}
        i = 0
        for output_info in OUTPUT_SHAPES:
            output_type = output_info[0]
            res_dict[output_type] = result[i]
            i += 1
        return res_dict

    @staticmethod
    def get_action(res_dict, observations):
        action_probs = res_dict['action']
        available_actions = observations['available_actions']
        action_probs = mask_unavailable_actions(available_actions, action_probs)
        action_id = np.argmax(action_probs)
        action = FUNCTIONS[action_id]
        return action

    @staticmethod
    def prepare_args(action, res_dict):
        action_args = []
        for needed_arg in action.args:
            needed_arg_name = needed_arg.name
            if needed_arg_name in ('minimap', 'screen'):
                prob_x = res_dict['x1']
                prob_y = res_dict['y1']
                x = np.argmax(prob_x)
                y = np.argmax(prob_y)
                arg_value = [x, y]
            elif needed_arg_name == 'screen2':
                prob_x = res_dict['x2']
                prob_y = res_dict['y2']
                x = np.argmax(prob_x)
                y = np.argmax(prob_y)
                arg_value = [x, y]
            else:
                probs = res_dict[needed_arg.name]
                arg_value = [np.argmax(probs)]
            action_args.append(arg_value)
        return action_args

    def run_model(self, model, model_number):
        obs = self.reset()
        print('Env {} reset completed'.format(self.pool_number))

        model.load_weights(self.save_dir + 'model_{}.h5'.format(model_number))
        print('Model {} loaded'.format(model_number))

        games_played = 0
        step = 0
        cumulative_score = 0

        while step < self.game_length:
            observations = obs[0].observation
            screen, minimap = self.translate_observations(observations)

            result = model.predict([np.array([screen]), np.array([minimap])])

            res_dict = self.result_to_dict(result)

            action = self.get_action(res_dict, observations)

            action_args = self.prepare_args(action, res_dict)

            # call action in a new step
            obs = self.step(actions=[FunctionCall(action.id, action_args)])
            step += 1

            if self.state == StepType.FIRST:
                games_played += 1
                current_reward = observations["score_cumulative"][0]
                print('Game {} has ended, score: {}'.format(games_played, current_reward))
                cumulative_score += current_reward

        # TODO: Нужно избегать двойного суммирования score
        games_played += 1
        observations = obs[0].observation
        current_reward = observations["score_cumulative"][0]
        print('Game {} has ended, score: {}'.format(games_played, current_reward))
        cumulative_score += current_reward

        print("Model: {}, cumulative_score: {}".format(model_number, cumulative_score))

        # TODO: Возможно, лучше будет использовать хитрые фичи multiprocessing, а не костыли с сохранением
        # TODO: Иногда score в одном пуле становится одинаковым. Нужно осознать, почему
        with open(self.save_dir + 'score_{}.txt'.format(model_number), mode='w') as f:
            print(cumulative_score, file=f)

    def run(self):
        from model import AtariModel
        model = AtariModel()

        for model_number in range(self.pool_number, self.population_size, self.envs_number):
            t = time.time()
            self.run_model(model, model_number)
            time_spent = time.time() - t
            print('Average time spent: {}'.format(time_spent / self.envs_number))
