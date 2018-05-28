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
                 game_steps_per_episode,
                 max_games,
                 visualize,
                 model_number,
                 generation,
                 save_dir,
                 activations):
        super(MyEnv, self).__init__(map_name=map_name,
                                    step_mul=step_mul,
                                    screen_size_px=(screen_size, screen_size),
                                    minimap_size_px=(minimap_size, minimap_size),
                                    game_steps_per_episode=0,
                                    visualize=visualize)
        self.game_steps_per_episode = game_steps_per_episode
        self.max_games = max_games
        self.model_number = model_number
        self.generation = generation
        self.save_dir = save_dir

        # Model parameters
        self.activations = activations

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

    def run(self):
        obs = self.reset()
        print('Env {} reset completed'.format(self.model_number))

        cumulative_score = 0

        from model import AtariModel
        model = AtariModel(
            activations=self.activations
        )
        model.load_weights(self.save_dir + 'model_{}.h5'.format(self.model_number))
        print('Model {} loaded'.format(self.model_number))

        games_played = 0
        while True:
            observations = obs[0].observation
            screen, minimap = self.translate_observations(observations)

            result = model.predict([np.array([screen]), np.array([minimap])])

            res_dict = self.result_to_dict(result)

            action = self.get_action(res_dict, observations)

            action_args = self.prepare_args(action, res_dict)

            # call action in new step
            obs = self.step(actions=[FunctionCall(action.id, action_args)])

            if self.state == StepType.FIRST:
                games_played += 1
                current_reward = observations["score_cumulative"][0]
                print('Game {} has ended, score: {}'.format(games_played, current_reward))
                cumulative_score += current_reward

            if games_played >= self.max_games:
                break

        return cumulative_score
