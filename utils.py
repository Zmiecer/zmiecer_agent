import numpy as np

from keras import backend as K

OUTPUT_SHAPES = (
    ('action', 524),
    ('x1', 64),
    ('y1', 64),
    ('x2', 64),
    ('y2', 64),
    ('select_unit_act', 4),
    ('queued', 2),
    ('control_group_act', 5),
    ('select_point_act', 4),
    ('unload_id', 500),
    ('select_worker', 4),
    ('build_queue_id', 10),
    ('select_add', 2),
    ('select_unit_id', 500),
    ('control_group_id', 10),
)


def reshape_channels(observation):
    data_format = K.image_data_format()
    if data_format == 'channels_last':
        observation = np.moveaxis(observation, 0, -1)
        return observation

    return observation


def mask_unavailable_actions(available_actions, action_probs):
    mask = np.zeros(action_probs[0].shape)
    mask[available_actions] = 1
    action_probs[0] *= mask
    return action_probs
