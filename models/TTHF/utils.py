from __future__ import division
import torch
import os
import glob


def save_one_model(model_path, max_to_save=8):
    total_models = glob.glob(model_path + '/*.pt')
    if len(total_models) >= max_to_save:
        total_models.sort()
        os.remove(total_models[0])


def only_model_saver(model_state_dict, model_path):
    state_dict = {}
    state_dict["model_state_dict"] = model_state_dict
    torch.save(state_dict, model_path)
    print('models {} save successfully!'.format(model_path))