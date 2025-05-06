import itertools
import math
import numpy as np
import os
import pickle
import torch

from collections import OrderedDict

import functools
import operator

import torch.nn as nn


def freeze_vit_backbone(model_type,model):
    if 'sam' in model_type :
        vit_model = model.sam_model.model.image_encoder
        for name,para in vit_model.named_parameters():
            para.requires_grad = False
        print(f'sam image-encoder is frozen ')
    # elif 'dinov2' in model_type:
    #     vit_model = model.vit_model
    #     for name,para in vit_model.named_parameters():
    #         para.requires_grad = False
    #     print(f'dinov2 image-encoder is frozen ')
    else:
        print(f'image-encoder is unfrezon ')
    
def prepare_optim_sched(model , optimcfg,  schedcfg ):
    # parameter to optimize
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
  
    # SGD
    if optimcfg.type == 'sgd':
        optimizer = optimcfg.cls(params_to_optimize, lr=optimcfg.lr, momentum=optimcfg.momentum , weight_decay = optimcfg.weight_decay)
    # Adam
    elif optimcfg.type == 'adam':
        optimizer = optimcfg.cls(params_to_optimize, lr=optimcfg.lr, weight_decay=optimcfg.weight_decay)

    # LinearLR and CosineAnnealingLR
    warm_sched = schedcfg.warm.cls(optimizer, schedcfg.warm.ini.start_factor, schedcfg.warm.ini.end_factor, schedcfg.warm.ini.total_iters)

    main_sched = schedcfg.main.cls(optimizer, schedcfg.main.ini.T_max ,schedcfg.main.ini.eta_min)
    lr_scheduler = schedcfg.cls(optimizer, [warm_sched, main_sched], [schedcfg.warm.warm_iters])
    return optimizer ,lr_scheduler

def load_checkpoint(cfg):
    dir_chk = os.path.join(cfg.output, 'checkpoints')
    # build file path
    if cfg.epoch != -1:
        path = os.path.join(dir_chk, 'model-{:02d}.pt'.format(cfg.epoch))
    else:
        raise FileNotFoundError()
    if not os.path.exists(path):
        raise FileNotFoundError()
    print('load file {}'.format(path))
    checkpoint = torch.load(path, map_location=cfg.device)
    # 判断是不是nn.DataParallel包装的模型
    if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
        weights = checkpoint['model_state_dict']
        checkpoint['model_state_dict'] = OrderedDict([(k[7:], v) for k, v in weights.items()])
            
    return checkpoint

'''
fpn_load(cfg, model)
'''
def fpn_load(cfg , model):
    checkpoint = torch.load(cfg.directly_load,map_location=torch.device('cpu'))
    # 判断是不是nn.DataParallel包装的模型
    if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
        weights = checkpoint['model_state_dict']
        checkpoint['model_state_dict'] = OrderedDict([(k[7:], v) for k, v in weights.items()])

    ckp = checkpoint['model_state_dict']
    ckp_update = {}
    for key in ckp.keys():
       tmp = key
       key = key.replace('fpn_model.backbone','vst_model')
       key = key.replace('fpn_model.fpn','fpn')

       key = key.replace('fpn_model.bn.weight','reducer.reducer.2.weight')
       key = key.replace('fpn_model.bn.bias','reducer.reducer.2.bias')
       key = key.replace('fpn_model.linear.weight','reducer.reducer.3.weight')
       key = key.replace('fpn_model.linear.bias','reducer.reducer.3.bias')
       key = key.replace('prompt_linear.weight','reducer.cls.0.weight')
       key = key.replace('prompt_linear.bias','reducer.cls.0.bias')


       key = key.replace('rnn_bn.weight','anomaly_regressor.ln.weight')
       key = key.replace('rnn_bn.bias','anomaly_regressor.ln.bias')

       key = key.replace('rnn.','anomaly_regressor.rnn.')
       key = key.replace('rnn_linear.weight','anomaly_regressor.cls.0.weight')
       key = key.replace('rnn_linear.bias','anomaly_regressor.cls.0.bias')
       key = key.replace('cls_linear.weight','anomaly_regressor.cls.3.weight')
       key = key.replace('cls_linear.bias','anomaly_regressor.cls.3.bias')
    
       ckp_update[key] = ckp[tmp]

    model.load_state_dict(ckp_update)
    return ckp_update
    
def direcrtly_load_checkpoint(cfg , model , whole_load = False):
    checkpoint = torch.load(cfg.directly_load,map_location=next(model.parameters()).device)
    # 判断是不是nn.DataParallel包装的模型 ,得到vst backbone 的参数
    checkpoint = checkpoint['model_state_dict']
    if 'module' in list(checkpoint.keys())[0]:
        checkpoint = dict([(k[7:], v) for k, v in checkpoint.items()])

    if not whole_load:
        ckp_update, fpn_update = {} , {}
        for key in checkpoint.keys():
            if 'vst_model' in key:
                key_up = key.replace('vst_model.','')
                ckp_update[key_up] = checkpoint[key]
            if 'fpn' in key:
                key_up = key.replace('fpn.','')
                fpn_update[key_up] = checkpoint[key]                             
        missing_keys, unexpected_keys = model.vst_model.load_state_dict(ckp_update,strict=True)
        assert not unexpected_keys and not missing_keys , f'{unexpected_keys=}\n{missing_keys=}'
        missing_keys, unexpected_keys = model.fpn.load_state_dict(fpn_update,strict=True)
        assert not unexpected_keys and not missing_keys , f'{unexpected_keys=}\n{missing_keys=}'
    else:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint,strict=True)
        assert not unexpected_keys and not missing_keys , f'{unexpected_keys=}\n{missing_keys=}'

# "/data/qh/DoTA/output/Prompt/standart_fpn_lr=0.002/checkpoints/model-200.pt"
def resume_from_checkpoint(cfg , model , optimizer , lr_scheduler):
    # directly load : this means Two models are not completely equal , customize load model_state_dict
    checkpoint = {}
    if cfg.directly_load and cfg.epoch == -1:
        whole_load = cfg.get('whole_load',False)
        print(f'directly load from {cfg.directly_load}')
        direcrtly_load_checkpoint(cfg,model,whole_load)
        print(f'epoch set to 0')
        cfg.epoch  = 0  # launch first epoch    
    else:
        try:
            checkpoint = load_checkpoint(cfg)   
            cfg.epoch = checkpoint['epoch'] + 1 # move to next epoch
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'],strict=False)
            if missing_keys:
                print(f'missing_keys:\n{unexpected_keys}')
            if unexpected_keys:
                print(f'unexpected_keys:\n{unexpected_keys}')
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if lr_scheduler:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])       
        except FileNotFoundError:
            print('no checkpoint found')

            if cfg.epoch != -1:
                raise Exception(f'epoch={cfg.epoch} but not find checkpoint')
            else:
                print(f'epoch set to 0')
                cfg.epoch  = 0  # launch first epoch 
    return checkpoint

def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)


def get_visual_directory(cfg, epoch):
    output_dir = os.path.join(cfg.output, 'vis-{:02d}'.format(epoch))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_result_filename(cfg, epoch):
    output_dir = os.path.join(cfg.output, 'eval')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, 'results-{:02d}.pkl'.format(epoch))


def load_results(filename):
    print('load file {}'.format(filename))
    with open(filename, 'rb') as f:
        content = pickle.load(f)
    return content

def get_last_epoch(filenames):
    epochs = [int(name.split('-')[1].split('.')[0]) for name in filenames]
    return filenames[np.array(epochs).argsort()[-1]]


def w_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flat_list(list_):
    if isinstance(list_, (np.ndarray, np.generic)):
        # to be retrocompatible
        return list_
    return list(itertools.chain(*list_))


def filter_by_class(outputs, info, cls):
    return [out for out, inf in zip(outputs, info.tolist()) if inf[0] == cls]


def filter_by_class_ego(outputs, info, cls, ego):
    return [out for out, inf in zip(outputs, info.tolist())
            if all([inf[0] == cls, inf[1] == ego])]

def split_by_class(outputs, targets, info):
    clss = np.unique(info[:, 0]).tolist()
    return {
        cls: {
            'outputs': np.array(
                flat_list(filter_by_class(outputs, info, cls))),
            'targets': np.array(
                flat_list(filter_by_class(targets, info, cls))),
        } for cls in clss
    }


def merge_oo_class(splitted):
    def _cat(v1, v2):
        return np.concatenate([v1, v2], axis=0)

    if (8.0, 0.0) in splitted and (9.0, 0.0) in splitted:
        oo0_outputs = _cat(splitted[(8.0, 0.0)]['outputs'],
                        splitted[(9.0, 0.0)]['outputs'])
        oo1_outputs = _cat(splitted[(8.0, 1.0)]['outputs'],
                        splitted[(9.0, 1.0)]['outputs'])
        oo0_targets = _cat(splitted[(8.0, 0.0)]['targets'],
                        splitted[(9.0, 0.0)]['targets'])
        oo1_targets = _cat(splitted[(8.0, 1.0)]['targets'],
                        splitted[(9.0, 1.0)]['targets'])
        
        lables = list(splitted.keys())
        ids = [x[0] for x in lables]
        next_id = max(ids) + 1
        # OO (11)  = OO-r (8) + OO-l (9)
        splitted[(next_id, 0.0)] = {
            'outputs': oo0_outputs,
            'targets': oo0_targets,
        }
        splitted[(next_id, 1.0)] = {
            'outputs': oo1_outputs,
            'targets': oo1_targets,
        }

    return splitted


def split_by_class_ego(outputs, targets, info):
    clss = np.unique(info[:, 0]).tolist()
    egos = np.unique(info[:, 1]).tolist()
    pairs = [itertools.chain(*li.tolist()) for li in np.meshgrid(clss, egos)]
    pairs = zip(*pairs)
    return merge_oo_class({
        (cls, ego): {
            'outputs': np.array(
                flat_list(filter_by_class_ego(outputs, info, cls, ego))),
            'targets': np.array(
                flat_list(filter_by_class_ego(targets, info, cls, ego))),
        } for cls, ego in pairs
    })


def get_abs_weights_grads(model):
    if isinstance(model , nn.Module):
        return torch.cat([
                p.grad.detach().view(-1) for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]).abs()
    elif isinstance(model, nn.Embedding):
        return model.weight.detach().view(-1)


def get_abs_weights(model):
    if isinstance(model , nn.Module):   
        return torch.cat([
            p.detach().view(-1) for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]).abs()
    elif isinstance(model, nn.Embedding):
        return model.weight.gard.detach().view(-1)

def log_vals(writer, model, global_key, key, fun, index_l):
    if w_count(model):
        vals = fun(model)
        writer.add_scalar(
            '{}_mean/{}'.format(global_key, key), vals.mean().item(), index_l)
        # writer.add_scalar(
        #     '{}_std/{}'.format(global_key, key), vals.std().item(), index_l)
        # writer.add_scalar(
        #     '{}_max/{}'.format(global_key, key), vals.max().item(), index_l)



def get_params(model, keys):
    return [param for name, param in model.named_parameters()
            if any([key in name for key in keys])]


def get_params_rest(model, keys):
    return [param for name, param in model.named_parameters()
            if all([key not in name for key in keys])]


def debug_weights(model_type, writer, model, loss_dict, index_l,
                  debug_train_weight, debug_train_grad, debug_loss,
                  debug_train_grad_level, debug_train_weight_level,
                  ):

    if debug_train_grad:
        scan_internal( model_type,
            writer, model, 'grads', get_abs_weights_grads,
            debug_train_grad_level, index_l)

    if debug_train_weight:
        scan_internal(model_type,
            writer, model, 'weights', get_abs_weights,
            debug_train_weight_level, index_l)

    if debug_loss:
        for key , value in loss_dict.items():
            writer.add_scalar(f'loss/{key}', value.item(), index_l)

def debug_guess(writer, outputs, targets, index):
    """Debug guess info."""
    f_tp = outputs == targets
    f_t_1 = targets == 1
    f_t_0 = targets == 0

    ok = len(outputs[f_tp])
    tpos = len(outputs[f_t_1])  # totally pos
    tneg = len(outputs[f_t_0])  # totally neg
    cpos = len(outputs[f_tp & f_t_1])  # true pos
    cneg = len(outputs[f_tp & f_t_0])  # true neg
    tot = prod(outputs.shape)

    g_all = ok / tot
    g_pos = (cpos / tpos) if tpos > 0 else 0
    g_neg = (cneg / tneg) if tneg > 0 else 0
    writer.add_scalar('guess/all', g_all, index)
    writer.add_scalar('guess/pos', g_pos, index)
    writer.add_scalar('guess/neg', g_neg, index)


def scan_internal(model_type, writer, model, global_key, fun, level, index_l):
  
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module
    
    # base_plain
    if model_type == 'base_plain':
        log_vals(writer, model.anomaly_regressor.reducer, global_key, 'ano_regressor.reducer', fun, index_l)
        log_vals(writer, model.anomaly_regressor.cls, global_key, 'ano_regressor.cls', fun, index_l)

    elif model_type == 'base_rnn': 
        log_vals(writer, model.sam_model.model.image_encoder.blocks[0], global_key, 'vit.block[0]', fun, index_l)
        log_vals(writer, model.sam_model.model.image_encoder.blocks[11], global_key, 'vit.block[11]', fun, index_l)
        log_vals(writer, model.sam_model.model.image_encoder.neck, global_key, 'vit.neck', fun, index_l)
        log_vals(writer, model.reducer, global_key, 'reducer', fun, index_l)
        log_vals(writer, model.anomaly_regressor.rnn, global_key, 'ano_regressor.rnn', fun, index_l)
        log_vals(writer, model.anomaly_regressor.cls, global_key, 'ano_regressor.cls', fun, index_l)

    elif model_type=='prompt_rnn':
        # level  base -> whole model 

        # level 1 -> sam_model bottle_aug bottle_regressor instance_decoder reducer anomaly_regressor
        if level > 0:
            log_vals(writer, model.sam_model.model.mask_decoder, global_key, 'sam_model.mask_decoder', fun, index_l)
            if model.use_bottle_aug:
                log_vals(writer, model.bottle_aug, global_key, 'bottle_aug', fun, index_l)
                log_vals(writer, model.bottle_regressor, global_key, 'bottle_regressor', fun, index_l)
            log_vals(writer, model.instance_decoder, global_key, 'instance_decoder', fun, index_l)
            log_vals(writer, model.reducer, global_key, 'reducer', fun, index_l)
            log_vals(writer, model.anomaly_regressor, global_key, 'anomaly_regressor', fun, index_l)

        # level 1 -> sam_model bottle_aug bottle_regressor instance_decoder reducer anomaly_regressor
        if  level > 1:
            log_vals(writer, model.sam_model.model.prompt_encoder.point_embeddings[2], global_key, 'sam_model.pnt_embeds[2]', fun, index_l)
            log_vals(writer, model.sam_model.model.prompt_encoder.point_embeddings[3], global_key, 'sam_model.pnt_embeds[3]', fun, index_l)
            log_vals(writer, model.instance_decoder.layer[0], global_key, 'instance_layer[0]', fun, index_l)
            log_vals(writer, model.instance_decoder.layer[1], global_key, 'instance_layer[1]', fun, index_l)
            log_vals(writer, model.anomaly_regressor.rnn, global_key, 'anomaly_regressor.rnn', fun, index_l)
            log_vals(writer, model.anomaly_regressor.cls, global_key, 'anomaly_regressor.cls', fun, index_l)
    
    elif model_type == 'poma_base_fpn_rnn':
        log_vals(writer, model.vst_model.patch_embed, global_key, 'vst_model.patch_embed', fun, index_l)
        log_vals(writer, model.vst_model.layers[0], global_key, 'vst_model.layers[0]', fun, index_l)
        log_vals(writer, model.vst_model.layers[1], global_key, 'vst_model.layers[1]', fun, index_l)
        log_vals(writer, model.vst_model.layers[2], global_key, 'vst_model.layers[2]', fun, index_l)
        log_vals(writer, model.vst_model.layers[3], global_key, 'vst_model.layers[3]', fun, index_l)
        log_vals(writer, model.fpn, global_key, 'fpn', fun, index_l)
        log_vals(writer, model.reducer, global_key, 'reducer', fun, index_l)
        log_vals(writer, model.anomaly_regressor, global_key, 'anomaly_regressor', fun, index_l)
    
    elif model_type == 'poma_prompt_fpn_rnn':
        log_vals(writer, model.vst_model.patch_embed, global_key, 'vst_model.patch_embed', fun, index_l)
        log_vals(writer, model.vst_model.layers[0], global_key, 'vst_model.layers[0]', fun, index_l)
        log_vals(writer, model.vst_model.layers[1], global_key, 'vst_model.layers[1]', fun, index_l)
        log_vals(writer, model.vst_model.layers[2], global_key, 'vst_model.layers[2]', fun, index_l)
        log_vals(writer, model.vst_model.layers[3], global_key, 'vst_model.layers[3]', fun, index_l)
        log_vals(writer, model.fpn, global_key, 'fpn', fun, index_l)
        log_vals(writer, model.instance_encoder.prompt_decoder.transformer.layers[0], global_key, 'prompt_decoder.layers[0]', fun, index_l)
        log_vals(writer, model.instance_encoder.prompt_decoder.transformer.layers[1], global_key, 'prompt_decoder.layers[1]', fun, index_l)
        log_vals(writer, model.instance_decoder.layer[0], global_key, 'instance_decoder.layers[0]', fun, index_l)
        log_vals(writer, model.instance_decoder.layer[1], global_key, 'instance_decoder.layers[1]', fun, index_l)
        log_vals(writer, model.reducer, global_key, 'reducer', fun, index_l)
        log_vals(writer, model.anomaly_regressor, global_key, 'anomaly_regressor', fun, index_l)
        

        