import torch
# optimizer
from torch.optim import SGD, Adam, AdamW
import torch_optimizer as optim
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR
from .warmup_scheduler import GradualWarmupScheduler

from .visualization import *

def get_parameters(models):
    """Get all model parameters recursively."""
    parameters = []
    if isinstance(models, list):
        for model in models:
            parameters += get_parameters(model)
    elif isinstance(models, dict):
        for model in models.values():
            parameters += get_parameters(model)
    else: # models is actually a single pytorch model
        parameters += list(models.parameters())
    return parameters

def get_optimizer_tcnn(hparams, model):
    optimizer = torch.optim.Adam(model['coarse'].get_params(hparams.lr), betas=(0.9, 0.99), eps=1e-6)
    return optimizer

def get_scheduler_tcnn(hparams, optimizer):
    scheduler = LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / hparams.iters, 1))
    return scheduler

def get_optimizer(hparams, models):
    eps = 1e-8
    parameters = get_parameters(models)
    if hparams.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=hparams.lr, 
                        momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'adam':
        # optimizer = Adam(parameters, lr=hparams.lr, eps=eps, 
        #                  weight_decay=hparams.weight_decay)
        optimizer = AdamW(parameters, lr=hparams.lr, eps=eps, 
                         weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'radam':
        optimizer = optim.RAdam(parameters, lr=hparams.lr, eps=eps, 
                                weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'ranger':
        optimizer = optim.Ranger(parameters, lr=hparams.lr, eps=eps, 
                                 weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer


def get_optimizer_latent(hparams, shape_latent, appearance_latent):
    latent_opt = torch.optim.AdamW([
            {'params':shape_latent.parameters(), 'lr': hparams.latent_lr},
            {'params':appearance_latent.parameters(), 'lr':hparams.latent_lr}
        ])

    return latent_opt

def get_optimizer_latent_opt(hparams, shape_latent, appearance_latent):
    latent_opt = torch.optim.AdamW([
            {'params':shape_latent, 'lr': hparams.latent_lr},
            {'params':appearance_latent, 'lr':hparams.latent_lr}
        ])

    return latent_opt

def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams.lr_scheduler == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step, 
                                gamma=hparams.decay_gamma)
    elif hparams.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)
    elif hparams.lr_scheduler == 'poly':
        scheduler = LambdaLR(optimizer, 
                             lambda epoch: (1-epoch/(hparams.num_epochs))**hparams.poly_exp)
    else:
        raise ValueError('scheduler not recognized!')

    if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier, 
                                           total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)

    return scheduler


def get_scheduler_latent(hparams, optimizer):
    eps = 1e-8
    if hparams.lr_scheduler_latent == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step, 
                                gamma=hparams.decay_gamma)
    elif hparams.lr_scheduler_latent == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)
    # elif hparams.lr_scheduler_latent == 'poly':
    #     scheduler = LambdaLR(optimizer, 
    #                          lambda epoch: (1-epoch/(hparams.num_epochs*2))**hparams.poly_exp)
    elif hparams.lr_scheduler_latent == 'poly':
        scheduler = LambdaLR(optimizer, 
                             lambda epoch: (1-epoch/(hparams.num_epochs))**hparams.poly_exp)
    else:
        raise ValueError('scheduler not recognized!')

    if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier, 
                                           total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)

    return scheduler

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v
    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[], load_latent = True):
    if not ckpt_path:
        return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def load_latent_codes(ckpt_path):
    checkpoint_latent_codes = torch.load(ckpt_path, map_location=torch.device('cpu'))
    shape_code_params = checkpoint_latent_codes['state_dict']['shape_codes.weight']
    texture_code_params = checkpoint_latent_codes['state_dict']['texture_codes.weight']
    return shape_code_params, texture_code_params