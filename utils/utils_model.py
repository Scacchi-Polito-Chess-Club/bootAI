import torch.nn as nn
import torch.optim

from models.autoencoder import AutoEncoder


def get_autoencoder(encoder: str, decoder: str, latent_dim: int) -> nn.Module:
    return AutoEncoder()


def get_optimizer(model: nn.Module, type: str, lr: float):
    if type == 'sgd':
        optim = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=10)
    elif type == 'adam':
        optim = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=10)
    else:
        raise NotImplementedError(f'{type} optimizer not implemented yet')
    return optim


def get_scheduler(optim: torch.optim.lr_scheduler, type: str):
    if type == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=0.95)
    else:
        raise NotImplementedError(f'{type} scheduler not implemented yet')
    # ADD HERE NEW SCHEDULERS
    return scheduler


def get_loss_func(type: str):
    if type == 'mse':
        loss_func = nn.MSELoss()
    elif type == 'cross_ent':
        loss_func = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f'{type} loss function not implemented yet')
    return loss_func
