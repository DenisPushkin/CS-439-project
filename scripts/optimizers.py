import random

import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import Optimizer


def closure():
    """
    Runs the inference of the model
    dual=True run the copied model
    """
    pass


class Extragrad_Var_Reduction():

    def __init__(
        self, parameters, dual_parameters,  *,
        lr=1e-3, betas=(0.5, 0.999),
        alpha=0.9, p=0.95,
        optimizer=Optimizer
    ):
        self.lr = lr
        self.alpha = alpha
        self.p = 0.95
        defaults = dict(lr=lr, betas=betas)
        self.optimizer = optimizer(parameters, **defaults)
        self.dual_optimizer = optimizer(dual_parameters, **defaults)
    
    @torch.no_grad()
    def step(self, closure):

        params_with_grad = []
        grads = []
        dual_params_with_grad = []
        dual_grads = []
        dual_params_with_grad_cp = []
        dual_grads_cp = []

        loss, dual_loss = None, None

        for group in self.dual_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    dual_params_with_grad.append(p)
                    dual_grads.append(p.grad)
                    dual_params_with_grad_cp.append(p.detach().clone())
                    dual_grads_cp.append(p.grad.detach().clone())
        
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
        
        proceed = True
        while proceed:
            self._update_start(params_with_grad, dual_params_with_grad_cp)
            self._set_grads(params_with_grad, dual_grads_cp)
            self.optimizer.step()
            
            with torch.enable_grad():
                loss, dual_loss = closure()
            
            for idx, param in enumerate(params_with_grad):
                param.grad.add_(dual_grads[idx], alpha=-1)
            self.optimizer.step()
            
            proceed = random.random() < self.p
        
        return loss, dual_loss

    def _update_start(self, params, dual_params):
        for idx, param in enumerate(params):
            param.mul_(self.alpha).add_(dual_params[idx], alpha=1-self.alpha)
    
    def _set_grads(self, params, new_grads):
        for idx, param in enumerate(params):
            param.grad = new_grads[idx]
            
    def zero_grad(self):
        self.optimizer.zero_grad()
        self.dual_optimizer.zero_grad()


        


