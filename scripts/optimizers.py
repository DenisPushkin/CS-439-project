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

    def __init__(self, parameters, dual_parameters, *, lr=1e-3, alpha=0.9, p=0.95):
        self.lr = lr
        self.alpha = alpha
        self.p = 0.95
        defaults = dict(lr=lr, p=p, alpha=alpha)
        self.optimizer = Optimizer(parameters, defaults)
        self.dual_optimizer = Optimizer(dual_parameters, defaults)
    
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
                    dual_params_with_grad_cp.append(p.detach.clone())
                    dual_grads_cp.append(p.grad.detach.clone())
        
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)
        
        proceed = True
        while proceed:
            self._update_start(params_with_grad, dual_params_with_grad_cp, dual_grads_cp)
            with torch.enable_grad():
                loss, dual_loss = closure()
            self._update_finish(params_with_grad, grads, dual_grads)
            proceed = random.random() < p
        
        return loss, dual_loss

    def update_start(self, params, dual_params, dual_grads):
        for idx, param in enumerate(params):
            param.mul_(self.alpha).add_(dual_params[idx], alpha=1-self.alpha)
            param.add_(dual_grads[idx], alpha=-self.lr)
    
    def update_end(self, params, grads, dual_grads):
        for idx, param in enumerate(params):
            param.add_(grads[idx], alpha=-self.lr)
            param.add_(dual_grads, alpha=self.lr)


        


