import random

import torch
from torch.optim import Optimizer
from .lookahead import Lookahead


class Extragrad_Var_Reduction():

    def __init__(
        self, parameters, dual_parameters,  *,
        lr=1e-3, betas=(0.9, 0.999),
        alpha=0.9, p=0.95, momentum=0,
        optimizer=Optimizer,
        use_lookahead=False
    ):
        self.lr = lr
        self.alpha = alpha
        self.p = p
        defaults = dict(lr=lr)
        if optimizer is torch.optim.Adam:
            defaults['betas'] = betas
        if optimizer is torch.optim.SGD:
            defaults['momentum'] = momentum
        self.optimizer = optimizer(parameters, **defaults)
        if use_lookahead:
            self.optimizer = Lookahead(self.optimizer, k=5, alpha=0.5)
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
            
            self.zero_grad()
            with torch.enable_grad():
                loss, dual_loss = closure()
            
            for idx, param in enumerate(params_with_grad):
                param.grad.add_(dual_grads[idx], alpha=-1)
            self.optimizer.step()
            
            proceed = random.random() < 1 - self.p
        
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


class Extragrad_Var_Reduction_Original():

    def __init__(
        self, parameters, dual_parameters,  *,
        lr=1e-3, alpha=0.9, p=0.95,
        optimizer=Optimizer
    ):
        self.lr = lr
        self.alpha = alpha
        self.p = p
        defaults = dict(lr=lr)
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
            self._update_start(params_with_grad, dual_params_with_grad_cp, dual_grads_cp)
            
            self.zero_grad()
            with torch.enable_grad():
                loss, dual_loss = closure()
            
            self._update_finish(params_with_grad, grads, dual_grads)
            
            proceed = random.random() < 1 - self.p
        
        return loss, dual_loss

    def _update_start(self, params, dual_params, dual_grads):
        for idx, param in enumerate(params):
            param.mul_(self.alpha).add_(dual_params[idx], alpha=1-self.alpha)
            param.add_(dual_grads[idx], alpha=-self.lr)
            
    def _update_finish(self, params, grads, dual_grads):
        for idx, param in enumerate(params):
            param.add_(grads[idx], alpha=-self.lr).add_(dual_grads[idx], alpha=self.lr)
            
    def zero_grad(self):
        self.optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        


