#based on https://github.com/201419/Optimizer-PyTorch/tree/ce5c0dc96dca0689af2e0a3f0b0bb3821c2a31b0, and modified

import torch
import math
from torch.optim import Optimizer

required = object()


class Extragradient(Optimizer):
    """Base class for optimizers with extrapolation step.
        Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
        extra_steps: (int): number of extrapolating steps per one parameter update.
            By default equals to 2
    """
    
    def __init__(self, params, defaults, extra_steps=1):
        super(Extragradient, self).__init__(params, defaults)
        self.extra_steps = extra_steps
        for group in self.param_groups:
            group['counter'] = 0
            group['params_copy'] = []
            for p in group['params']:
                group['params_copy'].append(p.data.clone())
    
    def update(self, p, group):
        raise NotImplementedError
    
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p, p_copy in zip(group['params'], group['params_copy']):
                u = self.update(p, group)
                if u is None:
                    continue
                if (group['counter'] + 1) % (self.extra_steps + 1) == 0:
                    p.data = p_copy.add_(u)
                    p_copy = p.data.clone()
                else:
                    p.data.add_(u)

            group['counter'] += 1
        
        return loss


class ExtraAdam(Extragradient):
    """Implements the Adam algorithm with extrapolation step.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(ExtraAdam, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(ExtraAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
    
    def update(self, p, group):
        if p.grad is None:
            return None
        grad = p.grad.data
        if grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
        amsgrad = group['amsgrad']
        
        state = self.state[p]
        
        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)
        
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']
        
        state['step'] += 1
        
        if group['weight_decay'] != 0:
            grad = grad.add(group['weight_decay'], p.data)
        
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])
        
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
        
        return -step_size * exp_avg / denom