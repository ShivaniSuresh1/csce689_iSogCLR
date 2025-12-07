import math
import torch
from torch.optim.optimizer import Optimizer

class AdafactorCustom(Optimizer):
    """
    Adafactor tailored for SogCLR:
    - Uses relative step size + parameter scaling.
    - Adds Gradient Centralization (GC) on weight-like tensors.
    - Adds global grad clipping via max_grad_norm.
    - Uses decoupled weight decay (AdamW style).
    """

    def __init__(self,
                 params,
                 lr=None,
                 eps2=(1e-30, 1e-3),
                 clip_threshold=1.0,
                 decay_rate=-0.8,
                 beta1=None,
                 weight_decay=0.0,
                 scale_parameter=True,
                 relative_step=True,
                 warmup_init=False,
                 gc=True,
                 max_grad_norm=None):
        if lr is not None and lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            eps2=eps2,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
            gc=gc,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _get_lr(self, group, state):
        lr = group['lr']
        if group['relative_step']:
            # time-dependent lr
            step = state['step']
            if group['warmup_init']:
                min_step = 1e-6 * step
            else:
                min_step = 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(step))
            param_scale = 1.0
            if group['scale_parameter']:
                param_scale = max(group['eps2'][1], state['RMS'])
            return param_scale * rel_step_sz
        else:
            return lr

    def _get_options(self, group, shape):
        factored = (len(shape) >= 2)
        use_first_moment = group['beta1'] is not None
        return factored, use_first_moment

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # global grad clipping
        max_norm = None
        for g in self.param_groups:
            if g['max_grad_norm'] is not None:
                max_norm = g['max_grad_norm']
                break
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.param_groups for p in g['params'] if p.grad is not None],
                max_norm
            )

        for group in self.param_groups:
            eps2 = group['eps2']
            clip_threshold = group['clip_threshold']
            decay_rate = group['decay_rate']
            beta1 = group['beta1']
            weight_decay = group['weight_decay']
            use_gc = group['gc']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                # Gradient Centralization for weight-like params
                if use_gc and grad.dim() > 1:
                    grad = grad - grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)

                state = self.state[p]
                shape = grad.shape
                factored, use_first_moment = self._get_options(group, shape)

                # state init
                if len(state) == 0:
                    state['step'] = 0
                    if use_first_moment:
                        state['exp_avg'] = torch.zeros_like(grad)
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(shape[:-1], dtype=grad.dtype, device=grad.device)
                        state['exp_avg_sq_col'] = torch.zeros(shape[:-2] + shape[-1:], dtype=grad.dtype, device=grad.device)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                    state['RMS'] = 0.0

                state['step'] += 1
                state['RMS'] = self._rms(p.data)

                lr = self._get_lr(group, state)

                # squared gradients
                update = grad ** 2 + eps2[0]

                beta2t = 1.0 - math.pow(state['step'], decay_rate)

                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)

                    # approximate squared gradient rms
                    r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
                    c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
                    update = r_factor * c_factor * grad
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = grad * exp_avg_sq.rsqrt_()

                # clip by RMS of update
                rms_update = self._rms(update)
                update.div_(max(1.0, rms_update / clip_threshold))

                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)
                    update = exp_avg

                update.mul_(lr)

                # decoupled weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-weight_decay * lr)

                p.data.add_(-update)

        return loss
