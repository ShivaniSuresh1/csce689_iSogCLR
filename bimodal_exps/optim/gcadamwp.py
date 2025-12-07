import math
import torch
from torch.optim import Optimizer

class GCAdamWP(Optimizer):
    """
    AdamW + Gradient Centralization + AdamP-style projection + global grad clipping.
    Designed to work well with SogCLR.
    """

    def __init__(self, params, lr=2e-4, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.02,
                 delta=0.1, wd_ratio=0.1,
                 gc=True, max_grad_norm=None):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio,
                        gc=gc, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        # 1) global grad clipping (helps SogCLR spikes)
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
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            delta = group['delta']
            wd_ratio = group['wd_ratio']
            use_gc = group['gc']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # 2) Gradient Centralization on weight-like tensors
                if use_gc and grad.dim() > 1:
                    grad = grad - grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                step = state['step']

                # Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_c1 = 1 - beta1 ** step
                bias_c2 = 1 - beta2 ** step
                denom = (exp_avg_sq / bias_c2).sqrt().add_(eps)
                step_size = lr / bias_c1

                # 3) AdamP-style projection to respect scale-invariance
                w_flat = p.data.view(p.data.size(0), -1)
                g_flat = exp_avg.view(exp_avg.size(0), -1)
                w_norm = w_flat.norm(dim=1, keepdim=True)           # [B,1]
                g_norm = g_flat.norm(dim=1, keepdim=True)           # [B,1]
                cosine = (w_flat * g_flat).sum(dim=1, keepdim=True) / (w_norm * g_norm + 1e-8)  # [B,1]

                proj = cosine.abs() < delta  # [B,1] boolean

                if proj.any():
                    # reshape cosine and w_norm back to broadcast over all non-batch dims
                    cos_b = cosine.view(-1, *([1] * (p.data.dim() - 1)))      # [B,1,1,1,...]
                    wnorm_b = w_norm.view(-1, *([1] * (p.data.dim() - 1)))    # [B,1,1,1,...]

                    # remove component along weight direction for projected rows
                    exp_avg.add_(-cos_b * p.data / (wnorm_b + 1e-8))

                # 4) Decoupled weight decay (AdamW) with projection-aware ratio
                if weight_decay != 0:
                    ratio = wd_ratio if proj.any() else 1.0
                    p.data.add_(p.data, alpha=-weight_decay * ratio * lr)

                # 5) Parameter update
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
