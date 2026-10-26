"""ValueNorm (MAPPO 방식) — value target 의 running 평균 / 분산으로 critic 출력 스케일을 정규화.

critic 은 "정규화된 값" 을 출력하고, rollout / GAE 에는 denormalize 한 실제 스케일을 쓴다.
running 통계는 nn.Module buffer 라 state_dict (= policy.get_weights()) 에 포함되어 워커와 동기화된다.

    debiased EMA:  m  <- β m  + (1-β) mean(x)
                   m2 <- β m2 + (1-β) mean(x^2)
                   w  <- β w  + (1-β)              (debiasing term)
                   μ = m / w,  σ² = max(m2 / w - μ², var_min)
β 가 1 에 가까워도 w 로 나누므로 초반 추정이 0 쪽으로 치우치지 않는다 (사실상 지금까지 본 데이터 평균에서 시작).
"""
import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    def __init__(self, beta=0.99999, epsilon=1e-5, var_min=1e-2):
        super().__init__()
        self.beta, self.epsilon, self.var_min = beta, epsilon, var_min
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_mean_sq", torch.zeros(1))
        self.register_buffer("debiasing_term", torch.zeros(1))

    def stats(self):
        """(μ, σ). 아직 한 번도 update 안 했으면 (0, 1) — 정규화가 항등이 된다."""
        if float(self.debiasing_term) <= 0.0:
            return torch.zeros_like(self.running_mean), torch.ones_like(self.running_mean)
        w = self.debiasing_term.clamp(min=1e-12)         # float32 에서 1-β 가 ε 보다 살짝 작을 수 있어 ε 대신 아주 작은 값
        mean = self.running_mean / w
        var = (self.running_mean_sq / w - mean ** 2).clamp(min=self.var_min)
        return mean, var.sqrt()

    @torch.no_grad()
    def update(self, x):
        x = x.detach().reshape(-1).to(self.running_mean)
        if x.numel() == 0:
            return
        b = self.beta
        self.running_mean.mul_(b).add_((1 - b) * x.mean())
        self.running_mean_sq.mul_(b).add_((1 - b) * (x ** 2).mean())
        self.debiasing_term.mul_(b).add_(1 - b)

    def normalize(self, x):
        mean, std = self.stats()
        return (x - mean.to(x)) / std.to(x)

    def denormalize(self, x):
        mean, std = self.stats()
        return x * std.to(x) + mean.to(x)
