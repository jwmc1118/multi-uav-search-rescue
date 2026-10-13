"""공용 인코더 블록.

GridCNN : grid 관측 (B, H, W, C) -> 특징벡터 (B, out_dim). 입력은 env 가 주는 channels-last 그대로.
    global_pool=True  : Global Average Pooling 종단 -> 맵 크기 독립 (global / state 맵, summary §8.2)
    global_pool=False : Flatten + FC 종단 -> 공간 배치 보존 (자기 중심 local 맵.
                        "장애물이 앞에 있는지 뒤에 있는지"가 primitive 행동에 중요하므로 GAP 금지)
MLP     : Linear-act 반복 + 출력 Linear
"""
import torch
import torch.nn as nn


def _conv_block(cin, cout, k, s, act):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2), nn.GroupNorm(1, cout), act())


class GridCNN(nn.Module):
    def __init__(self, in_channels, out_dim, filters=((16, 3, 2), (32, 3, 2), (64, 3, 1)),
                 activation=nn.ReLU, global_pool=True, input_hw=None):
        super().__init__()
        layers, cin = [], in_channels
        for (cout, k, s) in filters:
            layers.append(_conv_block(cin, cout, k, s, activation))
            cin = cout
        self.convs = nn.Sequential(*layers)
        self.global_pool = global_pool
        if global_pool:
            flat = cin
        else:
            assert input_hw is not None, "flatten 종단은 입력 크기(input_hw)가 필요"
            with torch.no_grad():
                flat = self.convs(torch.zeros(1, in_channels, *input_hw)).numel()
        self.out = nn.Sequential(nn.Linear(flat, out_dim), activation())
        self.out_dim = out_dim

    def forward(self, x):
        x = self.convs(x.permute(0, 3, 1, 2))           # (B,H,W,C) -> (B,C,H,W)
        x = x.mean(dim=(2, 3)) if self.global_pool else x.flatten(1)
        return self.out(x)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, activation=nn.ReLU):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), activation()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def per_agent(module, x):
    """(B, N, ...) 입력에 module 을 agent 축까지 펼쳐 적용 -> (B, N, out)."""
    B, N = x.shape[:2]
    return module(x.reshape(B * N, *x.shape[2:])).reshape(B, N, -1)
