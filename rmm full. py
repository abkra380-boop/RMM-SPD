import torch
import torch.nn as nn
import geoopt
from geoopt.manifolds import SymmetricPositiveDefinite

# RMM Optimizer
class RMM(geoopt.optim.RiemannianOptimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, lambd=0.1):
        defaults = dict(lr=lr, beta=beta, lambd=lambd)
        super(RMM, self).__init__(params, defaults)
        self.state = {}

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta, lambd, lr = group['beta'], group['lambd'], group['lr']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state.setdefault(p, {})
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)
                buf = state['momentum']
                buf.mul_(beta).add_(grad, alpha=1-beta)
                p.data = geoopt.Manifold.retraction(p, -lr * (buf + lambd * grad))

# SPDNet + Training
class SPDNet(nn.Module):
    def __init__(self, input_dim=10, num_classes=10):
        super(SPDNet, self).__init__()
        self.manifold = SymmetricPositiveDefinite()
        self.fc = nn.Sequential(
            nn.Linear(input_dim*(input_dim+1)//2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = self.manifold.log(x).view(x.size(0), -1)
        return self.fc(x)

if __name__ == "__main__":
    print("RMM-SPD Code Loaded. Ready for MSTAR, STAP, DoA")
