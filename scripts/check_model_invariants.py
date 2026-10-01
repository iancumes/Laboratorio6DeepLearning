"""Verifica forward, backward y SGD de la factorizacion espacial exacta."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from src.pipeline import ReplicatedSpatialLinear,ART

torch.set_num_threads(4);torch.manual_seed(23236)
fast=ReplicatedSpatialLinear(512*49,8,dtype=torch.float64)
original=nn.Linear(512*49,8,dtype=torch.float64)
original.load_state_dict(fast.state_dict())
opt1=torch.optim.SGD(fast.parameters(),lr=.01,momentum=.9,weight_decay=1e-4)
opt2=torch.optim.SGD(original.parameters(),lr=.01,momentum=.9,weight_decay=1e-4)
pool=nn.AdaptiveAvgPool2d((7,7));max_forward=0.
for step in range(3):
    a=torch.randn(3,512,1,1,dtype=torch.float64,requires_grad=True)
    b=a.detach().clone().requires_grad_(True)
    opt1.zero_grad(set_to_none=True);opt2.zero_grad(set_to_none=True)
    out1=fast(pool(a).flatten(1));out2=original(pool(b).flatten(1))
    max_forward=max(max_forward,(out1-out2).abs().max().item())
    torch.testing.assert_close(out1,out2,atol=1e-10,rtol=1e-10)
    out1.square().mean().backward();out2.square().mean().backward()
    torch.testing.assert_close(a.grad,b.grad,atol=1e-10,rtol=1e-10)
    for p,q in zip(fast.parameters(),original.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=1e-10,rtol=1e-10)
    opt1.step();opt2.step()
    for p,q in zip(fast.parameters(),original.parameters()):
        torch.testing.assert_close(p,q,atol=1e-10,rtol=1e-10)
result={'spatial_linear_forward_backward_sgd_equivalent':True,'steps':3,
        'dtype':'float64','maximum_forward_difference':max_forward,
        'tolerance':1e-10,'includes_momentum_and_weight_decay':True}
ART.mkdir(exist_ok=True);(ART/'model_invariants.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result)
