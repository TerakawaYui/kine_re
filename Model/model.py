"""Static PyTorch adaptation of kine3c345/code/kine/model.py."""
import torch
from torch import nn
class KineStatic(nn.Module):
    def __init__(self, depth=4, width=256, degrees=0):
        super().__init__()
        self.degrees=degrees
        self.layers=nn.ModuleList([nn.Linear(2+4*degrees if i==0 else width,width) for i in range(depth+1)])
        self.norms=nn.ModuleList([nn.BatchNorm1d(width,eps=1e-5,momentum=0.01) for _ in self.layers])
        self.output=nn.Linear(width,1)
        for layer in [*self.layers,self.output]:
            nn.init.kaiming_uniform_(layer.weight,nonlinearity='relu');nn.init.zeros_(layer.bias)
    def forward(self,coords):
        parts=[coords]
        for j in range(2):
            scales=2**torch.arange(self.degrees,device=coords.device)
            z=coords[:,j:j+1]*scales
            parts.extend([z.sin(),z.cos()])
        x=torch.cat(parts,dim=-1)
        for i,(layer,norm) in enumerate(zip(self.layers,self.norms)):
            x=torch.nn.functional.gelu(norm(layer(x)), approximate="tanh")
            if i==0: skip=x
        return torch.nn.functional.softplus(self.output(x+skip).squeeze(-1)-7)
