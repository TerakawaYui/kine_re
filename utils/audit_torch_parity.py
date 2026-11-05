"""Read-only numerical audit. Requires JAX/Flax/Optax in /tmp/kine_audit_deps."""
import sys,os,types,functools,json
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'
sys.path.append('/tmp/kine_audit_deps')
import numpy as np,torch,jax,jax.numpy as jnp,flax.linen as nn,optax
from typing import Tuple,Callable,Any
from Model import KineStatic

def main():
    cfg=types.ModuleType('config')
    cfg.__dict__.update(dict(nn=nn,functools=functools,jax=jax,jnp=jnp,Tuple=Tuple,Callable=Callable,Any=Any))
    sys.modules['config']=cfg
    namespace={};exec(Path('../kine3c345/code/kine/model.py').read_text(),namespace)
    torch.manual_seed(123);torch.set_num_threads(2)
    net=KineStatic(depth=4,width=32)
    coords=np.random.default_rng(3).uniform(0,1,(64,2)).astype('float32')
    original=namespace['MLP'](posenc_deg=(0,0),depth=4,width=32)
    v=original.init(jax.random.PRNGKey(1),jnp.asarray(coords),1.,train=True)
    for i,l in enumerate([*net.layers,net.output]):
        v['params'][f'Dense_{i}']['kernel']=jnp.array(l.weight.detach().numpy().T)
        v['params'][f'Dense_{i}']['bias']=jnp.array(l.bias.detach().numpy())
    for i,b in enumerate(net.norms):
        v['params'][f'BatchNorm_{i}']['scale']=jnp.array(b.weight.detach().numpy())
        v['params'][f'BatchNorm_{i}']['bias']=jnp.array(b.bias.detach().numpy())
    x=torch.tensor(coords)
    out=net(x)
    def loss(params):
        y,_=original.apply({'params':params,'batch_stats':v['batch_stats']},jnp.asarray(coords),1.,train=True,mutable=['batch_stats'])
        return jnp.mean((y[:,0]*1000)**2)
    jout,_=original.apply(v,jnp.asarray(coords),1.,train=True,mutable=['batch_stats'])
    forward=float(np.max(np.abs(out.detach().numpy()-np.asarray(jout[:,0]))))
    (out*1000).square().mean().backward();jg=jax.grad(loss)(v['params'])
    diffs=[];den=[]
    for i,l in enumerate([*net.layers,net.output]):
        diffs.extend([(l.weight.grad.numpy()-np.asarray(jg[f'Dense_{i}']['kernel']).T).ravel(),(l.bias.grad.numpy()-np.asarray(jg[f'Dense_{i}']['bias'])).ravel()])
        den.extend([np.asarray(jg[f'Dense_{i}']['kernel']).ravel(),np.asarray(jg[f'Dense_{i}']['bias']).ravel()])
    for i,b in enumerate(net.norms):
        for param,key in [(b.weight,'scale'),(b.bias,'bias')]:
            diffs.append((param.grad.numpy()-np.asarray(jg[f'BatchNorm_{i}'][key])).ravel());den.append(np.asarray(jg[f'BatchNorm_{i}'][key]).ravel())
    # Isolate Adamax semantics using identical prescribed gradients, not cross-framework roundoff.
    initial=np.array([1.,2.,3.],dtype='float32');param=torch.nn.Parameter(torch.tensor(initial));optim=torch.optim.Adamax([param],lr=0.01,eps=1e-8)
    tx=optax.adamax(0.01);jp=jnp.array(initial);state=tx.init(jp)
    for g in [np.array([1.,1e-9,0.],dtype='float32'),np.array([0.1,-2e-9,1e-12],dtype='float32')]*10:
        param.grad=torch.tensor(g);optim.step();updates,state=tx.update(jnp.array(g),state,jp);jp=optax.apply_updates(jp,updates)
    result={'same_weight_forward_max_abs':forward,'all_parameter_gradient_relative_l2':float(np.linalg.norm(np.concatenate(diffs))/np.linalg.norm(np.concatenate(den))),'adamax_20_steps_max_abs':float(np.max(abs(param.detach().numpy()-np.asarray(jp)))),'torch_parameter':param.detach().tolist(),'optax_parameter':np.asarray(jp).tolist(),'versions':{'torch':torch.__version__,'jax':jax.__version__,'optax':optax.__version__},'note':'Current available library versions; original repository does not pin these versions.'}
    Path('results/torch_parity_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__': main()
