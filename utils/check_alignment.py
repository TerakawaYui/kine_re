"""Run with: python -m utils.check_alignment"""
import numpy as np
import torch
from ehtim.imaging import imager_utils as iu
from utils.closure import prepare,closure_loss

def main():
    path='/home/terakawayui/projects/kine3c345/observations/m87/uvfits/hops_3597_M87_LO.dcal_full.uvfits'
    obs,proxy,prior,coords,products,flux=prepare(path,16)
    rng=np.random.default_rng(13)
    arr=prior.imvec*(1+0.1*rng.random(len(prior.imvec)))
    image=torch.tensor(arr,dtype=torch.float64,requires_grad=True)
    for name,(A,y,s) in zip(['cphase','logcamp'],products):
        ref=iu.chisq(arr,tuple(A),y*180/np.pi if name=='cphase' else y,s*180/np.pi if name=='cphase' else s,name,ttype='direct')
        actual=closure_loss(image,torch.tensor(A),torch.tensor(y),torch.tensor(s))
        np.testing.assert_allclose(actual.detach().numpy(),ref,rtol=1e-9,atol=1e-9)
        actual.backward();assert torch.isfinite(image.grad).all()
        reference_grad=iu.chisqgrad(arr,tuple(A),y*180/np.pi if name=='cphase' else y,s*180/np.pi if name=='cphase' else s,name,ttype='direct')
        np.testing.assert_allclose(image.grad.numpy(),reference_grad,rtol=1e-7,atol=1e-7)
        print(name,'analytic gradient relative L2:',np.linalg.norm(image.grad.numpy()-reference_grad)/np.linalg.norm(reference_grad))
        image.grad.zero_()
        print(name,'PyTorch/ehtim parity:',float(actual.detach()))
    if torch.cuda.is_available():
        A,y,s=products[0]
        cpu=closure_loss(image.detach().float(),torch.tensor(A,dtype=torch.complex64),torch.tensor(y,dtype=torch.float32),torch.tensor(s,dtype=torch.float32))
        gpu=closure_loss(image.detach().float().cuda(),torch.tensor(A,dtype=torch.complex64,device='cuda'),torch.tensor(y,dtype=torch.float32,device='cuda'),torch.tensor(s,dtype=torch.float32,device='cuda'))
        np.testing.assert_allclose(float(cpu),float(gpu),rtol=1e-3,atol=1e-3)
        print('CPU/CUDA parity passed')
if __name__=='__main__': main()
