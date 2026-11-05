"""Static PyTorch KINE: upstream-aligned closure training and configurable initialization."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from Model import KineStatic
from utils.closure import prepare,closure_loss

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--uvfits',default='/home/terakawayui/projects/kine3c345/observations/m87/uvfits/hops_3597_M87_LO.dcal_full.uvfits')
    for key,value in [('npix',64),('steps',10000),('init-steps',10000),('width',256),('depth',4),('degrees',0),('seed',1)]: p.add_argument('--'+key,type=int,default=value)
    for key,value in [('fov',200.),('lr',0.01),('tavg',120.),('syserr',0.01),('init-size',60.),('init-blur',20.)]: p.add_argument('--'+key,type=float,default=value)
    p.add_argument('--init-kind',choices=['disk','gauss'],default='disk')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--out',default='results/kine_aligned')
    args=p.parse_args()

    if args.npix<2 or args.fov<=0 or args.init_size<=0 or args.init_blur<0 or min(args.steps,args.init_steps)<0: p.error('Invalid grid or initialization')
    
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed)
    if device=='cuda': torch.backends.cuda.matmul.allow_tf32=False

    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)

    obs,proxy,prior,coords,products,flux=prepare(args.uvfits,args.npix,args.fov,args.tavg,args.syserr,args.init_kind,args.init_size,args.init_blur)
    
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def save_stage(values,name):
        values=np.asarray(values).reshape(args.npix,args.npix)
        np.save(out/(name+'.npy'),values)
        obj=proxy.copy();obj.imvec=values.ravel();obj.save_fits(str(out/(name+'.fits')))
        fig,ax=plt.subplots()
        im=ax.imshow(values,origin='upper',extent=[args.fov/2,-args.fov/2,-args.fov/2,args.fov/2],cmap='afmhot')
        ax.set(title=name,xlabel='RA offset (uas)',ylabel='Dec offset (uas)')
        fig.colorbar(im,ax=ax,label='Jy / pixel');fig.savefig(out/(name+'.png'),dpi=180);plt.close(fig)
    save_stage(prior.imvec,'init_target')

    coords=torch.as_tensor(coords,dtype=torch.float32,device=device)
    target=torch.as_tensor(prior.imvec,dtype=torch.float32,device=device)
    data=[tuple(torch.as_tensor(x,dtype=torch.complex64 if i==0 else torch.float32,device=device) for i,x in enumerate(item)) for item in products]
    model=KineStatic(args.depth,args.width,args.degrees).to(device)
    opt=torch.optim.Adamax(model.parameters(),lr=args.lr,betas=(0.9,0.999),eps=1e-8)
    history=[];start=time.time()

    print('Device:',device,'initial flux:',flux,'closure counts:',[len(x[1]) for x in products],flush=True)

    for step in range(args.init_steps+args.steps):
        lr=args.lr*0.01**sum(step>=b for b in [10000,20000])
        for group in opt.param_groups: group['lr']=lr
        image=model(coords)
        if step<args.init_steps:
            loss=(image-target).square().mean();metrics={'init_mse':float(loss.detach())}
        else:
            losses=[closure_loss(image,*item) for item in data]
            loss=sum(losses);metrics={'cphase':float(losses[0].detach()),'logcamp':float(losses[1].detach())}
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        opt.zero_grad();loss.backward();opt.step()
        if step==args.init_steps-1:
            with torch.no_grad(): save_stage(model(coords).cpu().numpy(),'pretrained')
        if step%100==0 or step==args.init_steps+args.steps-1:
            record={'step':step,'lr':lr,**metrics};history.append(record);print(record,flush=True)
    with torch.no_grad():
        image=model(coords);scores=[float(closure_loss(image,*item)) for item in data]
        
    arr=image.cpu().numpy().reshape(args.npix,args.npix);save_stage(arr,'image')
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'config':vars(args),'device':device},out/'model.pt')
    (out/'metrics.json').write_text(json.dumps({'config':vars(args),'device':device,'initial_flux_jy':flux,'final_flux_jy':float(arr.sum()),'counts':[len(x[1]) for x in products],'full_chi2':scores,'elapsed_seconds':time.time()-start,'history':history},indent=2))
    print('Full closure chi2:',scores,flush=True)
if __name__=='__main__': main()
