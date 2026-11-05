"""KINE static data adapter: use the same ehtim data products and pixel pulse."""
import numpy as np
import torch
import ehtim as eh
from ehtim.imaging import imager_utils as iu

def prepare(path,npix=64,fov=200,tavg=120,syserr=0.01,init_kind="disk",init_size=60.,init_blur=20.):
    # Match upstream loading, including ehtim's handling of missing parallel hands.
    obs=eh.obsdata.load_uvfits(path)
    if tavg>0: obs=obs.avg_coherent(tavg,scan_avg=False)
    if syserr>0: obs=obs.add_fractional_noise(syserr)
    proxy=eh.image.make_square(obs,npix,fov*eh.RADPERUAS)
    # Upstream get_zbl / static lightcurve: median amplitude on shortest baseline.
    pairs={}
    for row in obs.data:
        key=(str(row['t1']),str(row['t2']))
        pairs.setdefault(key,[]).append(row)
    key=min(pairs,key=lambda k:min(np.hypot(r['u'],r['v']) for r in pairs[k]))
    flux=float(np.median([abs(r['vis']) for r in pairs[key]]))
    # init_size is disk DIAMETER or Gaussian FWHM, in microarcseconds.
    if init_kind == "disk":
        prior=proxy.add_tophat(flux,init_size/2*eh.RADPERUAS)
    elif init_kind == "gauss":
        prior=proxy.add_gauss(flux,[init_size*eh.RADPERUAS,init_size*eh.RADPERUAS,0,0,0])
    else:
        raise ValueError("Unknown initialization: "+init_kind)
    if init_blur>0: prior=prior.blur_circ(init_blur*eh.RADPERUAS)
    prior.imvec*=flux/prior.total_flux()
    products=[]
    for kind in ['cphase','logcamp']:
        y,s,A=iu.chisqdata(obs,proxy,mask=[],dtype=kind,pol='I',weighting='natural',ttype='direct',debias=False,maxset=False)
        A=np.stack(A)
        if kind=='cphase': y,s=np.deg2rad(y),np.deg2rad(s)
        if not len(y) or not np.isfinite(y).all() or not np.isfinite(s).all() or (s<=0).any() or not np.isfinite(A).all():
            raise ValueError('Invalid closure products: '+kind)
        products.append((A,y,s))
    axis=np.linspace(0,1,npix)
    coords=np.stack(np.meshgrid(axis,axis,indexing='ij'),axis=-1).reshape(-1,2)
    return obs,proxy,prior,coords,products,flux

def closure_loss(image,A,target,sigma):
    vis=A@image.to(A.dtype)
    if A.shape[0]==3:
        phase=torch.angle(vis.prod(dim=0))
        return (2*(1-torch.cos(phase-target))/sigma.square()).mean()
    logs=vis.abs().clamp_min(1e-30).log()
    pred=logs[0]+logs[1]-logs[2]-logs[3]
    return ((pred-target)/sigma).square().mean()
