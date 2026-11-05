"""M87 closure-only diagnostic; Gaussian prior, no ring template. Run with MPLBACKEND=Agg."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import ehtim as eh
OUT=Path(__file__).parent / "m87_diagnostics"
OUT.mkdir(exist_ok=True)
p="/home/terakawayui/projects/kine3c345/observations/m87/uvfits/hops_3597_M87_LO.dcal_full.uvfits"
obs=eh.obsdata.load_uvfits(p).avg_coherent(120)
u=np.hypot(obs.data['u'],obs.data['v'])
w=1/obs.data['sigma']**2
print("short baseline weight fraction",w[u<1e8].sum()/w.sum(),flush=True)
# Intra-site baselines contain extended flux outside the compact imaging field.
obs=obs.flag_uvdist(uv_min=1e8)
results=[]
fig,axes=plt.subplots(1,2,figsize=(10,5))
for ax,width in zip(axes,[40,60]):
 prior=eh.image.make_square(obs,48,160*eh.RADPERUAS).add_gauss(0.6,(width*eh.RADPERUAS,width*eh.RADPERUAS,0,0,0))
 imgr=eh.imager.Imager(obs,prior,prior_im=prior,flux=0.6,
  data_term={'cphase':1,'logcamp':1,'amp':1},reg_term={'simple':1,'flux':100,'cm':100},
  maxit=500,ttype='direct',stop=1e-6)
 image=imgr.make_image_I(show_updates=False)
 for stage in range(2):
  imgr.init_next=image.blur_circ(obs.res()/2)
  imgr.reg_term_next={'simple':0.1/(stage+1),'flux':100,'cm':100}
  image=imgr.make_image_I(show_updates=False)
 metrics={k:float(obs.chisq(image,dtype=k,ttype='direct')) for k in ['cphase','logcamp']}
 metrics['prior_fwhm_uas']=width
 results.append(metrics)
 image.save_fits(str(OUT/f'rml_prior{width}.fits'))
 arr=image.blur_circ(20*eh.RADPERUAS).imvec.reshape(48,48)
 ax.imshow(arr,extent=[80,-80,-80,80],cmap='afmhot',origin='upper')
 ax.set(title=f'Gaussian prior {width} uas; 20 uas blur',xlabel='RA offset (uas)',ylabel='Dec offset (uas)')
 print(metrics,flush=True)
fig.tight_layout();fig.savefig(OUT/'closure_comparison.png',dpi=180)
(OUT/'metrics.json').write_text(json.dumps(results,indent=2))
