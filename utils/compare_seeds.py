"""Compare existing runs after flux normalization and translation alignment."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter,shift
from scipy.signal import correlate
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');p.add_argument('--out',default='results/seed_comparison');a=p.parse_args()
 paths=[Path(a.first),Path(a.second)];out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 meta=[json.loads((x/'metrics.json').read_text()) for x in paths]
 fov=meta[0]['config']['fov'];n=meta[0]['config']['npix'];pix=fov/n
 if any(d['config']['fov']!=fov or d['config']['npix']!=n for d in meta):raise ValueError('Different grids')
 images=[np.load(x/'image.npy').astype(float) for x in paths];images=[x/x.sum() for x in images]
 # Align on a common 20 uas resolution; do not rotate or reflect.
 blurred=[gaussian_filter(x,20/2.35482/pix,mode='constant') for x in images]
 c=correlate(blurred[0],blurred[1],mode='full',method='fft');delta=np.array(np.unravel_index(c.argmax(),c.shape))-(n-1)
 fig,axes=plt.subplots(2,2,figsize=(9,9));stats={}
 for row,(label,arrs) in enumerate([('native',images),('20uas',blurred)]):
  ref=arrs[0];other=shift(arrs[1],delta,order=1,mode='constant')
  stats[label]={'relative_l2':float(np.linalg.norm(ref-other)/np.linalg.norm(ref)),'cosine':float(np.sum(ref*other)/np.linalg.norm(ref)/np.linalg.norm(other)),'shifted_flux_retained':float(other.sum())}
  vmax=max(ref.max(),other.max())
  for col,im in enumerate([ref,other]):
   axes[row,col].imshow(im,origin='upper',extent=[fov/2,-fov/2,-fov/2,fov/2],vmin=0,vmax=vmax,cmap='afmhot')
   axes[row,col].set_title(f'{label}; seed {meta[col]["config"]["seed"]}')
 fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180)
 stats['alignment_shift_row_col_pixels']=delta.tolist();stats['runs']=[{'path':str(x),'chi2':d['full_chi2'],'flux':d.get('final_flux_jy')} for x,d in zip(paths,meta)]
 (out/'comparison.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
