"""Visualize a saved normalized placement without changing coordinates."""
from pathlib import Path
import argparse,json
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_sample(folder):
    folder=Path(folder)
    d=torch.load(folder/'sample.pt',map_location='cpu',weights_only=False)
    metrics=json.loads((folder/'metrics.json').read_text())
    geometry=json.loads((folder/'cache_geometry.json').read_text())
    width,height=geometry['canvas_size_cache_units']
    fig,ax=plt.subplots(figsize=(7,7*height/width))
    for i,(center,size) in enumerate(zip(d['centers'].numpy(),d['sizes'].numpy())):
        color=plt.get_cmap('tab20')(i%20)
        ax.add_patch(Rectangle(center-size/2,*size,facecolor=color,edgecolor='black',lw=.5,alpha=.8))
        if len(d['centers'])<40:
            ax.text(*center,str(d['macro_names'][i]),fontsize=6,ha='center',va='center')
    ax.set(xlim=(-1,1),ylim=(-1,1),xlabel='Normalized x',ylabel='Normalized y')
    ax.set_title(f"{d['design']}: {len(d['centers'])} macros; legal={metrics['is_legal']}")
    fig.tight_layout();fig.savefig(folder/'placement.png',dpi=180);fig.savefig(folder/'placement.svg');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args();plot_sample(args.folder)
