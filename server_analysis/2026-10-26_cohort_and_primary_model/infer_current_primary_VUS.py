"""Score locked current BRCA1 VUS using the selected primary checkpoint."""
import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torchvision

BASE=Path('/home/bio1/workdata/GeneMutation2/Revision_2026-10-26')
MODEL=BASE/'retraining/global_gene_holdout'
DATA=BASE/'paired_analysis/current_BRCA1_VUS_input'
OUT=DATA/'primary_VUS_new_figure'
OUT.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
assert json.loads((MODEL/'training_complete.json').read_text())['complete']
df=pd.read_parquet(DATA/'current_BRCA1_strict_VUS_fixed46.parquet')
cache=np.load(DATA/'images_current_BRCA1_VUS.npy',mmap_mode='r')
assert cache.shape==(1511,3,300,300) and df.variant_id.is_unique
assert df.label.isna().all() and df.gene.eq('BRCA1').all()
manifest=pd.read_csv(DATA/'current_VUS_cache_rowmanifest.csv')
assert manifest.variant_id.tolist()==df.variant_id.tolist()
train=pd.read_csv(MODEL/'gene_split_manifest.csv')
assert train.loc[train.gene.eq('BRCA1'),'split'].eq('test').all()
checkpoint=MODEL/'cnn_full_seed42_best_weights.pth'
torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
model=torchvision.models.resnet34(weights=None,num_classes=2).cuda()
model.load_state_dict(torch.load(checkpoint,map_location='cpu'))
model.eval();scores=[]
with torch.inference_mode():
    for start in range(0,len(df),64):
        x=torch.from_numpy(np.array(cache[start:start+64],copy=True)).cuda().float().div_(255)
        with torch.autocast('cuda',dtype=torch.float16):logits=model(x)
        scores.extend(torch.softmax(logits.float(),dim=1)[:,1].cpu().numpy())
out=df[['variant_id','variant_key','gene']].copy()
out['prob_pathogenic']=np.asarray(scores,dtype=float)
assert out.prob_pathogenic.between(0,1).all()
target=OUT/'current_primary_BRCA1_VUS_predictions.csv'
out.to_csv(target,index=False)
report=dict(complete=True,n=len(df),checkpoint_sha256=sha(checkpoint),input_sha256=sha(DATA/'current_BRCA1_strict_VUS_fixed46.parquet'),cache_sha256=sha(DATA/'images_current_BRCA1_VUS.npy'),predictions_sha256=sha(target),torch=torch.__version__,torchvision=torchvision.__version__,mode='eval; inference_mode; autocast CUDA float16; float32 softmax; batch64; divide uint8 by255',BRCA1_gene_partition='test',no_VUS_labels_or_refitting=True)
(OUT/'primary_VUS_inference_provenance.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
