"""CIFAR-10: CNN propia, VGG-16 feature extraction y fine-tuning.

No se utiliza test para seleccionar configuraciones. La cache contiene solamente
el forward del prefijo congelado; los bloques descongelados siempre reciben
gradientes y se recalculan en cada minibatch.
"""
from __future__ import annotations
import argparse, copy, gc, hashlib, json, os, platform, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
FIG = ART / 'figures'
SEED = 23236
SIZE = 32
IM_MEAN = (0.485, 0.456, 0.406)
IM_STD = (0.229, 0.224, 0.225)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COLORS = {'cnn':'#147d92', 'fe':'#bc7626', 'ft':'#704aad'}
LABELS = {'cnn':'CNN propia', 'fe':'VGG-16 extractor', 'ft':'VGG-16 fine-tuning'}

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def sync():
    if DEVICE.type == 'cuda': torch.cuda.synchronize()

def setup():
    ART.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    torch.set_num_threads(int(os.getenv('LAB6_THREADS', '6')))
    seed_everything()
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'figure.dpi':140})
    hardware = {'cpu':platform.processor(), 'cpu_model':'Intel Core i9-13900H',
                'logical_cpus':os.cpu_count(), 'threads':torch.get_num_threads(),
                'ram_gb':round(psutil.virtual_memory().total/2**30,2),
                'device':str(DEVICE), 'gpu':torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None,
                'graphics':'Intel Iris Xe (no CUDA)', 'torch':torch.__version__,
                'torchvision':__import__('torchvision').__version__, 'python':platform.python_version(),
                'seed':SEED, 'vgg_resolution':SIZE, 'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    save_json(hardware, ART/'hardware.json')
    return hardware

def ensure_cifar():
    """Descarga verificable que evita el servidor oficial cuando esta lento."""
    if (ROOT/'data/cifar-10-batches-py/data_batch_1').exists(): return
    import urllib.request, shutil
    target=ROOT/'data/cifar-10-python.tar.gz'; target.parent.mkdir(exist_ok=True)
    expected='c58f30108f718f92721af3b95e74349a'
    if target.exists() and hashlib.md5(target.read_bytes()).hexdigest()==expected: return
    urls=['https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz',
          'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz']
    for url in urls:
        try:
            with urllib.request.urlopen(url,timeout=30) as r,open(target,'wb') as f:
                shutil.copyfileobj(r,f,length=1024*1024)
            if hashlib.md5(target.read_bytes()).hexdigest()!=expected: raise ValueError('MD5 incorrecto')
            return
        except Exception as exc: print('Descarga alternativa:',type(exc).__name__,flush=True)
    raise RuntimeError('No se pudo descargar CIFAR-10 con el MD5 oficial')

def load_data():
    ensure_cifar()
    train = datasets.CIFAR10(ROOT/'data', train=True, download=True)
    test = datasets.CIFAR10(ROOT/'data', train=False, download=True)
    x, y = train.data, np.asarray(train.targets)
    tr, va = train_test_split(np.arange(len(y)), test_size=5000, stratify=y, random_state=SEED)
    reduced, _ = train_test_split(tr, train_size=4500, stratify=y[tr], random_state=SEED)
    assert not set(tr)&set(va) and len(tr)==45000 and len(va)==5000
    assert np.bincount(y[tr]).tolist()==[4500]*10
    assert np.bincount(y[reduced]).tolist()==[450]*10
    np.savez(ART/'split_indices.npz', train=tr, validation=va, reduced=reduced)
    # Estadisticas descriptivas del train oficial (solicitadas); la normalizacion
    # de CNN se ajusta solamente a las 45k imagenes de entrenamiento.
    mean = x.mean(axis=(0,1,2), dtype=np.float64)/255
    std = np.sqrt((x.astype(np.float64)**2).mean(axis=(0,1,2))/255**2-mean**2)
    mx = x[tr].mean(axis=(0,1,2), dtype=np.float64)/255
    sx = np.sqrt((x[tr].astype(np.float64)**2).mean(axis=(0,1,2))/255**2-mx**2)
    stats = {'official_train':50000, 'test':10000, 'train':len(tr), 'validation':len(va),
             'reduced_train':len(reduced), 'classes':train.classes,
             'class_counts_official':np.bincount(y).tolist(),
             'mean_official':mean.tolist(), 'std_official':std.tolist(),
             'mean_fit':mx.tolist(), 'std_fit':sx.tolist(),
             'imagenet_mean':list(IM_MEAN), 'imagenet_std':list(IM_STD),
             'split_sha256':hashlib.sha256(tr.tobytes()+va.tobytes()).hexdigest()}
    save_json(stats, ART/'data_summary.json')
    fig, axes=plt.subplots(10,5,figsize=(7,12))
    for c in range(10):
        for j,idx in enumerate(np.flatnonzero(y==c)[:5]):
            axes[c,j].imshow(x[idx]); axes[c,j].set_xticks([]); axes[c,j].set_yticks([])
            if j==0: axes[c,j].set_ylabel(train.classes[c], rotation=0, ha='right', va='center')
    fig.suptitle('CIFAR-10: cinco ejemplos por clase (train oficial)')
    fig.tight_layout(); fig.savefig(FIG/'examples.png',bbox_inches='tight'); plt.close(fig)
    return x,y,test.data,np.asarray(test.targets),tr,va,reduced,stats

class SmallCNN(nn.Module):
    def __init__(self, width=16, dropout=.25):
        super().__init__()
        blocks=[]; cin=3
        for cout in (width,width*2,width*4):
            blocks += [nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(),
                       nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(),
                       nn.MaxPool2d(2),nn.Dropout2d(dropout/2)]
            cin=cout
        self.features=nn.Sequential(*blocks)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Dropout(dropout),nn.Linear(cin,10))
    def forward(self,x): return self.classifier(self.avgpool(self.features(x)))

class ReplicatedSpatialLinear(nn.Linear):
    """Identica a Linear(512*7*7,4096) cuando avgpool replica un mapa 1x1.

    Para entradas 32x32, cinco max-pools producen 1x1. AdaptiveAvgPool(7,7)
    repite ese valor; sumamos los 49 pesos correspondientes antes del GEMM.
    Los 102760448 pesos originales siguen siendo Parameters entrenables, con
    autograd y SGD normales. No se reduce ni congela el clasificador.
    Para medir el costo nominal/inferencia se usa el Linear original.
    """
    collapse = True
    def forward(self,x):
        if not self.collapse: return F.linear(x,self.weight,self.bias)
        z=x.reshape(x.shape[0],512,7,7)[:,:,0,0]
        w=self.weight.reshape(self.out_features,512,49).sum(dim=2)
        return F.linear(z,w,self.bias)

def make_vgg(config, accelerated=True):
    weights=ART/'vgg16-397923af.pth'
    if weights.exists():
        # Evita otra descarga y mantiene exactamente los pesos IMAGENET1K_V1.
        model=models.vgg16(weights=None,init_weights=False)
        model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    else:
        model=models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6]=nn.Linear(4096,10)
    for layer in model.classifier:
        if isinstance(layer,nn.Dropout): layer.p=config.get('dropout',.5)
    for p in model.features.parameters(): p.requires_grad=False
    if config['model']=='ft':
        start=24 if config['blocks']=='5' else 17
        for p in model.features[start:].parameters(): p.requires_grad=True
    if accelerated:
        old=model.classifier[0]
        new=ReplicatedSpatialLinear(old.in_features,old.out_features,device='meta')
        new.weight=old.weight; new.bias=old.bias
        model.classifier[0]=new
    return model

def preprocess(x, indices, kind, mean=None, std=None, augment=False):
    xb=torch.from_numpy(np.asarray(x[indices])).permute(0,3,1,2).to(DEVICE,dtype=torch.float32)/255
    if augment:
        # Crop con padding reflectante: desplazamiento compartido por minibatch,
        # flip independiente por imagen. Solo se aplica a train.
        xb=F.pad(xb,(4,4,4,4),mode='reflect')
        a,b=np.random.randint(0,9,size=2); xb=xb[:,:,a:a+32,b:b+32]
        mask=torch.rand(len(xb),device=DEVICE)<.5
        xb[mask]=xb[mask].flip(-1)
    if kind=='vgg':
        xb=F.interpolate(xb,size=(SIZE,SIZE),mode='bilinear',align_corners=False,antialias=True)
        mean,std=IM_MEAN,IM_STD
    m=torch.tensor(mean,device=DEVICE).view(1,3,1,1)
    s=torch.tensor(std,device=DEVICE).view(1,3,1,1)
    return ((xb-m)/s).contiguous(memory_format=torch.channels_last)

def build_cache(x,y,xt,yt,tr,va):
    """Una vista de train con flips aleatorios fijados por semilla; val/test sin augmentation.

    La cache del prefijo (features[:17]) puede compartirse entre FT B5 y B4+B5.
    La cache de features completas solo se utiliza por feature extraction.
    """
    manifest=ART/'cache_manifest.json'
    if manifest.exists(): return json.loads(manifest.read_text(encoding='utf-8'))
    seed_everything(); model=make_vgg({'model':'fe'}).to(DEVICE).to(memory_format=torch.channels_last).eval()
    prefix=model.features[:17]; suffix=model.features[17:]
    meta={'resolution':SIZE,'seed':SEED,'views_train':['fixed_seed_horizontal_flip_p0.5'],
          'prefix_end_exclusive':17,'seconds':{},'counts':{}}
    for name,raw,indices in [('train',x,tr),('validation',x,va),('test',xt,np.arange(len(xt)))]:
        t=time.perf_counter(); n=len(indices); views=1
        ppath=ART/f'cache_{name}_prefix.npy'; fpath=ART/f'cache_{name}_features.npy'
        pmap=np.lib.format.open_memmap(ppath,mode='w+',dtype='float16',shape=(views,n,256,SIZE//8,SIZE//8))
        fmap=np.lib.format.open_memmap(fpath,mode='w+',dtype='float32',shape=(views,n,512,1,1))
        with torch.no_grad():
            for start in range(0,n,128):
                sl=slice(start,min(start+128,n))
                z=preprocess(raw,indices[sl],'vgg')
                if name=='train':
                    mask=torch.rand(len(z),device=DEVICE)<.5
                    z[mask]=z[mask].flip(-1)
                for view in range(views):
                    pp=prefix(z if view==0 else z.flip(-1))
                    # Se cuantiza el prefijo tambien para FE: mismas features
                    # de entrada al suffix que FT, evitando un sesgo numerico.
                    pp=pp.to(torch.float16).float()
                    ff=suffix(pp)
                    pmap[view,sl]=pp.cpu().numpy(); fmap[view,sl]=ff.cpu().numpy()
                if start%4096==0: print(f'cache {name}: {start}/{n}',flush=True)
        pmap.flush(); fmap.flush(); del pmap,fmap
        sync(); meta['seconds'][name]=time.perf_counter()-t; meta['counts'][name]=n
        print('cache terminada',name,round(meta['seconds'][name],1),'s',flush=True)
    save_json(meta,manifest); del model; gc.collect()
    return meta

def metrics(y,pred):
    p,r,f,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),'precision_macro':float(p),
            'recall_macro':float(r),'f1_macro':float(f)}

def build_block4_cache():
    """Cache adicional valida para FT con solamente B5 descongelado."""
    path=ART/'block4_cache_manifest.json'
    if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
    model=make_vgg({'model':'fe'}).to(DEVICE).to(memory_format=torch.channels_last).eval()
    block=model.features[17:24];del model;gc.collect();seconds={}
    for name in ['train','validation','test']:
        source=np.load(ART/f'cache_{name}_prefix.npy',mmap_mode='r')
        n=source.shape[1]; t=time.perf_counter()
        target=np.lib.format.open_memmap(ART/f'cache_{name}_block4.npy',mode='w+',
              dtype='float32',shape=(1,n,512,SIZE//16,SIZE//16))
        with torch.no_grad():
            for start in range(0,n,256):
                z=torch.from_numpy(np.array(source[0,start:start+256],copy=True)).to(DEVICE,dtype=torch.float32)
                out=block(z.contiguous(memory_format=torch.channels_last))
                target[0,start:start+256]=out.cpu().numpy()
        target.flush();sync();seconds[name]=time.perf_counter()-t
        print('cache B4',name,round(seconds[name],1),'s',flush=True)
        del source,target
    del block;gc.collect();save_json(seconds,path);return seconds

def cache_training_seconds(summary,kind,config):
    if kind=='cnn': return 0.
    seconds=sum(summary['cache']['seconds'][k] for k in ['train','validation'])
    if kind=='ft' and config.get('use_block4_cache'):
        seconds+=sum(summary['cache'].get('block4_seconds',{}).get(k,0.) for k in ['train','validation'])
    return seconds

def parameter_counts(model):
    return {'params_total':sum(p.numel() for p in model.parameters()),
            'params_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}

def batch_inputs(model,config,raw,indices,cache,positions,train=False):
    if config['model']=='cnn':
        return preprocess(raw,indices[positions],'cnn',config['mean'],config['std'],train)
    v=0
    z=torch.from_numpy(np.array(cache[v,positions],copy=True)).to(DEVICE,dtype=torch.float32)
    if config['model']=='ft':
        start=24 if config.get('use_block4_cache') else 17
        z=model.features[start:](z.contiguous(memory_format=torch.channels_last))
    return torch.flatten(model.avgpool(z),1)

def logits(model,kind,xb):
    return model(xb) if kind=='cnn' else model.classifier(xb)

@torch.no_grad()
def evaluate(model,config,raw,indices,y,cache,batch=512,return_predictions=False):
    model.eval(); losses=0.; pred=[]
    for start in range(0,len(indices),batch):
        positions=np.arange(start,min(start+batch,len(indices)))
        xb=batch_inputs(model,config,raw,indices,cache,positions)
        target=torch.as_tensor(y[indices[positions]],device=DEVICE)
        out=logits(model,config['model'],xb)
        losses += F.cross_entropy(out,target,reduction='sum').item()
        pred.append(out.argmax(1).cpu().numpy())
    pred=np.concatenate(pred); result=metrics(y[indices],pred)
    result['loss']=losses/len(indices)
    return (result,pred) if return_predictions else result

def run_training(config,raw,y,tr,va,stats,cache_name='train',fraction=1.,epochs=None):
    config=copy.deepcopy(config); seed_everything(config.get('seed',SEED))
    config['threads']=torch.get_num_threads()
    config['use_block4_cache']=bool(config['model']=='ft' and config.get('blocks')=='5' and (ART/'block4_cache_manifest.json').exists())
    kind=config['model']; run_id=config['id']+(('_10pct') if fraction<1 else '')
    run_path=ART/'runs'/f'{run_id}.json'
    if run_path.exists():
        saved=json.loads(run_path.read_text(encoding='utf-8'))
        if len(saved['history'])==config['epochs']: return saved
        raise RuntimeError(f'Run incompleto: {run_id}; preservar y mover JSON antes de reentrenar')
    config['mean'],config['std']=stats['mean_fit'],stats['std_fit']
    model=SmallCNN(config['width'],config['dropout']) if kind=='cnn' else make_vgg(config)
    model=model.to(DEVICE).to(memory_format=torch.channels_last)
    train_cache=val_cache=None
    positions_all=np.arange(len(tr))
    if kind!='cnn':
        suffix='features' if kind=='fe' else ('block4' if config['use_block4_cache'] else 'prefix')
        train_cache=np.load(ART/f'cache_train_{suffix}.npy',mmap_mode='r')
        val_cache=np.load(ART/f'cache_validation_{suffix}.npy',mmap_mode='r')
        # Cache train esta en orden del split original de 45k. Un subconjunto
        # necesita traducir sus indices a posiciones de esa cache.
        full=np.load(ART/'split_indices.npz')['train']
        lookup={int(idx):j for j,idx in enumerate(full)}
        positions_all=np.asarray([lookup[int(idx)] for idx in tr])
    if kind=='cnn':
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['wd'])
    else:
        groups=[{'params':model.classifier.parameters(),'lr':config['lr']}]
        if kind=='ft': groups.append({'params':[p for p in model.features.parameters() if p.requires_grad],
                                      'lr':config['backbone_lr']})
        optimizer=torch.optim.SGD(groups,momentum=.9,weight_decay=config['wd'],foreach=False)
    count=parameter_counts(model); history=[]; best=-1.; best_epoch=0
    checkpoint=ART/'checkpoints'/f'{run_id}.pt'; checkpoint.parent.mkdir(exist_ok=True)
    epochs=epochs or config['epochs']; batch=config['batch']
    peak_rss=psutil.Process().memory_info().rss
    if DEVICE.type=='cuda': torch.cuda.reset_peak_memory_stats()
    sync(); all_start=time.perf_counter(); cumulative=0.
    for epoch in range(1,epochs+1):
        sync(); t=time.perf_counter(); model.train()
        if kind!='cnn': model.features[:17].eval()
        perm=np.random.permutation(len(tr)); losses=0.; correct=0
        for start in range(0,len(perm),batch):
            local=perm[start:start+batch]
            cache_positions=positions_all[local]
            if kind=='cnn':
                xb=batch_inputs(model,config,raw,tr,train_cache,local,True)
            else:
                xb=batch_inputs(model,config,raw,tr,train_cache,cache_positions,True)
            target=torch.as_tensor(y[tr[local]],device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            out=logits(model,kind,xb); loss=F.cross_entropy(out,target)
            loss.backward(); optimizer.step()
            losses+=loss.item()*len(local); correct+=(out.argmax(1)==target).sum().item()
            peak_rss=max(peak_rss,psutil.Process().memory_info().rss)
        sync(); train_seconds=time.perf_counter()-t
        val=evaluate(model,config,raw,va,y,val_cache,batch)
        sync(); elapsed=time.perf_counter()-t; cumulative+=elapsed
        row={'epoch':epoch,'train_loss':losses/len(tr),'train_accuracy':correct/len(tr),
             'val_loss':val['loss'],'val_accuracy':val['accuracy'],'val_precision_macro':val['precision_macro'],
             'val_recall_macro':val['recall_macro'],'val_f1_macro':val['f1_macro'],
             'train_seconds':train_seconds,'epoch_seconds':elapsed,'cumulative_seconds':cumulative}
        history.append(row)
        if val['f1_macro']>best:
            best=val['f1_macro']; best_epoch=epoch
            torch.save(model.state_dict(),checkpoint)
        result={'id':run_id,'config':config,'fraction':fraction,'n_train':len(tr),'n_validation':len(va),
                **count,'history':history,'best_epoch':best_epoch,'best_val_f1':best,
                'total_seconds':time.perf_counter()-all_start,'cpu_peak_rss_mb':peak_rss/2**20,
                'gpu_peak_mb':torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=='cuda' else None,
                'checkpoint':str(checkpoint.relative_to(ROOT))}
        save_json(result,run_path)
        print(f'{run_id} epoch {epoch}/{epochs}: loss={row["train_loss"]:.3f}/{row["val_loss"]:.3f} '
              f'acc={row["val_accuracy"]:.3f} F1={row["val_f1_macro"]:.3f} time={elapsed:.1f}s',flush=True)
    del optimizer,model,train_cache,val_cache; gc.collect()
    if DEVICE.type=='cuda': torch.cuda.empty_cache()
    return result

def default_configs():
    return [
      dict(id='CNN1',model='cnn',width=16,dropout=.15,lr=.003,wd=1e-4,batch=256,epochs=3),
      dict(id='CNN2',model='cnn',width=24,dropout=.25,lr=.003,wd=1e-4,batch=256,epochs=3),
      dict(id='CNN3',model='cnn',width=24,dropout=.35,lr=.001,wd=5e-4,batch=256,epochs=3),
      dict(id='FE1',model='fe',blocks='ninguno',dropout=.5,lr=.01,wd=1e-4,batch=512,epochs=2),
      dict(id='FE2',model='fe',blocks='ninguno',dropout=.3,lr=.02,wd=1e-4,batch=512,epochs=2),
      dict(id='FE3',model='fe',blocks='ninguno',dropout=.5,lr=.005,wd=5e-4,batch=512,epochs=2),
      dict(id='FT1',model='ft',blocks='5',dropout=.5,lr=.01,backbone_lr=.0001,wd=1e-4,batch=512,epochs=2),
      dict(id='FT2',model='ft',blocks='4+5',dropout=.5,lr=.01,backbone_lr=.0001,wd=1e-4,batch=1024,epochs=2),
      dict(id='FT3',model='ft',blocks='5',dropout=.3,lr=.02,backbone_lr=.0005,wd=1e-4,batch=1024,epochs=2),
    ]

def resource_profile(model,kind):
    """MACs de Conv/Linear nominales; FLOPs=2*MACs, excluye operaciones elementales.

    Se desactiva la factorizacion espacial para medir la arquitectura VGG
    original. Los FLOPs de entrenamiento son estimaciones, no contadores HW.
    """
    for layer in model.modules():
        if isinstance(layer,ReplicatedSpatialLinear): layer.collapse=False
    model.eval(); shape=(1,3,32,32) if kind=='cnn' else (1,3,SIZE,SIZE)
    per_layer=[]; hooks=[]
    def hook(name):
        def fn(module,inp,out):
            if isinstance(module,nn.Conv2d):
                mac=out.numel()*module.kernel_size[0]*module.kernel_size[1]*module.in_channels/module.groups
            else: mac=out.numel()*module.in_features
            per_layer.append({'layer':name,'macs':int(mac),
                              'trainable':any(p.requires_grad for p in module.parameters())})
        return fn
    for name,m in model.named_modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)): hooks.append(m.register_forward_hook(hook(name)))
    x=torch.randn(shape,device=DEVICE).contiguous(memory_format=torch.channels_last)
    with torch.no_grad(): model(x)
    for h in hooks: h.remove()
    with torch.no_grad():
        for _ in range(5): model(x)
        samples=[]
        for _ in range(30):
            sync(); t=time.perf_counter(); model(x); sync(); samples.append((time.perf_counter()-t)*1000)
    result={'macs_inference':sum(r['macs'] for r in per_layer),
            'flops_inference':2*sum(r['macs'] for r in per_layer),
            'macs_trainable':sum(r['macs'] for r in per_layer if r['trainable']),
            'latency_ms':float(np.mean(samples)),'latency_std_ms':float(np.std(samples)),
            'latency_device':str(DEVICE),'latency_batch':1,'latency_repetitions':30,'layers':per_layer}
    result['latency_threads']=torch.get_num_threads()
    # Validacion contra una herramienta independiente. THOP agrega BN; el
    # contador nominal Conv/Linear excluye esas operaciones elementales.
    from thop import profile
    from thop.vision.basic_hooks import count_linear, zero_ops
    thop_macs,_=profile(model,inputs=(x,),custom_ops={ReplicatedSpatialLinear:count_linear,
        nn.AdaptiveAvgPool2d:zero_ops},verbose=False)
    result['thop_macs_including_bn']=int(thop_macs)
    if kind!='cnn': assert int(thop_macs)==result['macs_inference']
    # Aproximacion forward + 2 forwards equivalentes para backward en capas
    # entrenables. Es conservadora: primera capa entrenable no propaga al prefijo.
    result['estimated_flops_train_per_image']=result['flops_inference']+4*result['macs_trainable']
    for layer in model.modules():
        if isinstance(layer,ReplicatedSpatialLinear): layer.collapse=True
    return result

def final_evaluation(run,xt,yt,stats):
    out_path=ART/'evaluations'/f'{run["id"]}.json'
    if out_path.exists(): return json.loads(out_path.read_text(encoding='utf-8'))
    config=run['config']; seed_everything(); kind=config['model']
    model=SmallCNN(config['width'],config['dropout']) if kind=='cnn' else make_vgg(config)
    model.load_state_dict(torch.load(ROOT/run['checkpoint'],map_location='cpu',weights_only=True))
    model=model.to(DEVICE).to(memory_format=torch.channels_last)
    cache_suffix='features' if kind=='fe' else ('block4' if config.get('use_block4_cache') else 'prefix')
    cache=None if kind=='cnn' else np.load(ART/f'cache_test_{cache_suffix}.npy',mmap_mode='r')
    m,pred=evaluate(model,config,xt,np.arange(len(yt)),yt,cache,config['batch'],True)
    cm=confusion_matrix(yt,pred,labels=np.arange(10))
    out={'id':run['id'],'model':kind,'fraction':run['fraction'],'test':m,
         'confusion_matrix':cm.tolist(),'n_test':len(yt),'profile':resource_profile(model,kind)}
    save_json(out,out_path); np.savez(ART/'evaluations'/f'{run["id"]}_predictions.npz',y_true=yt,y_pred=pred)
    del model,cache; gc.collect()
    print('TEST',run['id'],m,flush=True)
    return out

def run_all(epochs=None):
    hardware=setup(); x,y,xt,yt,tr,va,reduced,stats=load_data()
    configs=default_configs()
    if epochs:
        for c in configs: c['epochs']=epochs
    save_json(configs,ART/'experiment_configs.json')
    # CNN comienza mientras se garantiza primero un resultado end-to-end.
    runs=[]
    for c in configs[:3]: runs.append(run_training(c,x,y,tr,va,stats))
    cache=build_cache(x,y,xt,yt,tr,va)
    cache['block4_seconds']=build_block4_cache()
    for c in configs[3:]: runs.append(run_training(c,x,y,tr,va,stats))
    winners={k:max([r for r in runs if r['config']['model']==k],key=lambda r:r['best_val_f1']) for k in LABELS}
    # Se fija la seleccion exclusivamente por F1 macro de validacion antes de test.
    save_json({k:r['id'] for k,r in winners.items()},ART/'selection.json')
    finals={k:final_evaluation(r,xt,yt,stats) for k,r in winners.items()}
    reduced_runs={}; reduced_tests={}
    for kind,run in winners.items():
        r=run_training(run['config'],x,y,reduced,va,stats,fraction=.1)
        reduced_runs[kind]=r; reduced_tests[kind]=final_evaluation(r,xt,yt,stats)
    summary={'hardware':hardware,'data':stats,'cache':cache,'runs':runs,'winners':winners,
             'finals':finals,'reduced_runs':reduced_runs,'reduced_tests':reduced_tests}
    save_json(summary,ART/'summary.json'); create_figures(summary)
    return summary

def create_figures(s):
    rows=[]
    fig,axes=plt.subplots(3,3,figsize=(12,9),sharex=True)
    for ax,r in zip(axes.flat,s['runs']):
        h=pd.DataFrame(r['history'])
        ax.plot(h.epoch,h.train_loss,'o-',label='Entrenamiento')
        ax.plot(h.epoch,h.val_loss,'s--',label='Validacion')
        ax.set_title(f'{r["id"]}: {LABELS[r["config"]["model"]]}')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Entropia cruzada'); ax.grid(alpha=.15)
        if ax==axes[0,0]: ax.legend()
        b=r['history'][r['best_epoch']-1]
        rows.append({'id':r['id'],'modelo':r['config']['model'],'bloques':r['config'].get('blocks','3 propios'),
                     'lr':r['config']['lr'],'backbone_lr':r['config'].get('backbone_lr'),
                     'dropout':r['config']['dropout'],'weight_decay':r['config']['wd'],
                     'batch':r['config']['batch'],'epochs':len(r['history']),'mejor_epoch':r['best_epoch'],
                     'threads':r['config'].get('threads',8),'block4_cache':r['config'].get('use_block4_cache',False),
                     'val_accuracy':b['val_accuracy'],'val_precision':b['val_precision_macro'],
                     'val_recall':b['val_recall_macro'],'val_f1':b['val_f1_macro'],
                     'params_total':r['params_total'],'params_trainable':r['params_trainable'],
                     'seconds':r['total_seconds'],'gpu_peak_mb':r['gpu_peak_mb'],'cpu_rss_mb':r['cpu_peak_rss_mb']})
    fig.tight_layout(); fig.savefig(FIG/'loss_curves_9.png',bbox_inches='tight'); plt.close(fig)
    pd.DataFrame(rows).to_csv(ART/'iterations.csv',index=False)
    comp=[]
    for k,r in s['winners'].items():
        f=s['finals'][k]; p=f['profile']; n=r['n_train']; e=len(r['history'])
        comp.append({'modelo':LABELS[k],'id':r['id'],'params_total':r['params_total'],
                     'params_trainable':r['params_trainable'],'macs_inference':p['macs_inference'],
                     'flops_inference':p['flops_inference'],'latency_ms':p['latency_ms'],
                     'seconds_per_epoch':np.mean([h['train_seconds'] for h in r['history']]),
                     'best_epoch':r['best_epoch'],'epochs':e,'seconds_total':r['total_seconds'],
                     'gpu_peak_mb':r['gpu_peak_mb'],'cpu_peak_rss_mb':r['cpu_peak_rss_mb'],
                     'estimated_flops_training_nominal':p['estimated_flops_train_per_image']*n*e,
                     **{f'test_{a}':b for a,b in f['test'].items()},
                     **{f'test_10pct_{a}':b for a,b in s['reduced_tests'][k]['test'].items()}})
    pd.DataFrame(comp).to_csv(ART/'comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for k,r in s['winners'].items():
        h=pd.DataFrame(r['history']); offset=cache_training_seconds(s,k,r['config'])
        ax.plot((h.cumulative_seconds+offset)/60,h.val_accuracy*100,label=LABELS[k],color=COLORS[k],
                marker={'cnn':'o','fe':'s','ft':'^'}[k],linestyle={'cnn':'-','fe':'--','ft':'-.'}[k])
    ax.set_xlabel('Tiempo acumulado (min; incluye cache train + validacion en VGG)')
    ax.set_ylabel('Accuracy de validacion (%)'); ax.set_ylim(0,100); ax.legend(); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(FIG/'accuracy_time.png',bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for k,f in s['finals'].items():
        xx=f['profile']['flops_inference']/1e9; yy=f['test']['f1_macro']*100
        ax.scatter(xx,yy,s=75,color=COLORS[k],marker={'cnn':'o','fe':'s','ft':'^'}[k])
        ax.annotate(LABELS[k],(xx,yy),xytext=(7,8 if k!='fe' else -16),textcoords='offset points',fontsize=9)
    ax.set_xscale('log'); ax.set_xlabel('FLOPs nominales de inferencia por imagen (GFLOPs, escala log)')
    ax.set_ylabel('F1 macro de test (%)'); ax.set_ylim(0,100); ax.margins(x=.25); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(FIG/'f1_cost.png',bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,(k,f) in zip(axes,s['finals'].items()):
        cm=np.asarray(f['confusion_matrix']); im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=1000)
        for i in range(10):
            for j in range(10): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=6,color='white' if cm[i,j]>500 else '#14232d')
        ax.set_xticks(range(10),s['data']['classes'],rotation=45,ha='right',fontsize=8)
        ax.set_yticks(range(10),s['data']['classes'],fontsize=8)
        ax.set_title(LABELS[k]); ax.set_xlabel('Prediccion'); ax.set_ylabel('Clase real')
    fig.tight_layout(); fig.savefig(FIG/'confusion_matrices.png',bbox_inches='tight'); plt.close(fig)
    for k,f in s['finals'].items():
        fig,ax=plt.subplots(figsize=(7,5.5));cm=np.asarray(f['confusion_matrix'])
        im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=1000)
        for i in range(10):
            for j in range(10):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=9,
                        color='white' if cm[i,j]>500 else '#14232d')
        ax.set_xticks(range(10),s['data']['classes'],rotation=45,ha='right',fontsize=10)
        ax.set_yticks(range(10),s['data']['classes'],fontsize=10)
        ax.set_title(LABELS[k]+' - test oficial (10,000 imágenes)')
        ax.set_xlabel('Clase predicha');ax.set_ylabel('Clase real')
        fig.colorbar(im,ax=ax,label='Número de imágenes')
        fig.tight_layout();fig.savefig(FIG/f'confusion_{k}.png',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); xx=np.arange(3)
    a=[s['finals'][k]['test']['f1_macro']*100 for k in LABELS]
    b=[s['reduced_tests'][k]['test']['f1_macro']*100 for k in LABELS]
    ax.bar(xx-.18,a,.36,label='45,000 imagenes'); ax.bar(xx+.18,b,.36,label='4,500 imagenes',hatch='//')
    ax.set_xticks(xx,[LABELS[k] for k in LABELS]); ax.set_ylabel('F1 macro de test (%)'); ax.set_ylim(0,100); ax.legend()
    fig.tight_layout(); fig.savefig(FIG/'data_fraction.png',bbox_inches='tight'); plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--epochs',type=int)
    args=parser.parse_args(); run_all(args.epochs)
