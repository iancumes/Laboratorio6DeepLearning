"""Descarga local verificable de CIFAR-10 y pesos de ImageNet."""
from pathlib import Path
import sys,urllib.request,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pipeline import ensure_cifar
if __name__=='__main__':
    ensure_cifar()
    path=ROOT/'artifacts/vgg16-397923af.pth'
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve('https://download.pytorch.org/models/vgg16-397923af.pth',path)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha.startswith('397923af'), 'Pesos VGG16 no corresponden al SHA oficial'
    print('CIFAR-10 MD5 oficial verificado; VGG16 SHA256:',sha)
