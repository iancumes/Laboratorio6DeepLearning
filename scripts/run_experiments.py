"""Ejecutar desde la raiz del repositorio: python scripts/run_experiments.py."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pipeline import run_all
if __name__ == '__main__':
    run_all()
