"""Ejecutar notebook top-to-bottom y auditar los resultados guardados."""
from pathlib import Path
import json,sys,html,base64
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
import fitz
ROOT=Path(__file__).resolve().parents[1]
NB=ROOT/'notebooks/Laboratorio6_TransferLearning_Cumes_Valladares.ipynb'
PDF=ROOT/'output/pdf/Laboratorio6_TransferLearning_Cumes_Valladares.pdf'

def check_results():
    s=json.loads((ROOT/'artifacts/summary.json').read_text(encoding='utf-8'))
    assert len(s['runs'])==9
    assert all(len(r['history'])==r['config']['epochs'] for r in s['runs'])
    for k,winner in s['winners'].items():
        competitors=[r for r in s['runs'] if r['config']['model']==k]
        assert winner['best_val_f1']==max(r['best_val_f1'] for r in competitors)
        assert winner['params_trainable']<=winner['params_total']
    for group in ['finals','reduced_tests']:
        for k,e in s[group].items():
            p=np.load(ROOT/'artifacts/evaluations'/f'{e["id"]}_predictions.npz')
            y,yp=p['y_true'],p['y_pred'];cm=confusion_matrix(y,yp,labels=np.arange(10))
            assert len(y)==10000 and np.all(cm.sum(1)==1000)
            assert cm.tolist()==e['confusion_matrix']
            pm,rm,fm,_=precision_recall_fscore_support(y,yp,average='macro',zero_division=0)
            for key,val in [('accuracy',accuracy_score(y,yp)),('precision_macro',pm),('recall_macro',rm),('f1_macro',fm)]:
                assert abs(e['test'][key]-val)<1e-12
    split=np.load(ROOT/'artifacts/split_indices.npz')
    assert not set(split['train'])&set(split['validation'])
    assert set(split['reduced']).issubset(set(split['train']))
    print('Splits, selección y seis evaluaciones auditadas.',flush=True)

def execute_notebook():
    nb=nbformat.read(NB,as_version=4);nbformat.validate(nb)
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(nb,timeout=240,resources={'metadata':{'path':str(ROOT)}},km=km)
    client.execute();nbformat.write(nb,NB)
    assert not any(o.get('output_type')=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs)
    print('Notebook ejecutado y guardado sin errores.',flush=True)

def render_pdf():
    out=ROOT/'tmp/pdfs';out.mkdir(parents=True,exist_ok=True)
    doc=fitz.open(PDF);assert len(doc)==4
    for i,page in enumerate(doc):
        text=page.get_text();assert len(text)>400
        for line in ['23236','23045']:assert line in text
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(out/f'page-{i+1}.png')
    assert 'https://github.com/iancumes/Laboratorio6DeepLearning' in doc[-1].get_text()
    print('PDF: cuatro páginas, texto y enlace verificados; PNG listos para revisión.',flush=True)

def export_html():
    # Lectura cómoda sin kernel; los outputs se conservan también en el ipynb.
    nb=nbformat.read(NB,as_version=4)
    parts=['<!doctype html><html lang="es"><meta charset="utf-8"><title>Laboratorio 6</title>',
           '<style>body{max-width:1100px;margin:40px auto;font:16px/1.55 Arial;color:#153447;padding:0 24px}pre{white-space:pre-wrap;background:#f0f5f7;padding:14px;font:12px/1.45 Consolas}img{max-width:100%}table{border-collapse:collapse;font-size:12px}td,th{padding:5px;border:1px solid #ddd}h1,h2{color:#147d92}details{margin:8px 0}summary{cursor:pointer}</style>']
    import markdown
    for c in nb.cells:
        if c.cell_type=='markdown':
            parts.append(markdown.markdown(c.source,extensions=['tables','fenced_code']))
        else:
            parts.append('<details><summary>Ver código</summary><pre>'+html.escape(c.source)+'</pre></details>')
            for o in c.get('outputs',[]):
                data=o.get('data',{})
                if 'image/png' in data:parts.append('<img src="data:image/png;base64,'+data['image/png']+'">')
                elif 'text/html' in data:parts.append('<div style="overflow-x:auto">'+data['text/html']+'</div>')
                elif 'text/plain' in data:parts.append('<pre>'+html.escape(data['text/plain'])+'</pre>')
                elif o.get('output_type')=='stream':parts.append('<pre>'+html.escape(o['text'])+'</pre>')
    parts.append('</html>');(ROOT/'notebooks/Laboratorio6_vista.html').write_text('\n'.join(parts),encoding='utf-8')

if __name__=='__main__':
    check_results();execute_notebook();render_pdf();export_html()
    (ROOT/'artifacts/validation.json').write_text(json.dumps({'nine_runs':True,'six_test_evaluations':True,
        'notebook_executed':True,'pdf_pages':4,'pdf_visual_review':'PNG generado; revisar visualmente'},indent=2),encoding='utf-8')
