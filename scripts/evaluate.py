"""Export actual metrics, protocol, splits, and standalone SVG sensitivity plot."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import BookModel

def main():
    p=argparse.ArgumentParser();p.add_argument('--data');p.add_argument('--output',default='outputs/demo');a=p.parse_args()
    m=BookModel(a.data);r=m.evaluate();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps({'dataset':m.stats(),**r},indent=2))
    (out/'splits.json').write_text(json.dumps({'seed':42,'train':m.train,'validation':m.val,'test':m.test}))
    for name,rows in [('comparison',r['models']),('validation-sweep',r['sweep']),('reader-groups',r['groups'])]:
        with (out/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="450" viewBox="0 0 900 450"><rect width="900" height="450" fill="#fffefb"/><g font-family="sans-serif" fill="#29352b"><text x="60" y="35" font-size="22">Validation relevance and long-tail discovery</text><text x="60" y="60" font-size="13">'+('Synthetic demo interactions — not Goodreads findings' if m.source.get('synthetic') else 'Imported subset; see results.json for provenance')+'</text>']
    for v in [0,.25,.5,.75,1]:
        y=350-v*250;parts.append(f'<line x1="60" y1="{y}" x2="840" y2="{y}" stroke="#ddd"/><text x="15" y="{y+5}">{v:.0%}</text>')
    for i,s in enumerate(r['sweep']):parts.append(f'<text x="{60+i*97.5}" y="375" text-anchor="middle">{s["lambda"]:.1f}</text>')
    for key,color in [('ndcg','#df643f'),('tail_share','#566c4a')]:
        points=' '.join(f'{60+i*97.5},{350-s[key]*250}' for i,s in enumerate(r['sweep']))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
    parts.append('<text x="300" y="410">Reranking strength λ</text><text x="60" y="435" fill="#df643f">NDCG@10</text><text x="200" y="435" fill="#566c4a">Long-tail share</text></g></svg>')
    (out/'sensitivity.svg').write_text(''.join(parts))
    print(json.dumps({'output':str(out),'selected_lambda':r['selected_lambda'],'models':r['models']},indent=2))
if __name__=='__main__':main()
