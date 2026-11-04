"""Local-only application server. No external services or API keys required."""
import argparse, csv, io, json, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from model import BookModel
ROOT=Path(__file__).resolve().parent
class Handler(BaseHTTPRequestHandler):
    def send(self,body,status=200,kind='application/json; charset=utf-8'):
        data=json.dumps(body,ensure_ascii=False).encode() if kind.startswith('application/json') else body
        self.send_response(status); self.send_header('Content-Type',kind); self.send_header('Content-Length',str(len(data))); self.send_header('X-Content-Type-Options','nosniff'); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        url=urlparse(self.path); q=parse_qs(url.query)
        try:
            if url.path=='/api/stats':return self.send(self.server.model.stats())
            if url.path=='/api/books':return self.send([self.server.model.book(i) for i in range(self.server.model.n)])
            if url.path=='/api/readers':return self.send(self.server.model.readers)
            if url.path=='/api/evaluation':
                with self.server.eval_lock: result=self.server.model.evaluate()
                return self.send(result)
            if url.path=='/api/export':
                with self.server.eval_lock: result=self.server.model.evaluate()
                out=io.StringIO(); writer=csv.DictWriter(out,fieldnames=list(result['models'][0]));writer.writeheader();writer.writerows(result['models'])
                return self.send(out.getvalue().encode(),kind='text/csv; charset=utf-8')
            if url.path=='/api/recommend':
                m=self.server.model; user=int(q.get('user',['0'])[0]); strength=float(q.get('strength',['.3'])[0]); mode=q.get('mode',['reranked'])[0]
                if not 0<=user<len(m.train) or not 0<=strength<=.8 or mode not in ['popular','cf','reranked']:raise ValueError('Invalid recommendation settings')
                history=None
                if 'history' in q or 'custom' in q:
                    requested=q.get('history',[''])[0].split(','); history=[m.ids[x] for x in requested if x in m.ids]
                return self.send(m.recommend(user,history,mode,strength,q.get('personalized',['false'])[0]=='true',q.get('genre',[None])[0] or None))
            paths={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if url.path not in paths:return self.send({'error':'Not found'},404)
            file=ROOT/'static'/paths[url.path]
            kind={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8'}[file.suffix]
            return self.send(file.read_bytes(),kind=kind)
        except (ValueError,KeyError) as e:return self.send({'error':str(e)},400)
        except Exception:
            import traceback; traceback.print_exc();return self.send({'error':'The request could not be completed. Check the server log.'},500)
    def log_message(self,fmt,*args):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8000);p.add_argument('--data',help='Directory produced by scripts/import_data.py');args=p.parse_args()
    model=BookModel(args.data);server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.model=model;server.eval_lock=threading.Lock()
    print(f'Between the Lines is ready at http://localhost:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:server.server_close()
if __name__=='__main__':main()
