"""Reproducible implicit item-item CF and popularity-aware reranking."""
import json, math
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent
class BookModel:
    def __init__(self, data_dir=None):
        self.seed = 42
        self.candidate_size = 40
        if data_dir:
            path = Path(data_dir)
            self.books = json.loads((path/'books.json').read_text())
            raw = json.loads((path/'interactions.json').read_text())
            self.source = json.loads((path/'manifest.json').read_text())
            self.histories = [sorted(set(x)) for x in raw]
        else:
            rows = json.loads((ROOT/'data/catalog.json').read_text())
            self.books = [dict(id=str(i),title=r[0],author=r[1],genre=r[2],description=r[3],year=r[4]) for i,r in enumerate(rows)]
            rng = np.random.default_rng(self.seed)
            self.histories = []
            for u in range(360):
                genre = u % 5
                affinity = np.array([5 if i//12 == genre else (2 if i//12 == (genre+1)%5 else .55) for i in range(len(rows))])
                popularity = np.array([4.5,3.5,3,2.3,1.8,1.4,1.1,.9,.7,.6,.45,.35]*5)
                weight = affinity * popularity ** (.25 if u%3==0 else 1.1)
                self.histories.append(sorted(rng.choice(len(rows),size=int(rng.integers(14,25)),replace=False,p=weight/weight.sum()).tolist()))
            self.source = {'name':'Synthetic demo dataset','synthetic':True,'note':'Real book titles with generated reading interactions. These results are not findings about Goodreads.','seed':42}
        self.n = len(self.books)
        self.ids = {b['id']: i for i,b in enumerate(self.books)}
        rng = np.random.default_rng(self.seed)
        self.train, self.val, self.test = [],[],[]
        for history in self.histories:
            items = rng.permutation(history).tolist()
            hold = max(1, int(len(items)*.15))
            self.test.append(items[:hold]); self.val.append(items[hold:2*hold]); self.train.append(items[2*hold:])
        rr,cc = [],[]
        for u,h in enumerate(self.train): rr.extend([u]*len(h)); cc.extend(h)
        matrix = sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(self.train),self.n))
        self.pop = np.asarray(matrix.sum(axis=0)).ravel()
        self.eligible = np.flatnonzero(self.pop>0)
        co = (matrix.T @ matrix).tocoo()
        norm = np.sqrt(self.pop[co.row]*self.pop[co.col])
        sim = co.data / np.maximum(norm,1) * co.data/(co.data+10)  # support shrinkage
        self.sim = sparse.csr_matrix((sim,(co.row,co.col)),shape=(self.n,self.n))
        self.sim.setdiag(0); self.sim.eliminate_zeros()
        self.novelty = 1-np.log1p(self.pop)/max(float(np.log1p(self.pop).max()),1)
        ordered = sorted(self.eligible,key=lambda i:(-self.pop[i],i))
        head_count = max(1,math.ceil(len(ordered)*.2))
        self.head = set(ordered[:head_count])
        self.tail = set(ordered[head_count:])
        self.readers = []
        for u,h in enumerate(self.train):
            counts = {}
            for i in h: counts[self.books[i]['genre']] = counts.get(self.books[i]['genre'],0)+1
            genre = max(counts,key=counts.get)
            self.readers.append({'id':u,'name':f'Reader {u+1:03d}','genre':genre,'count':len(h),'group':'Niche' if np.mean([i in self.tail for i in h])>=.6 else 'Mainstream'})
        self._evaluation = None

    def rank(self, history, mode='reranked', strength=.3, personalized=False, limit=10, genre=None, exclude=()):
        history = list(dict.fromkeys(history))
        excluded = set(history)|set(exclude)
        eligible = np.array([i for i in self.eligible if i not in excluded and (not genre or self.books[i]['genre']==genre)],dtype=int)
        if not len(eligible): return [],float(strength)
        scores = np.asarray(self.sim[history].sum(axis=0)).ravel() if history else self.pop.copy()
        if mode=='popular':
            candidates = sorted(eligible,key=lambda i:(-self.pop[i],i))[:limit]
            return [(int(i),float(self.pop[i]/max(self.pop.max(),1)),float(self.pop[i]/max(self.pop.max(),1))) for i in candidates],0.
        candidates = np.array(sorted(eligible,key=lambda i:(-scores[i],i))[:self.candidate_size],dtype=int)
        raw = scores[candidates]
        rel = (raw-raw.min())/(raw.max()-raw.min()) if raw.max()>raw.min() else np.zeros(len(raw))
        lam = 0. if mode=='cf' else float(strength)
        if personalized and history and mode=='reranked':
            tail_fraction = np.mean([i in self.tail for i in history])
            lam = min(.8,max(0.,lam*(.5+tail_fraction)))
        final = (1-lam)*rel + lam*self.novelty[candidates]
        ix = sorted(range(len(candidates)),key=lambda j:(-final[j],int(candidates[j])))[:limit]
        return [(int(candidates[j]),float(final[j]),float(rel[j])) for j in ix],float(lam)

    def book(self,i):
        return {**self.books[i], 'popularity':int(self.pop[i]),'tail':i in self.tail,'novelty':round(float(self.novelty[i]),4)}

    def recommend(self,user=0,history=None,mode='reranked',strength=.3,personalized=False,genre=None):
        h = self.train[user] if history is None else history
        ranked,lam = self.rank(h,mode,strength,personalized,genre=genre)
        result=[]
        for i,score,rel in ranked:
            neighbors = sorted(h,key=lambda j:float(self.sim[i,j]),reverse=True)
            anchor = next((j for j in neighbors if self.sim[i,j]>0),None)
            reason = 'Popular in the training data' if mode=='popular' else ('Start with a popular book; add books to personalize' if not h else (f"Readers of {self.books[anchor]['title']} also explored this" if anchor is not None else 'Exploration pick with limited co-reading evidence'))
            result.append({**self.book(i),'score':round(score,4),'relevance':round(rel,4),'reason':reason})
        return {'books':result,'lambda':round(lam,3),'history':[self.book(i) for i in h],'cold_start':not bool(h)}

    def metrics(self,mode,strength,split='validation',personalized=False,users=None):
        truth = self.val if split=='validation' else self.test
        exposures = np.zeros(self.n)
        precision=[]; recall=[]; ndcg=[]; pops=[]; tails=[]
        for u in (range(len(self.train)) if users is None else users):
            # Validation positives are excluded at test time, never added to the fitted model.
            ranked,_ = self.rank(self.train[u],mode,strength,personalized,exclude=self.val[u] if split=='test' else ())
            ids = [i for i,_,_ in ranked]
            target = set(truth[u]); hits=[int(i in target) for i in ids]
            precision.append(sum(hits)/10); recall.append(sum(hits)/len(target))
            ideal=sum(1/math.log2(j+2) for j in range(min(10,len(target))))
            ndcg.append(sum(h/math.log2(j+2) for j,h in enumerate(hits))/ideal if ideal else 0.)
            for i in ids: exposures[i]+=1; pops.append(self.pop[i]); tails.append(i in self.tail)
        x=np.sort(exposures[self.eligible]); n=len(x)
        gini=float((2*np.sum(np.arange(1,n+1)*x)/(n*x.sum())-(n+1)/n)) if x.sum() else 0.
        return {'precision':float(np.mean(precision)), 'recall':float(np.mean(recall)), 'ndcg':float(np.mean(ndcg)), 'coverage':float(np.count_nonzero(exposures)/len(self.eligible)), 'tail_share':float(np.mean(tails)) if tails else 0.,'avg_popularity':float(np.mean(pops)) if pops else 0.,'gini':gini,'users':len(precision)}

    def evaluate(self):
        if self._evaluation is not None:return self._evaluation
        sweep=[{'lambda':v,**self.metrics('reranked',v)} for v in [0,.1,.2,.3,.4,.5,.6,.7,.8]]
        floor=sweep[0]['ndcg']*.95
        viable=[s for s in sweep if s['ndcg']>=floor]
        best=max(viable,key=lambda s:(s['tail_share'],s['ndcg'],-s['lambda']))['lambda']
        rows=[]
        for label,mode,lam,personal in [('Popularity','popular',0,False),('Collaborative filtering','cf',0,False),('Discovery reranking','reranked',best,False),('Personalized reranking','reranked',best,True)]:
            rows.append({'name':label,**self.metrics(mode,lam,'test',personal)})
        groups=[]
        for group in ['Mainstream','Niche']:
            users=[r['id'] for r in self.readers if r['group']==group]
            if users:groups.append({'name':group,**self.metrics('reranked',best,'test',users=users)})
        self._evaluation={'models':rows,'sweep':sweep,'selected_lambda':best,'groups':groups,'protocol':{'seed':42,'split':'Seeded per-user random holdout: 15% validation, 15% test (rounded down, minimum one each); remaining interactions train. No reliable timestamps used.','candidates':self.candidate_size,'head':'Top 20% of training-observed books by interaction count; deterministic ID tie-break. Remaining 80% are long-tail.','selection':'Highest validation long-tail share among strengths retaining at least 95% of baseline CF NDCG; ties prefer NDCG then lower strength.','test':'Test is evaluated only after validation selection. Test candidates exclude train and validation items. Displayed results are cached per server run.','normalization':'Candidate-list min-max relevance; novelty = 1 − log(1 + training count) / max log(1 + training count).','similarity':'Cosine co-interaction similarity with support shrinkage c / (c + 10). Diagonal removed.','personalization':'λu = clip(λ × (0.5 + training-history tail fraction), 0, 0.8). The personalized variant uses the fixed variant’s selected λ; it is not separately tuned.'}}
        return self._evaluation

    def stats(self):
        return {'books':self.n,'eligible_books':len(self.eligible),'users':len(self.train),'interactions':sum(map(len,self.histories)),'train':sum(map(len,self.train)),'validation':sum(map(len,self.val)),'test':sum(map(len,self.test)),'source':self.source,'genres':sorted(set(b['genre'] for b in self.books)),'tail_books':len(self.tail)}
