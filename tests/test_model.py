import math,unittest
import numpy as np
from model import BookModel

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=BookModel()
    def test_disjoint_holdouts(self):
        for all_items,tr,va,te in zip(self.m.histories,self.m.train,self.m.val,self.m.test):
            self.assertFalse(set(tr)&set(va));self.assertFalse(set(tr)&set(te));self.assertFalse(set(va)&set(te))
            self.assertEqual(set(all_items),set(tr+va+te))
    def test_train_popularity_only(self):
        counts=np.zeros(self.m.n)
        for h in self.m.train:
            for i in h:counts[i]+=1
        np.testing.assert_array_equal(counts,self.m.pop)
    def test_no_seen_recommendations(self):
        for mode in ['popular','cf','reranked']:
            for u in range(30):
                ranking,_=self.m.rank(self.m.train[u],mode,.5)
                self.assertEqual(len(ranking),10)
                self.assertFalse(set(i for i,_,_ in ranking)&set(self.m.train[u]))
    def test_zero_strength_equals_cf(self):
        for u in range(10):self.assertEqual(self.m.rank(self.m.train[u],'cf')[0],self.m.rank(self.m.train[u],'reranked',0)[0])
    def test_reranking_changes_results(self):
        self.assertTrue(any(self.m.rank(h,'cf')[0]!=self.m.rank(h,'reranked',.8)[0] for h in self.m.train[:10]))
    def test_cold_start_and_exhausted_catalogue(self):
        self.assertTrue(self.m.recommend(history=[])['cold_start'])
        self.assertEqual(self.m.rank(list(range(self.m.n)))[0],[])
    def test_genre_filter(self):
        ranked,_=self.m.rank([],'reranked',.3,genre='Fantasy')
        self.assertTrue(all(self.m.books[i]['genre']=='Fantasy' for i,_,_ in ranked))
    def test_repeatable(self):
        second=BookModel();self.assertEqual(second.train,self.m.train);self.assertEqual(second.recommend(),self.m.recommend())
    def test_metrics_hand_calculated(self):
        m=BookModel();m.val=[[0,1]];m.train=[[3]];m.eligible=np.arange(4);m.pop=np.array([2,1,1,1]);m.tail={1,2,3};m.n=4
        m.rank=lambda *a,**kw: ([(0,1.,1.),(2,.5,.5)],0.)
        r=m.metrics('cf',0)
        self.assertAlmostEqual(r['precision'],.1);self.assertAlmostEqual(r['recall'],.5)
        self.assertAlmostEqual(r['ndcg'],1/(1+1/math.log2(3)));self.assertAlmostEqual(r['coverage'],.5)
        self.assertAlmostEqual(r['gini'],.5);self.assertAlmostEqual(r['tail_share'],.5)
    def test_validation_selection_and_metrics(self):
        r=self.m.evaluate();baseline=r['sweep'][0]['ndcg'];chosen=next(x for x in r['sweep'] if x['lambda']==r['selected_lambda'])
        self.assertGreaterEqual(chosen['ndcg'],.95*baseline)
        for row in r['models']:
            for key in ['precision','recall','ndcg','coverage','tail_share','gini']:self.assertTrue(0<=row[key]<=1,(key,row[key]))
if __name__=='__main__':unittest.main()
