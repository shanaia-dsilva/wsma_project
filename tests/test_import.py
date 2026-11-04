import csv,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from model import BookModel
class ImportTests(unittest.TestCase):
    def test_csv_import_deduplicates_and_removes_raw_ids(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)
            with (d/'books.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['book_id','title','author','genre'])
                for i in range(20):w.writerow([i,f'Book {i}','Author','Test'])
            with (d/'interactions.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['user_id','book_id'])
                for u in range(12):
                    for i in range(12):w.writerow([f'private-user-{u}',(u+i)%20])
                    w.writerow([f'private-user-{u}',u])
            result=subprocess.run([sys.executable,'scripts/import_data.py','--books',str(d/'books.csv'),'--interactions',str(d/'interactions.csv'),'--output',str(d/'out')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads((d/'out/manifest.json').read_text());self.assertEqual(manifest['duplicate_pairs_removed'],12)
            self.assertNotIn('private-user-',(d/'out/interactions.json').read_text())
            m=BookModel(d/'out');self.assertEqual(len(m.train),12);self.assertFalse(m.source['synthetic']);self.assertTrue(m.recommend()['books'])
    def test_bad_join_fails_without_output(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);(d/'books.csv').write_text('book_id,title\n1,One\n');(d/'interactions.csv').write_text('user_id,book_id\nu,missing\n')
            r=subprocess.run([sys.executable,'scripts/import_data.py','--books',str(d/'books.csv'),'--interactions',str(d/'interactions.csv'),'--output',str(d/'out')],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertFalse((d/'out').exists())
