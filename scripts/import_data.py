"""Import bounded local UCSD JSONL(.gz) or canonical CSV inputs; never downloads data."""
import argparse, csv, gzip, hashlib, json
from collections import defaultdict
from pathlib import Path

def records(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8-sig') as f:
        if str(path).removesuffix('.gz').endswith('.csv'):
            yield from csv.DictReader(f)
        else:
            for line in f:
                if line.strip():yield json.loads(line)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--books',required=True);p.add_argument('--interactions',required=True)
    p.add_argument('--output',default='data/imported');p.add_argument('--genre',default='Books')
    p.add_argument('--max-books',type=int,default=1000);p.add_argument('--max-users',type=int,default=1000)
    p.add_argument('--max-records',type=int,default=1000000);p.add_argument('--min-user-items',type=int,default=8)
    args=p.parse_args()
    if not 20<=args.max_books<=5000 or not 10<=args.max_users<=10000 or args.min_user_items<5 or args.max_records<1:
        p.error('Use 20–5000 books, 10–10000 users, at least 5 user items, and a positive record limit.')
    out=Path(args.output)
    if out.exists() and any(out.iterdir()):p.error('Output directory must be empty; choose a new path to preserve existing data.')
    books=[];lookup={}
    for row in records(args.books):
        bid=str(row.get('book_id','')).strip();title=str(row.get('title','')).strip()
        if not bid or not title or bid in lookup:continue
        author=row.get('author','')
        if not author:
            authors=row.get('authors',[])
            if isinstance(authors,list):author=', '.join(a.get('name') or 'Author ID '+str(a.get('author_id','unknown')) for a in authors if isinstance(a,dict))
        lookup[bid]=len(books)
        books.append({'id':bid,'title':title,'author':author or 'Author not supplied','genre':row.get('genre') or args.genre,'description':row.get('description') or 'No description supplied in the imported metadata.','year':str(row.get('publication_year') or row.get('year') or '')})
        if len(books)>=args.max_books:break
    histories=defaultdict(set);scanned=0;duplicates=0;matched=0
    for row in records(args.interactions):
        if scanned>=args.max_records:break
        scanned+=1;bid=str(row.get('book_id','')).strip();uid=str(row.get('user_id','')).strip()
        if bid not in lookup or not uid:continue
        # All valid shelf interactions count as implicit positives, irrespective of rating/read status.
        if uid not in histories and len(histories)>=args.max_users:continue
        matched+=1;i=lookup[bid]
        if i in histories[uid]:duplicates+=1
        histories[uid].add(i)
    retained=[sorted(items) for items in histories.values() if len(items)>=args.min_user_items]
    if len(retained)<10:p.error(f'Only {len(retained)} eligible readers. Increase limits or use a denser subset. Check that book IDs match; UCSD full CSV requires book_id_map.csv conversion first.')
    used=sorted(set(i for h in retained for i in h));remap={old:new for new,old in enumerate(used)}
    histories_out=[[remap[i] for i in h] for h in retained];books_out=[books[i] for i in used]
    manifest={'name':'Imported local research subset','synthetic':False,'note':'Imported subset; results apply to these historical interactions, not current Goodreads behaviour.','files':{'books':Path(args.books).name,'interactions':Path(args.interactions).name},'sha256':{'books':digest(args.books),'interactions':digest(args.interactions)},'sampling':'First valid distinct metadata records; first admitted users within the scan limit. This bounded convenience sample is not representative.','interaction_policy':'Every valid shelf interaction is a binary positive; repeated user-book pairs collapse to one. Ratings and is_read are not used.','split':'Seeded random per-user holdout; timestamps are not parsed. No claim of temporal validation.','limits':vars(args),'scanned_records':scanned,'matched_records':matched,'duplicate_pairs_removed':duplicates,'readers_before_minimum':len(histories),'readers_retained':len(retained),'books_retained':len(books_out),'interactions_retained':sum(map(len,histories_out)),'seed':42}
    # Raw user IDs are intentionally omitted from output.
    out.mkdir(parents=True,exist_ok=True)
    for name,obj in [('books',books_out),('interactions',histories_out),('manifest',manifest)]:
        (out/(name+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Imported {len(books_out)} books, {len(retained)} anonymized readers, {sum(map(len,histories_out))} interactions into {out}')
if __name__=='__main__':main()
