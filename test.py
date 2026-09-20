import csv
import re
from pathlib import Path

D = Path('data/return-refund')
REQ = ['doc_id', 'title', 'source_url', 'retrieved_at', 'document_version', 'audience']
mds = sorted(D.glob('*.md'))
rows = list(csv.DictReader(open(D / 'sources.csv', encoding='utf-8')))
ids, auds = [], {}

for p in mds:
    parts = p.read_text(encoding='utf-8').split('---')
    if len(parts) >= 3:
        raw_fm = re.findall(r'^(\w+):\s*(.+)$', parts[1], re.M)
        fm = {k: v.strip('"\'') for k, v in raw_fm}
    else:
        fm = {}
    doc_id = fm.get('doc_id')
    ids.append(doc_id)
    aud = fm.get('audience')
    auds[aud] = auds.get(aud, 0) + 1
    status = 'OK' if all(k in fm for k in REQ) and doc_id == p.stem else 'THIEU METADATA'
    print(f'{p.name:40} {status}')

print('so file :', len(mds), '(can 5-10)')
print('csv     :', 'khop' if sorted(r['doc_id'] for r in rows) == sorted(ids) else 'LECH')
print('audience:', auds)