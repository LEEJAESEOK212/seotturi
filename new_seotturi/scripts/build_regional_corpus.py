"""Merge all supplied records by region/split; remove exact contextual duplicates only.
No model loading, sampling, training, text normalization, or raw-data edits.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import re
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAIR = re.compile(r'\(([^()]*)\)/\(([^()]*)\)')
SLUGS = {'강원도':'gangwon', '경상도':'gyeongsang', '전라도':'jeolla', '제주도':'jeju', '충청도':'chungcheong'}

def dump(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'))

def parse_txt(line):
    speaker, sep, form = line.partition('\t')
    if not sep:
        form, speaker = line, None
    dialect = PAIR.sub(lambda m: m.group(1), form)
    standard = PAIR.sub(lambda m: m.group(2), form)
    unresolved = '/(' in dialect or '/(' in standard
    return {'speaker_id': speaker, 'form': form, 'dialect_form': dialect,
            'standard_form': standard, 'parse_status': 'review_required' if unresolved or not sep else 'derived_from_txt'}

def contextual_key(rows, i):
    # Adjacent original speech is provenance for dedup, NOT a future model input.
    # Speaker identifiers and original form are retained to avoid over-merging.
    def view(u):
        if u is None:
            return None
        return {k: u.get(k) for k in ('speaker_id', 'form', 'dialect_form', 'standard_form', 'parse_status')}
    return dump([view(rows[i-1]) if i else None, view(rows[i]), view(rows[i+1]) if i+1<len(rows) else None])

def build():
    dest = ROOT/'data/processed/regional-v1'
    dest.mkdir(parents=True, exist_ok=True)
    archives = defaultdict(list)
    for p in (ROOT/'data/raw').rglob('*.part0'):
        name = unicodedata.normalize('NFC', str(p.relative_to(ROOT/'data/raw')))
        region = re.search(r'데이터\((.*?)\)', name).group(1)
        split = 'validation' if '2.Validation' in name else 'train'
        archives[(region, split)].append(p)
    summary = {'complete': False, 'training_performed': False, 'normalization': 'none',
               'dedup_key': 'Exact previous/current/next records: speaker_id, form, dialect_form, standard_form, parse_status; within region and split only',
               'identity_pairs': 'retained', 'limits': None, 'partitions': []}
    manifest = dest/'manifest.json'
    if manifest.exists():
        summary = json.loads(manifest.read_text())
        if summary.get('complete'):
            raise SystemExit('Corpus is already complete; refusing overwrite')
    completed = {(x['region'], x['split']) for x in summary['partitions']}
    for (region, split), paths in sorted(archives.items()):
        if (region, split) in completed:
            print('Already complete:', region, split, flush=True)
            continue
        folder = dest/SLUGS[region]
        folder.mkdir(exist_ok=True)
        output = folder/f'{split}.jsonl'
        db_path = folder/f'{split}.dedup-index.sqlite'
        seen = {}
        counts = Counter(); source_id = 0
        with output.with_suffix('.jsonl.partial').open('w') as out, (folder/f'{split}.sources.jsonl').open('w') as sources, (folder/f'{split}.duplicates.jsonl').open('w') as dup:
            for archive in sorted(paths):
                archive_sha = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
                with zipfile.ZipFile(archive) as z:
                    names = sorted(n for n in z.namelist() if not n.endswith('/'))
                    has_json = any(n.lower().endswith('.json') for n in names)
                    for name in names:
                        if not (name.lower().endswith('.json') or (not has_json and name.lower().endswith('.txt'))):
                            continue
                        source_id += 1
                        raw = z.read(name)
                        if name.lower().endswith('.json'):
                            document = json.loads(raw)
                            rows = document.get('utterance', [])
                            metadata = {k:v for k,v in document.items() if k!='utterance'}
                        else:
                            text = raw.decode('utf-8-sig')
                            rows = []
                            for lineno, line in enumerate(text.splitlines(), 1):
                                if not line.strip():
                                    counts['blank_txt_lines'] += 1
                                    continue
                                u = parse_txt(line);u['id'] = f'{Path(name).stem}:{lineno}'
                                rows.append(u)
                            metadata = {'id':Path(name).stem, 'format':'txt', 'note':'Parenthesized paired notation parsed mechanically; labels not human-verified.'}
                        sources.write(dump({'source_id':source_id, 'archive':str(archive.relative_to(ROOT/'data/raw')), 'archive_sha256':archive_sha, 'member':name, 'metadata':metadata})+'\n')
                        for i,u in enumerate(rows):
                            counts['input_records'] += 1
                            key = contextual_key(rows,i)
                            h = hashlib.sha256(key.encode()).digest()
                            record_id = counts['output_records']+1
                            existing = seen.get(h)
                            if existing is not None:
                                old_key, old_id = existing
                                if old_key != key:
                                    raise RuntimeError('Hash collision; refusing to discard nonidentical record')
                                counts['exact_context_duplicates'] += 1
                                dup.write(dump({'source_id':source_id,'utterance_index':i,'utterance_id':u.get('id'),'kept_record_id':old_id})+'\n')
                                continue
                            seen[h] = (key, record_id)
                            a=u.get('dialect_form');b=u.get('standard_form')
                            if isinstance(a,str) and isinstance(b,str) and a.strip() and b.strip():
                                counts['identity_pairs' if a==b else 'changed_pairs'] += 1
                            else:
                                counts['invalid_pair_preserved'] += 1
                            if u.get('parse_status')=='review_required':counts['txt_review_required'] += 1
                            out.write(dump({'record_id':record_id,'source_id':source_id,'utterance_index':i,
                                'utterance':u,'adjacent_context':{
                                    'previous':rows[i-1].get('dialect_form') if i else None,
                                    'next':rows[i+1].get('dialect_form') if i+1<len(rows) else None},
                                'context_usage':'dedup/provenance only; next utterance is not a default model input'})+'\n')
                            counts['output_records'] += 1
                        if source_id % 200 == 0:
                            print(region,split,'documents',source_id,'records',counts['input_records'],flush=True)
        del seen
        # Disposable index created exclusively by this run; not a source artifact.
        if db_path.exists():
            db_path.unlink()
        output.with_suffix('.jsonl.partial').rename(output)
        assert counts['input_records'] == counts['output_records']+counts['exact_context_duplicates']
        entry={'region':region,'split':split,'counts':dict(counts),'sources':source_id,'output':str(output.relative_to(dest))}
        summary['partitions'].append(entry)
        manifest.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print('FINISHED',dump(entry),flush=True)
    summary['complete'] = True
    manifest.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    build()
