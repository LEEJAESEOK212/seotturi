"""Read-only corpus audit. Writes reports, never trains or alters raw data."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib, json, re, unicodedata, zipfile, random
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'
norm=lambda x:unicodedata.normalize('NFC',x)
def digest(x):return hashlib.blake2b(x.encode(),digest_size=16).digest()
def clean(x):return ' '.join(norm(x).split())
def quantiles(hist):
 total=sum(hist.values()); out={}
 for q in [.5,.9,.95,.99,1]:
  acc=0
  for size,count in sorted(hist.items()):
   acc+=count
   if acc>=total*q:out[str(q)]=size;break
 return out
regions=defaultdict(list)
for p in RAW.rglob('*.part0'):
 region=re.search(r'데이터\((.*?)\)',norm(str(p.relative_to(RAW)))).group(1)
 regions[region].append(p)
report={'scope':'All supplied archives; no training; comparisons are within region. NFC and whitespace normalization for duplicates. Semantic correctness and near duplicates require further review.','regions':{}}
for region,paths in sorted(regions.items()):
 stats={s:Counter() for s in ['train','validation']}; lengths={s:Counter() for s in stats}
 pairs={s:set() for s in stats}; inputs={s:{} for s in stats}; docs={s:set() for s in stats}; speakers={s:set() for s in stats}
 changed_pairs={s:set() for s in stats}; samples={s:[] for s in stats}; rng=random.Random(42); archives=[]; errors=[]
 txt_only=Counter(); txt_samples=[]
 for p in sorted(paths):
  split='validation' if '2.Validation' in str(p) else 'train'; c=stats[split]
  with zipfile.ZipFile(p) as z:
   members=[m for m in z.infolist() if not m.is_dir()]; exts=Counter(Path(m.filename).suffix.lower() for m in members)
   archives.append({'path':norm(str(p.relative_to(RAW))),'split':split,'bytes':p.stat().st_size,'extensions':dict(exts)})
   for m in members:
    try:
     raw=z.read(m)
     if m.filename.lower().endswith('.txt') and not exts['.json']:
      text=raw.decode('utf-8-sig');txt_only['files']+=1
      for line in text.splitlines():
       if not line.strip():continue
       txt_only['nonempty_lines']+=1
       if '\t' in line:txt_only['speaker_tab_lines']+=1
       if re.search(r'\([^()]+\)/\([^()]+\)',line):txt_only['lines_with_paired_notation']+=1
       if len(txt_samples)<5 and re.search(r'\([^()]+\)/\([^()]+\)',line):txt_samples.append(line)
      continue
     if not m.filename.lower().endswith('.json'):continue
     d=json.loads(raw);c['json_files']+=1
     doc=str(d.get('id') or Path(m.filename).stem)
     if doc in docs[split]:c['duplicate_document_ids']+=1
     docs[split].add(doc)
     for sp in d.get('speaker',[]):
      if isinstance(sp,dict) and sp.get('id') is not None:speakers[split].add(str(sp['id']))
     for u in d.get('utterance',[]):
      c['utterances']+=1;a=u.get('dialect_form');b=u.get('standard_form')
      if not isinstance(a,str) or not isinstance(b,str) or not a.strip() or not b.strip():c['invalid_pairs']+=1;continue
      a=clean(a);b=clean(b);c['valid_pairs']+=1
      ph=digest(a+'\0'+b); ih=digest(a); bh=digest(b)
      if ph in pairs[split]:c['duplicate_pair_rows']+=1
      pairs[split].add(ph)
      if ih in inputs[split] and inputs[split][ih]!=bh:c['rows_with_alternative_target_for_same_input']+=1
      inputs[split][ih]=bh
      changed=a!=b;c['changed' if changed else 'unchanged']+=1
      if changed:changed_pairs[split].add(ph)
      length=max(len(a),len(b));lengths[split][length]+=1
      if len(a)<=3:c['input_le_3_chars']+=1
      if length>512:c['pair_over_512_chars']+=1
      if '@' in a+b:c['has_at_placeholder']+=1
      if re.search(r'[()/~]|\(\(',a+b):c['has_transcription_notation']+=1
      if re.search(r'\([^()]+\)/\([^()]+\)',a+b):c['has_unresolved_pair_notation']+=1
      if '\ufffd' in a+b:c['replacement_character']+=1
      if not u.get('speaker_id'):c['missing_speaker_id']+=1
      if changed:
       item={'input':a,'target':b,'document':doc,'utterance':u.get('id')}
       n=c['changed']
       if len(samples[split])<12:samples[split].append(item)
       else:
        j=rng.randrange(n)
        if j<12:samples[split][j]=item
    except Exception as e:errors.append({'archive':p.name,'member':m.filename,'error':str(e)})
  print(region,split,p.name,'done',flush=True)
 overlap={
  'document_ids':len(docs['train']&docs['validation']),
  'speaker_ids_raw_unverified_scope':len(speakers['train']&speakers['validation']),
  'unique_exact_pairs':len(pairs['train']&pairs['validation']),
  'unique_changed_pairs':len(changed_pairs['train']&changed_pairs['validation']),
  'unique_input_texts':len(inputs['train'].keys()&inputs['validation'].keys())}
 report['regions'][region]={'archives':archives,'stats':{s:dict(v) for s,v in stats.items()},'unique_pairs':{s:len(v) for s,v in pairs.items()},'max_pair_character_length_quantiles':{s:quantiles(v) for s,v in lengths.items()},'overlap':overlap,'changed_samples':samples,'txt_only_stats':dict(txt_only),'txt_samples':txt_samples,'errors':errors}
 (ROOT/'outputs/data-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(region,json.dumps({'stats':report['regions'][region]['stats'],'overlap':overlap,'txt_only':dict(txt_only)},ensure_ascii=False),flush=True)
