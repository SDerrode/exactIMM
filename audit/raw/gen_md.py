import json, sys
# args: extracted.json  out.md  "Titre"  "ligne sous-titre"
xpath, outpath, title, subtitle = sys.argv[1:5]
with open(xpath) as f: data = json.load(f)
SEV = {'critical':0,'high':1,'medium':2,'low':3,'info':4}
ICON = {'confirmed':'✅','uncertain':'❓','refuted':'❌','unverified':'⚠️'}
vmap = {}
for v in data['verdicts']: vmap.setdefault(v['title'], []).append(v)
def status_of(f):
    votes = vmap.get(f['title'], [])
    if not votes: return 'unverified', []
    yes = sum(1 for v in votes if v['verdict'].get('isReal'))
    return ('confirmed' if yes==len(votes) else 'refuted' if yes==0 else 'uncertain'), votes
def clean(s): return (s or '').replace('/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/','')
L = [f'# {title}', '', subtitle, '']
allf = [(d['dimension'], f) for d in data['dimensions'] for f in d['findings']]
conf = [(d,f) for d,f in allf if status_of(f)[0]=='confirmed']
from collections import Counter
c = Counter(f['severity'] for _,f in conf)
nbref = sum(1 for _,f in allf if status_of(f)[0]=='refuted')
nbunc = sum(1 for _,f in allf if status_of(f)[0]=='uncertain')
L.append(f"**Bilan : {len(allf)} trouvailles — {len(conf)} confirmées** "
         f"({c.get('critical',0)} critical, {c.get('high',0)} high, {c.get('medium',0)} medium, "
         f"{c.get('low',0)} low, {c.get('info',0)} info), {nbunc} incertaines, {nbref} réfutées.")
L.append('')
L.append('## Trouvailles majeures (critical + high confirmées)'); L.append('')
for dim, f in sorted(allf, key=lambda t: SEV[t[1]['severity']]):
    st,_ = status_of(f)
    if f['severity'] in ('critical','high') and st=='confirmed':
        L.append(f"- **[{f['severity'].upper()}] {f['title']}** — `{clean(f['file'])}:{f.get('line','')}`")
L.append('')
dimtitles = data.get('_dimtitles', {})
for d in data['dimensions']:
    L.append(f"## {dimtitles.get(d['dimension'], d['dimension'])}"); L.append('')
    L.append(f"_{d.get('scope_summary','')}_"); L.append('')
    for f in sorted(d['findings'], key=lambda f: SEV[f['severity']]):
        st, votes = status_of(f)
        loc = clean(f['file']) + (f":{f['line']}" if f.get('line') else '')
        L.append(f"### {ICON.get(st,'•')} [{f['severity'].upper()}] {f['title']}"); L.append('')
        L.append(f"`{loc}` — statut : {st} ({len(votes)} vote(s)) — catégorie : {f.get('category','')}"); L.append('')
        L.append(f['description']); L.append('')
        if f.get('evidence'): L.append(f"**Preuve :** {f['evidence']}"); L.append('')
        if f.get('suggestion'): L.append(f"**Suggestion :** {f['suggestion']}"); L.append('')
        adj = [v['verdict'].get('severityAdjustment','') for v in votes
               if v['verdict'].get('severityAdjustment','') and v['verdict'].get('severityAdjustment','').strip().lower() not in ('','aucun','none','n/a')
               and not v['verdict'].get('severityAdjustment','').lower().startswith('aucun')]
        if adj: L.append(f"**Ajustement de sévérité (vérificateurs) :** {' | '.join(adj[:2])}"); L.append('')
L.append('---'); L.append(f"_Généré automatiquement ; raisonnements complets dans `{clean(xpath)}`._")
open(outpath,'w').write('\n'.join(L))
print('écrit:', outpath, '—', len(L), 'lignes')
for k in sorted(set(f['severity'] for _,f in allf)):
    n = sum(1 for _,f in allf if f['severity']==k)
    nc = sum(1 for _,f in conf if f['severity']==k)
    print(f'  {k}: {n} ({nc} confirmées)')
