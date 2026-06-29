import json, re, os
RAW='/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/audit/raw'
SEV=['critical','high','medium','low','info']
def clean(s): return (s or '').replace('/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/FofGss/exactIMM/','')

def parse_adj(adjs, orig):
    """Sévérité effective : si les votes convergent vers un downgrade, l'appliquer (cible la plus haute parmi downgrades = prudent)."""
    targets=[]
    for a in adjs:
        if not a: continue
        al=a.lower()
        # motif "X -> Y", "X → Y", "medium to low", "downgrade ... to Y"
        m=re.findall(r'(?:->|→|–>|-->|\bto\b|\bvers\b|\ben\b)\s*(critical|high|medium|low|info)', al)
        if m:
            targets.append(m[-1]); continue
        if any(k in al for k in ['keep','maint.','maintenu','inchang','aucun','none','correct','appropri','justifi','well calibrated','well-calibrated','bien calibr','approprié']):
            targets.append(orig)
    if not targets: return orig
    # prudent : prendre la sévérité la PLUS BASSE proposée (les vérificateurs tendent à downgrader les artefacts)
    return max(targets, key=lambda s: SEV.index(s) if s in SEV else 0)

def status_from_votes(votes):
    if not votes: return 'unverified'
    yes=sum(1 for v in votes if v.get('isReal'))
    return 'confirmed' if yes==len(votes) else ('refuted' if yes==0 else 'uncertain')

findings=[]

# Vagues 1-2 : votes inline (status + votes dans chaque finding)
for wave,fn in [('1 — Code cœur','01-code-core-result.json'),('2 — Code périphérie','02-code-periphery-result.json')]:
    res=json.load(open(f'{RAW}/{fn}'))
    for d in res:
        for f in d['findings']:
            votes=f.get('votes',[])
            adjs=[v.get('severityAdjustment','') for v in votes]
            findings.append({'wave':wave,'dim':d['dimension'],'sev':f['severity'],
                'sev_eff':parse_adj(adjs, f['severity']) if f.get('status')=='confirmed' else f['severity'],
                'status':f.get('status','unverified'),'title':f['title'],'file':clean(f.get('file','')),
                'line':f.get('line',''),'cat':f.get('category',''),'desc':f.get('description','')})

# Vagues 3,4a,4b,4c : verdicts séparés à matcher par titre
for wave,fn in [('3 — Papier maths','03-paper-math-extracted.json'),
                ('4a — Algo↔code','04a-extracted.json'),
                ('4b — Chiffres↔résultats','04b-extracted.json'),
                ('4c — Éditorial/biblio','04c-extracted.json')]:
    data=json.load(open(f'{RAW}/{fn}'))
    vmap={}
    for v in data['verdicts']: vmap.setdefault(v['title'],[]).append(v)
    for d in data['dimensions']:
        for f in d['findings']:
            vs=vmap.get(f['title'],[])
            votes=[v['verdict'] for v in vs]
            adjs=[v.get('severityAdjustment','') for v in votes]
            st=status_from_votes(votes)
            findings.append({'wave':wave,'dim':d['dimension'],'sev':f['severity'],
                'sev_eff':parse_adj(adjs, f['severity']) if st=='confirmed' else f['severity'],
                'status':st,'title':f['title'],'file':clean(f.get('file','')),
                'line':f.get('line',''),'cat':f.get('category',''),'desc':f.get('description','')})

json.dump(findings, open(f'{RAW}/all-findings.json','w'), ensure_ascii=False, indent=1)

# Stats
from collections import Counter
print('TOTAL trouvailles:', len(findings))
print('Par statut:', dict(Counter(f['status'] for f in findings)))
conf=[f for f in findings if f['status']=='confirmed']
print('Confirmées:', len(conf))
print('  par sévérité FINDER :', {s:sum(1 for f in conf if f['sev']==s) for s in SEV})
print('  par sévérité EFFECTIVE (post-ajustement):', {s:sum(1 for f in conf if f['sev_eff']==s) for s in SEV})
print()
print('=== CONFIRMÉES critical/high EN SÉVÉRITÉ EFFECTIVE ===')
for f in sorted(conf, key=lambda f: (SEV.index(f['sev_eff']), f['wave'])):
    if f['sev_eff'] in ('critical','high'):
        dn = ' [rétrogradé depuis '+f['sev']+']' if f['sev_eff']!=f['sev'] else ''
        print(f"[{f['sev_eff'].upper()}] (v{f['wave'][0]}) {f['title'][:95]}")
        print(f"      {f['file']}:{f['line']}{dn}")
print()
print('=== Trouvailles dont sev_eff=medium (confirmées), comptage par vague ===')
print({w:sum(1 for f in conf if f['sev_eff']=='medium' and f['wave']==w) for w in sorted(set(f['wave'] for f in conf))})
