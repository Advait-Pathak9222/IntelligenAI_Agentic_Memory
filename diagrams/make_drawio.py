import html, sys
from pathlib import Path

class D:
    def __init__(s): s.cells=[]; s.n=2
    def id(s): s.n+=1; return f"c{s.n}"
    def box(s,x,y,w,h,label,style,parent="1"):
        i=s.id(); s.cells.append(f'<mxCell id="{i}" value="{html.escape(label)}" style="{style}" vertex="1" parent="{parent}"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'); return i
    def edge(s,a,b,label="",style="",pts=None):
        i=s.id(); base="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeWidth=2;endArrow=block;endFill=1;fontSize=12;labelBackgroundColor=#ffffff;fontColor=#333333;"
        s.cells.append(f'<mxCell id="{i}" value="{html.escape(label)}" style="{base}{style}" edge="1" parent="1" source="{a}" target="{b}"><mxGeometry relative="1" as="geometry">'+(('<Array as="points">'+''.join(f'<mxPoint x="{x}" y="{y}"/>' for x,y in pts)+'</Array>') if pts else '')+'</mxGeometry></mxCell>'); return i
    def xml(s,name):
        body="".join(s.cells)
        return f'<mxfile host="drawio"><diagram name="{name}" id="{name}"><mxGraphModel dx="1400" dy="900" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" pageWidth="1600" pageHeight="1000" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root></mxGraphModel></diagram></mxfile>'

F="fontFamily=Helvetica;fontSize=13;html=1;whiteSpace=wrap;fontColor=#1a1a1a;"
HOT=F+"rounded=1;fillColor=#dae8fc;strokeColor=#6c8ebf;strokeWidth=2;"
EXT=F+"rounded=1;fillColor=#f8cecc;strokeColor=#b85450;strokeWidth=2;dashed=1;"
ZERO=F+"rounded=1;fillColor=#d5e8d4;strokeColor=#82b366;strokeWidth=2;"
LLM=F+"rounded=1;fillColor=#ffe6cc;strokeColor=#d79b00;strokeWidth=2;"
HUM=F+"rounded=1;fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=2;"
GREY=F+"rounded=1;fillColor=#f5f5f5;strokeColor=#666666;strokeWidth=2;"
LANE=F+"rounded=1;arcSize=3;fillColor=none;strokeWidth=2;dashed=1;verticalAlign=top;align=left;spacingLeft=12;spacingTop=6;fontStyle=1;fontSize=15;"
DB=F+"shape=cylinder;fillColor=#fff2cc;strokeColor=#d6b656;strokeWidth=2;"
TITLE="text;html=1;fillColor=none;strokeColor=none;fontColor=#1a1a1a;fontSize=20;fontStyle=1;align=center;verticalAlign=middle;fontFamily=Helvetica;"
NOTE="text;html=1;fillColor=none;strokeColor=none;fontColor=#1a1a1a;fontSize=12;align=left;verticalAlign=middle;fontFamily=Helvetica;whiteSpace=wrap;"

def b(t,sub): return f"<b>{t}</b><br><font style='font-size:11px'>{sub}</font>"

# ---------- Figure 1: architecture ----------
d=D()
d.box(0,0,1500,40,"Epistemic Provenance Graph: two-lane memory architecture",TITLE)
d.box(20,55,1460,215,"HOT PATH: critical path (0 LLM calls, ≤150 ms p95)",LANE+"strokeColor=#6c8ebf;fontColor=#1f4e8c;")
q=d.box(50,110,250,120,b("Task Queue","5,000 tasks/day · 200 orgs<br>org_id + task_context"),HOT)
r=d.box(390,110,280,120,b("Hybrid Recall + IFC Gate","BM25 + vector + 1-hop graph<br>score = sim × conf × decay<br>pack ≤4k tokens, tag [id], no T3 actions"),ZERO)
a=d.box(770,110,270,120,b("Headless Agent Loop","Black-box LLM + tools<br>reads untrusted inputs"),HOT)
w=d.box(1120,110,330,120,b("Write Staging Push","≤3 candidate claims + cited [id]s<br>emitted in final step (0 extra LLM)"),ZERO)
ext=d.box(770,290,270,60,b("External inputs (untrusted)","email · web · uploads · tickets"),EXT)
d.edge(q,r); d.edge(r,a,"≤4k tokens"); d.edge(a,w)
d.edge(ext,a,"",  "dashed=1;strokeColor=#b85450;")
dag=d.box(150,380,1200,90,b("Epistemic Provenance DAG (org-partitioned)","Claim nodes + DERIVED_FROM / CONTESTED_BY / SUPERSEDES edges · bi-temporal validity · CAS version"),DB)
d.edge(dag,r,"recall (ACTIVE / CONTESTED)","exitX=0.2;exitY=0;entryX=0.5;entryY=1;")
d.box(20,510,1460,230,"SLEEP-TIME ASYNC LANE: off critical path (~0.08 to 0.30 LLM calls/run avg)",LANE+"strokeColor=#d79b00;fontColor=#8a5a00;")
g1=d.box(45,565,330,140,b("1. Deterministic Gate + CAS","verify tool logs → T0–T3 (non-amplifying)<br>exact match → pos++<br>T3 → QUARANTINED · 0 LLM"),ZERO)
g2=d.box(400,565,330,140,b("2. Batched Conflict Revision","only entity collisions / sim > 0.82<br>5 pairs per LLM call →<br>CONFIRM / SUPERSEDE / COEXIST / REJECT"),LLM)
g3=d.box(755,565,330,140,b("3. JTMS Error Containment","1 LLM attribution per flagged report<br>walk back → root, forward → BFS<br>quarantine unsupported subgraph (0 LLM)"),LLM)
g4=d.box(1110,565,345,140,b("4. Global k-anon & Decay","PII scrub · k ≥ 10 orgs → global (1 batch LLM)<br>nightly Ebbinghaus pruning<br>cap 10k/org · compact leaves > 90 d"),LLM)
d.edge(w,g1,"async queue","exitX=0.92;exitY=1;entryX=0;entryY=0.5;dashed=1;",[(1424,492),(30,492),(30,635)])
d.edge(g1,g2); d.edge(g2,g3); d.edge(g3,g4)
for g,cx in ((g1,210),(g2,565),(g3,920),(g4,1282)):
    d.edge(g,dag,"","dashed=1;strokeColor=#999999;endArrow=open;exitX=0.5;exitY=0;entryY=1;entryX=%.3f;"%((cx-150)/1200))
hf=d.box(755,770,330,60,b("Delayed human feedback","~2% of outputs, days later"),HUM)
d.edge(hf,g3,"","strokeColor=#9673a6;")
# legend
d.box(45,770,22,22,"",ZERO); d.box(72,770,180,22,"0 LLM calls (rules / stats)",NOTE)
d.box(45,802,22,22,"",LLM); d.box(72,802,220,22,"Batched, off-path LLM calls",NOTE)
d.box(300,770,22,22,"",EXT); d.box(327,770,200,22,"Untrusted input (T3)",NOTE)
d.box(300,802,22,22,"",HUM); d.box(327,802,200,22,"Human signal (T0)",NOTE)
d.box(1110,770,345,60,"<b>Budget:</b> ~400 LLM calls/day ≈ 0.08 per run (cap 2.0)<br><b>Latency:</b> ≤150 ms p95 added (cap 300 ms)",NOTE+"fillColor=#f5f5f5;strokeColor=#666666;rounded=1;spacingLeft=8;")
Path(sys.argv[1],"fig1_architecture.drawio").write_text(d.xml("Architecture"))

# ---------- Figure 2: trust gate + rollback ----------
d=D()
d.box(0,0,1500,40,"Trust gating and causal rollback",TITLE)
d.box(20,55,720,560,"A. IFC non-amplification and quarantine (scenarios 1 and 2)",LANE+"strokeColor=#b85450;fontColor=#8c2a26;")
t=d.box(40,120,180,90,b("External ticket","T3_UNTRUSTED<br>“escalate to billing-help@acme-support.co”"),EXT)
qz=d.box(290,120,180,90,b("QUARANTINED claim","prior (0,2) · conf 0.25<br>N × T3 repeats cannot promote"),GREY)
sk=d.box(540,120,180,90,b("Action sink gate","blocks auto-CC to domains<br>not on Acme's T1 allowlist"),ZERO)
d.edge(t,qz); d.edge(qz,sk,"hint only")
v=d.box(40,330,180,100,b("Independent check","T1 CRM / DKIM / tool read<br>or T0 human approval"),HUM)
ac=d.box(290,330,180,100,b("ACTIVE claim","new valid_from window<br>trust = min_trust(sources)"),ZERO)
ol=d.box(540,330,180,100,b("Old claim","valid_to closed<br>→ SUPERSEDED (kept for audit)"),GREY)
d.edge(sk,v,"triggered on next relevant task","exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
d.edge(v,ac,"corroborates"); d.edge(ac,ol,"SUPERSEDES")
d.box(50,470,670,120,"<b>Rule:</b> a T3 claim may only add a CONTESTED_BY edge to a T0–T2 claim; it never silently overwrites one. Real updates are not lost: the sink gate turns them into a T1 check or a T0 approval request instead of an action. (0 LLM calls)",NOTE+"fillColor=#fff2cc;strokeColor=#d6b656;rounded=1;spacingLeft=10;")
d.box(770,55,720,560,"B. Bidirectional JTMS rollback (scenario 3)",LANE+"strokeColor=#9673a6;fontColor=#5b3b73;")
hf=d.box(790,120,200,90,b("Human flag","report #40 marked wrong<br>(days later)"),HUM)
at=d.box(1030,120,200,90,b("Fault attribution","1 async LLM call over<br>the report's recalled [id]s"),LLM)
rt=d.box(1270,120,190,90,b("C_utc (root)","walk DERIVED_FROM back<br>→ RETRACTED"),EXT)
d.edge(hf,at); d.edge(at,rt)
dq=d.box(1030,330,200,100,b("Derived claims A..M","supported only by C_utc<br>→ QUARANTINED (0 LLM)"),GREY)
ds=d.box(1270,330,190,100,b("Derived claims N..O","have independent T1 support<br>→ stay ACTIVE, conf penalised"),ZERO)
d.edge(rt,dq,"forward BFS","exitX=0.2;exitY=1;entryX=0.5;entryY=0;")
d.edge(rt,ds,"keep if alt. support","exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
it=d.box(1290,450,150,50,b("Independent T1","alternate proof"),HOT)
d.edge(it,ds,"","dashed=1;")
lg=d.box(790,330,200,100,b("Audit log","40 affected run_ids<br>listed for review"),GREY)
d.edge(dq,lg,"","dashed=1;")
d.box(790,520,670,70,"<b>Effect:</b> future recall stops using every claim that rests only on the bad root; past reports are flagged by run_id, not rewritten.",NOTE+"fillColor=#fff2cc;strokeColor=#d6b656;rounded=1;spacingLeft=10;")
Path(sys.argv[1],"fig2_trust_and_rollback.drawio").write_text(d.xml("TrustRollback"))
