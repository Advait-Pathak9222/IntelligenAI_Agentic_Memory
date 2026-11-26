import html, sys
from pathlib import Path
class D:
    def __init__(s): s.cells=[]; s.n=2
    def id(s): s.n+=1; return f"c{s.n}"
    def box(s,x,y,w,h,label,style):
        i=s.id(); s.cells.append(f'<mxCell id="{i}" value="{html.escape(label)}" style="{style}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'); return i
    def edge(s,a,b,label="",style="",pts=None):
        i=s.id(); base="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeWidth=1.5;strokeColor=#333333;endArrow=block;endFill=1;endSize=7;fontSize=14;fontFamily=Inter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DInter;fontColor=#333333;labelBackgroundColor=#ffffff;"
        p=('<Array as="points">'+''.join(f'<mxPoint x="{x}" y="{y}"/>' for x,y in pts)+'</Array>') if pts else ''
        s.cells.append(f'<mxCell id="{i}" value="{html.escape(label)}" style="{base}{style}" edge="1" parent="1" source="{a}" target="{b}"><mxGeometry relative="1" as="geometry">{p}</mxGeometry></mxCell>'); return i
    def xml(s,name):
        return f'<mxfile host="drawio"><diagram name="{name}" id="{name}"><mxGraphModel grid="0" page="0" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>{"".join(s.cells)}</root></mxGraphModel></diagram></mxfile>'
F="html=1;whiteSpace=wrap;fontFamily=Inter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DInter;fontColor=#1a1a1a;"
BOX=F+"rounded=0;fillColor=#ffffff;strokeColor=#333333;strokeWidth=1.5;fontSize=16;spacing=6;"
LLM=F+"rounded=0;fillColor=#fbe9c6;strokeColor=#333333;strokeWidth=1.5;fontSize=16;spacing=6;"
EXT=F+"rounded=0;fillColor=#ffffff;strokeColor=#333333;strokeWidth=1.5;dashed=1;dashPattern=6 4;fontSize=16;spacing=6;"
STORE=F+"shape=cylinder;fillColor=#f2f2f2;strokeColor=#333333;strokeWidth=1.5;fontSize=16;"
LANE=F+"rounded=0;fillColor=none;strokeColor=#9a9a9a;strokeWidth=1;verticalAlign=top;align=left;spacingLeft=10;spacingTop=4;fontSize=15;fontStyle=1;fontColor=#444444;"
TXT="text;html=1;fillColor=none;strokeColor=none;fontFamily=Inter;fontSource=https%3A%2F%2Ffonts.googleapis.com%2Fcss%3Ffamily%3DInter;fontColor=#333333;fontSize=14;align=left;verticalAlign=middle;whiteSpace=wrap;"
def L(t,sub): return f"<span style='font-weight:600'>{t}</span><br><span style='font-size:14px;color:#444444'>{sub}</span>"
W=170; G=26; X0=20

# ---- Figure 1 ----
d=D()
d.box(0,0,800,250,"Critical path: 0 LLM calls, ≤150 ms p95",LANE)
y=55; xs=[X0+i*(W+G) for i in range(4)]
q=d.box(xs[0],y,W,80,L("Task queue","one org per task"),BOX)
r=d.box(xs[1],y,W,80,L("Recall","hybrid, ≤4k tokens"),BOX)
a=d.box(xs[2],y,W,80,L("Agent run","black-box LLM + tools"),BOX)
w=d.box(xs[3],y,W,80,L("Write staging","≤3 cited claims"),BOX)
d.edge(q,r); d.edge(r,a); d.edge(a,w)
ext=d.box(xs[2],170,W,56,L("Untrusted input","email, web, tickets"),EXT)
d.edge(ext,a)
g=d.box(20,290,760,70,L("Provenance graph, one partition per org","claims with trust tier, validity window and DERIVED_FROM edges"),F+"rounded=0;fillColor=#efefef;strokeColor=#333333;strokeWidth=1.5;fontSize=16;")
d.edge(g,r,"read",'exitX=%.3f;exitY=0;entryX=0.5;entryY=1;'%((xs[1]+W/2-20)/760))
d.edge(w,g,"stage (async)",'exitX=0.5;exitY=1;entryX=%.3f;entryY=0;dashed=1;'%((xs[3]+W/2-20)/760))
lane=d.box(0,400,800,235,"Background lane: batched, off the critical path",LANE)
d.edge(lane,g,"revise and commit",'exitX=0.5;exitY=0;entryX=0.5;entryY=1;startArrow=block;startFill=1;')
ys=450
s1=d.box(xs[0],ys,W,80,L("Trust gate","assign tier T0–T3"),BOX)
s2=d.box(xs[1],ys,W,80,L("Conflict review","5 pairs per LLM call"),LLM)
s3=d.box(xs[2],ys,W,80,L("Rollback","1 LLM call per flag"),LLM)
s4=d.box(xs[3],ys,W,80,L("Sharing and decay","nightly"),LLM)
d.edge(s1,s2); d.edge(s2,s3); d.edge(s3,s4)
hf=d.box(xs[2],556,W,62,L("Human feedback","2% of outputs, delayed"),EXT)
d.edge(hf,s3)
d.box(0,655,30,20,"",LLM); d.box(38,653,280,24,"Step that calls an LLM",TXT)
d.box(300,655,30,20,"",EXT); d.box(338,653,300,24,"Input from outside the memory layer",TXT)
Path(sys.argv[1],"fig1_architecture.drawio").write_text(d.xml("Architecture"))

# ---- Figure 2 ----
d=D(); W2=230; G2=35; xs=[X0+i*(W2+G2) for i in range(3)]
d.box(0,0,800,300,"A. Write-time trust gate (scenarios 1 and 2)",LANE)
t=d.box(xs[0],55,W2,80,L("Untrusted claim","from ticket text (T3)"),EXT)
qz=d.box(xs[1],55,W2,80,L("Quarantined","stored as a hint only"),BOX)
gate=d.box(xs[2],55,W2,80,L("Action gate","blocks external actions"),BOX)
d.edge(t,qz); d.edge(qz,gate)
v=d.box(xs[0],195,W2,80,L("Verification","tool read (T1) or human (T0)"),BOX)
ac=d.box(xs[1],195,W2,80,L("Active claim","new validity window"),BOX)
ol=d.box(xs[2],195,W2,80,L("Old claim","superseded, kept for audit"),BOX)
d.edge(gate,v,"next relevant task",'exitX=0.5;exitY=1;entryX=0.5;entryY=0;',[(xs[2]+W2/2,165),(xs[0]+W2/2,165)])
d.edge(v,ac); d.edge(ac,ol)
d.box(0,330,800,320,"B. Rollback after a human flag (scenario 3)",LANE)
hf=d.box(xs[0],385,W2,80,L("Human flag","one report marked wrong"),EXT)
at=d.box(xs[1],385,W2,80,L("Fault attribution","1 LLM call"),LLM)
rt=d.box(xs[2],385,W2,80,L("Root claim","retracted"),BOX)
d.edge(hf,at); d.edge(at,rt)
dq=d.box(xs[1],545,W2,80,L("Claims built only on root","quarantined"),BOX)
ds=d.box(xs[2],545,W2,80,L("Claims with other support","kept, confidence lowered"),BOX)
lg=d.box(xs[0],545,W2,80,L("Affected runs","listed for review"),BOX)
d.edge(rt,dq,"",'exitX=0.25;exitY=1;entryX=0.5;entryY=0;',[(xs[2]+W2*0.25,505),(xs[1]+W2/2,505)])
d.edge(rt,ds,"",'exitX=0.75;exitY=1;entryX=0.5;entryY=0;')
d.edge(dq,lg)
d.box(0,670,30,20,"",LLM); d.box(38,668,330,24,"Step that calls an LLM",TXT)
d.box(300,670,30,20,"",EXT); d.box(338,668,300,24,"Input from outside the memory layer",TXT)
Path(sys.argv[1],"fig2_trust_and_rollback.drawio").write_text(d.xml("TrustRollback"))
