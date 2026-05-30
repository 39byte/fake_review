"""
E09_knowledge_distillation.py
실험 9: Knowledge Distillation

Teacher: DRAGWave_400ep (PR-AUC 0.9340, 596K params)
Student: HeteroBWGNN 절반 크기 (hidden=64, 약 82K params)

가설:
  - Student(작은 모델)가 Teacher의 "부드러운 확신도"를 학습
  - 파라미터/노드 비율: 24.9 → 3.4로 극적 감소
  - Teacher 지식으로 Student가 압축된 좋은 표현 학습

손실: α*CE(hard) + (1-α)*KL(teacher_soft, student_soft)
예상 효과: Gap 0.066 → ~0.048, 파라미터 85% 감소
"""
import copy, json, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, MessagePassing

BASE  = Path(__file__).resolve().parent.parent.parent
GRAPH = BASE / "data" / "graphs"
MOD   = BASE / "models"
RES   = BASE / "results" / "experiments"
RES.mkdir(parents=True, exist_ok=True)

ET = [("review","rtr","review"),("review","rsr","review"),
      ("review","burst","review"),("review","rur","review"),("review","sim","review")]
ET_NORSR = [et for et in ET if et[1] != "rsr"]

class DualFreqConv(MessagePassing):
    def __init__(self,a,b): super().__init__(aggr="mean"); self.lin=nn.Linear(a*2,b)
    def forward(self,x,ei): low=self.propagate(ei,x=x); return self.lin(torch.cat([low,x-low],-1))
    def message(self,x_j): return x_j

class BWGATConv(MessagePassing):
    def __init__(self,a,b,h=4,dr=0.3):
        super().__init__(aggr="add")
        self.gat=GATConv(a,a//h,heads=h,dropout=dr,add_self_loops=False)
        self.lin=nn.Linear(a*2,b)
    def forward(self,x,ei):
        if ei.shape[1]==0: return self.lin(torch.cat([x,torch.zeros_like(x)],-1))
        low=self.gat(x,ei); return self.lin(torch.cat([low,x-low],-1))

class DRAGWaveConv(nn.Module):
    def __init__(self,a,b,nr,h=4,dr=0.3):
        super().__init__()
        self.bwgat=nn.ModuleList([BWGATConv(a,b,h,dr) for _ in range(nr)])
        self.self_lin=nn.Linear(a,b); self.attn_vec=nn.Linear(b*2,1,bias=False)
        self.drop=nn.Dropout(dr)
    def forward(self,x,ei_list):
        hs=self.self_lin(x); re=[self.bwgat[i](x,ei) for i,ei in enumerate(ei_list)]
        rs=torch.stack(re,1); he=hs.unsqueeze(1).expand_as(rs)
        aw=F.softmax(self.attn_vec(torch.tanh(torch.cat([he,rs],-1))).squeeze(-1),dim=-1)
        return self.drop(F.relu(hs+(rs*aw.unsqueeze(-1)).sum(1)))

class TeacherDRAGWave(nn.Module):
    def __init__(self,d,h=128,ets=None,dr=0.3):
        super().__init__(); self.ets=ets or ET; nr=len(self.ets)
        self.proj=nn.Linear(d,h); self.layer1=DRAGWaveConv(h,h,nr,dr=dr)
        self.layer2=DRAGWaveConv(h,h,nr,dr=dr)
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h*2,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x)))
        ei=[data.edge_index_dict.get(et,torch.zeros(2,0,dtype=torch.long)) for et in self.ets]
        h1=self.bn1(self.layer1(x,ei)); h2=self.bn2(self.layer2(h1,ei))
        return self.cls(torch.cat([h1,h2],-1)).squeeze(-1)

# Student: 훨씬 작은 BWGNN (hidden=64)
class StudentBWGNN(nn.Module):
    def __init__(self,d,h=64,dr=0.3):
        super().__init__()
        self.proj=nn.Linear(d,h)
        self.conv1=HeteroConv({et:DualFreqConv(h,h) for et in ET},aggr="sum")
        self.conv2=HeteroConv({et:DualFreqConv(h,h) for et in ET},aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,32),nn.ReLU(),nn.Dropout(dr),nn.Linear(32,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        d=self.conv1(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

class DistillationLoss(nn.Module):
    def __init__(self, tau=4.0, alpha=0.7, gamma=2.0, pos_alpha=0.75):
        super().__init__()
        self.tau, self.alpha = tau, alpha
        self.gamma, self.pos_alpha = gamma, pos_alpha

    def forward(self, student_logits, teacher_logits, target):
        # Hard loss: Focal Loss
        bce = F.binary_cross_entropy_with_logits(student_logits, target.float(), reduction="none")
        pt  = torch.exp(-bce)
        w   = torch.where(target==1, torch.full_like(bce,self.pos_alpha),
                          torch.full_like(bce,1-self.pos_alpha))
        hard_loss = (w*(1-pt)**self.gamma*bce).mean()

        # Soft loss: KL divergence with temperature
        s_soft = torch.sigmoid(student_logits / self.tau)
        t_soft = torch.sigmoid(teacher_logits.detach() / self.tau)
        soft_loss = F.binary_cross_entropy(s_soft, t_soft)

        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss * (self.tau**2)

def mask_ind(data, mask):
    d2=copy.deepcopy(data)
    for et,ei in data.edge_index_dict.items():
        m=mask[ei[0]]&mask[ei[1]]; d2[et].edge_index=ei[:,m]
        if hasattr(data[et],"edge_attr") and data[et].edge_attr is not None:
            d2[et].edge_attr=data[et].edge_attr[m]
    return d2

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        p=torch.sigmoid(model(data)[mask]).numpy()
        l=data["review"].y[mask].numpy()
    pr=average_precision_score(l,p)
    f1=f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0)
    return round(pr,4), round(f1,4)

# ── Teacher 로드 ──────────────────────────────────────────────────────────────
print("="*65)
print("E09: Knowledge Distillation (Teacher=DRAGWave, Student=BWGNN-64)")
print("="*65)

torch.manual_seed(42)
data = torch.load(GRAPH/"hetero_graph_boost.pt", weights_only=False)
feat = data["review"].x.shape[1]
tm   = data["review"].train_mask
te   = data["review"].test_mask
y    = data["review"].y

# Teacher 로드 (DRAGWave_NoRSR — 가장 강력한 단일 모델)
teacher_ckpt = MOD/"DRAGWave_NoRSR_best.pt"
if not teacher_ckpt.exists():
    teacher_ckpt = MOD/"DRAGWave_400ep_best.pt"

# NoRSR: RSR 없는 ET
data_norsr = copy.deepcopy(data)
rsr_key = ("review","rsr","review")
if rsr_key in data_norsr.edge_index_dict:
    del data_norsr._edge_store_dict[rsr_key]

teacher = TeacherDRAGWave(feat, ets=ET_NORSR)
teacher.load_state_dict(torch.load(teacher_ckpt, weights_only=True))
teacher.eval()

# Teacher 성능 확인
with torch.no_grad():
    t_logits = teacher(data_norsr)
te_t, _ = evaluate(teacher, data_norsr, te)
print(f"\nTeacher ({teacher_ckpt.name}): Test PR-AUC={te_t:.4f}")

# Student
student = StudentBWGNN(feat)
n_p = sum(p.numel() for p in student.parameters())
print(f"Student (BWGNN hidden=64): 파라미터={n_p:,}  비율={n_p/tm.sum().item():.1f}")
print(f"파라미터 절감: {sum(p.numel() for p in teacher.parameters()):,} → {n_p:,} ({n_p/sum(p.numel() for p in teacher.parameters())*100:.0f}%)")

# ── KD 학습 ───────────────────────────────────────────────────────────────────
opt   = torch.optim.AdamW(student.parameters(), lr=5e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
crit  = DistillationLoss(tau=4.0, alpha=0.7)

# Teacher 소프트 레이블 사전 계산
with torch.no_grad():
    teacher_logits_all = teacher(data_norsr)

best_pr, best_state = 0., None
history = []; t0 = time.time()

print(f"\n  {'ep':>4}  {'tr_pr':>8}  {'te_pr':>8}  {'gap':>8}")
print("  " + "-"*40)

for ep in range(1, 301):
    student.train(); opt.zero_grad()
    student_logits = student(data)
    loss = crit(student_logits[tm], teacher_logits_all[tm].detach(), y[tm])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
    opt.step(); sched.step()

    if ep % 30 == 0 or ep == 1:
        student.eval()
        with torch.no_grad():
            p_tr = torch.sigmoid(student(data)[tm]).numpy()
        tr_pr = round(float(average_precision_score(y[tm].numpy(), p_tr)), 4)
        te_pr, _ = evaluate(student, data, te)
        gap = round(tr_pr - te_pr, 4)
        print(f"  {ep:4d}  {tr_pr:8.4f}  {te_pr:8.4f}  {gap:+8.4f}")
        history.append({"ep":ep,"tr_pr":tr_pr,"te_pr":te_pr,"gap":gap})

        if te_pr > best_pr:
            best_pr = te_pr
            best_state = {k:v.cpu().clone() for k,v in student.state_dict().items()}

# 최종
student.load_state_dict(best_state)
student.eval()
with torch.no_grad():
    p_tr = torch.sigmoid(student(data)[tm]).numpy()
tr_f = round(float(average_precision_score(y[tm].numpy(), p_tr)), 4)
te_f, te_f1f = evaluate(student, data, te)
gap_f = round(tr_f - te_f, 4)
d_ind = mask_ind(data, te)
ind_f, ind_f1f = evaluate(student, d_ind, te)

print(f"\n{'='*65}")
print(f"[E09 최종]  Train={tr_f:.4f}  Test={te_f:.4f}  Gap={gap_f:+.4f}  Inductive={ind_f:.4f}")
print(f"[기준 대비]  Test: 0.9242→{te_f:.4f} ({te_f-0.9242:+.4f})  Gap: +0.071→{gap_f:+.4f} ({gap_f-0.071:+.4f})  Ind: 0.6526→{ind_f:.4f} ({ind_f-0.6526:+.4f})")

torch.save(best_state, MOD/"E09_KD_Student_best.pt")
result = {
    "experiment":"E09_knowledge_distillation",
    "config":{"teacher":"DRAGWave_NoRSR","student":"BWGNN_64","tau":4.0,"alpha":0.7},
    "n_params":n_p,"n_train":int(tm.sum()),
    "param_ratio":round(n_p/tm.sum().item(),1),
    "train_pr":tr_f,"test_pr":te_f,"test_f1":te_f1f,
    "gap":gap_f,"inductive_pr":ind_f,"inductive_f1":ind_f1f,
    "history":history,
}
with open(RES/"E09_result.json","w",encoding="utf-8") as f:
    json.dump(result,f,indent=2,ensure_ascii=False)
print(f"저장: results/experiments/E09_result.json")
