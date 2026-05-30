"""배포용 추론 결과 사전 저장"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from pathlib import Path
from torch_geometric.nn import HeteroConv, MessagePassing

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
MOD   = BASE / "models"
RES   = BASE / "results"

ET = [("review","rtr","review"),("review","rsr","review"),
      ("review","burst","review"),("review","rur","review"),("review","sim","review")]

class DualFreqConv(MessagePassing):
    def __init__(self,a,b): super().__init__(aggr="mean"); self.lin=nn.Linear(a*2,b)
    def forward(self,x,ei): low=self.propagate(ei,x=x); return self.lin(torch.cat([low,x-low],-1))
    def message(self,x_j): return x_j

class HeteroBWGNN(nn.Module):
    def __init__(self,d,h=128,dr=0.3):
        super().__init__()
        self.proj=nn.Linear(d,h)
        self.conv1=HeteroConv({et:DualFreqConv(h,h) for et in ET},aggr="sum")
        self.conv2=HeteroConv({et:DualFreqConv(h,h) for et in ET},aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        d=self.conv1(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

print("그래프 로드 중...")
data = torch.load(GRAPH/"hetero_graph_boost.pt", weights_only=False)
feat = data["review"].x.shape[1]
m = HeteroBWGNN(feat)
m.load_state_dict(torch.load(MOD/"HeteroBWGNN_boost_best.pt", weights_only=True))
m.eval()

print("추론 중...")
with torch.no_grad():
    probs = torch.sigmoid(m(data)).numpy()

np.save(str(RES/"all_probs.npy"),  probs)
np.save(str(RES/"all_labels.npy"), data["review"].y.numpy())
np.save(str(RES/"test_mask.npy"),  data["review"].test_mask.numpy())

burst_ei = data["review","burst","review"].edge_index.numpy()
burst_dt = data["review","burst","review"].edge_attr.squeeze().numpy()
sim_ei   = data["review","sim","review"].edge_index.numpy()
rur_ei   = data["review","rur","review"].edge_index.numpy()
np.save(str(RES/"burst_ei.npy"), burst_ei)
np.save(str(RES/"burst_dt.npy"), burst_dt)
np.save(str(RES/"sim_ei.npy"),   sim_ei)
np.save(str(RES/"rur_ei.npy"),   rur_ei)

print(f"probs shape: {probs.shape}")
print(f"test spam rate: {probs[data['review'].test_mask.numpy()].mean():.3f}")
print("완료")
