"""
快速重新生成对比实验图表
- 使用已训练的模型检查点直接评估
- 用极短训练(2 epoch)获取曲线趋势
- 生成6幅高质量对比图
"""
import os, random, numpy as np, pandas as pd, ast, time
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, hamming_loss, roc_curve, auc)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# ==================== 配置 ====================
IMG_DIR = "data/all_images"
CSV_PATH = "data/full_df.csv"
OUTPUT_DIR = "compare_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")
print(f"使用设备: {DEVICE}")

NUM_CLASSES = 8
CLASS_NAMES = ["正常", "糖尿病", "青光眼", "白内障", "AMD", "高血压", "近视", "其他疾病/异常"]
BATCH_SIZE = 8
THRESHOLD = 0.5
SEED = 42
SUBSET_SIZE = 1000  # 小样本快速评估
QUICK_EPOCHS = 3   # 快速训练epochs

# 颜色方案
C1 = '#2196F3'; C2 = '#FF5722'

def set_seed(s=42):
    random.seed(s); os.environ['PYTHONHASHSEED']=str(s); np.random.seed(s)
    torch.manual_seed(s)
set_seed(SEED)

# ==================== 数据集 ====================
class PairedEyeDataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.df = df.reset_index(drop=True); self.img_dir = img_dir
        t = [transforms.Resize((224,224)), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
        self.transform = transforms.Compose(t)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        L = self.transform(Image.open(os.path.join(self.img_dir,row['Left-Fundus'])).convert('RGB'))
        R = self.transform(Image.open(os.path.join(self.img_dir,row['Right-Fundus'])).convert('RGB'))
        return (L,R), torch.tensor(ast.literal_eval(row['target']),dtype=torch.float32)

class SingleEyeDataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.img_dir = img_dir
        self.samples = []
        for _, row in df.iterrows():
            tgt = ast.literal_eval(row['target'])
            self.samples.append((row['Left-Fundus'],tgt))
            self.samples.append((row['Right-Fundus'],tgt))
        t = [transforms.Resize((224,224)), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
        self.transform = transforms.Compose(t)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        fn, tgt = self.samples[idx]
        return self.transform(Image.open(os.path.join(self.img_dir,fn)).convert('RGB')), torch.tensor(tgt,dtype=torch.float32)

# ==================== 模型 ====================
class ResNet50Classifier(nn.Module):
    def __init__(self, n=8):
        super().__init__()
        r = models.resnet50(weights='IMAGENET1K_V1'); r.fc = nn.Linear(r.fc.in_features,n)
        self.backbone = r
    def forward(self, x): return self.backbone(x)

class SharedFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        r = models.resnet50(weights='IMAGENET1K_V1')
        for p in r.parameters(): p.requires_grad=False
        self.features=nn.Sequential(*list(r.children())[:-1])
    def forward(self,x): return self.features(x).view(x.size(0),-1)

class Expert(nn.Module):
    def __init__(self,i,o):
        super().__init__()
        self.fc=nn.Sequential(nn.Linear(i,1024),nn.ReLU(),nn.BatchNorm1d(1024),nn.Dropout(0.3),
            nn.Linear(1024,512),nn.ReLU(),nn.BatchNorm1d(512),nn.Dropout(0.3),nn.Linear(512,o))
    def forward(self,x): return self.fc(x)

class GatingNetwork(nn.Module):
    def __init__(self,i,n):
        super().__init__()
        self.fc=nn.Sequential(nn.Linear(i,512),nn.ReLU(),nn.Dropout(0.3),nn.Linear(512,n),nn.Softmax(dim=1))
    def forward(self,x): return self.fc(x)

class MixtureOfExperts(nn.Module):
    def __init__(self,ne=8,nt=8,id=4096):
        super().__init__()
        self.se=SharedFeatureExtractor()
        self.experts=nn.ModuleList([Expert(id,nt) for _ in range(ne)])
        self.gating=GatingNetwork(id,ne)
    def forward(self,xl,xr):
        f=torch.cat([self.se(xl),self.se(xr)],dim=1)
        g=self.gating(f); o=torch.stack([e(f) for e in self.experts],dim=1)
        return torch.sum(g.unsqueeze(-1)*o,dim=1)

class FocalLoss(nn.Module):
    def __init__(self,a=1,g=2): super().__init__(); self.a=a; self.g=g
    def forward(self,i,t):
        b=F.binary_cross_entropy_with_logits(i,t,reduction='none')
        return (self.a*(1-torch.exp(-b))**self.g*b).mean()

# ==================== 评估 ====================
def compute_all_metrics(labels, probs, threshold=0.5):
    preds = (probs>threshold).astype(int)
    m = {'subset_acc':accuracy_score(labels,preds), 'hamming_loss':hamming_loss(labels,preds),
         'micro_precision':precision_score(labels,preds,average='micro',zero_division=0),
         'micro_recall':recall_score(labels,preds,average='micro',zero_division=0),
         'micro_f1':f1_score(labels,preds,average='micro',zero_division=0),
         'macro_precision':precision_score(labels,preds,average='macro',zero_division=0),
         'macro_recall':recall_score(labels,preds,average='macro',zero_division=0),
         'macro_f1':f1_score(labels,preds,average='macro',zero_division=0),
         'sample_f1':f1_score(labels,preds,average='samples',zero_division=0)}
    pf = f1_score(labels,preds,average=None,zero_division=0)
    for i,n in enumerate(CLASS_NAMES): m[f'f1_{n}']=pf[i]
    try:
        aucs=roc_auc_score(labels,probs,average=None); m['macro_auc']=np.mean(aucs)
        for i,n in enumerate(CLASS_NAMES): m[f'auc_{n}']=aucs[i]
    except: m['macro_auc']=0
    return m, preds

def evaluate_paired(model, loader, method='resnet'):
    model.eval(); all_probs,all_labels,times=[],[],[]
    with torch.no_grad():
        for (L,R), labels in tqdm(loader, desc=f"评估 {method}"):
            L,R=L.to(DEVICE),R.to(DEVICE)
            t0=time.time()
            if method=='resnet':
                probs=torch.max(torch.sigmoid(model(L)),torch.sigmoid(model(R)))
            else:
                probs=torch.sigmoid(model(L,R))
            times.append(time.time()-t0)
            all_probs.append(probs.cpu().numpy()); all_labels.append(labels.cpu().numpy())
    return np.vstack(all_labels), np.vstack(all_probs), np.mean(times)

# ==================== 快速训练曲线 ====================
def quick_train_resnet(train_loader, val_loader):
    model = ResNet50Classifier(8).to(DEVICE)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    hist = {'train_loss':[], 'val_loss':[], 'val_f1':[]}
    print("\n快速训练 ResNet50 (获取训练曲线)...")
    for ep in range(QUICK_EPOCHS):
        model.train(); tl=0
        for im,lb in tqdm(train_loader, desc=f"RN E{ep+1}/{QUICK_EPOCHS}"):
            im,lb=im.to(DEVICE),lb.to(DEVICE)
            loss=crit(model(im),lb); opt.zero_grad(); loss.backward(); opt.step()
            tl+=loss.item()
        tl/=len(train_loader)
        model.eval(); vl=0; aps,als=[],[]
        with torch.no_grad():
            for im,lb in val_loader:
                im,lb=im.to(DEVICE),lb.to(DEVICE); out=model(im)
                vl+=crit(out,lb).item(); aps.append(torch.sigmoid(out).cpu().numpy()); als.append(lb.cpu().numpy())
        vl/=len(val_loader)
        vf=f1_score(np.vstack(als),(np.vstack(aps)>0.5).astype(int),average='macro',zero_division=0)
        hist['train_loss'].append(tl); hist['val_loss'].append(vl); hist['val_f1'].append(vf)
        print(f"  TL:{tl:.4f} VL:{vl:.4f} VF1:{vf:.4f}")
    return model, hist

def quick_train_moe(train_loader, val_loader):
    model = MixtureOfExperts(8,8,4096).to(DEVICE)
    crit = FocalLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    hist = {'train_loss':[], 'val_loss':[], 'val_f1':[]}
    print("\n快速训练 MoE (获取训练曲线)...")
    for ep in range(QUICK_EPOCHS):
        model.train(); tl=0
        for (L,R), lb in tqdm(train_loader, desc=f"MoE E{ep+1}/{QUICK_EPOCHS}"):
            L,R,lb=L.to(DEVICE),R.to(DEVICE),lb.to(DEVICE)
            loss=crit(model(L,R),lb); opt.zero_grad(); loss.backward(); opt.step()
            tl+=loss.item()
        tl/=len(train_loader)
        model.eval(); vl=0; aps,als=[],[]
        with torch.no_grad():
            for (L,R), lb in val_loader:
                L,R,lb=L.to(DEVICE),R.to(DEVICE),lb.to(DEVICE)
                out=model(L,R); vl+=crit(out,lb).item()
                aps.append(torch.sigmoid(out).cpu().numpy()); als.append(lb.cpu().numpy())
        vl/=len(val_loader)
        vf=f1_score(np.vstack(als),(np.vstack(aps)>0.5).astype(int),average='macro',zero_division=0)
        hist['train_loss'].append(tl); hist['val_loss'].append(vl); hist['val_f1'].append(vf)
        print(f"  TL:{tl:.4f} VL:{vl:.4f} VF1:{vf:.4f}")
    return model, hist

# ==================== 可视化 ====================
def plot_all(rn_m, me_m, rn_p, me_p, rn_l, me_l, rn_h, me_h):
    # 配置中文字体 - 直接注册字体文件
    import matplotlib.font_manager as fm
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    try:
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Microsoft YaHei'
    except:
        pass
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False; plt.rcParams['font.size']=11
    sns.set_style("whitegrid")

    # ---- 图1: 训练曲线 ----
    fig,axes=plt.subplots(1,2,figsize=(15,5.5))
    er=range(1,len(rn_h['train_loss'])+1); em=range(1,len(me_h['train_loss'])+1)
    ax=axes[0]
    ax.plot(er,rn_h['train_loss'],'-',color=C1,lw=2,alpha=0.5,marker='o',ms=4,label='ResNet50 Train')
    ax.plot(er,rn_h['val_loss'],'-',color=C1,lw=2.5,marker='s',ms=5,label='ResNet50 Val')
    ax.plot(em,me_h['train_loss'],'-',color=C2,lw=2,alpha=0.5,marker='o',ms=4,label='MoE Train')
    ax.plot(em,me_h['val_loss'],'-',color=C2,lw=2.5,marker='s',ms=5,label='MoE Val')
    ax.set_xlabel('Epoch',fontsize=12); ax.set_ylabel('Loss',fontsize=12)
    ax.set_title('Loss 曲线对比',fontsize=14,fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax=axes[1]
    ax.plot(er,rn_h['val_f1'],'-',color=C1,lw=2.5,marker='o',ms=6,label='ResNet50')
    ax.plot(em,me_h['val_f1'],'-',color=C2,lw=2.5,marker='s',ms=6,label='MoE')
    # 标注最优
    for h,c,off in [(rn_h,C1,0.02),(me_h,C2,-0.04)]:
        bf=max(h['val_f1']); be=h['val_f1'].index(bf)+1
        ax.annotate(f'{bf:.4f}',xy=(be,bf),xytext=(be-0.5,bf+off),fontsize=9,color=c,fontweight='bold',
                    arrowprops=dict(arrowstyle='->',color=c,alpha=0.5))
    ax.set_xlabel('Epoch',fontsize=12); ax.set_ylabel('Macro F1',fontsize=12)
    ax.set_title('验证集 Macro F1 曲线',fontsize=14,fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout(pad=2)
    plt.savefig(os.path.join(OUTPUT_DIR,'01_training_curves.png'),dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(); print("[1/6] 训练曲线 → 01_training_curves.png")

    # ---- 图2: 总体指标 ----
    mn=['macro_f1','micro_f1','macro_precision','macro_recall','macro_auc']
    ml=['Macro F1','Micro F1','Macro Precision','Macro Recall','Macro AUC']
    fig,ax=plt.subplots(figsize=(11,6.5))
    x=np.arange(len(mn)); w=0.32
    rv=[rn_m.get(k,0) for k in mn]; mv=[me_m.get(k,0) for k in mn]
    diffs=[m-r for r,m in zip(rv,mv)]
    b1=ax.bar(x-w/2,rv,w,label='ResNet50 (后融合)',color=C1,edgecolor='white',lw=0.8,alpha=0.9)
    b2=ax.bar(x+w/2,mv,w,label='MoE (门控融合)',color=C2,edgecolor='white',lw=0.8,alpha=0.9)
    for b,v in zip(b1,rv):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.015,f'{v:.4f}',ha='center',fontsize=9,fontweight='bold',color='#333' if v>0.15 else '#999')
    for b,v in zip(b2,mv):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.015,f'{v:.4f}',ha='center',fontsize=9,fontweight='bold',color='#333')
    for i,d in enumerate(diffs):
        ax.annotate(f'Δ={d:+.3f}',xy=(x[i],max(rv[i],mv[i])+0.08),ha='center',fontsize=8,color='#D84315' if d>0 else '#1565C0',
                    fontweight='bold',bbox=dict(boxstyle='round,pad=0.2',facecolor='#FFF9C4',alpha=0.7))
    ax.set_xticks(x); ax.set_xticklabels(ml,fontsize=11)
    ax.set_ylim(0,max(max(rv),max(mv))*1.35); ax.set_ylabel('Score',fontsize=12)
    ax.set_title('ResNet50 vs MoE — 总体指标对比',fontsize=14,fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='y',alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,'02_overall_metrics.png'),dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(); print("[2/6] 总体指标 → 02_overall_metrics.png")

    # ---- 图3: 逐类F1 ----
    fig,ax=plt.subplots(figsize=(12,7))
    y=np.arange(len(CLASS_NAMES)); h=0.32
    rf=[rn_m.get(f'f1_{n}',0) for n in CLASS_NAMES]
    mf=[me_m.get(f'f1_{n}',0) for n in CLASS_NAMES]
    ax.barh(y-h/2,rf,h,label='ResNet50 (后融合)',color=C1,edgecolor='white',lw=0.8)
    ax.barh(y+h/2,mf,h,label='MoE (门控融合)',color=C2,edgecolor='white',lw=0.8)
    for b,v in zip(ax.containers[0],rf):
        ax.text(max(v+0.008,0.005),b.get_y()+b.get_height()/2,f'{v:.3f}',va='center',fontsize=9,fontweight='bold',color='#333' if v>0.05 else '#999')
    for b,v in zip(ax.containers[1],mf):
        ax.text(v+0.008,b.get_y()+b.get_height()/2,f'{v:.3f}',va='center',fontsize=9,fontweight='bold',color='#333')
    ax.set_yticks(y); ax.set_yticklabels(CLASS_NAMES,fontsize=11)
    ax.set_xlim(0,max(max(rf),max(mf))*1.4); ax.set_xlabel('F1-Score',fontsize=12)
    ax.set_title('ResNet50 vs MoE — 逐类 F1 对比',fontsize=14,fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='x',alpha=0.3); ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,'03_per_class_f1.png'),dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(); print("[3/6] 逐类F1 → 03_per_class_f1.png")

    # ---- 图4: 逐类AUC ----
    fig,ax=plt.subplots(figsize=(12,7))
    ra=[rn_m.get(f'auc_{n}',0) for n in CLASS_NAMES]
    ma=[me_m.get(f'auc_{n}',0) for n in CLASS_NAMES]
    ax.barh(y-h/2,ra,h,label='ResNet50 (后融合)',color=C1,edgecolor='white',lw=0.8)
    ax.barh(y+h/2,ma,h,label='MoE (门控融合)',color=C2,edgecolor='white',lw=0.8)
    for b,v in zip(ax.containers[0],ra):
        ax.text(v+0.008,b.get_y()+b.get_height()/2,f'{v:.4f}',va='center',fontsize=9,fontweight='bold',color='#333')
    for b,v in zip(ax.containers[1],ma):
        ax.text(v+0.008,b.get_y()+b.get_height()/2,f'{v:.4f}',va='center',fontsize=9,fontweight='bold',color='#333')
    ax.set_yticks(y); ax.set_yticklabels(CLASS_NAMES,fontsize=11)
    ax.set_xlim(0,max(max(ra),max(ma))*1.3); ax.set_xlabel('AUC',fontsize=12)
    ax.set_title('ResNet50 vs MoE — 逐类 AUC 对比',fontsize=14,fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='x',alpha=0.3); ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,'04_per_class_auc.png'),dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(); print("[4/6] 逐类AUC → 04_per_class_auc.png")

    # ---- 图5: ROC曲线 ----
    fig=plt.figure(figsize=(18,12))
    ax1=plt.subplot(3,3,(1,3))
    fpr_r,tpr_r,_=roc_curve(rn_l.ravel(),rn_p.ravel()); auc_r=auc(fpr_r,tpr_r)
    fpr_m,tpr_m,_=roc_curve(me_l.ravel(),me_p.ravel()); auc_m=auc(fpr_m,tpr_m)
    ax1.plot(fpr_r,tpr_r,'-',color=C1,lw=2.5,label=f'ResNet50 (AUC={auc_r:.4f})')
    ax1.plot(fpr_m,tpr_m,'-',color=C2,lw=2.5,label=f'MoE (AUC={auc_m:.4f})')
    ax1.plot([0,1],[0,1],'k--',label='Random',alpha=0.4,lw=1)
    ax1.fill_between(fpr_r,tpr_r,alpha=0.05,color=C1)
    ax1.fill_between(fpr_m,tpr_m,alpha=0.05,color=C2)
    ax1.set_xlabel('False Positive Rate',fontsize=12); ax1.set_ylabel('True Positive Rate',fontsize=12)
    ax1.set_title('ROC Curve (Micro-average)',fontsize=14,fontweight='bold')
    ax1.legend(fontsize=10); ax1.grid(alpha=0.3); ax1.set_xlim([-0.02,1.02]); ax1.set_ylim([-0.02,1.02])
    # 逐类ROC: 6 classes in 3x3 grid (positions 4-9)
    for idx,cn in enumerate(CLASS_NAMES[:6]):
        ci=CLASS_NAMES.index(cn); ax=plt.subplot(3,3,idx+4)
        fr_c,tr_c,_=roc_curve(rn_l[:,ci],rn_p[:,ci]); ar_c=auc(fr_c,tr_c)
        fm_c,tm_c,_=roc_curve(me_l[:,ci],me_p[:,ci]); am_c=auc(fm_c,tm_c)
        ax.plot(fr_c,tr_c,'-',color=C1,lw=1.8,label=f'ResNet50 ({ar_c:.3f})')
        ax.plot(fm_c,tm_c,'-',color=C2,lw=1.8,label=f'MoE ({am_c:.3f})')
        ax.plot([0,1],[0,1],'k--',alpha=0.3,lw=0.8)
        ax.set_title(f'{cn}',fontsize=12,fontweight='bold'); ax.legend(fontsize=8); ax.grid(alpha=0.2)
        ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.02])
    plt.suptitle('ResNet50 vs MoE — ROC 曲线分析',fontsize=16,fontweight='bold',y=0.98)
    plt.tight_layout(pad=3)
    plt.savefig(os.path.join(OUTPUT_DIR,'05_roc_curves.png'),dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(); print("[5/6] ROC曲线 → 05_roc_curves.png")

    # ---- 图6: 推理速度+参数量 ----
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    ax=axes[0]
    tms=[rn_m['inference_time']*1000,me_m['inference_time']*1000]
    bars=ax.bar(['ResNet50\n(后融合)','MoE\n(门控融合)'],tms,color=[C1,C2],edgecolor='white',lw=1.5,width=0.55)
    for b,t in zip(bars,tms):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,f'{t:.1f} ms',ha='center',fontsize=13,fontweight='bold',color='#333')
    ax.set_ylabel('推理时间 (ms/对)',fontsize=11)
    ax.set_title('单对眼底图推理速度',fontsize=13,fontweight='bold')
    ax.grid(axis='y',alpha=0.3); ax.set_ylim(0,max(tms)*1.25)
    ax=axes[1]
    rt=ResNet50Classifier(8); mt=MixtureOfExperts(8,8,4096)
    rp=sum(p.numel() for p in rt.parameters())
    mp=sum(p.numel() for p in mt.parameters())
    mtp=sum(p.numel() for p in mt.parameters() if p.requires_grad)
    pd_data={'ResNet50\n(后融合)':rp,'MoE\n(总参数)':mp,'MoE\n(可训练)':mtp}
    pcol=[C1,C2,'#FFAB91']
    pb=ax.bar(pd_data.keys(),pd_data.values(),color=pcol,edgecolor='white',lw=1.5,width=0.55)
    for b,v in zip(pb,pd_data.values()):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.5e6,f'{v/1e6:.1f}M',ha='center',fontsize=11,fontweight='bold',color='#333')
    ax.set_ylabel('参数量',fontsize=11); ax.set_title('模型参数量对比',fontsize=13,fontweight='bold'); ax.grid(axis='y',alpha=0.3)
    fig.tight_layout(pad=2)
    fig.savefig(os.path.join(OUTPUT_DIR,'06_model_comparison.png'),dpi=150,facecolor='white')
    plt.close(); print("[6/6] 推理+参数量 → 06_model_comparison.png")

# ==================== 主流程 ====================
def main():
    print("="*60)
    print("快速重新生成对比实验图表")
    print("="*60)

    # 1. 数据
    print("\n[1/4] 加载数据...")
    df = pd.read_csv(CSV_PATH).sample(n=SUBSET_SIZE, random_state=SEED).reset_index(drop=True)
    print(f"  样本: {len(df)}")
    n=int(0.8*len(df)); indices=list(range(len(df))); random.shuffle(indices)
    tr_idx=indices[:n]; te_idx=indices[n:]
    tr_df=df.iloc[tr_idx].reset_index(drop=True); te_df=df.iloc[te_idx].reset_index(drop=True)
    n2=int(0.8*len(tr_df))
    va_df=tr_df.iloc[n2:].reset_index(drop=True); tr_df=tr_df.iloc[:n2].reset_index(drop=True)
    print(f"  训练:{len(tr_df)} 验证:{len(va_df)} 测试:{len(te_df)}")

    # 2. DataLoader
    rn_train=DataLoader(SingleEyeDataset(tr_df,IMG_DIR,augment=True),batch_size=BATCH_SIZE,shuffle=True)
    rn_val=DataLoader(SingleEyeDataset(va_df,IMG_DIR),batch_size=BATCH_SIZE,shuffle=False)
    me_train=DataLoader(PairedEyeDataset(tr_df,IMG_DIR,augment=True),batch_size=BATCH_SIZE,shuffle=True)
    me_val=DataLoader(PairedEyeDataset(va_df,IMG_DIR),batch_size=BATCH_SIZE,shuffle=False)
    te_loader=DataLoader(PairedEyeDataset(te_df,IMG_DIR),batch_size=BATCH_SIZE,shuffle=False)

    # 3. 快速训练(获取曲线) + 完整评估
    print("\n[2/4] 快速训练获取收敛曲线...")
    rn_model, rn_hist = quick_train_resnet(rn_train, rn_val)
    me_model, me_hist = quick_train_moe(me_train, me_val)

    print("\n[3/4] 统一评估两个模型...")
    rn_l, rn_p, rn_t = evaluate_paired(rn_model, te_loader, 'resnet')
    rn_m, _ = compute_all_metrics(rn_l, rn_p, THRESHOLD); rn_m['inference_time']=rn_t
    me_l, me_p, me_t = evaluate_paired(me_model, te_loader, 'moe')
    me_m, _ = compute_all_metrics(me_l, me_p, THRESHOLD); me_m['inference_time']=me_t

    print(f"\n  ResNet50: Macro F1={rn_m['macro_f1']:.4f} AUC={rn_m['macro_auc']:.4f} Time={rn_t*1000:.1f}ms")
    print(f"  MoE:      Macro F1={me_m['macro_f1']:.4f} AUC={me_m['macro_auc']:.4f} Time={me_t*1000:.1f}ms")

    # 4. 生成图表
    print("\n[4/4] 生成对比图表...")
    plot_all(rn_m, me_m, rn_p, me_p, rn_l, me_l, rn_hist, me_hist)

    print("\n" + "="*60)
    print("图表重新生成完成！")
    print(f"  输出目录: {OUTPUT_DIR}/")
    print("="*60)

if __name__=='__main__':
    main()
