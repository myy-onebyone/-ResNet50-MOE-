"""
眼底疾病智能诊断系统 — ResNet50 + MoE 双模型集成对比
================================================================
集成策略: 根据两种模型在各类疾病上的验证表现，智能选择最优模型结果
  - ResNet50 优势: 正常、糖尿病、AMD、其他异常 (后融合简单有效)
  - MoE 优势:     青光眼、白内障、近视 (双流特征级融合更敏感)
  - 高血压:       两种模型均偏弱，综合参考
"""
import sys, os, torch, torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QWidget, QFileDialog, QMessageBox, QSizePolicy, QFrame, QGroupBox,
    QListWidget, QListWidgetItem, QScrollArea, QSplitter, QTextEdit, QTabWidget,
    QDialog
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QLinearGradient, QMovie
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image

# MATLAB 风格量化分析模块
from matlab_style_analysis import MatlabStylePlotter, FundusImageAnalyzer

# ==================== 模型定义(与训练保持一致) ====================
class SharedFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        r = models.resnet50(weights=None)
        for p in r.parameters(): p.requires_grad = False
        self.features = nn.Sequential(*list(r.children())[:-1])
    def forward(self, x): return self.features(x).view(x.size(0), -1)

class Expert(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(i,1024),nn.ReLU(),nn.BatchNorm1d(1024),nn.Dropout(0.3),
            nn.Linear(1024,512),nn.ReLU(),nn.BatchNorm1d(512),nn.Dropout(0.3),
            nn.Linear(512,o))
    def forward(self,x): return self.fc(x)

class GatingNetwork(nn.Module):
    def __init__(self,i,n):
        super().__init__()
        self.fc=nn.Sequential(nn.Linear(i,512),nn.ReLU(),nn.Dropout(0.3),nn.Linear(512,n),nn.Softmax(dim=1))
    def forward(self,x): return self.fc(x)

class MixtureOfExperts(nn.Module):
    def __init__(self,ne=8,nt=8,id=4096):
        super().__init__()
        self.shared_extractor=SharedFeatureExtractor()
        self.experts=nn.ModuleList([Expert(id,nt) for _ in range(ne)])
        self.gating=GatingNetwork(id,ne)
    def forward(self,xl,xr):
        f=torch.cat([self.shared_extractor(xl),self.shared_extractor(xr)],dim=1)
        g=self.gating(f); o=torch.stack([e(f) for e in self.experts],dim=1)
        return torch.sum(g.unsqueeze(-1)*o,dim=1)

class ResNet50Classifier(nn.Module):
    def __init__(self,n=8):
        super().__init__()
        r=models.resnet50(weights=None); r.fc=nn.Linear(r.fc.in_features,n)
        self.backbone=r
    def forward(self,x): return self.backbone(x)

# ==================== 配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["正常","糖尿病","青光眼","白内障","AMD","高血压","近视","其他疾病/异常"]
THRESHOLD = 0.5

# 颜色 - 柔和现代色调
C_PRIMARY="#263238"; C_ACCENT="#00ACC1"
C_RN="#2E7D32"; C_MOE="#EF6C00"; C_EN="#5E35B1"
C_RN_BG="#E8F5E9"; C_MOE_BG="#FFF3E0"; C_EN_BG="#EDE7F6"
C_BG="#ECEFF1"; C_CARD="#FFFFFF"
C_SHADOW = "#CFD8DC"

NORM = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
T = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), NORM])

# ==================== 智能集成诊断引擎 ====================
# 各模型逐类F1(基于严谨训练验证集)
RN_F1 = [0.561, 0.630, 0.301, 0.578, 0.419, 0.222, 0.746, 0.563]
MOE_F1 = [0.408, 0.519, 0.438, 0.745, 0.143, 0.250, 0.810, 0.246]

# 归一化权重: w_rn[i] = RN_F1[i] / (RN_F1[i] + MOE_F1[i])
RN_W = np.zeros(8); MOE_W = np.zeros(8)
for i in range(8):
    total = RN_F1[i] + MOE_F1[i]
    if total > 0:
        RN_W[i] = RN_F1[i] / total
        MOE_W[i] = MOE_F1[i] / total
    else:
        RN_W[i] = MOE_W[i] = 0.5

# 自适应阈值(基于F1表现)
def get_adaptive_threshold(f1):
    """F1越高→阈值越严格; F1越低→阈值放松但标记不确定"""
    if f1 >= 0.6: return 0.50   # 高性能: 标准阈值
    elif f1 >= 0.3: return 0.40 # 中等: 放宽
    else: return 0.35            # 弱: 更宽但需标记

THRESHOLDS = [get_adaptive_threshold(max(RN_F1[i], MOE_F1[i])) for i in range(8)]

# 共病关联矩阵(医学先验)
# 糖尿病↔高血压, 糖尿病↔AMD, 青光眼↔近视, 高血压↔AMD
COMORBIDITY = {
    (1,5):0.08, (5,1):0.08,  # 糖尿病↔高血压 (共病关联)
    (1,4):0.10, (4,1):0.10,  # 糖尿病↔AMD
    (2,6):0.10, (6,2):0.10,  # 青光眼↔近视
    (5,4):0.10, (4,5):0.10,  # 高血压↔AMD
}

class IntegratedDiagnosisEngine:
    """集成诊断引擎: 加权融合 + 一致性检验 + 不确定性量化 + 共病分析"""

    @staticmethod
    def integrate(rn_probs, me_probs):
        """
        返回:
          diagnosis: List[(name, final_score, confidence_level, source_info)]
          report: 结构化报告文本
        """
        # === 1. 动态加权融合 ===
        base_rn_w = RN_W.copy()
        base_me_w = MOE_W.copy()

        for i in range(8):
            divergence = abs(rn_probs[i] - me_probs[i])
            rn_better = RN_F1[i] > MOE_F1[i]
            # ResNet50 在其优势疾病(正常/糖尿病/AMD/其他)上F1显著更高 → 增强动态权重
            rn_dominant = (rn_better and RN_F1[i] > 0.4 and RN_F1[i] > MOE_F1[i] * 1.2)

            if rn_dominant and rn_probs[i] > me_probs[i]:
                # ResNet50强势类别+ResNet50更自信 → 大幅加权
                boost = min(0.6, divergence * 0.6 + 0.15)
                base_rn_w[i] = min(0.95, base_rn_w[i] + boost * (1 - base_rn_w[i]))
                base_me_w[i] = 1.0 - base_rn_w[i]
            elif rn_better and rn_probs[i] > me_probs[i]:
                boost = divergence * 0.4
                base_rn_w[i] = min(0.90, base_rn_w[i] + boost * (1 - base_rn_w[i]))
                base_me_w[i] = 1.0 - base_rn_w[i]
            elif not rn_better and me_probs[i] > rn_probs[i]:
                boost = divergence * 0.4
                base_me_w[i] = min(0.90, base_me_w[i] + boost * (1 - base_me_w[i]))
                base_rn_w[i] = 1.0 - base_me_w[i]

        weighted_prob = base_rn_w * rn_probs + base_me_w * me_probs

        # === 2. 模型一致性 ===
        agreement = 1.0 - np.abs(rn_probs - me_probs)  # 0=完全不一致, 1=完全一致

        # === 3. 校准置信度 ===
        # 高一致性+高概率=高置信度; 低一致性=置信度打折
        calibrated = weighted_prob * (0.4 + 0.6 * agreement)

        # === 4. 共病增强 ===
        enhanced = calibrated.copy()
        for (a, b), boost in COMORBIDITY.items():
            if weighted_prob[a] > 0.4:
                enhanced[b] = min(1.0, enhanced[b] + boost * weighted_prob[a])

        # === 5. 自适应阈值判定 ===
        results = []
        for i, name in enumerate(CLASS_NAMES):
            score = float(enhanced[i])
            threshold = THRESHOLDS[i]
            rn_c = float(rn_probs[i]); me_c = float(me_probs[i])
            agr = float(agreement[i])
            best_f1 = max(RN_F1[i], MOE_F1[i])

            if score >= threshold:
                # 确定置信度等级
                if score >= 0.8 and agr >= 0.7:
                    level = "🟢 高置信度"
                    detail = "双模型高度一致"
                elif score >= 0.6 or (agr >= 0.6 and score >= threshold):
                    level = "🟡 中等置信度"
                    detail = "双模型基本一致" if agr >= 0.6 else "单模型主导"
                elif agr < 0.4:
                    level = "🔴 需复核"
                    detail = f"模型意见分歧(ResNet50:{rn_c:.0%}, MoE:{me_c:.0%})"
                else:
                    level = "🟠 低置信度"
                    detail = f"置信度偏低(模型一致性:{agr:.0%})"

                results.append({
                    'name': name,
                    'score': score,
                    'threshold': threshold,
                    'level': level,
                    'detail': detail,
                    'rn_prob': rn_c,
                    'me_prob': me_c,
                    'agreement': agr,
                    'best_model': "ResNet50" if RN_F1[i] > MOE_F1[i] else "MoE",
                    'best_f1': best_f1,
                    'rn_weight': float(base_rn_w[i]),
                    'me_weight': float(base_me_w[i]),
                })

        # 按优先级+分数排序: 先特异性疾病，后兜底分类
        results.sort(key=lambda x: (-DISEASE_PRIORITY.get(x['name'],0), -x['score']))
        return results

    @staticmethod
    def generate_report(results, rn_probs, me_probs):
        lines = []
        lines.append("═" * 52)
        lines.append("   🔬 智能医学分析报告")
        lines.append("═" * 52)

        positive = [r for r in results if r['score'] >= r['threshold']]
        high_conf = [r for r in positive if '高置信度' in r['level']]
        mid_conf = [r for r in positive if '中等置信度' in r['level']]
        low_conf = [r for r in positive if '低置信度' in r['level'] or '需复核' in r['level']]

        # ---- A. 核心诊断结论 ----
        lines.append("\n┌─ 核心诊断结论")
        if high_conf:
            names = "、".join([r['name'] for r in high_conf[:3]])
            lines.append(f"│  🎯 高置信度阳性: {names}")
        elif mid_conf:
            names = "、".join([r['name'] for r in mid_conf[:3]])
            lines.append(f"│  📋 中等置信度阳性: {names}")
        elif low_conf:
            names = "、".join([r['name'] for r in low_conf[:3]])
            lines.append(f"│  ⚠ 低置信度阳性(建议复核): {names}")
        else:
            lines.append(f"│  ✅ 未见明确阳性病变")

        # ---- B. 模型一致性分析 ----
        agreements = [r['agreement'] for r in results]
        avg_agr = np.mean(agreements) if agreements else 0
        disagreements = [(i, rn_probs[i], me_probs[i])
                        for i in range(8) if abs(rn_probs[i]-me_probs[i]) > 0.3]
        lines.append("│")
        lines.append(f"│  📊 模型一致性: {avg_agr:.0%}")
        if avg_agr > 0.7:
            lines.append("│     → 双模型判断高度一致，结果可靠")
        elif avg_agr > 0.5:
            lines.append("│     → 双模型基本一致，部分类别存在差异")
        else:
            lines.append("│     → 双模型存在明显分歧，建议结合临床判断")
        if disagreements:
            lines.append(f"│  ⚠ 分歧类别({len(disagreements)}项):")
            for i, rn, me in disagreements[:3]:
                lines.append(f"│     {CLASS_NAMES[i]}: ResNet50={rn:.0%} MoE={me:.0%}")

        # ---- C. 阳性发现详情 ----
        if positive:
            lines.append("└" + "─" * 51)
            lines.append("\n┌─ 阳性发现详情")
            for r in positive[:5]:
                lines.append(f"│")
                lines.append(f"│  {r['level']}")
                lines.append(f"│  ├ 疾病: {r['name']}")
                lines.append(f"│  ├ 综合评分: {r['score']:.1%} (阈值: {r['threshold']:.0%})")
                lines.append(f"│  ├ ResNet50: {r['rn_prob']:.1%} | MoE: {r['me_prob']:.1%}")
                lines.append(f"│  ├ 模型一致性: {r['agreement']:.0%}")
                lines.append(f"│  └ 分析: {r['detail']}")
                if r['best_f1'] < 0.3:
                    lines.append(f"│     ⚠ 该疾病检测能力有限(F1={r['best_f1']:.2f})，结果仅供参考")
            lines.append("└" + "─" * 51)

        # ---- D. 临床病例说明 (top-3阳性详细说明) ----
        lines.append("\n┌─ 临床病例说明 (Top-3阳性)")
        for r in positive[:3]:
            name = r['name']
            info = DISEASE_INFO.get(name, "")
            lines.append(f"│\n{info}")
            lines.append(f"│  ├ AI评分: {r['score']:.1%} | 一致性: {r['agreement']:.0%}")
            lines.append(f"│  └ 置信等级: {r['level']}")
            lines.append("│")
        if not positive:
            # 即使全阴性，也显示最可能的印象
            name = results[0]['name']
            info = DISEASE_INFO.get(name, "")
            lines.append(f"│\n{info}")
            lines.append(f"│  └ AI评分: {results[0]['score']:.1%} (未达阳性阈值)")
            lines.append("│")
        lines.append("└" + "─" * 51)

        # ---- E. 鉴别诊断 (所有类别按优先级排列) ----
        lines.append("\n┌─ 鉴别诊断列表 (特异性优先)")
        lines.append("│  (按置信度与特异性排序)")
        for rank, r in enumerate(results):
            mark = "✓ 确诊" if r['score'] >= r['threshold'] else "? 待排"
            lines.append(f"│  {rank+1}. {mark} {r['name']}: {r['score']:.1%} [{r['level']}]")
        lines.append("└" + "─" * 51)

        # ---- F. 多病共存提示 ----
        multi_disease = [r for r in positive if r['name'] != "正常"]
        if len(multi_disease) >= 2:
            lines.append("\n┌─ 多病共存提示")
            lines.append(f"│  ⚠ 检测到{len(multi_disease)}种特异性疾病阳性")
            lines.append("│  眼底疾病常合并存在，需综合考虑诊疗方案")
            lines.append("│  建议多学科联合评估(眼科+内科/内分泌科)")
            lines.append("└" + "─" * 51)

        # ---- G. 免责 ----
        lines.append("\n" + "═" * 52)
        lines.append("⚠ 本报告由AI辅助生成，仅供临床参考")
        lines.append("   最终诊断请以执业医师综合判断为准")
        lines.append("   如症状持续或加重，请及时就医")
        lines.append("═" * 52)

        return "\n".join(lines)

def _remap_state_dict(state_dict, old_prefix, new_prefix):
    """重映射state_dict的key前缀"""
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith(old_prefix):
            new_sd[new_prefix + k[len(old_prefix):]] = v
        else:
            new_sd[k] = v
    return new_sd

def load_models():
    import os as _os
    _base = _os.path.dirname(_os.path.abspath(__file__))
    rn_path = _os.path.join(_base, "..", "models", "best_resnet50.pth")
    moe_path = _os.path.join(_base, "..", "models", "best_moe.pth")

    # ResNet50: 训练时保存为resnet50.xxx, 模型类用backbone.xxx
    rn=ResNet50Classifier(8).to(DEVICE)
    rn_sd = torch.load(rn_path, map_location=DEVICE, weights_only=True)
    rn_sd = _remap_state_dict(rn_sd, "resnet50.", "backbone.")
    rn.load_state_dict(rn_sd)
    rn.eval()

    # MoE: shared_extractor内部也是resnet50.xxx, 也需要remap
    me=MixtureOfExperts(8,8,4096).to(DEVICE)
    me_sd = torch.load(moe_path, map_location=DEVICE, weights_only=True)
    me_sd = _remap_state_dict(me_sd, "shared_extractor.resnet50.", "shared_extractor.features.")
    # MoE的shared_extractor是用nn.Sequential包装的，key是features.xxx而非resnet50.xxx
    # 检查实际key前缀
    sample_keys = list(me_sd.keys())[:5]
    if any("shared_extractor.resnet50" in k for k in sample_keys):
        me_sd = _remap_state_dict(me_sd, "shared_extractor.resnet50.", "shared_extractor.features.")
    me.load_state_dict(me_sd)
    me.eval()
    return rn,me

# ==================== 置信度条 ====================
class ConfidenceBar(QFrame):
    def __init__(self,name,parent=None):
        super().__init__(parent); self.setFixedHeight(36)
        self.name=name; self.value=0.0; self.bar_color=C_RN; self.star=""
    def set_result(self,value,bar_color,star=""):
        self.value=value; self.bar_color=bar_color; self.star=star; self.update()
    def paintEvent(self,e):
        p=QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w,h=self.width(),self.height()
        p.setPen(Qt.NoPen); p.setBrush(QColor("#ECEFF1"))
        p.drawRoundedRect(0,0,w,h,7,7)
        bw=int(w*min(self.value,1.0))
        if bw>0:
            grad=QLinearGradient(0,0,bw,0)
            c=QColor(self.bar_color); grad.setColorAt(0,c.lighter(140)); grad.setColorAt(1,c)
            p.setBrush(grad); p.drawRoundedRect(0,0,bw,h,7,7)
        # 疾病名(左)
        p.setPen(QColor("#263238")); p.setFont(QFont("Microsoft YaHei",9,QFont.Bold))
        p.drawText(10,0,w-110,h,Qt.AlignVCenter|Qt.AlignLeft,self.name)
        # 权重标记(中右)
        if self.star:
            p.setPen(QColor("#78909C")); p.setFont(QFont("Microsoft YaHei",7))
            p.drawText(w-110,0,48,h,Qt.AlignVCenter|Qt.AlignRight,self.star)
        # 百分比(右)
        p.setPen(QColor(self.bar_color)); p.setFont(QFont("Consolas",9,QFont.Bold))
        p.drawText(w-60,0,56,h,Qt.AlignCenter,f"{self.value:.1%}")

# ==================== 疾病数据 ====================
DISEASE_INFO = {
    "糖尿病": (
        "【糖尿病视网膜病变(DR)】\n"
        "━━ 病理特征 ━━\n"
        "微动脉瘤、点状/火焰状出血斑、硬性渗出(黄色蜡样)、棉絮斑(软性渗出)、\n"
        "新生血管形成、静脉串珠状改变、视网膜内微血管异常(IRMA)。\n"
        "━━ 临床表现 ━━\n"
        "早期多无症状，进展期出现视力模糊、飞蚊症、视野缺损，\n"
        "增殖期可致玻璃体积血、牵拉性视网膜脱离，严重可失明。\n"
        "━━ 医学建议 ━━\n"
        "① 严格控制血糖(HbA1c<7%)、血压(<130/80mmHg)、血脂\n"
        "② 每6-12个月散瞳眼底检查，增殖期需全视网膜光凝(PRP)\n"
        "③ 黄斑水肿考虑抗VEGF玻璃体腔注射(雷珠单抗/阿柏西普)\n"
        "④ 晚期增殖性DR需玻璃体切割手术"
    ),
    "青光眼": (
        "【青光眼(Glaucoma)】\n"
        "━━ 病理特征 ━━\n"
        "杯盘比增大(>0.6)、视盘边缘切迹/不规则、视网膜神经纤维层缺损(RNFLD)、\n"
        "视盘周围萎缩弧、盘沿线状出血(Drance出血)、血管向鼻侧移位。\n"
        "━━ 临床表现 ━━\n"
        "开角型: 早期无症状，逐渐出现旁中心暗点→弓形暗点→管状视野→失明\n"
        "闭角型: 急性发作时眼痛、头痛、恶心呕吐、虹视、视力骤降\n"
        "━━ 医学建议 ━━\n"
        "① 一线治疗: 前列腺素类滴眼液(拉坦前列素)降低眼压\n"
        "② 二线: β受体阻滞剂/碳酸酐酶抑制剂/α2受体激动剂\n"
        "③ 激光: 选择性激光小梁成形术(SLT)或激光周边虹膜切开术(LPI)\n"
        "④ 手术: 小梁切除术/引流阀植入术\n"
        "⑤ 终身随访，每3-6个月复查眼压、视野、OCT"
    ),
    "白内障": (
        "【白内障(Cataract)】\n"
        "━━ 病理特征 ━━\n"
        "晶状体蛋白变性导致混浊，按部位分: 核性(中央混浊)、皮质性(轮辐状)、\n"
        "后囊下性(后极部盘状)。混浊程度影响眼底观察清晰度。\n"
        "━━ 临床表现 ━━\n"
        "渐进性无痛性视力下降、视物模糊如隔雾、对比敏感度下降、\n"
        "眩光/畏光(尤其夜间开车)、色彩感知减弱(偏黄)、单眼复视。\n"
        "━━ 医学建议 ━━\n"
        "① 早期: 配镜矫正、增加照明、防紫外线(墨镜)\n"
        "② 影响日常生活时行超声乳化白内障吸除+人工晶体(IOL)植入术\n"
        "③ 术后1天/1周/1月复查，术后使用抗生素+激素眼液\n"
        "④ 手术成功率>95%，是目前最成功的眼科手术之一"
    ),
    "AMD": (
        "【年龄相关性黄斑变性(AMD)】\n"
        "━━ 病理特征 ━━\n"
        "干性(萎缩性): 玻璃膜疣(RPE下黄色沉积)、地图状萎缩(大片RPE丧失)\n"
        "湿性(新生血管性): 黄斑区CNV、视网膜下/内出血、浆液性脱离、盘状瘢痕\n"
        "━━ 临床表现 ━━\n"
        "中心视力下降、视物变形(Amsler表检查阳性)、中心暗点、\n"
        "阅读困难、面部识别障碍。周边视野通常保留。\n"
        "━━ 医学建议 ━━\n"
        "① 干性: AREDS2配方(维生素C/E+锌+叶黄素+玉米黄质)、戒烟、控制血压\n"
        "② 湿性: 尽早抗VEGF玻璃体腔注射(雷珠单抗/阿柏西普/康柏西普)\n"
        "③ 每月复查OCT，按需(PRN)或治疗并延长(T&E)方案\n"
        "④ 居家Amsler表自测，发现变形立即就诊"
    ),
    "高血压": (
        "【高血压视网膜病变(Hypertensive Retinopathy)】\n"
        "━━ 病理特征 ━━\n"
        "1级: 视网膜动脉轻度狭窄 | 2级: 明显狭窄+动静脉交叉压迫\n"
        "3级: 视网膜出血、棉絮斑、硬性渗出 | 4级: 3级+视盘水肿\n"
        "━━ 临床表现 ━━\n"
        "早期多无症状，严重时视力模糊、视野缺损。\n"
        "视盘水肿提示恶性高血压，属急症。\n"
        "眼底改变反映全身微血管损伤，预示卒中/心梗风险增加。\n"
        "━━ 医学建议 ━━\n"
        "① 严格控制血压(目标<130/80mmHg)，规律服用降压药\n"
        "② 低盐饮食(<5g/天)、戒烟限酒、控制体重、规律运动\n"
        "③ 每6-12个月眼底检查监测血管变化\n"
        "④ 合并心内科管理，筛查心脑肾靶器官损害\n"
        "⑤ 出现视盘水肿需紧急降压(住院治疗)"
    ),
    "近视": (
        "【病理性近视(Pathologic Myopia)】\n"
        "━━ 病理特征 ━━\n"
        "豹纹状眼底、视盘倾斜+颞侧弧形斑/萎缩弧、后巩膜葡萄肿、\n"
        "视网膜脉络膜萎缩、漆裂纹(Lacquer cracks)、Fuchs斑。\n"
        "━━ 临床表现 ━━\n"
        "高度近视(等效球镜>-6D或眼轴>26.5mm)患者视力矫正困难，\n"
        "可并发视网膜脱离、近视性黄斑劈裂/CNV、开角型青光眼。\n"
        "━━ 医学建议 ━━\n"
        "① 每年散瞳眼底检查(尤其周边视网膜)，OCT监测黄斑\n"
        "② 避免剧烈运动(蹦极/拳击等)以降低视网膜脱离风险\n"
        "③ 近视性CNV需抗VEGF治疗\n"
        "④ 儿童青少年: 户外活动>2h/天、低浓度阿托品、角膜塑形镜"
    ),
    "其他疾病/异常": (
        "【其他眼部疾病/异常】\n"
        "━━ 说明 ━━\n"
        "此类别涵盖训练数据中除上述明确疾病外的其他眼底异常表现，\n"
        "可能包括: 视网膜静脉/动脉阻塞、视神经病变、葡萄膜炎、\n"
        "视网膜色素变性、黄斑前膜、黄斑裂孔等。\n"
        "当明确疾病已确诊时，此类的诊断权重会适度降低。\n"
        "━━ 医学建议 ━━\n"
        "① 建议眼科专科进一步检查(FFA/OCT/视野等)\n"
        "② 根据具体临床表现确定诊断和治疗方案\n"
        "③ 如仅有此类阳性而无明确疾病，建议短期内复查"
    ),
    "正常": (
        "【正常眼底(Normal Fundus)】\n"
        "━━ 表现 ━━\n"
        "视盘边界清晰、杯盘比正常(<0.5)、视网膜血管走行自然、\n"
        "黄斑中心凹反光可见、无出血/渗出/新生血管/色素异常。\n"
        "━━ 医学建议 ━━\n"
        "✅ 眼底健康，建议每年一次常规眼科体检\n"
        "✅ 保持良好用眼习惯，控制屏幕时间\n"
        "✅ 如有糖尿病/高血压等全身性疾病，定期眼底筛查"
    ),
}

# 疾病优先级(特异性越高越靠前，用于诊断排序)
DISEASE_PRIORITY = {"正常":1,"其他疾病/异常":6,"高血压":3,"AMD":4,"近视":5,"青光眼":7,"白内障":7,"糖尿病":8}

DISEASE_IMAGES = {
    "正常":("../data/影像数据/各类疾病代表医学影像/01_正常_left.jpg","../data/影像数据/各类疾病代表医学影像/01_正常_right.jpg"),
    "糖尿病":("../data/影像数据/各类疾病代表医学影像/02_糖尿病_轻度_left.jpg","../data/影像数据/各类疾病代表医学影像/02_糖尿病_轻度_right.jpg"),
    "青光眼":("../data/影像数据/各类疾病代表医学影像/04_青光眼_left.jpg","../data/影像数据/各类疾病代表医学影像/04_青光眼_right.jpg"),
    "白内障":("../data/影像数据/各类疾病代表医学影像/05_白内障_left.jpg","../data/影像数据/各类疾病代表医学影像/05_白内障_right.jpg"),
    "AMD":("../data/影像数据/各类疾病代表医学影像/06_AMD_干性_left.jpg","../data/影像数据/各类疾病代表医学影像/06_AMD_干性_right.jpg"),
    "高血压":("../data/影像数据/各类疾病代表医学影像/08_高血压_left.jpg","../data/影像数据/各类疾病代表医学影像/08_高血压_right.jpg"),
    "近视":("../data/影像数据/各类疾病代表医学影像/09_近视_left.jpg","../data/影像数据/各类疾病代表医学影像/09_近视_right.jpg"),
    "其他疾病/异常":("../data/影像数据/各类疾病代表医学影像/10_其他_视网膜静脉阻塞_left.jpg","../data/影像数据/各类疾病代表医学影像/10_其他_视网膜静脉阻塞_right.jpg"),
}

# ==================== 后台线程 Worker ====================
class DiagnosisWorker(QThread):
    """后台运行模型推理, 避免界面卡顿"""
    finished = pyqtSignal(dict)   # 结果字典
    error = pyqtSignal(str)

    def __init__(self, resnet, moe, left_raw, right_raw):
        super().__init__()
        self.resnet = resnet
        self.moe = moe
        self.left_raw = left_raw
        self.right_raw = right_raw

    def run(self):
        try:
            lt = T(self.left_raw).unsqueeze(0).to(DEVICE)
            rt = T(self.right_raw).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                lp = torch.sigmoid(self.resnet(lt)).cpu().numpy()[0]
                rp = torch.sigmoid(self.resnet(rt)).cpu().numpy()[0]
            rn_probs = np.maximum(lp, rp)

            lm = T(self.left_raw).unsqueeze(0).to(DEVICE)
            rm = T(self.right_raw).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                mp = torch.sigmoid(self.moe(lm, rm)).cpu().numpy()[0]

            results = IntegratedDiagnosisEngine.integrate(rn_probs, mp)
            en_probs = np.array([r['score'] for r in results])
            en_probs_ordered = np.zeros(8)
            for i, r in enumerate(results):
                en_probs_ordered[CLASS_NAMES.index(r['name'])] = r['score']

            self.finished.emit({
                'rn_probs': rn_probs,
                'me_probs': mp,
                'en_probs': en_probs_ordered,
                'results': results,
            })
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


class AnalysisWorker(QThread):
    """后台生成MATLAB分析图表"""
    done = pyqtSignal()

    def __init__(self, left_np, right_np, output_dir=None):
        super().__init__()
        if output_dir is None:
            import os as _os
            _base = _os.path.dirname(_os.path.abspath(__file__))
            output_dir = _os.path.join(_base, "..", "output", "analysis_output")
        self.left_np = left_np
        self.right_np = right_np
        self.output_dir = output_dir

    def run(self):
        try:
            import os, glob
            os.makedirs(self.output_dir, exist_ok=True)
            for old in glob.glob(os.path.join(self.output_dir, "*.png")):
                try: os.remove(old)
                except: pass
            MatlabStylePlotter.analyze_and_plot(
                self.left_np, self.right_np, output_dir=self.output_dir)
        except Exception:
            pass
        self.done.emit()


# ==================== 主窗口 ====================
class DualCompareWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resnet, self.moe = load_models()
        self.init_ui()
        self.left_path=None; self.right_path=None
        self.left_raw=None; self.right_raw=None
        self._diag_worker = None
        self._analysis_worker = None

    def init_ui(self):
        self.setWindowTitle("眼底疾病智能诊断系统")
        self.setGeometry(40,20,1880,960)
        self.setStyleSheet(f"QMainWindow{{background:{C_BG};}} *{{font-family:'Microsoft YaHei';}}")

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(1)
        main_splitter.setStyleSheet(f"QSplitter::handle{{background:{C_SHADOW};}}")

        # ===== 左侧: 上传区 =====
        left=QWidget()
        left_panel=QVBoxLayout(left); left_panel.setContentsMargins(16,16,16,16); left_panel.setSpacing(10)

        logo=QLabel("眼底疾病\n智能诊断系统")
        logo.setFont(QFont("Microsoft YaHei",15,QFont.Bold)); logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(f"color:{C_PRIMARY};padding:14px;background:white;"
                          f"border-radius:10px;border:1px solid #E0E0E0;")

        img_style=("border:2px dashed #BDBDBD;border-radius:10px;background:white;"
                   "color:#9E9E9E;font-size:11px;")

        # 左眼
        gb_l=QGroupBox("左眼 (Left Eye)")
        gb_l.setStyleSheet(self._gb_style())
        gl=QVBoxLayout(); gl.setSpacing(6)
        self.left_img=QLabel("点击按钮上传\n左眼眼底图像")
        self.left_img.setAlignment(Qt.AlignCenter); self.left_img.setFixedSize(270,270)
        self.left_img.setStyleSheet(img_style)
        bl=QPushButton("上传左眼图片"); bl.clicked.connect(lambda:self.upload("left"))
        bl.setStyleSheet(self._btn(C_RN)); gl.addWidget(self.left_img,alignment=Qt.AlignCenter); gl.addWidget(bl)
        gb_l.setLayout(gl)

        # 右眼
        gb_r=QGroupBox("右眼 (Right Eye)")
        gb_r.setStyleSheet(self._gb_style())
        gr=QVBoxLayout(); gr.setSpacing(6)
        self.right_img=QLabel("点击按钮上传\n右眼眼底图像")
        self.right_img.setAlignment(Qt.AlignCenter); self.right_img.setFixedSize(270,270)
        self.right_img.setStyleSheet(img_style)
        br=QPushButton("上传右眼图片"); br.clicked.connect(lambda:self.upload("right"))
        br.setStyleSheet(self._btn("#0277BD")); gr.addWidget(self.right_img,alignment=Qt.AlignCenter); gr.addWidget(br)
        gb_r.setLayout(gr)

        # 诊断按钮
        self.btn_pred=QPushButton("开始智能诊断")
        self.btn_pred.clicked.connect(self.predict_both); self.btn_pred.setEnabled(False)
        self.btn_pred.setCursor(Qt.PointingHandCursor)
        self.btn_pred.setStyleSheet(f"""
            QPushButton{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {C_ACCENT},stop:1 {C_PRIMARY});
            color:white;border:none;padding:14px;border-radius:8px;font-size:15px;font-weight:bold;}}
            QPushButton:hover{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #26C6DA,stop:1 #37474F);}}
            QPushButton:disabled{{background:#B0BEC5;color:#ECEFF1;}}
        """)

        # MATLAB 量化分析按钮
        self.btn_analyze = QPushButton("图像量化分析")
        self.btn_analyze.clicked.connect(self.run_matlab_analysis)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setCursor(Qt.PointingHandCursor)
        self.btn_analyze.setStyleSheet(f"""
            QPushButton{{background:#FFF;color:{C_MOE};border:2px solid {C_MOE};padding:11px;
            border-radius:8px;font-size:13px;font-weight:bold;}}
            QPushButton:hover{{background:{C_MOE_BG};}}
            QPushButton:disabled{{background:#CFD8DC;color:#90A4AE;border:2px solid #CFD8DC;}}
        """)

        hint=QLabel("上传左右眼图像后开始诊断，可查看 AI 诊断与图像量化分析")
        hint.setAlignment(Qt.AlignCenter); hint.setStyleSheet("color:#90A4AE;font-size:10px;")

        left_panel.addWidget(logo); left_panel.addWidget(gb_l); left_panel.addWidget(gb_r)
        left_panel.addWidget(self.btn_pred); left_panel.addWidget(self.btn_analyze)
        left_panel.addWidget(hint); left_panel.addStretch()

        # ===== 右侧: 结果区 =====
        right=QWidget()
        right_panel=QVBoxLayout(right); right_panel.setContentsMargins(12,12,12,12); right_panel.setSpacing(6)

        header=QLabel("双模型对比 · 智能集成诊断结果")
        header.setFont(QFont("Microsoft YaHei",14,QFont.Bold)); header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet(f"color:{C_PRIMARY};padding:8px;background:white;"
                            f"border-radius:8px;border:1px solid #E0E0E0;")
        right_panel.addWidget(header)

        # ---- 三列结果 ----
        result_row=QHBoxLayout(); result_row.setSpacing(8)

        # ResNet50
        rn_col, rn_con, rn_bars = self._build_column("ResNet50","后融合(Post-fusion)",
            "左/右眼独立推理→Max概率融合", C_RN, C_RN_BG)
        self.rn_conclusion=rn_con; self.rn_bars_list=rn_bars
        result_row.addLayout(rn_col)

        # MoE
        me_col, me_con, me_bars = self._build_column("MoE","门控融合(Gated Fusion)",
            "双流特征拼接+8专家门控投票", C_MOE, C_MOE_BG)
        self.me_conclusion=me_con; self.me_bars_list=me_bars
        result_row.addLayout(me_col)

        # 集成诊断
        en_col, en_con = self._build_ensemble_column()
        self.en_conclusion=en_con
        result_row.addLayout(en_col)

        right_panel.addLayout(result_row, 55)

        # ---- 底部: 分析报告 + 参考图 ----
        # ---- 底部: 三栏 (诊断列表 | 病例详情 | 参考图例) ----
        bottom=QHBoxLayout(); bottom.setSpacing(8)

        # 栏1: 诊断结果列表
        list_gb=QGroupBox("诊断结果")
        list_gb.setStyleSheet(self._gb_style())
        ll=QVBoxLayout()
        self.diag_list=QListWidget()
        self.diag_list.setStyleSheet(
            "QListWidget{border:1px solid #E0E0E0;border-radius:8px;background:white;padding:4px;font-size:12px;}"
            "QListWidget::item{padding:12px 14px;border-radius:6px;margin:3px 4px;font-size:13px;min-height:28px;}"
            "QListWidget::item:selected{background:#6A1B9A;color:white;}"
            "QListWidget::item:hover{background:#F3E5F5;}")
        self.diag_list.currentItemChanged.connect(self._on_disease_selected)
        # 初始占位
        self.diag_list.addItem("诊断后将显示结果...")
        ll.addWidget(self.diag_list); list_gb.setLayout(ll)

        # 栏2: 病例详情
        detail_gb=QGroupBox("临床病例说明")
        detail_gb.setStyleSheet(self._gb_style())
        dl=QVBoxLayout()
        self.detail_text=QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setStyleSheet(
            "QTextEdit{background:white;border:1px solid #E0E0E0;border-radius:8px;"
            "padding:14px;color:#263238;}"
            "QTextEdit:focus{border:1px solid #5E35B1;}")
        self.detail_text.document().setDefaultStyleSheet(
            "body{font-size:14px;line-height:1.8;}")
        self.detail_text.setHtml(
            "<div style='color:#90A4AE;text-align:center;padding:50px 20px;'>"
            "<p style='font-size:28px;margin:0 0 12px 0;'>--</p>"
            "<p style='font-size:14px;'>上传左右眼图像并点击诊断后<br>"
            "点击左侧疾病名称查看详细病例说明</p></div>")
        dl.addWidget(self.detail_text); detail_gb.setLayout(dl)

        # 栏3: 参考图例
        r_gb=QGroupBox("参考图例")
        r_gb.setStyleSheet(self._gb_style())
        rl=QHBoxLayout()
        self.ref_list=QListWidget()
        self.ref_list.addItems(CLASS_NAMES); self.ref_list.setFixedWidth(120)
        self.ref_list.setStyleSheet(
            "QListWidget{border:1px solid #E0E0E0;border-radius:8px;background:white;padding:4px;}"
            "QListWidget::item{padding:7px 10px;border-radius:4px;margin:2px 3px;font-size:11px;}"
            "QListWidget::item:selected{background:#FF9800;color:white;}"
            "QListWidget::item:hover{background:#FFF3E0;}")
        self.ref_list.currentItemChanged.connect(self._update_ref)
        ri=QVBoxLayout(); rt=QHBoxLayout()
        self.ref_left=QLabel("左眼"); self.ref_left.setFixedSize(200,200); self.ref_left.setAlignment(Qt.AlignCenter)
        self.ref_left.setStyleSheet("border:1px solid #E0E0E0;border-radius:8px;background:#FAFAFA;")
        self.ref_right=QLabel("右眼"); self.ref_right.setFixedSize(200,200); self.ref_right.setAlignment(Qt.AlignCenter)
        self.ref_right.setStyleSheet("border:1px solid #E0E0E0;border-radius:8px;background:#FAFAFA;")
        rt.addWidget(self.ref_left); rt.addWidget(self.ref_right); ri.addLayout(rt)
        rl.addWidget(self.ref_list); rl.addLayout(ri); r_gb.setLayout(rl)

        bottom.addWidget(list_gb, 22); bottom.addWidget(detail_gb, 50); bottom.addWidget(r_gb, 28)
        right_panel.addLayout(bottom, 50)

        main_splitter.addWidget(left); main_splitter.addWidget(right)
        main_splitter.setSizes([400,1450])
        self.setCentralWidget(main_splitter)
        self.ref_list.setCurrentRow(0)

        # 存储诊断结果用于列表点击
        self.current_diag_results = []

    def _on_disease_selected(self):
        """点击诊断列表中的疾病时显示病例详情"""
        item = self.diag_list.currentItem()
        if not item or not hasattr(self, 'current_diag_results'): return
        data = item.data(Qt.UserRole)
        if not data: return

        name = data.get('name', '')
        info = DISEASE_INFO.get(name, "暂无详细信息")
        score = data.get('score', 0)
        level = data.get('level', '')
        agreement = data.get('agreement', 0)
        rn_p = data.get('rn_prob', 0)
        me_p = data.get('me_prob', 0)
        best_model = data.get('best_model', '')
        idx = CLASS_NAMES.index(name)

        if '高' in level: lc = '#2E7D32'; status = '高置信度阳性'
        elif '中等' in level: lc = '#EF6C00'; status = '中等置信度'
        elif '需复核' in level: lc = '#C62828'; status = '需进一步检查'
        else: lc = '#78909C'; status = '低置信度'

        dynamic_rn = data.get('rn_weight', RN_W[idx])
        agree_text = ('双模型判断一致，诊断可靠' if agreement > 0.7 else
                     '双模型基本一致，可参考' if agreement > 0.5 else
                     '双模型存在分歧，建议进一步检查')
        agree_color = '#2E7D32' if agreement > 0.7 else '#EF6C00' if agreement > 0.5 else '#C62828'

        ai_summary = f"""
        <div style='background:#F5F7FA;border-radius:8px;padding:16px;margin-bottom:14px;'>
        <h3 style='margin:0 0 12px 0;color:#263238;font-size:16px;'>AI 诊断摘要</h3>
        <table style='font-size:14px;width:100%;border-collapse:collapse;'>
        <tr><td style='padding:6px 10px;width:28%;color:#546E7A;'>诊断状态</td>
            <td style='padding:6px 10px;color:{lc};font-size:15px;font-weight:bold;'>{status}</td></tr>
        <tr><td style='padding:6px 10px;color:#546E7A;'>综合评分</td>
            <td style='padding:6px 10px;font-size:15px;font-weight:bold;color:{lc};'>{score:.1%}</td></tr>
        <tr style='background:#ECEFF1;'><td style='padding:6px 10px;color:#546E7A;'>ResNet50</td>
            <td style='padding:6px 10px;'>{rn_p:.1%}</td></tr>
        <tr style='background:#FFF8E1;'><td style='padding:6px 10px;color:#546E7A;'>MoE 门控融合</td>
            <td style='padding:6px 10px;'>{me_p:.1%}</td></tr>
        <tr><td style='padding:6px 10px;color:#546E7A;'>模型一致性</td>
            <td style='padding:6px 10px;color:{agree_color};font-weight:bold;'>{agreement:.0%} — {agree_text}</td></tr>
        <tr><td style='padding:6px 10px;color:#546E7A;'>最优模型</td>
            <td style='padding:6px 10px;'>{best_model} (验证F1={max(RN_F1[idx],MOE_F1[idx]):.3f})</td></tr>
        </table></div>
        """

        # 疾病信息格式化: 【标题】→h4, ━━ 小标题 ━━→加粗彩色
        info_html = info.replace('\n', '<br>')
        import re
        info_html = re.sub(r'【(.+?)】', r'<h4 style="margin:16px 0 8px 0;color:#37474F;font-size:15px;">\1</h4>', info_html)
        info_html = re.sub(r'━━(.+?)━━', r'<p style="margin:12px 0 4px 0;color:#5E35B1;font-size:14px;font-weight:bold;">\1</p>', info_html)

        html = f"""
        <div style='font-size:14px;line-height:1.9;color:#263238;'>
        {ai_summary}
        <div style='background:white;border-radius:8px;padding:16px;border:1px solid #E0E0E0;'>
        <h3 style='margin:0 0 12px 0;color:#263238;font-size:16px;'>临床病例详情 — {name}</h3>
        {info_html}
        </div>
        <div style='margin-top:12px;padding:12px 14px;background:#FFF8E1;border-radius:6px;
        font-size:12px;color:#795548;border-left:3px solid #FF9800;'>
        <b>提示</b>&nbsp; 本AI诊断基于眼底图像特征，不替代裂隙灯、OCT、视野等临床检查。
        诊断结果需结合病史、症状、体征综合判断。</div></div>
        """
        self.detail_text.setHtml(html)

    def _gb_style(self):
        return ("QGroupBox{border:1px solid #E0E0E0;border-radius:10px;"
                "margin-top:14px;padding-top:20px;background:white;font-weight:bold;font-size:12px;}"
                "QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 6px;color:#37474F;}")

    def _btn(self,color):
        return (f"QPushButton{{background:{color};color:white;border:none;padding:9px 12px;"
                f"border-radius:6px;font-size:12px;font-weight:bold;}}"
                f"QPushButton:hover{{background:{color}DD;}}"
                f"QPushButton:pressed{{background:{color};}}")

    def _build_column(self, title, method, desc, color, bg):
        col=QVBoxLayout(); col.setSpacing(2)
        t=QLabel(title); t.setAlignment(Qt.AlignCenter)
        t.setFont(QFont("Microsoft YaHei",11,QFont.Bold))
        t.setStyleSheet(f"color:white;background:{color};border-radius:6px;padding:6px;")
        con=QLabel("等待诊断..."); con.setAlignment(Qt.AlignCenter); con.setWordWrap(True)
        con.setFont(QFont("Microsoft YaHei",10,QFont.Bold))
        con.setStyleSheet(f"background:{bg};border:1px solid {color}33;"
                         f"border-radius:6px;padding:6px;color:#333;min-height:44px;")
        bl=QVBoxLayout(); bl.setSpacing(2); bars=[]
        for name in CLASS_NAMES:
            bar=ConfidenceBar(name); bars.append(bar); bl.addWidget(bar)
        col.addWidget(t); col.addWidget(con); col.addLayout(bl); col.addStretch()
        return col, con, bars

    def _build_ensemble_column(self):
        col=QVBoxLayout(); col.setSpacing(2)
        t=QLabel("集成诊断"); t.setAlignment(Qt.AlignCenter)
        t.setFont(QFont("Microsoft YaHei",11,QFont.Bold))
        t.setStyleSheet(f"color:white;background:{C_EN};border-radius:6px;padding:6px;")
        con=QLabel("等待诊断..."); con.setAlignment(Qt.AlignCenter); con.setWordWrap(True)
        con.setFont(QFont("Microsoft YaHei",10,QFont.Bold))
        con.setStyleSheet(f"background:{C_EN_BG};border:1px solid {C_EN}33;"
                         f"border-radius:6px;padding:8px;color:#333;min-height:48px;")
        detail=QLabel(""); detail.setWordWrap(True); detail.setFont(QFont("Microsoft YaHei",8))
        detail.setStyleSheet("color:#90A4AE;padding:2px;")
        col.addWidget(t); col.addWidget(con); col.addWidget(detail); col.addStretch()
        self.en_detail=detail
        return col, con

    # ===== 事件处理 =====
    def _update_ref(self):
        name=self.ref_list.currentItem().text()
        paths=DISEASE_IMAGES.get(name,("",""))
        for lbl,p in [(self.ref_left,paths[0]),(self.ref_right,paths[1])]:
            if os.path.exists(str(p)):
                pm=QPixmap(str(p)).scaled(230,230,Qt.KeepAspectRatio,Qt.SmoothTransformation)
                lbl.setPixmap(pm)
            else:
                lbl.setText("图片缺失"); lbl.setStyleSheet("border:1px solid #E0E0E0;border-radius:8px;background:#EEE;color:#999;")

    def upload(self,side):
        p,_=QFileDialog.getOpenFileName(self,f"选择{side}眼图片","","Images (*.png *.jpg *.jpeg)")
        if not p: return
        img=Image.open(p).convert("RGB")
        if side=="left": self.left_path=p; self.left_raw=img; self._show(self.left_img,img)
        else: self.right_path=p; self.right_raw=img; self._show(self.right_img,img)
        if self.left_path and self.right_path:
            self.btn_pred.setEnabled(True)
            self.btn_analyze.setEnabled(True)

    def _show(self,lbl,img):
        d=img.convert("RGB").tobytes("raw","RGB")
        pm=QPixmap.fromImage(QImage(d,img.size[0],img.size[1],QImage.Format_RGB888))
        pm=pm.scaled(lbl.width()-16,lbl.height()-16,Qt.KeepAspectRatio,Qt.SmoothTransformation)
        lbl.setPixmap(pm)

    def resizeEvent(self,e):
        super().resizeEvent(e)
        if hasattr(self,'left_raw') and self.left_raw: self._show(self.left_img,self.left_raw)
        if hasattr(self,'right_raw') and self.right_raw: self._show(self.right_img,self.right_raw)

    # ===== 预测逻辑 =====
    def _set_loading(self, loading):
        """切换加载状态: 禁用按钮, 修改光标"""
        self.btn_pred.setEnabled(not loading)
        self.btn_analyze.setEnabled(not loading)
        if loading:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.btn_pred.setText("诊断中...")
        else:
            QApplication.restoreOverrideCursor()
            self.btn_pred.setText("开始智能诊断")

    def predict_both(self):
        """后台线程运行诊断, 避免界面卡顿"""
        self._set_loading(True)
        self._diag_worker = DiagnosisWorker(
            self.resnet, self.moe, self.left_raw, self.right_raw)
        self._diag_worker.finished.connect(self._on_diagnosis_done)
        self._diag_worker.error.connect(self._on_diagnosis_error)
        self._diag_worker.start()

    def _on_diagnosis_done(self, data):
        """诊断完成回调 (主线程)"""
        self._set_loading(False)
        self._display(data['rn_probs'], data['me_probs'],
                     data['en_probs'], data['results'])
        # 后台生成分析图表
        self._update_analysis_charts()

    def _on_diagnosis_error(self, err):
        """诊断出错回调"""
        self._set_loading(False)
        QMessageBox.critical(self, "诊断错误", err)

    def _update_analysis_charts(self):
        """后台线程生成 MATLAB 分析图"""
        left_np = np.array(self.left_raw)
        right_np = np.array(self.right_raw)
        self._analysis_worker = AnalysisWorker(left_np, right_np)
        self._analysis_worker.start()  # 静默运行, 不阻塞UI

    def run_matlab_analysis(self):
        """弹出 MATLAB 量化分析图表预览窗口 (先确保图表已生成)"""
        if not self.left_raw or not self.right_raw:
            QMessageBox.warning(self, "提示", "请先上传左右眼图像！")
            return

        import os
        _base = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(_base, "..", "output", "analysis_output")
        os.makedirs(output_dir, exist_ok=True)
        v1 = os.path.join(output_dir, "matlab_style_analysis_1_visual.png")
        v2 = os.path.join(output_dir, "matlab_style_analysis_2_metrics.png")

        # 如果图表不存在或需要更新, 先生成
        if not os.path.exists(v1) or not os.path.exists(v2):
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                left_np = np.array(self.left_raw)
                right_np = np.array(self.right_raw)
                MatlabStylePlotter.analyze_and_plot(left_np, right_np, output_dir=output_dir)
            finally:
                QApplication.restoreOverrideCursor()

        if not os.path.exists(v1) or not os.path.exists(v2):
            QMessageBox.warning(self, "错误", "分析图表生成失败")
            return

        # ── 图表预览弹窗 ──
        dlg = QDialog(self)
        dlg.setWindowTitle("眼底图像量化分析 - 原始 vs 组合流水线")
        dlg.resize(1100, 800)
        dlg.setStyleSheet("background:#FAFAFA;")

        layout = QVBoxLayout(dlg)
        tabs = QTabWidget()
        tabs.setStyleSheet(
            "QTabWidget::pane{border:1px solid #E0E0E0;background:white;}"
            "QTabBar::tab{padding:8px 24px;font-size:13px;}"
            "QTabBar::tab:selected{font-weight:bold;color:#D84315;"
            "border-bottom:2px solid #D84315;}")

        for path, label in [(v1, "视觉特征对比"), (v2, "量化指标对比")]:
            pix = QPixmap(path)
            lbl = QLabel(); lbl.setPixmap(pix.scaled(
                1050, 680, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            lbl.setAlignment(Qt.AlignCenter)
            scroll = QScrollArea(); scroll.setWidget(lbl)
            scroll.setWidgetResizable(True)
            tabs.addTab(scroll, label)

        layout.addWidget(tabs)
        hint = QLabel("原始图像 vs 组合流水线 的视觉特征与量化指标对比，每次诊断后自动更新")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color:#9E9E9E;font-size:11px;padding:6px;")
        layout.addWidget(hint)
        dlg.exec_()

    def _display(self, rn_probs, me_probs, en_probs, results):
        rn_sort=sorted(enumerate(rn_probs),key=lambda x:-x[1])
        me_sort=sorted(enumerate(me_probs),key=lambda x:-x[1])

        # 构建results查找字典
        result_map = {r['name']: r for r in results}

        # ResNet50列
        rn_top=[f"{CLASS_NAMES[i]}({v:.0%})" for i,v in rn_sort[:2]]
        self.rn_conclusion.setText(f"诊断：{CLASS_NAMES[rn_sort[0][0]]}\n置信度：{rn_sort[0][1]:.0%}\nTop2：{', '.join(rn_top)}")
        for i in range(8):
            name=CLASS_NAMES[i]
            r=result_map.get(name,{})
            star = f"★(权重{r.get('rn_weight',0):.0%})" if r.get('best_model','')=='ResNet50' else ""
            self.rn_bars_list[i].set_result(rn_probs[i],C_RN,star)

        # MoE列
        me_top=[f"{CLASS_NAMES[i]}({v:.0%})" for i,v in me_sort[:2]]
        self.me_conclusion.setText(f"诊断：{CLASS_NAMES[me_sort[0][0]]}\n置信度：{me_sort[0][1]:.0%}\nTop2：{', '.join(me_top)}")
        for i in range(8):
            name=CLASS_NAMES[i]
            r=result_map.get(name,{})
            star = f"★(权重{r.get('me_weight',0):.0%})" if r.get('best_model','')=='MoE' else ""
            self.me_bars_list[i].set_result(me_probs[i],C_MOE,star)

        # 集成诊断
        positive=[r for r in results if r['score']>=r['threshold']]
        high=[r for r in positive if '高置信度' in r['level']]
        if high:
            self.en_conclusion.setText(f"🎯 高置信度阳性:\n{', '.join([r['name'] for r in high[:3]])}")
        elif positive:
            self.en_conclusion.setText(f"📋 阳性发现({len(positive)}项):\n{', '.join([r['name'] for r in positive[:3]])}")
        else:
            self.en_conclusion.setText(f"✅ 未见明确阳性\n倾向: {results[0]['name']}({results[0]['score']:.0%})")

        # 来源详情
        agreements=[r['agreement'] for r in results]
        avg_agr=np.mean(agreements) if agreements else 0
        sources_parts=[]
        for r in results[:5]:
            sources_parts.append(f"{r['name']}:{'RN' if r['best_model']=='ResNet50' else 'MoE'}({r['agreement']:.0%})")
        self.en_detail.setText(f"模型一致性:{avg_agr:.0%} | {' | '.join(sources_parts)}")

        # 填充诊断列表
        self.current_diag_results = results
        self.diag_list.clear()
        positive=[r for r in results if r['score']>=r['threshold']]
        for r in results:
            mark = "✓" if r['score']>=r['threshold'] else "?"
            if '高' in r['level']: icon="🟢"
            elif '中等' in r['level']: icon="🟡"
            elif '需复核' in r['level']: icon="🔴"
            else: icon="⚪"
            item_text = f"{icon} {mark} {r['name']}  {r['score']:.1%}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, r)
            # 高置信度项加粗
            if '高' in r['level']:
                f=item.font(); f.setBold(True); item.setFont(f)
            self.diag_list.addItem(item)

        # 自动选中第一个阳性或最高评分项
        if positive: self.diag_list.setCurrentRow(results.index(positive[0]))
        else: self.diag_list.setCurrentRow(0)

        # 参考图
        top_d=results[0]['name']
        for i in range(self.ref_list.count()):
            if self.ref_list.item(i).text()==top_d:
                self.ref_list.setCurrentRow(i); break


if __name__=='__main__':
    app=QApplication(sys.argv)
    app.setStyle('Fusion')
    w=DualCompareWindow(); w.show()
    sys.exit(app.exec_())
