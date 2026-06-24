"""
================================================================================
眼底图像 MATLAB 风格量化分析模块
================================================================================
提供数字图像处理课程要求的传统图像定量分析功能。

功能:
  1. 图像基础统计 (均值/方差/偏度/峰度/熵)
  2. 直方图分析 (灰度/RGB三通道)
  3. 血管密度估计
  4. 对比度/清晰度分析
  5. 频域分析 (FFT频谱)
  6. MATLAB 风格可视化
  7. 量化分析报告 (文本输出)

输出: 与现有 GUI 集成, 在诊断后一键对眼底图像进行量化分析。
================================================================================
"""
import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import os

from fundus_preprocess import FundusPreprocessor

# 条件导入 matplotlib (允许在无 GUI 环境使用)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


# ============================================================================
# 颜色常量: 模仿 MATLAB 默认配色
# ============================================================================
MATLAB_BLUE   = '#0072BD'
MATLAB_RED    = '#D95319'
MATLAB_YELLOW = '#EDB120'
MATLAB_PURPLE = '#7E2F8E'
MATLAB_GREEN  = '#4DBEEE'
MATLAB_CYAN   = '#77AC30'
MATLAB_COLORS = [MATLAB_BLUE, MATLAB_RED, MATLAB_YELLOW,
                 MATLAB_PURPLE, MATLAB_GREEN, MATLAB_CYAN]

# 中文字体 (MATLAB 风格用英文避免字体问题, 内部标签用中文)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class FundusImageStats:
    """眼底图像综合统计"""
    # 基本属性
    width: int = 0
    height: int = 0
    channels: int = 3

    # 灰度统计
    gray_mean: float = 0.0
    gray_std: float = 0.0
    gray_median: float = 0.0
    gray_min: int = 0
    gray_max: int = 0
    gray_skewness: float = 0.0     # 偏度 (直方图不对称度)
    gray_kurtosis: float = 0.0     # 峰度 (直方图尖锐度)
    gray_entropy: float = 0.0      # 信息熵

    # RGB 三通道统计
    r_mean: float = 0.0;  r_std: float = 0.0
    g_mean: float = 0.0;  g_std: float = 0.0
    b_mean: float = 0.0;  b_std: float = 0.0
    rgb_entropy: float = 0.0

    # 对比度与清晰度
    contrast_rms: float = 0.0      # RMS 对比度
    sharpness: float = 0.0         # 平均梯度清晰度
    laplacian_var: float = 0.0     # Laplacian 方差 (模糊检测)

    # 血管相关
    vessel_density: float = 0.0    # 估计血管密度
    vessel_mean_width: float = 0.0 # 估计平均血管宽度

    # 频域特征
    fft_energy_low: float = 0.0    # 低频能量占比
    fft_energy_mid: float = 0.0    # 中频能量占比
    fft_energy_high: float = 0.0   # 高频能量占比

    # 亮度分布
    brightness_percentiles: List[float] = field(default_factory=list)  # [p5, p25, p50, p75, p95]


# ============================================================================
# 核心分析引擎
# ============================================================================
class FundusImageAnalyzer:
    """眼底图像 MATLAB 风格量化分析引擎"""

    @staticmethod
    def compute_all_stats(image: np.ndarray) -> FundusImageStats:
        """
        计算眼底图像的全部统计特征。

        参数:
            image: RGB 图像 (H, W, 3), uint8

        返回:
            FundusImageStats 对象
        """
        h, w = image.shape[:2]
        c = image.shape[2] if len(image.shape) == 3 else 1
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        stats = FundusImageStats(width=w, height=h, channels=c)

        # ── 1. 灰度基本统计 ──
        gray_flat = gray.ravel().astype(float)
        stats.gray_mean = float(np.mean(gray_flat))
        stats.gray_std  = float(np.std(gray_flat))
        stats.gray_median = float(np.median(gray_flat))
        stats.gray_min = int(gray.min())
        stats.gray_max = int(gray.max())

        # 偏度 = E[(X-μ)³] / σ³
        z = (gray_flat - stats.gray_mean) / (stats.gray_std + 1e-8)
        stats.gray_skewness = float(np.mean(z ** 3))
        # 峰度 = E[(X-μ)⁴] / σ⁴ - 3 (超额峰度, 正态=0)
        stats.gray_kurtosis = float(np.mean(z ** 4) - 3)

        # 信息熵
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist_norm = hist / hist.sum()
        hist_norm = hist_norm[hist_norm > 0]
        stats.gray_entropy = float(-np.sum(hist_norm * np.log2(hist_norm)))

        # ── 2. RGB 三通道统计 ──
        for ch_name, ch_idx in [('r',0), ('g',1), ('b',2)]:
            ch = image[:,:,ch_idx].astype(float)
            setattr(stats, f'{ch_name}_mean', float(np.mean(ch)))
            setattr(stats, f'{ch_name}_std', float(np.std(ch)))

        # RGB 联合熵 (简化: 各通道熵平均)
        rgb_ents = []
        for ch_idx in range(3):
            ch = image[:,:,ch_idx]
            h_ch = cv2.calcHist([ch], [0], None, [256], [0,256])
            h_ch = h_ch / h_ch.sum()
            h_ch = h_ch[h_ch > 0]
            if len(h_ch) > 0:
                rgb_ents.append(float(-np.sum(h_ch * np.log2(h_ch))))
        stats.rgb_entropy = float(np.mean(rgb_ents)) if rgb_ents else 0.0

        # ── 3. 对比度与清晰度 ──
        # RMS 对比度
        stats.contrast_rms = float(stats.gray_std)

        # Laplacian 方差 (模糊检测: 越低越模糊)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        stats.laplacian_var = float(np.var(lap))

        # 平均梯度 (Sobel 清晰度)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        stats.sharpness = float(np.mean(grad_mag))

        # ── 4. 血管密度估计 ──
        vessel_mask, vessel_width = FundusImageAnalyzer._estimate_vessels(gray)
        stats.vessel_density = float(np.sum(vessel_mask) / vessel_mask.size)
        stats.vessel_mean_width = vessel_width

        # ── 5. 频域特征 ──
        fft = np.fft.fftshift(np.fft.fft2(gray.astype(float)))
        fft_mag = np.abs(fft)
        h_center, w_center = h // 2, w // 2
        max_radius = min(h_center, w_center)

        # 三个频段: 低频(r<0.1R), 中频(0.1R<r<0.4R), 高频(r>0.4R)
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((y_grid - h_center)**2 + (x_grid - w_center)**2)

        low_mask = dist < max_radius * 0.1
        mid_mask = (dist >= max_radius * 0.1) & (dist < max_radius * 0.4)
        high_mask = dist >= max_radius * 0.4

        total_energy = np.sum(fft_mag)
        stats.fft_energy_low  = float(np.sum(fft_mag[low_mask]) / total_energy)
        stats.fft_energy_mid  = float(np.sum(fft_mag[mid_mask]) / total_energy)
        stats.fft_energy_high = float(np.sum(fft_mag[high_mask]) / total_energy)

        # ── 6. 亮度百分位数 ──
        stats.brightness_percentiles = [
            float(np.percentile(gray_flat, p)) for p in [5, 25, 50, 75, 95]
        ]

        return stats

    # ------------------------------------------------------------------
    # 血管估计 (基于形态学 + 阈值)
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_vessels(gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        估计眼底图像血管掩膜和平均血管宽度。
        使用多尺度 Gabor 滤波 + Otsu 阈值。
        """
        h, w = gray.shape

        # CLAHE 增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # 顶帽变换提取细小结构
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_small)

        # Otsu 二值化
        _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学清理
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_clean)

        vessel_mask = binary > 0

        # 估计平均血管宽度 (基于距离变换)
        dist = ndimage.distance_transform_edt(vessel_mask)
        # 取骨架上的距离值 × 2 = 血管宽度
        from skimage.morphology import skeletonize
        skeleton = skeletonize(vessel_mask)
        widths = dist[skeleton]
        mean_width = float(np.mean(widths) * 2) if len(widths) > 0 else 0.0

        return vessel_mask, mean_width

    # ------------------------------------------------------------------
    # 格式化统计报告
    # ------------------------------------------------------------------
    @staticmethod
    def format_report(stats: FundusImageStats,
                      left_name: str = "左眼",
                      right_name: str = "右眼") -> str:
        """生成可打印的量化分析报告文本"""
        def fmt_stats(s: FundusImageStats, title: str) -> str:
            return f"""
  ╔══════════════════════════════════════════╗
  ║  {title} 量化分析                        ║
  ╠══════════════════════════════════════════╣
  ║  图像尺寸: {s.width} × {s.height} ({s.channels}通道)
  ║
  ║  [灰度统计]
  ║    均值 μ = {s.gray_mean:.2f}
  ║    标准差 σ = {s.gray_std:.2f}
  ║    中位数 = {s.gray_median:.1f}
  ║    范围 = [{s.gray_min}, {s.gray_max}]
  ║    偏度 γ1 = {s.gray_skewness:+.4f} {"(左偏/暗区多)" if s.gray_skewness > 0 else "(右偏/亮区多)"}
  ║    峰度 γ2 = {s.gray_kurtosis:+.4f} {"(尖峰/对比度高)" if s.gray_kurtosis > 0 else "(扁平/对比度低)"}
  ║    信息熵 H = {s.gray_entropy:.4f} bit
  ║
  ║  [RGB三通道]
  ║    R: μ={s.r_mean:.1f}  σ={s.r_std:.1f}
  ║    G: μ={s.g_mean:.1f}  σ={s.g_std:.1f}
  ║    B: μ={s.b_mean:.1f}  σ={s.b_std:.1f}
  ║    RGB联合熵: {s.rgb_entropy:.4f} bit
  ║
  ║  [对比度 & 清晰度]
  ║    RMS对比度 = {s.contrast_rms:.2f}
  ║    Sobel清晰度 = {s.sharpness:.2f}
  ║    Laplacian方差 = {s.laplacian_var:.2f}
  ║
  ║  [血管分析]
  ║    估计血管密度 = {s.vessel_density*100:.2f}%
  ║    估计平均血管宽度 = {s.vessel_mean_width:.2f} px
  ║
  ║  [频域分析]
  ║    低频能量 = {s.fft_energy_low*100:.1f}%
  ║    中频能量 = {s.fft_energy_mid*100:.1f}%
  ║    高频能量 = {s.fft_energy_high*100:.1f}%
  ║
  ║  [亮度分布 (百分位数)]
  ║    P5={s.brightness_percentiles[0]:.0f}  P25={s.brightness_percentiles[1]:.0f}
  ║    P50={s.brightness_percentiles[2]:.0f}  P75={s.brightness_percentiles[3]:.0f}
  ║    P95={s.brightness_percentiles[4]:.0f}
  ╚══════════════════════════════════════════╝
"""
        left_rpt = fmt_stats(stats[0], left_name) if isinstance(stats, (list, tuple)) else ""
        right_rpt = fmt_stats(stats[1], right_name) if isinstance(stats, (list, tuple)) else ""
        if not isinstance(stats, (list, tuple)):
            return fmt_stats(stats, "")
        return left_rpt + "\n" + right_rpt


# ============================================================================
# MATLAB 风格可视化
# ============================================================================
class MatlabStylePlotter:
    """生成 MATLAB 风格的眼底图像量化分析图表"""

    @staticmethod
    @staticmethod
    def analyze_and_plot(left_img: np.ndarray, right_img: np.ndarray,
                         left_path: str = "左眼", right_path: str = "右眼",
                         output_dir: str = "analysis_output") -> str:
        """
        生成两张独立的分析图:
          图1 - 视觉特征对比: 原图/组合流水线/直方图/差异图/Sobel梯度
          图2 - 量化指标对比: 统计表/频域分析/改善率
        """
        os.makedirs(output_dir, exist_ok=True)

        # 使用组合流水线 (CLAHE → 高斯滤波 → 反锐化掩膜) 替代单一CLAHE
        left_pipeline  = FundusPreprocessor.pipeline_full(left_img)
        right_pipeline = FundusPreprocessor.pipeline_full(right_img)

        left_orig_s  = FundusImageAnalyzer.compute_all_stats(left_img)
        left_pipeline_s = FundusImageAnalyzer.compute_all_stats(left_pipeline)
        right_orig_s  = FundusImageAnalyzer.compute_all_stats(right_img)
        right_pipeline_s = FundusImageAnalyzer.compute_all_stats(right_pipeline)

        left_diff  = np.clip(cv2.absdiff(left_img, left_pipeline).astype(float)*5, 0, 255).astype(np.uint8)
        right_diff = np.clip(cv2.absdiff(right_img, right_pipeline).astype(float)*5, 0, 255).astype(np.uint8)

        gray_l, gray_lc = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY), cv2.cvtColor(left_pipeline, cv2.COLOR_RGB2GRAY)
        gray_r, gray_rc = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY), cv2.cvtColor(right_pipeline, cv2.COLOR_RGB2GRAY)

        # ================================================================
        # 图1: 视觉特征对比 (3行x4列)
        # ================================================================
        fig1 = plt.figure(figsize=(18, 12))
        fig1.suptitle('眼底图像预处理 - 视觉特征对比 (原始 vs 组合流水线)',
                     fontsize=16, fontweight='bold')
        gs1 = GridSpec(3, 4, figure=fig1, hspace=0.35, wspace=0.3)

        for col, (title, orig, pipeline, diff, gray_o, gray_c) in enumerate([
            ('左眼', left_img, left_pipeline, left_diff, gray_l, gray_lc),
            ('右眼', right_img, right_pipeline, right_diff, gray_r, gray_rc),
        ]):
            c0 = col * 2
            # (0, c0): 原始图像
            ax = fig1.add_subplot(gs1[0, c0]); ax.imshow(orig)
            ax.set_title(f'{title} - 原始图像', fontsize=13, fontweight='bold', color='#333')
            ax.axis('off')
            # (0, c0+1): 组合流水线增强
            ax = fig1.add_subplot(gs1[0, c0+1]); ax.imshow(pipeline)
            ax.set_title(f'{title} - 组合流水线', fontsize=13, fontweight='bold', color=MATLAB_RED)
            ax.axis('off')

            # (1, c0): 灰度直方图
            ax = fig1.add_subplot(gs1[1, c0])
            ax.hist(gray_o.ravel(), bins=128, range=(0,255), color='#555', alpha=0.6, label='原始')
            ax.hist(gray_c.ravel(), bins=128, range=(0,255), color=MATLAB_RED, alpha=0.45, label='组合流水线')
            ax.set_xlabel('像素值'); ax.set_ylabel('频数')
            ax.set_title(f'{title}灰度直方图', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8); ax.grid(alpha=0.2)

            # (1, c0+1): 差异图
            ax = fig1.add_subplot(gs1[1, c0+1]); ax.imshow(diff)
            ax.set_title(f'{title}差异图 |流水线-原始|x5', fontsize=12, fontweight='bold', color=MATLAB_RED)
            ax.axis('off')

            # (2, c0): Sobel 原始
            so_o = np.abs(cv2.Sobel(gray_o, cv2.CV_64F, 1, 1, ksize=3))
            ax = fig1.add_subplot(gs1[2, c0]); ax.imshow(so_o, cmap='hot')
            ax.set_title(f'{title}Sobel梯度 (原始)\n|G|均值={np.mean(so_o):.1f}', fontsize=11, fontweight='bold')
            ax.axis('off')

            # (2, c0+1): Sobel 组合流水线
            so_c = np.abs(cv2.Sobel(gray_c, cv2.CV_64F, 1, 1, ksize=3))
            ax = fig1.add_subplot(gs1[2, c0+1]); ax.imshow(so_c, cmap='hot')
            ax.set_title(f'{title}Sobel梯度 (组合流水线)\n|G|均值={np.mean(so_c):.1f}', fontsize=11, fontweight='bold', color=MATLAB_RED)
            ax.axis('off')

        out1 = os.path.join(output_dir, 'matlab_style_analysis_1_visual.png')
        plt.savefig(out1, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig1)

        # ================================================================
        # 图2: 量化指标对比 (3行x3列)
        # ================================================================
        fig2 = plt.figure(figsize=(16, 12))
        fig2.suptitle('眼底图像预处理 - 量化指标对比 (原始 vs 组合流水线)',
                     fontsize=16, fontweight='bold')
        gs2 = GridSpec(3, 3, figure=fig2, hspace=0.4, wspace=0.35)

        # (0, 0:2): 统计特征对比表
        ax = fig2.add_subplot(gs2[0, :2]); ax.axis('off')
        table_data = [
            ['指标', '左眼原始', '左眼流水线', '右眼原始', '右眼流水线'],
            ['均值 mu', f'{left_orig_s.gray_mean:.1f}', f'{left_pipeline_s.gray_mean:.1f}',
                       f'{right_orig_s.gray_mean:.1f}', f'{right_pipeline_s.gray_mean:.1f}'],
            ['标准差 sigma', f'{left_orig_s.gray_std:.1f}', f'{left_pipeline_s.gray_std:.1f}',
                            f'{right_orig_s.gray_std:.1f}', f'{right_pipeline_s.gray_std:.1f}'],
            ['偏度 gamma1', f'{left_orig_s.gray_skewness:+.3f}', f'{left_pipeline_s.gray_skewness:+.3f}',
                          f'{right_orig_s.gray_skewness:+.3f}', f'{right_pipeline_s.gray_skewness:+.3f}'],
            ['峰度 gamma2', f'{left_orig_s.gray_kurtosis:+.3f}', f'{left_pipeline_s.gray_kurtosis:+.3f}',
                          f'{right_orig_s.gray_kurtosis:+.3f}', f'{right_pipeline_s.gray_kurtosis:+.3f}'],
            ['信息熵 (bit)', f'{left_orig_s.gray_entropy:.2f}', f'{left_pipeline_s.gray_entropy:.2f}',
                           f'{right_orig_s.gray_entropy:.2f}', f'{right_pipeline_s.gray_entropy:.2f}'],
            ['RMS对比度', f'{left_orig_s.contrast_rms:.1f}', f'{left_pipeline_s.contrast_rms:.1f}',
                         f'{right_orig_s.contrast_rms:.1f}', f'{right_pipeline_s.contrast_rms:.1f}'],
            ['清晰度(Sobel)', f'{left_orig_s.sharpness:.1f}', f'{left_pipeline_s.sharpness:.1f}',
                            f'{right_orig_s.sharpness:.1f}', f'{right_pipeline_s.sharpness:.1f}'],
        ]
        tbl = ax.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.18, 0.14, 0.14, 0.14, 0.14])
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.4)
        for j in range(5): tbl[0,j].set_facecolor('#37474F'); tbl[0,j].set_text_props(color='white', fontweight='bold')
        for i in range(1, len(table_data)):
            tbl[i,1].set_facecolor('#F5F5F5'); tbl[i,3].set_facecolor('#F5F5F5')
            tbl[i,2].set_facecolor('#FFF3E0'); tbl[i,4].set_facecolor('#FFF3E0')
        ax.set_title('灰度统计特征对比', fontsize=13, fontweight='bold', color=MATLAB_BLUE)

        # (0, 2): FFT频段能量分布
        ax = fig2.add_subplot(gs2[0, 2])
        bands_labels = ['低频\n(背景)', '中频\n(纹理)', '高频\n(边缘)']
        orig_b = [left_orig_s.fft_energy_low*100, left_orig_s.fft_energy_mid*100, left_orig_s.fft_energy_high*100]
        pipeline_b = [left_pipeline_s.fft_energy_low*100, left_pipeline_s.fft_energy_mid*100, left_pipeline_s.fft_energy_high*100]
        x = np.arange(3); w = 0.3
        ax.bar(x-w/2, orig_b, w, label='原始', color='#9E9E9E', edgecolor='white')
        ax.bar(x+w/2, pipeline_b, w, label='组合流水线', color=MATLAB_RED, edgecolor='white')
        ax.set_xticks(x); ax.set_xticklabels(bands_labels, fontsize=9)
        ax.set_title('频段能量分布 (左眼)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.2)

        # (1, 0): RGB三通道直方图 (左眼)
        ax = fig2.add_subplot(gs2[1, 0])
        for ci, cc, cn in zip([0,1,2], ['#FF4444','#44AA44','#4444FF'], ['R','G','B']):
            ax.hist(left_img[:,:,ci].ravel(), bins=64, range=(0,255), color=cc, alpha=0.3, label=f'{cn}_原始')
            ax.hist(left_pipeline[:,:,ci].ravel(), bins=64, range=(0,255), color=cc, alpha=0.65, label=f'{cn}_流水线')
        ax.set_xlabel('像素值'); ax.set_ylabel('频数')
        ax.set_title('左眼RGB通道直方图对比', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.15)

        # (1, 1): RGB三通道直方图 (右眼)
        ax = fig2.add_subplot(gs2[1, 1])
        for ci, cc, cn in zip([0,1,2], ['#FF4444','#44AA44','#4444FF'], ['R','G','B']):
            ax.hist(right_img[:,:,ci].ravel(), bins=64, range=(0,255), color=cc, alpha=0.3, label=f'{cn}_原始')
            ax.hist(right_pipeline[:,:,ci].ravel(), bins=64, range=(0,255), color=cc, alpha=0.65, label=f'{cn}_流水线')
        ax.set_xlabel('像素值'); ax.set_ylabel('频数')
        ax.set_title('右眼RGB通道直方图对比', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.15)

        # (1, 2): FFT频谱对比
        ax = fig2.add_subplot(gs2[1, 2])
        fft_o = np.fft.fftshift(np.fft.fft2(gray_l))
        fft_c = np.fft.fftshift(np.fft.fft2(gray_lc))
        fft_ratio = np.clip(np.log(1+np.abs(fft_c))/(np.log(1+np.abs(fft_o))+1e-8), 0.5, 2.0)
        im = ax.imshow(fft_ratio, cmap='RdBu_r')
        ax.set_title('FFT幅度比 (流水线/原始)\n红=增强 蓝=减弱', fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='比值')

        # (2, 0:2): 组合流水线效果量化
        ax = fig2.add_subplot(gs2[2, :2])
        def eme_calc(img):
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
            bh, bw = g.shape[0]//8, g.shape[1]//8
            if bh<2: return 0.0
            s, n = 0.0, 0
            for i in range(8):
                for j in range(8):
                    b = g[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                    if b.min()>0: s+=20*np.log10(b.max()/b.min()); n+=1
            return s/max(n,1)
        eme_o, eme_c = eme_calc(left_img), eme_calc(left_pipeline)
        names = ['信息熵(bit)', 'RMS对比度', 'Sobel清晰度', 'EME增强度量']
        orig_v = [left_orig_s.gray_entropy, left_orig_s.contrast_rms, left_orig_s.sharpness, eme_o]
        pipe_v = [left_pipeline_s.gray_entropy, left_pipeline_s.contrast_rms, left_pipeline_s.sharpness, eme_c]
        x = np.arange(4); w = 0.3
        b1 = ax.bar(x-w/2, orig_v, w, label='原始', color='#9E9E9E', edgecolor='white')
        b2 = ax.bar(x+w/2, pipe_v, w, label='组合流水线', color=MATLAB_RED, edgecolor='white')
        for i, (o, c) in enumerate(zip(orig_v, pipe_v)):
            imp = (c-o)/max(o,1e-8)*100
            ax.text(i, max(o,c)*1.06, f'+{imp:.0f}%' if imp>0 else f'{imp:.0f}%',
                   ha='center', fontsize=9, fontweight='bold', color='#D84315')
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
        ax.set_title('左眼 组合流水线 效果量化 (改善率%%)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.2)

        # (2, 2): Sobel梯度直方图 (右眼)
        ax = fig2.add_subplot(gs2[2, 2])
        so_r = np.abs(cv2.Sobel(gray_r, cv2.CV_64F, 1, 1, ksize=3))
        so_rc = np.abs(cv2.Sobel(gray_rc, cv2.CV_64F, 1, 1, ksize=3))
        ax.hist(so_r.ravel(), bins=100, range=(0,200), color='#9E9E9E', alpha=0.6, label=f'原始 (|G|={np.mean(so_r):.0f})')
        ax.hist(so_rc.ravel(), bins=100, range=(0,200), color=MATLAB_RED, alpha=0.5, label=f'组合流水线 (|G|={np.mean(so_rc):.0f})')
        ax.set_xlabel('梯度幅值 |G|'); ax.set_ylabel('频数')
        ax.set_title('右眼Sobel梯度分布对比', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.15)

        out2 = os.path.join(output_dir, 'matlab_style_analysis_2_metrics.png')
        plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig2)

        return out1

    def quick_analysis_plot(left_img: np.ndarray, right_img: np.ndarray,
                            output_path: str) -> str:
        """
        生成简化的 MATLAB 风格分析图 (适合 GUI 展示, 2x3 布局)。
        所有标签使用中文。
        """
        left_stats  = FundusImageAnalyzer.compute_all_stats(left_img)
        right_stats = FundusImageAnalyzer.compute_all_stats(right_img)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('眼底图像量化分析 (MATLAB 风格)', fontsize=16, fontweight='bold')

        # (0,0): 左眼原图
        axes[0,0].imshow(left_img)
        axes[0,0].set_title(f'左眼 — 原始图像\n'
                           f'{left_stats.width}x{left_stats.height}')
        axes[0,0].axis('off')

        # (0,1): 右眼原图
        axes[0,1].imshow(right_img)
        axes[0,1].set_title(f'右眼 — 原始图像\n'
                           f'{right_stats.width}x{right_stats.height}')
        axes[0,1].axis('off')

        # (0,2): 统计对比表 (中文)
        axes[0,2].axis('off')
        table_data = [
            ['指标', '左眼', '右眼'],
            ['灰度均值', f'{left_stats.gray_mean:.1f}', f'{right_stats.gray_mean:.1f}'],
            ['灰度标准差', f'{left_stats.gray_std:.1f}', f'{right_stats.gray_std:.1f}'],
            ['偏度(γ1)', f'{left_stats.gray_skewness:+.3f}', f'{right_stats.gray_skewness:+.3f}'],
            ['峰度(γ2)', f'{left_stats.gray_kurtosis:+.3f}', f'{right_stats.gray_kurtosis:+.3f}'],
            ['信息熵(bit)', f'{left_stats.gray_entropy:.3f}', f'{right_stats.gray_entropy:.3f}'],
            ['RMS对比度', f'{left_stats.contrast_rms:.1f}', f'{right_stats.contrast_rms:.1f}'],
            ['清晰度(Sobel)', f'{left_stats.sharpness:.2f}', f'{right_stats.sharpness:.2f}'],
            ['血管密度(%)', f'{left_stats.vessel_density*100:.2f}', f'{right_stats.vessel_density*100:.2f}'],
        ]
        table = axes[0,2].table(cellText=table_data, loc='center',
                               cellLoc='center',
                               colWidths=[0.25, 0.18, 0.18])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.35)
        # 表头着色
        for j in range(3):
            table[0, j].set_facecolor('#0072BD')
            table[0, j].set_text_props(color='white', fontweight='bold')
        axes[0,2].set_title('统计指标对比', fontsize=12, fontweight='bold',
                           color=MATLAB_BLUE)

        # (1,0): 灰度直方图对比
        gray_left = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
        gray_right = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
        axes[1,0].hist(gray_left.ravel(), bins=128, range=(0,255),
                      color=MATLAB_BLUE, alpha=0.5, label='左眼',
                      edgecolor='white', linewidth=0.3)
        axes[1,0].hist(gray_right.ravel(), bins=128, range=(0,255),
                      color=MATLAB_RED, alpha=0.5, label='右眼',
                      edgecolor='white', linewidth=0.3)
        axes[1,0].set_xlabel('像素值')
        axes[1,0].set_ylabel('频数')
        axes[1,0].set_title('灰度直方图对比', fontweight='bold')
        axes[1,0].legend(fontsize=9)
        axes[1,0].grid(alpha=0.2)

        # (1,1): 图像指标对比 (条形图)
        metrics = ['血管\n密度', '对比度\nRMS', '清晰度\n(Sobel)', '信息熵\n(bit)']
        left_vals = [left_stats.vessel_density * 100,
                    left_stats.contrast_rms,
                    left_stats.sharpness,
                    left_stats.gray_entropy]
        right_vals = [right_stats.vessel_density * 100,
                     right_stats.contrast_rms,
                     right_stats.sharpness,
                     right_stats.gray_entropy]
        # 归一化到 [0,1]
        max_vals = [max(l,r)*1.15 for l,r in zip(left_vals, right_vals)]
        left_norm = [l/m for l,m in zip(left_vals, max_vals)]
        right_norm = [r/m for r,m in zip(right_vals, max_vals)]

        x = np.arange(len(metrics)); w = 0.35
        b1 = axes[1,1].bar(x-w/2, left_norm, w, label='左眼',
                          color=MATLAB_BLUE, edgecolor='white')
        b2 = axes[1,1].bar(x+w/2, right_norm, w, label='右眼',
                          color=MATLAB_RED, edgecolor='white')
        # 数值标注
        for bar, val in zip(b1, left_vals):
            axes[1,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                          f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')
        for bar, val in zip(b2, right_vals):
            axes[1,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                          f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')
        axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(metrics, fontsize=9)
        axes[1,1].set_ylim(0, 1.25)
        axes[1,1].set_title('图像质量指标对比', fontweight='bold')
        axes[1,1].legend(fontsize=9)
        axes[1,1].grid(axis='y', alpha=0.2)

        # (1,2): RGB 三通道均值对比
        rgb_left = [left_stats.r_mean, left_stats.g_mean, left_stats.b_mean]
        rgb_right = [right_stats.r_mean, right_stats.g_mean, right_stats.b_mean]
        x = np.arange(3)
        w = 0.35
        axes[1,2].bar(x-w/2, rgb_left, w, label='左眼',
                     color=MATLAB_BLUE, edgecolor='white')
        axes[1,2].bar(x+w/2, rgb_right, w, label='右眼',
                     color=MATLAB_RED, edgecolor='white')
        axes[1,2].set_xticks(x); axes[1,2].set_xticklabels(['R通道','G通道','B通道'])
        axes[1,2].set_ylabel('平均像素值')
        axes[1,2].set_title('RGB三通道均值对比', fontweight='bold')
        axes[1,2].legend(fontsize=9)
        axes[1,2].grid(axis='y', alpha=0.2)

        plt.tight_layout(pad=2)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()

        return output_path


# ============================================================================
# 测试入口
# ============================================================================
if __name__ == '__main__':
    import sys

    img_dir = "data/all_images"
    test_files = ["0_left.jpg", "0_right.jpg"]

    imgs = []
    for f in test_files:
        path = os.path.join(img_dir, f)
        if os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
            print(f"Loaded: {f} ({img.shape})")
        else:
            print(f"Missing: {path}")
            sys.exit(1)

    # 完整分析
    output = MatlabStylePlotter.analyze_and_plot(
        imgs[0], imgs[1],
        left_path=test_files[0],
        right_path=test_files[1],
        output_dir="analysis_output"
    )
    print(f"\nFull analysis saved to: {output}")

    # 简化分析
    output2 = MatlabStylePlotter.quick_analysis_plot(
        imgs[0], imgs[1],
        output_path="analysis_output/gui_analysis.png"
    )
    print(f"GUI analysis saved to: {output2}")

    # 打印统计报告
    stats_left  = FundusImageAnalyzer.compute_all_stats(imgs[0])
    stats_right = FundusImageAnalyzer.compute_all_stats(imgs[1])
    report = FundusImageAnalyzer.format_report([stats_left, stats_right],
                                                test_files[0], test_files[1])
    print(report)

    print("\n[OK] MATLAB 风格量化分析模块测试完成!")
