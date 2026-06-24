"""
================================================================================
眼底图像预处理模块 — Fundus Image Preprocessing Module
================================================================================
专为眼底图像设计的数字图像处理预处理流水线。

方法列表:
  1. CLAHE (对比度受限自适应直方图均衡化) — 增强血管对比度
  2. Gaussian Filtering (高斯滤波) — 去噪平滑
  3. Unsharp Masking (反锐化掩膜) — 边缘/血管锐化
  4. Median Filtering (中值滤波) — 椒盐噪声去除
  5. Bilateral Filtering (双边滤波) — 保边平滑
  6. Gamma Correction (伽马校正) — 亮度调节
  7. Histogram Equalization (直方图均衡化) — 全局对比度增强
  8. Morphological Top-Hat (形态学顶帽变换) — 血管结构增强
  9. 组合流水线 — CLAHE + Gaussian + UnsharpMask

定量评估指标:
  - PSNR (峰值信噪比)
  - SSIM (结构相似性指数)
  - Contrast Improvement Index (对比度改善指数)
  - Entropy (信息熵)
  - Mean Gradient (平均梯度 / 清晰度)
  - EME (增强度量)

作者: 数字图像处理课程项目
================================================================================
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class PreprocessResult:
    """预处理结果数据类"""
    name: str                          # 方法名称
    image: np.ndarray                  # 处理后的图像 (RGB, uint8)
    params: Dict = field(default_factory=dict)  # 使用的参数
    metrics: Dict = field(default_factory=dict) # 定量指标

    def __repr__(self):
        return (f"PreprocessResult(name='{self.name}', "
                f"PSNR={self.metrics.get('psnr', 'N/A'):.2f}, "
                f"SSIM={self.metrics.get('ssim', 'N/A'):.4f})")


# ============================================================================
# 核心预处理方法
# ============================================================================

class FundusPreprocessor:
    """眼底图像预处理类 — 包含所有预处理方法"""

    def __init__(self):
        self.results: List[PreprocessResult] = []

    # ------------------------------------------------------------------
    # 1. CLAHE — 对比度受限自适应直方图均衡化
    # ------------------------------------------------------------------
    @staticmethod
    def clahe(image: np.ndarray, clip_limit: float = 2.0,
              tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        对眼底图像应用 CLAHE 增强。

        原理: 将图像分为小块，在每个块内做直方图均衡化，
              同时限制对比度放大倍数以防止噪声过度放大。
              对眼底血管和病变区域的对比度增强效果显著。

        参数:
            image: RGB 图像 (H, W, 3)
            clip_limit: 对比度限制阈值 (越大对比度越强，推荐1.5-4.0)
            tile_grid_size: 分块大小 (越小局部性越强)

        返回:
            增强后的 RGB 图像
        """
        # 转换到 LAB 色彩空间 (仅对亮度通道 L 做 CLAHE)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE 应用于 L 通道
        clahe_obj = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )
        l_enhanced = clahe_obj.apply(l)

        # 合并并转回 RGB
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # ------------------------------------------------------------------
    # 2. Gaussian Filtering — 高斯滤波去噪
    # ------------------------------------------------------------------
    @staticmethod
    def gaussian_filter(image: np.ndarray, kernel_size: int = 5,
                        sigma: float = 1.0) -> np.ndarray:
        """
        高斯滤波平滑去噪。

        原理: 使用高斯核对图像进行卷积，权值随距离中心
              的距离呈高斯分布衰减，在去噪的同时较好地保留边缘。

        参数:
            kernel_size: 核大小 (必须为奇数)
            sigma: 标准差 (越大平滑越强)
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    # ------------------------------------------------------------------
    # 3. Unsharp Masking — 反锐化掩膜
    # ------------------------------------------------------------------
    @staticmethod
    def unsharp_mask(image: np.ndarray, kernel_size: int = 5,
                     amount: float = 1.5, threshold: int = 0) -> np.ndarray:
        """
        反锐化掩膜锐化。

        原理: 原始图像 + amount * (原始图像 - 高斯模糊图像)
              即增强图像中高频分量（边缘、血管等细节）。

        参数:
            kernel_size: 高斯模糊核大小
            amount: 锐化强度 (1.0-3.0)
            threshold: 差异阈值 (低于此值的差异不增强，抑制噪声)
        """
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        if threshold > 0:
            # 带阈值的锐化: 仅增强差异大于阈值的像素
            diff = cv2.subtract(image, blurred)
            mask = cv2.absdiff(image, blurred) > threshold
            sharpened = image.copy()
            sharpened[mask] = cv2.addWeighted(
                image, 1.0 + amount, blurred, -amount, 0
            )[mask]
            return sharpened
        else:
            return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    # ------------------------------------------------------------------
    # 4. Median Filtering — 中值滤波
    # ------------------------------------------------------------------
    @staticmethod
    def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        中值滤波。

        原理: 用邻域像素的中值替代中心像素值。
              对椒盐噪声效果极佳，同时能保持边缘。

        参数:
            kernel_size: 核大小 (必须为奇数)
        """
        return cv2.medianBlur(image, kernel_size)

    # ------------------------------------------------------------------
    # 5. Bilateral Filtering — 双边滤波
    # ------------------------------------------------------------------
    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int = 9,
                         sigma_color: float = 75,
                         sigma_space: float = 75) -> np.ndarray:
        """
        双边滤波 — 保边去噪。

        原理: 同时考虑像素的空间距离和颜色相似度。
              空间上远的权重小，颜色差异大的权重也小。
              因此在平滑噪声的同时能保持血管边缘。

        参数:
            d: 滤波直径
            sigma_color: 颜色空间标准差 (越大，颜色差异大的像素也参与平均)
            sigma_space: 坐标空间标准差 (越大，更远的像素也参与平均)
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    # ------------------------------------------------------------------
    # 6. Gamma Correction — 伽马校正
    # ------------------------------------------------------------------
    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """
        伽马校正用于亮度调节。

        原理: I_out = I_in^gamma。
              gamma < 1: 增亮 (提升暗部细节)
              gamma > 1: 变暗 (抑制过曝区域)
              眼底图像通常偏暗，gamma=0.8-1.0 可提亮暗区。

        参数:
            gamma: 伽马值
        """
        inv_gamma = 1.0 / gamma
        table = np.array([
            (i / 255.0) ** inv_gamma * 255 for i in range(256)
        ]).astype(np.uint8)
        return cv2.LUT(image, table)

    # ------------------------------------------------------------------
    # 7. Histogram Equalization — 全局直方图均衡化
    # ------------------------------------------------------------------
    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """
        全局直方图均衡化。

        原理: 将图像的直方图拉伸到整个灰度范围，
              增强全局对比度。但可能过度放大噪声。
              通常 CLAHE (局部均衡化) 效果更好。
        """
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.equalizeHist(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # ------------------------------------------------------------------
    # 8. Morphological Top-Hat — 形态学顶帽变换 (血管增强)
    # ------------------------------------------------------------------
    @staticmethod
    def morphological_tophat(image: np.ndarray,
                             kernel_size: int = 15) -> np.ndarray:
        """
        形态学顶帽变换用于血管结构增强。

        原理: TopHat = 原图 - 开运算(原图)。
              开运算(先腐蚀后膨胀)会移除比结构元素小的亮结构，
              相减后得到这些细小亮结构 = 血管等细节增强。

        参数:
            kernel_size: 结构元素大小 (应大于血管宽度)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        # 顶帽变换
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # 叠加回原图 (增强血管)
        tophat_color = cv2.cvtColor(tophat, cv2.COLOR_GRAY2RGB)
        enhanced = cv2.addWeighted(image, 1.0, tophat_color, 0.5, 0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # 9. 组合预处理流水线 — CLAHE + 高斯滤波 + 反锐化掩膜
    # ------------------------------------------------------------------
    @classmethod
    def pipeline_full(cls, image: np.ndarray,
                      clahe_clip: float = 2.0,
                      gaussian_kernel: int = 3,
                      unsharp_amount: float = 1.2) -> np.ndarray:
        """
        推荐的眼底图像完整预处理流水线:

        CLAHE → 高斯滤波 → 反锐化掩膜

        这三步的组合效果:
        1. CLAHE 增强局部对比度 (突出血管和病变)
        2. 轻度高斯滤波去除 CLAHE 可能放大的噪声
        3. 反锐化掩膜锐化血管边缘
        """
        # Step 1: CLAHE
        enhanced = cls.clahe(image, clip_limit=clahe_clip)

        # Step 2: 轻度高斯滤波
        smoothed = cls.gaussian_filter(enhanced, kernel_size=gaussian_kernel,
                                       sigma=0.8)

        # Step 3: 反锐化掩膜
        sharpened = cls.unsharp_mask(smoothed, kernel_size=5,
                                     amount=unsharp_amount, threshold=3)

        return sharpened


# ============================================================================
# 定量评估指标
# ============================================================================

class ImageQualityMetrics:
    """图像质量定量评估"""

    @staticmethod
    def psnr(original: np.ndarray, processed: np.ndarray) -> float:
        """峰值信噪比 (越高越好)"""
        mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
        if mse == 0:
            return 100.0
        return float(20 * np.log10(255.0 / np.sqrt(mse)))

    @staticmethod
    def ssim_score(original: np.ndarray, processed: np.ndarray) -> float:
        """结构相似性指数 (越接近1越好)"""
        return float(ssim(original, processed,
                         data_range=processed.max() - processed.min(),
                         channel_axis=2))

    @staticmethod
    def contrast_improvement_index(original: np.ndarray,
                                   processed: np.ndarray) -> float:
        """
        对比度改善指数 (CII)

        CII = C_processed / C_original
        其中 C = std / mean (局部对比度的变差系数)
        CII > 1 表示对比度提升
        """
        gray_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(float)
        gray_proc = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY).astype(float)

        c_orig = np.std(gray_orig) / (np.mean(gray_orig) + 1e-8)
        c_proc = np.std(gray_proc) / (np.mean(gray_proc) + 1e-8)

        return float(c_proc / (c_orig + 1e-8))

    @staticmethod
    def entropy(image: np.ndarray) -> float:
        """信息熵 (越高表示信息量越大)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # 避免 log(0)
        return float(-np.sum(hist * np.log2(hist)))

    @staticmethod
    def mean_gradient(image: np.ndarray) -> float:
        """
        平均梯度 (清晰度指标, 越高越清晰)

        用 Sobel 算子计算梯度幅值的均值
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        return float(np.mean(gradient_mag))

    @staticmethod
    def eme(image: np.ndarray, block_size: int = 8) -> float:
        """
        增强度量 (EME — Measure of Enhancement)

        将图像分块，每块计算 I_max/I_min 的对数平均。
        越高表示局部对比度越强。

        公式: EME = (1/(k1*k2)) * Σ 20*log10(I_max / I_min)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
        h, w = gray.shape

        # 调整分块
        bh = h // block_size
        bw = w // block_size

        if bh < 2 or bw < 2:
            return 0.0

        eme_sum = 0.0
        count = 0

        for i in range(block_size):
            for j in range(block_size):
                block = gray[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                i_max = block.max()
                i_min = block.min()
                if i_min > 0:
                    eme_sum += 20 * np.log10(i_max / i_min)
                    count += 1

        return float(eme_sum / max(count, 1))

    @classmethod
    def compute_all(cls, original: np.ndarray,
                    processed: np.ndarray) -> Dict[str, float]:
        """计算所有定量指标"""
        return {
            'psnr': cls.psnr(original, processed),
            'ssim': cls.ssim_score(original, processed),
            'cii': cls.contrast_improvement_index(original, processed),
            'entropy_orig': cls.entropy(original),
            'entropy_proc': cls.entropy(processed),
            'entropy_gain': cls.entropy(processed) - cls.entropy(original),
            'mean_gradient_orig': cls.mean_gradient(original),
            'mean_gradient_proc': cls.mean_gradient(processed),
            'gradient_gain': (cls.mean_gradient(processed) /
                            max(cls.mean_gradient(original), 1e-8)),
            'eme_orig': cls.eme(original),
            'eme_proc': cls.eme(processed),
            'eme_gain': (cls.eme(processed) /
                        max(cls.eme(original), 1e-8)),
        }


# ============================================================================
# 批量预处理 + 定量分析
# ============================================================================

class FundusAnalysisPipeline:
    """眼底图像预处理 + 定量分析流水线"""

    # 核心预处理方法 (精简为3种 — 覆盖增强/去噪/组合流水线)
    METHODS = {
        'CLAHE': lambda img: FundusPreprocessor.clahe(img, clip_limit=2.0),
        'Gaussian': lambda img: FundusPreprocessor.gaussian_filter(img, kernel_size=5),
        'FullPipeline': lambda img: FundusPreprocessor.pipeline_full(img),
    }

    @classmethod
    def analyze_single_image(cls, image: np.ndarray,
                             methods: Optional[List[str]] = None
                             ) -> List[PreprocessResult]:
        """
        对单张图像应用所有预处理方法并计算定量指标。

        参数:
            image: RGB 图像 (H, W, 3)
            methods: 要使用的方法列表 (None = 全部)

        返回:
            PreprocessResult 列表
        """
        if methods is None:
            methods = list(cls.METHODS.keys())

        results = []
        for name in methods:
            if name not in cls.METHODS:
                continue
            try:
                processed = cls.METHODS[name](image)
                metrics = ImageQualityMetrics.compute_all(image, processed)
                results.append(PreprocessResult(
                    name=name,
                    image=processed,
                    params={},
                    metrics=metrics,
                ))
            except Exception as e:
                print(f"  [警告] {name} 处理失败: {e}")

        return results

    @classmethod
    def analyze_batch(cls, image_paths: List[str],
                      methods: Optional[List[str]] = None,
                      max_images: int = 50
                      ) -> Dict[str, Dict[str, float]]:
        """
        批量分析多张图像，返回各方法的平均指标。

        返回:
            {method_name: {metric_name: avg_value}}
        """
        if methods is None:
            methods = list(cls.METHODS.keys())

        # 初始化累加器
        accum: Dict[str, Dict[str, List[float]]] = {
            m: {} for m in methods
        }

        n_processed = 0
        for path in image_paths[:max_images]:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = cls.analyze_single_image(img, methods)
            for r in results:
                for k, v in r.metrics.items():
                    if k not in accum[r.name]:
                        accum[r.name][k] = []
                    accum[r.name][k].append(v)
            n_processed += 1

        # 计算平均值
        avg_metrics: Dict[str, Dict[str, float]] = {}
        for method_name, metrics_dict in accum.items():
            avg_metrics[method_name] = {
                k: np.mean(v) for k, v in metrics_dict.items()
            }

        return avg_metrics


# ============================================================================
# 测试入口
# ============================================================================
if __name__ == '__main__':
    import sys
    import os

    # 测试: 找一张示例图像
    img_dir = "data/all_images"
    test_imgs = sorted(os.listdir(img_dir))[:2]

    for fname in test_imgs:
        path = os.path.join(img_dir, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"\n测试图像: {fname} ({img.shape})")

        # 测试 CLAHE
        result = FundusPreprocessor.clahe(img)
        metrics = ImageQualityMetrics.compute_all(img, result)
        print(f"  CLAHE → PSNR={metrics['psnr']:.2f}dB, "
              f"SSIM={metrics['ssim']:.4f}, CII={metrics['cii']:.3f}, "
              f"ΔEntropy={metrics['entropy_gain']:.4f}")

        # 测试完整流水线
        result2 = FundusPreprocessor.pipeline_full(img)
        metrics2 = ImageQualityMetrics.compute_all(img, result2)
        print(f"  Full  → PSNR={metrics2['psnr']:.2f}dB, "
              f"SSIM={metrics2['ssim']:.4f}, CII={metrics2['cii']:.3f}, "
              f"ΔEntropy={metrics2['entropy_gain']:.4f}")

    print("\n[OK] 眼底预处理模块测试完成!")
