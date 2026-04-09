import numpy as np
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from pysted import base, utils
import argparse

class RobustSTEDGenerator:
    """
    升级版 STED 数据生成器 (V2)
    特性：
    1. 基于物理的光子计数 (Photon Counts) - 符合泊松统计
    2. 包含光漂白效应 (Photobleaching) - STED 的关键特征
    3. 域随机化 (Domain Randomization) - 随机化采集参数，提升模型泛化能力
    """
    
    def __init__(self, output_dir="./sted_dataset_v2"):
        self.output_dir = output_dir
        self.setup_directories()
        self.setup_microscope()
        
    def setup_directories(self):
        subdirs = ["images", "labels", "visualization", "metadata"]
        for d in subdirs:
            os.makedirs(os.path.join(self.output_dir, d), exist_ok=True)
            
    def setup_microscope(self):
        print("初始化 STED 物理模型...")
        # ==================== 1. 荧光分子 (EGFP) ====================
        # 增加漂白速率 k1，使漂白效应在短时间内更明显，增加训练难度
        self.egfp = {
            "lambda_": 535e-9,
            "qy": 0.6,
            "sigma_abs": {488: 0.08e-21, 575: 0.02e-21},
            "sigma_ste": {575: 3.0e-22},
            "tau": 3e-09,
            "tau_vib": 1.0e-12,
            "tau_tri": 1.2e-6,
            "k1": 2.0e-15,  # 稍微调高漂白速率
            "b": 1.4,
            "triplet_dynamics_frac": 0,
        }
        
        # ==================== 2. 硬件参数 ====================
        self.pixelsize = 20e-9
        self.image_size = 64
        
        # 激光与探测器
        self.laser_ex = base.GaussianBeam(488e-9)
        self.laser_sted = base.DonutBeam(575e-9, zero_residual=0, rate=40e6, tau=400e-12)
        
        # 探测器：noise=True 开启散粒噪声
        # background=0.5 模拟探测器暗电流（每秒计数）
        self.detector = base.Detector(noise=True, det_delay=750e-12, det_width=8e-9, background=0.5)
        
        self.objective = base.Objective()
        self.fluo = base.Fluorescence(**self.egfp)
        
        self.microscope = base.Microscope(
            self.laser_ex, self.laser_sted, self.detector, 
            self.objective, self.fluo, load_cache=True
        )
        
        # 缓存 PSF
        print("计算并缓存 PSF...")
        self.i_ex, self.i_sted, _ = self.microscope.cache(self.pixelsize, save_cache=True)

    def get_random_positions(self, n_emitters):
        """生成避免重叠的随机位置"""
        positions = []
        pad = 5 # 边缘留白
        
        for _ in range(n_emitters * 10): # 尝试次数
            if len(positions) >= n_emitters:
                break
            # 随机坐标
            pos = np.random.randint(pad, self.image_size - pad, size=2)
            
            # 检查距离
            if positions:
                dists = np.linalg.norm(np.array(positions) - pos, axis=1)
                if np.min(dists) < 2.0: # 最小间距 2 像素
                    continue
            positions.append(pos)
            
        return np.array(positions)

    def create_datamap(self, positions):
        """创建样本分布图"""
        dmap_arr = np.zeros((self.image_size, self.image_size), dtype=int)
        for pos in positions:
            # 每个点不仅仅是1，而是随机分子数 (1-3个)，模拟亮度不均
            dmap_arr[pos[1], pos[0]] = np.random.randint(1, 4)
            
        dmap = base.Datamap(dmap_arr, self.pixelsize)
        dmap.set_roi(self.i_ex, "max")
        return dmap

    def generate_heatmap(self, positions, sigma=1.0):
        """生成高斯热图作为 Label (Deep-STORM 标准)"""
        heatmap = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        y_grid, x_grid = np.ogrid[:self.image_size, :self.image_size]
        
        for pos in positions:
            # x, y 注意顺序
            dist_sq = (x_grid - pos[0])**2 + (y_grid - pos[1])**2
            # 生成高斯斑
            gauss = np.exp(-dist_sq / (2 * sigma**2))
            # 取最大值叠加（避免重叠处数值过大）或累加均可，这里用最大值保持峰值为1
            heatmap = np.maximum(heatmap, gauss)
            
        return heatmap

    def generate_sample(self, idx):
        # 1. 随机化物理参数 (Domain Randomization)
        # STED 功率范围：从低分辨到高分辨变化，让网络适应不同分辨率
        p_sted = np.random.uniform(0.0, 150e-3) 
        # 激发光功率
        p_ex = np.random.uniform(10e-6, 100e-6)
        # 像素停留时间：影响信噪比
        pdt = np.random.uniform(20e-6, 100e-6)
        
        # 2. 随机分子生成
        n_emitters = np.random.randint(1, 10) # 1到9个分子
        positions = self.get_random_positions(n_emitters)
        datamap = self.create_datamap(positions)
        
        # 3. 仿真采集 (开启 Bleaching!)
        # bleach=True 是关键，模拟扫描过程中的荧光淬灭
        acq, _, _ = self.microscope.get_signal_and_bleach(
            datamap, self.pixelsize, pdt, p_ex, p_sted,
            bleach=True, update=False
        )
        
        # 4. 后处理：添加额外的背景杂散光 (Stray light)
        # 真实的实验图像底噪通常不为0
        background_counts = np.random.poisson(np.random.uniform(1, 5), size=acq.shape)
        noisy_image = acq + background_counts
        
        # 5. 生成标签
        label = self.generate_heatmap(positions, sigma=1.0)
        
        return {
            "image": noisy_image.astype(np.int32), # 原始光子数
            "label": label.astype(np.float32),     # 0-1 热图
            "positions": positions,
            "params": {"p_sted": p_sted, "p_ex": p_ex, "pdt": pdt}
        }

    def run(self, num_samples=1000):
        print(f"开始生成 {num_samples} 个样本...")
        metadata = []
        
        for i in tqdm(range(num_samples)):
            sample = self.generate_sample(i)
            
            # 保存
            img_name = f"img_{i:05d}.npy"
            lbl_name = f"lbl_{i:05d}.npy"
            
            np.save(os.path.join(self.output_dir, "images", img_name), sample["image"])
            np.save(os.path.join(self.output_dir, "labels", lbl_name), sample["label"])
            
            metadata.append({
                "id": i,
                "n_emitters": len(sample["positions"]),
                "params": sample["params"]
            })
            
            # 可视化前 5 张
            if i < 5:
                self.visualize(sample, i)
        
        # 保存元数据
        with open(os.path.join(self.output_dir, "metadata", "meta.json"), "w") as f:
            json.dump(metadata, f)
            
        print("生成完毕。")

    def visualize(self, sample, idx):
        plt.figure(figsize=(12, 4))
        
        # 原始 STED 图像 (Raw Counts)
        plt.subplot(1, 3, 1)
        im = plt.imshow(sample["image"], cmap="magma")
        plt.colorbar(im, label="Photon Counts")
        plt.title(f"STED Input (P_sted={sample['params']['p_sted']*1e3:.1f}mW)")
        
        # 标签 (Heatmap)
        plt.subplot(1, 3, 2)
        plt.imshow(sample["label"], cmap="inferno")
        plt.title(f"Ground Truth ({len(sample['positions'])} emitters)")
        
        # 叠加
        plt.subplot(1, 3, 3)
        plt.imshow(sample["image"], cmap="gray")
        # 标出真实位置
        pos = sample["positions"]
        if len(pos) > 0:
            plt.plot(pos[:, 0], pos[:, 1], 'gx', markersize=8)
        plt.title("Overlay")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "visualization", f"vis_{idx}.png"))
        plt.close()

if __name__ == "__main__":
    # 实例化并运行
    gen = RobustSTEDGenerator()
    gen.run(num_samples=2000) # 本科训练建议生成 1000-2000 张