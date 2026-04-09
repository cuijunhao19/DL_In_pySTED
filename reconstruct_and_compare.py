import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.ndimage import median_filter
from skimage import restoration

# ================= 1. 网络结构 (必须与训练时一致) =================
class BetterUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(1, 32)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = conv_block(64+32, 32)
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        u1 = self.up(e2)
        cat = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat)
        out = self.final(d1)
        return out

# ================= 2. 图像处理与重建逻辑 =================
def traditional_deconvolution(img):
    """
    传统方法：Richardson-Lucy (RL) 去卷积
    这是光学显微镜最常用的非AI分辨率提升算法
    """
    # 构造一个简单的STED点扩散函数(PSF)用于去卷积
    psf_size = 11
    x = np.arange(-psf_size//2 + 1., psf_size//2 + 1.)
    X, Y = np.meshgrid(x, x)
    sigma_psf = 1.2  # 假设的PSF宽度
    psf = np.exp(-(X**2 + Y**2) / (2 * sigma_psf**2))
    psf /= psf.sum()
    
    # 归一化输入并执行RL去卷积 (迭代15次)
    img_norm = img / (img.max() + 1e-8)
    deconvolved = restoration.richardson_lucy(img_norm, psf, num_iter=15)
    return deconvolved

def reconstruct_with_background(raw_img, ai_pred):
    """
    AI图像重建：将AI的稀疏定位图与原始环境背景融合
    """
    # 1. 提取原始背景：使用中值滤波抹平分子亮点和高频散粒噪声，保留低频环境光
    background = median_filter(raw_img, size=5)
    # 归一化背景
    if background.max() > 0:
        background = background / background.max()
        
    # 2. 对齐AI预测图的动态范围
    if ai_pred.max() > 0:
        ai_pred = ai_pred / ai_pred.max()
        
    # 3. 图像融合：突出分子(主导) + 保留背景(辅助)
    # 这里的 0.3 是背景保留权重，你可以根据视觉效果自行微调 (0.1 ~ 0.5)
    alpha_bg = 0.3 
    reconstructed = ai_pred + alpha_bg * background
    
    return reconstructed

# ================= 3. 主程序 =================
def main():
    MODEL_PATH = "best_sted_model.pth"
    DATA_DIR = "./sted_dataset_v2/images"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在加载模型...")
    
    model = BetterUNet().to(device)
    if os.path.exists(MODEL_PATH):
        # 消除安全警告
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
    else:
        print("未找到模型文件！请确保 best_sted_model.pth 存在。")
        return

    # 随机选择一张测试图
    all_images = glob.glob(os.path.join(DATA_DIR, "*.npy"))
    test_img_path = np.random.choice(all_images)
    print(f"处理图像: {os.path.basename(test_img_path)}")
    
    # 读取图像
    raw_img = np.load(test_img_path).astype(np.float32)
    
    # --- A. 获取 AI 预测 ---
    # 预处理
    img_tensor = raw_img.copy()
    if img_tensor.max() > 0:
        img_tensor = img_tensor / img_tensor.max()
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        ai_pred = model(img_tensor).squeeze().cpu().numpy()
        
    # --- B. 获取传统去卷积结果 ---
    trad_deconv = traditional_deconvolution(raw_img)
    
    # --- C. 构建带背景的最终高分辨图 ---
    ai_reconstructed = reconstruct_with_background(raw_img, ai_pred)

    # ================= 4. 可视化绘图 (2x2 对比图) =================
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.2) # 调整子图间距
    
    # 统一下色彩映射
    cmap = 'inferno'
    
    # 1. 原始 STED 输入
    axes[0, 0].imshow(raw_img, cmap=cmap)
    axes[0, 0].set_title("1. Raw STED Input", fontsize=14)
    axes[0, 0].axis('off')
    
    # 2. 传统方法 (RL 去卷积)
    axes[0, 1].imshow(trad_deconv, cmap=cmap)
    axes[0, 1].set_title("2. Traditional (RL Deconvolution)", fontsize=14)
    axes[0, 1].axis('off')
    
    # 3. AI 原始热图 (稀疏输出)
    axes[1, 0].imshow(ai_pred, cmap=cmap)
    axes[1, 0].set_title("3. AI Heatmap (Raw Output)", fontsize=14)
    axes[1, 0].axis('off')
    
    # 4. AI 重建的高分辨图像 (融合背景)
    axes[1, 1].imshow(ai_reconstructed, cmap=cmap)
    axes[1, 1].set_title("4. AI Reconstructed (Signal + Background)", fontsize=14, color='darkred')
    axes[1, 1].axis('off')
    
    # 添加大标题
    plt.suptitle("STED Image Reconstruction: Traditional vs. Deep Learning", fontsize=16, y=0.95)
    
    save_path = "comparison_reconstruction.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n对比图已保存为: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()