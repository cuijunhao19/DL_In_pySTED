import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# ================= 1. 数据集定义 =================
class STEDDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # 确保路径正确，兼容 Windows 路径分隔符
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.npy")))
        self.label_paths = sorted(glob.glob(os.path.join(data_dir, "labels", "*.npy")))
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"在 {data_dir} 下没找到数据，请检查路径！")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载 .npy
        img = np.load(self.image_paths[idx]).astype(np.float32)
        label = np.load(self.label_paths[idx]).astype(np.float32)
        
        # === 归一化 (关键) ===
        # 将输入图像缩放到 0-1 之间
        max_val = img.max()
        if max_val > 0:
            img = img / max_val
        
        # 转为 Tensor (增加 Channel 维度: H,W -> 1,H,W)
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        label_tensor = torch.from_numpy(label).unsqueeze(0)
        
        return img_tensor, label_tensor

# ================= 2. 网络架构 (Simple U-Net) =================
class BetterUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个基础的小模块：Conv -> BN -> ReLU
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),  # <--- 新加的魔法层
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),  # <--- 新加的魔法层
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

# ================= 3. 辅助功能：训练后可视化 =================
def visualize_result(model, dataset, device):
    """随机抽取一张图进行预测并画图"""
    model.eval() # 切换到评估模式
    idx = np.random.randint(0, len(dataset))
    img_tensor, label_tensor = dataset[idx]
    
    # 增加 Batch 维度 (1, C, H, W) 传给网络
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(input_tensor)
    
    # 转回 numpy
    img_np = img_tensor.squeeze().numpy()
    label_np = label_tensor.squeeze().numpy()
    pred_np = pred.cpu().squeeze().numpy()
    
    # 画图
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Input STED (Normalized)")
    plt.imshow(img_np, cmap='magma')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title("Prediction (Network Output)")
    plt.imshow(pred_np, cmap='inferno')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.title("Ground Truth (Target)")
    plt.imshow(label_np, cmap='inferno')
    plt.colorbar()
    
    save_path = "train_result_preview.png"
    plt.savefig(save_path)
    print(f"\n[完成] 预测预览图已保存为: {save_path}")
    plt.close()

# ================= 4. 主训练循环 =================
def train():
    # 自动检测设备 (没有显卡会自动用 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前运行设备: {device}")
    if device.type == 'cpu':
        print("提示: 使用 CPU 训练速度较慢，但对于 64x64 图像完全没问题。")
    
    # 路径设置
    data_path = "./sted_dataset_v2"
    if not os.path.exists(data_path):
        print(f"错误: 找不到文件夹 {data_path}")
        print("请确保你已经运行了数据生成脚本，并且文件夹在当前目录下。")
        return

    dataset = STEDDataset(data_path)
    # Windows下 num_workers 建议设为 0，否则可能报错
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    model = BetterUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    epochs = 20
    print(f"开始训练，共 {epochs} 轮，数据量: {len(dataset)}...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 打印进度
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    # 保存模型
    torch.save(model.state_dict(), "sted_model.pth")
    print("\n训练完成！模型已保存为 sted_model.pth")
    
    # 马上验证效果
    visualize_result(model, dataset, device)

if __name__ == "__main__":
    train()