import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt

# ================= 1. 增强版数据集 (加入数据增强) =================
class STEDDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.npy")))
        self.label_paths = sorted(glob.glob(os.path.join(data_dir, "labels", "*.npy")))
        self.augment = augment # 是否开启数据增强
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"错误：在 {data_dir} 下没找到数据！")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 1. 读取数据
        img = np.load(self.image_paths[idx]).astype(np.float32)
        label = np.load(self.label_paths[idx]).astype(np.float32)
        
        # 2. 归一化 (Input 0-1)
        if img.max() > 0:
            img = img / img.max()
            
        # 3. 数据增强 (Data Augmentation) - 关键提分点
        # 只有在 augment=True 时才执行 (通常训练集开启，测试集关闭)
        if self.augment:
            # 50% 概率水平翻转
            if random.random() > 0.5:
                img = np.flip(img, axis=1).copy()
                label = np.flip(label, axis=1).copy()
            # 50% 概率垂直翻转
            if random.random() > 0.5:
                img = np.flip(img, axis=0).copy()
                label = np.flip(label, axis=0).copy()
        
        # 4. 转 Tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        label_tensor = torch.from_numpy(label).unsqueeze(0)
        
        return img_tensor, label_tensor

# ================= 2. 升级版网络 (加入 BN 层) =================
class BetterUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 定义基础卷积块：Conv -> BN -> ReLU
        # BN 层能显著提升收敛速度和对噪声的抵抗力
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
            
        # 编码器 (Encoder)
        self.enc1 = conv_block(1, 32)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        
        # 解码器 (Decoder)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = conv_block(64+32, 32) # 64来自up, 32来自enc1的skip connection
        
        # 输出层
        self.final = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        
        u1 = self.up(e2)
        # Skip Connection (拼接特征)
        cat = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat)
        out = self.final(d1)
        return out

# ================= 3. 可视化工具 =================
def plot_loss_curve(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    print("Loss 曲线已保存为 loss_curve.png")

def visualize_prediction(model, dataset, device, filename="result_final.png"):
    model.eval()
    # 随机找一张有点东西的图 (防止随机到全黑的图看不出效果)
    max_attempts = 10
    for _ in range(max_attempts):
        idx = np.random.randint(0, len(dataset))
        img_tensor, label_tensor = dataset[idx]
        if label_tensor.max() > 0.1: # 确保这张图里有分子
            break
            
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(input_tensor)
        
    # 归一化显示以便观察 (防止预测值过小导致全黑)
    img_np = img_tensor.squeeze().numpy()
    label_np = label_tensor.squeeze().numpy()
    pred_np = pred.cpu().squeeze().numpy()
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input (Noisy STED)")
    plt.imshow(img_np, cmap='magma')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Prediction (AI Output)")
    plt.imshow(pred_np, cmap='inferno')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Ground Truth")
    plt.imshow(label_np, cmap='inferno')
    plt.axis('off')
    
    plt.savefig(filename)
    plt.close()
    print(f"预测对比图已保存为 {filename}")

# ================= 4. 主训练流程 =================
def train():
    # 配置参数
    BATCH_SIZE = 16
    EPOCHS = 30          # 增加训练轮数
    LR = 1e-4            # 学习率
    DATA_DIR = "./sted_dataset_v2"
    MODEL_PATH = "best_sted_model.pth" # 之前保存的最好模型

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device} 进行训练...")

    # 准备数据 (开启增强)
    dataset = STEDDataset(DATA_DIR, augment=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 初始化模型
    model = BetterUNet().to(device)

    # ================= 核心修改：加载预训练权重 =================
    if os.path.exists(MODEL_PATH):
        print(f"发现预训练模型: {MODEL_PATH}，正在加载...")
        # 加载权重
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print("成功加载！将在该模型基础上继续优化 (Fine-tuning)。")
    else:
        print("未找到预训练模型，将从头开始训练 (Scratch)。")
    

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # 记录最佳模型
    best_loss = float('inf')
    loss_history = []
    
    print(f"开始训练 (共 {EPOCHS} 轮)...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)
        
        # 打印日志
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f}")
        
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_sted_model.pth")
            # print("  -> 发现更优模型，已保存。")
            
    print("-" * 30)
    print(f"训练完成！最佳 Loss: {best_loss:.6f}")
    print("模型已保存为 best_sted_model.pth")
    
    # 结果可视化
    plot_loss_curve(loss_history)
    
    # 加载最佳模型进行最终测试
    model.load_state_dict(torch.load("best_sted_model.pth",  map_location=device, weights_only=True))
    visualize_prediction(model, dataset, device, filename="final_prediction_check.png")

if __name__ == "__main__":
    if not os.path.exists("./sted_dataset_v2"):
        print("错误：请先运行数据生成代码！")
    else:
        train()