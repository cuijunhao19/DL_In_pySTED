import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from torch.utils.data import Dataset, DataLoader

# ================= 1. 必须复制模型结构 =================
# 注意：这里的模型定义必须和训练时完全一致，否则权重加载会报错
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

# ================= 2. 预测函数 =================
def predict_single_image(model, img_path, device):
    # 读取并预处理
    img = np.load(img_path).astype(np.float32)
    original_img = img.copy() # 备份一份用于画图
    
    # 归一化 (必须与训练时一致！)
    if img.max() > 0:
        img = img / img.max()
    
    # 转 Tensor: (H, W) -> (1, 1, H, W)
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    
    # 推理
    model.eval() # 切换到评估模式 (关闭 Dropout, 锁定 BN)
    with torch.no_grad():
        pred = model(img_tensor)
    
    # 取回结果
    pred_np = pred.squeeze().cpu().numpy()
    return original_img, pred_np

def main():
    # 配置
    MODEL_PATH = "best_sted_model.pth"
    DATA_DIR = "./sted_dataset_v2/images" # 测试图片来源
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    # 1. 加载模型
    model = BetterUNet().to(device)
    if os.path.exists(MODEL_PATH):
        # weights_only=True 消除警告
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print("模型加载成功！")
    else:
        print("错误：找不到模型文件！")
        return

    # 2. 随机选一张图进行测试
    all_images = glob.glob(os.path.join(DATA_DIR, "*.npy"))
    if not all_images:
        print("找不到测试图片")
        return
        
    # 这里随机选一张，你也可以指定具体的文件名
    test_img_path = np.random.choice(all_images)
    print(f"正在测试图片: {test_img_path}")
    
    # 3. 执行预测
    input_img, pred_img = predict_single_image(model, test_img_path, device)
    
    # 4. 找对应的 Label (为了对比)
    # 假设文件名是 image_00123.npy -> label_00123.npy
    filename = os.path.basename(test_img_path)
    label_filename = filename.replace("img_", "lbl_").replace("image_", "label_") # 兼容你的命名习惯
    label_path = os.path.join("./sted_dataset_v2/labels", label_filename)
    
    has_label = False
    if os.path.exists(label_path):
        label_img = np.load(label_path)
        has_label = True
    
    # 5. 可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(input_img, cmap='magma')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title("AI Prediction")
    plt.imshow(pred_img, cmap='inferno')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    if has_label:
        plt.title("Ground Truth")
        plt.imshow(label_img, cmap='inferno')
    else:
        plt.title("No Ground Truth")
        plt.text(0.5, 0.5, "Real Data", ha='center')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show() # 直接弹窗显示，或者用 plt.savefig 保存

if __name__ == "__main__":
    main()