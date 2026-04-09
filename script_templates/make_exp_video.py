import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
from tqdm import tqdm
import argparse
import time
import subprocess

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Video making script")
parser.add_argument("--pdt", type=float, default=1e-6, help="Pixel dwell time used for the experiment (in s)")
parser.add_argument("--save_path", type=str, default="", help="Path to the saved npy and ffconcant files")
parser.add_argument("--delete_after", type=str2bool, default=True, help="Wether or not the figures are deleted after")
args = parser.parse_args()

# ==================== 修复路径处理 ====================
# 使用绝对路径并正确处理Windows路径
files_path = os.path.abspath(args.save_path)
pdt = args.pdt
delete_figures_after = args.delete_after

print(f"工作目录: {files_path}")

# 修复目录创建
figures_dir = os.path.join(files_path, "figures")
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    print(f"创建图像目录: {figures_dir}")

# 检查必要文件
required_files = ["datamaps.npy", "confocals.npy", "steds.npy", "idx_type_dict.npy"]
for file in required_files:
    file_path = os.path.join(files_path, file)
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        exit(1)

# 读取数据
try:
    datamaps = np.load(os.path.join(files_path, "datamaps.npy"))
    confocals = np.load(os.path.join(files_path, "confocals.npy"))
    steds = np.load(os.path.join(files_path, "steds.npy"))
    idx_type = np.load(os.path.join(files_path, "idx_type_dict.npy"), allow_pickle=True).item()
    print("数据加载成功!")
except Exception as e:
    print(f"加载数据时出错: {e}")
    exit(1)

min_datamap, max_datamap = np.min(datamaps), np.max(datamaps)
min_confocal, max_confocal = np.min(confocals), np.max(confocals)
min_sted, max_sted = np.min(steds), np.max(steds)

d_idx, c_idx, s_idx = 0, 0, 0

# 生成第一帧
fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)

dmap_imshow = axes[0].imshow(datamaps[d_idx], vmin=min_datamap, vmax=max_datamap)
axes[0].set_title(f"Datamap")
fig.colorbar(dmap_imshow, ax=axes[0], fraction=0.04, pad=0.05)

confocal_imshow = axes[1].imshow(confocals[c_idx], vmin=min_confocal, vmax=max_confocal)
axes[1].set_title(f"Confocal (Ground truth) \n 1/3 the resolution of STED")
fig.colorbar(confocal_imshow, ax=axes[1], fraction=0.04, pad=0.05)

sted_imshow = axes[2].imshow(steds[s_idx], vmin=min_sted, vmax=max_sted)
axes[2].set_title(f"STED acquisition")
fig.colorbar(sted_imshow, ax=axes[2], fraction=0.04, pad=0.05)

fig.suptitle(f"Acquisition will start in 5 seconds")
plt.savefig(os.path.join(figures_dir, "0.png"))
plt.close()

# 生成所有帧
keys_list = sorted(idx_type.keys())
print(f"生成 {len(keys_list)} 帧图像...")

for idx, key in enumerate(tqdm(keys_list)):
    if idx_type[key] == "datamap":
        d_idx += 1
    elif idx_type[key] == "confocal":
        c_idx += 1
    elif idx_type[key] == "sted":
        s_idx += 1
    else:
        print(f"未知类型: {idx_type[key]}")

    if key != keys_list[-1]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)

        dmap_imshow = axes[0].imshow(datamaps[d_idx], vmin=min_datamap, vmax=max_datamap)
        axes[0].set_title(f"Datamap")
        fig.colorbar(dmap_imshow, ax=axes[0], fraction=0.04, pad=0.05)

        confocal_imshow = axes[1].imshow(confocals[c_idx], vmin=min_confocal, vmax=max_confocal)
        axes[1].set_title(f"Confocal (Ground truth) \n 1/3 the resolution of STED")
        fig.colorbar(confocal_imshow, ax=axes[1], fraction=0.04, pad=0.05)

        sted_imshow = axes[2].imshow(steds[s_idx], vmin=min_sted, vmax=max_sted)
        axes[2].set_title(f"STED acquisition")
        fig.colorbar(sted_imshow, ax=axes[2], fraction=0.04, pad=0.05)

        fig.suptitle(f"Acquisition in progress...")
        plt.savefig(os.path.join(figures_dir, f"{key + 1}.png"))
        plt.close()
    else:
        # 结束帧
        for i in range(2):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)

            dmap_imshow = axes[0].imshow(datamaps[d_idx], vmin=min_datamap, vmax=max_datamap)
            axes[0].set_title(f"Datamap")
            fig.colorbar(dmap_imshow, ax=axes[0], fraction=0.04, pad=0.05)

            confocal_imshow = axes[1].imshow(confocals[c_idx], vmin=min_confocal, vmax=max_confocal)
            axes[1].set_title(f"Confocal (Ground truth) \n 1/3 the resolution of STED")
            fig.colorbar(confocal_imshow, ax=axes[1], fraction=0.04, pad=0.05)

            sted_imshow = axes[2].imshow(steds[s_idx], vmin=min_sted, vmax=max_sted)
            axes[2].set_title(f"STED acquisition")
            fig.colorbar(sted_imshow, ax=axes[2], fraction=0.04, pad=0.05)

            fig.suptitle(f"Acquisition has ended")
            plt.savefig(os.path.join(figures_dir, f"{key + 1}.png"))
            plt.close()

# 计算总时长
total_duration = 5 + (keys_list[-1] * pdt * 10) + 10

# ==================== 修复FFmpeg调用 ====================
try:
    # 复制ffconcat文件
    concat_source = os.path.join(files_path, "in.ffconcat")
    concat_dest = os.path.join(figures_dir, "in.ffconcat")
    if os.path.exists(concat_source):
        shutil.copy(concat_source, concat_dest)
    else:
        print(f"警告: 找不到 {concat_source}")
        # 如果没有ffconcat文件，创建一个简单的
        with open(concat_dest, 'w') as f:
            f.write("ffconcat version 1.0\n")
            for i in range(len(keys_list) + 2):  # +2 为了开始和结束帧
                f.write(f"file {i}.png\n")
                if i == 0:
                    f.write("duration 5\n")
                elif i <= len(keys_list):
                    f.write(f"duration {pdt * 10}\n")
                else:
                    f.write("duration 10\n")
    
    print("使用FFmpeg生成视频...")
    
    # 第一步：生成初始视频
    cmd1 = [
        'ffmpeg', '-y',
        '-i', 'in.ffconcat',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',  # 添加这个以提高兼容性
        '-vf', 'fps=25',
        'out.avi'
    ]
    
    # 在figures目录中运行第一个命令
    result1 = subprocess.run(cmd1, cwd=figures_dir, capture_output=True, text=True)
    if result1.returncode != 0:
        print(f"第一步FFmpeg错误: {result1.stderr}")
    else:
        print("第一步视频生成成功")
    
    # 第二步：裁剪视频
    cmd2 = [
        'ffmpeg', '-y',
        '-ss', '0',
        '-i', 'out.avi',
        '-t', str(total_duration),
        '-c', 'copy',
        'experiment_video.avi'
    ]
    
    # 在figures目录中运行第二个命令
    result2 = subprocess.run(cmd2, cwd=figures_dir, capture_output=True, text=True)
    if result2.returncode != 0:
        print(f"第二步FFmpeg错误: {result2.stderr}")
    else:
        print("第二步视频裁剪成功")
    
    # 移动最终视频到实验目录
    final_video_src = os.path.join(figures_dir, "experiment_video.avi")
    final_video_dest = os.path.join(files_path, "experiment_video.avi")
    if os.path.exists(final_video_src):
        shutil.move(final_video_src, final_video_dest)
        print(f"最终视频已保存: {final_video_dest}")
    else:
        print("警告: 找不到生成的视频文件")
        
except Exception as e:
    print(f"视频生成过程中出错: {e}")

# 清理临时文件
try:
    temp_files = ["in.ffconcat", "out.avi"]
    for temp_file in temp_files:
        temp_path = os.path.join(figures_dir, temp_file)
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"已删除临时文件: {temp_file}")
except Exception as e:
    print(f"清理临时文件时出错: {e}")

# 可选：删除图像文件
if delete_figures_after:
    try:
        # 先删除所有PNG文件
        png_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        for png_file in png_files:
            os.remove(os.path.join(figures_dir, png_file))
            print(f"已删除图像文件: {png_file}")
        
        # 然后删除空目录
        if not os.listdir(figures_dir):
            os.rmdir(figures_dir)
            print(f"已删除空目录: {figures_dir}")
    except Exception as e:
        print(f"删除图像文件时出错: {e}")

print("处理完成!")