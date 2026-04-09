import numpy as np
import tqdm
from pysted import base, utils
import os
import argparse
from matplotlib import pyplot as plt

"""
通用STED实验采集脚本
这个脚本实现了完整的STED实验流程，包括时间动态样本、交替采集和漂白效应
支持命令行参数配置，可用于批量数据生成
"""

def str2bool(v):
    """命令行参数布尔值转换"""
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ==================== 命令行参数解析 ====================
# add arg parser handling
parser = argparse.ArgumentParser(description="Example of experiment script")
parser.add_argument("--save_path", type=str, default="", help="Where to save the files")
parser.add_argument("--bleach", type=str2bool, default=False, help="Whether or not bleaching is on or not")
parser.add_argument("--dmap_seed", type=int, default=None, help="Whether or not the dmap is created using a seed")
parser.add_argument("--flash_seed", type=int, default=None, help="Whether or not the flashes are controlled by a seed")
parser.add_argument("--acq_time", type=int, default=1, help="Acquisition time (in seconds)")
args = parser.parse_args()

# ==================== 初始化设置 ====================
save_path = args.save_path
if not os.path.exists(save_path):
    os.mkdir(save_path)

print("Setting up the datamap and its flashes ...")
# Get light curves stuff to generate the flashes later
# event_file_path = "flash_files/stream1_events.txt"
# video_file_path = "flash_files/stream1.tif"
curves_path = "flash_files/events_curves.npy"  # 荧光闪烁曲线数据路径

# ==================== 样本生成 ====================
# Generate a datamap
frame_shape = (64, 64)       # 图像尺寸
# 生成突触纤维：框架形状、突触数量范围、分支数量范围、纤维长度范围
ensemble_func, synapses_list = utils.generate_synaptic_fibers(frame_shape, (9, 55), (3, 10), (2, 5),
                                                              seed=args.dmap_seed)

# 展平突触列表便于处理
# Build a dictionnary corresponding synapses to a bool saying if they are currently flashing or not
# They all start not flashing
flat_synapses_list = [item for sublist in synapses_list for item in sublist]

poils_frame = ensemble_func.return_frame().astype(int)

# plt.imshow(poils_frame)
# plt.show()
# exit()

print("Setting up the microscope ...")

# ==================== 显微镜参数配置 ====================
# Microscope stuff

# 荧光分子参数 (与教程略有不同)
egfp = {"lambda_": 535e-9,
        "qy": 0.6,
        "sigma_abs": {488: 1.15e-20,
                      575: 6e-21},
        "sigma_ste": {560: 1.2e-20,
                      575: 6.0e-21,
                      580: 5.0e-21},
        "sigma_tri": 1e-21,
        "tau": 3e-09,
        "tau_vib": 1.0e-12,
        "tau_tri": 5e-6,
        "phy_react": {488: 1e-8,   # 1e-4
                      575: 1e-12},   # 1e-8
        "k_isc": 0.26e6}
# 采集参数
pixelsize = 10e-9           # STED像素尺寸 10nm
confoc_pxsize = 30e-9       # 共聚焦像素尺寸 30nm (分辨率较低)
dpxsz = 10e-9               # 数据图像素尺寸
bleach = args.bleach        # 是否启用漂白
p_ex = np.ones(frame_shape) * 1e-6  # 激发功率分布
p_sted = 30e-3              # STED功率
min_pdt = 1e-6              # 最小像素停留时间 1μs
pdt = np.ones(frame_shape) * 10e-6  # 像素停留时间分布
roi = 'max'                 # 感兴趣区域设置
acquisition_time = args.acq_time  # 采集总时间
flash_prob = 0.05           # 闪烁概率 - 每个时间步突触有5%概率开始闪烁
flash_seed = args.flash_seed

# ==================== 显微镜对象创建 ===================
# Generating objects necessary for acquisition simulation
laser_ex = base.GaussianBeam(488e-9)
laser_sted = base.DonutBeam(575e-9, zero_residual=0)
detector = base.Detector(noise=True, background=0)
objective = base.Objective()
fluo = base.Fluorescence(**egfp)

# 创建时间动态数据图
temporal_datamap = base.TemporalDatamap(poils_frame, dpxsz, flat_synapses_list)
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)
i_ex, _, _ = microscope.cache(temporal_datamap.pixelsize)

# 重新初始化并设置时间动态
temporal_datamap = base.TemporalDatamap(poils_frame, dpxsz, flat_synapses_list)
temporal_datamap.set_roi(i_ex, roi)
temporal_datamap.create_t_stack_dmap(acquisition_time, min_pdt, (10, 1.5), curves_path, flash_prob)

# ==================== 采集循环变量初始化 ====================
# set up variables for acquisition loop
t_stack_idx = 0
frozen_datamap = np.copy(temporal_datamap.whole_datamap[temporal_datamap.roi])

# 计算时间步对应关系
n_time_steps, n_tsteps_per_flash_step = utils.compute_time_correspondances((10, 1.5), acquisition_time, min_pdt, mode="pdt")

# 共聚焦与STED分辨率比例计算
ratio = utils.pxsize_ratio(confoc_pxsize, temporal_datamap.pixelsize)
confoc_n_rows, confoc_n_cols = int(np.ceil(frame_shape[0] / ratio)), int(np.ceil(frame_shape[1] / ratio))

# 动作所需像素数
actions_required_pixels = {"confocal": confoc_n_rows * confoc_n_cols, "sted": frame_shape[0] * frame_shape[1]}

# 初始化采集状态变量
imaged_pixels = 0
action_selected = "confocal"     # 初始动作为共聚焦
action_completed = False
pixels_for_current_action = actions_required_pixels[action_selected]

# 初始化采集结果存储
confoc_intensity = np.zeros((confoc_n_rows, confoc_n_cols)).astype(float)
sted_intensity = np.zeros(frozen_datamap.shape).astype(float)
list_datamaps = [np.copy(frozen_datamap)]
list_confocals = [np.zeros(confoc_intensity.shape)]
list_steds = [np.zeros(sted_intensity.shape)]
idx_type = {}

# 起始像素位置
confocal_starting_pixel, sted_starting_pixel = [0, 0], [0, 0]


# ==================== 像素列表和时间计算 ====================
# verif that no values in the pdt_array are lower than the min pdt
# 验证像素停留时间有效性
min_pdt_selected = np.min(pdt)
if min_pdt_selected < min_pdt:
    # TODO : raise error or something not sure how I want to handle it
    print("hey!")
    exit()

# 生成共聚焦和STED的像素扫描列表
confocal_pixel_list = utils.generate_raster_pixel_list(frame_shape[0] * frame_shape[1], [0, 0], frozen_datamap)
confocal_pixel_list = utils.pixel_list_filter(frozen_datamap, confocal_pixel_list, confoc_pxsize,
                                              temporal_datamap.pixelsize, output_empty=True)

# 创建共聚焦PDT数组
confoc_pdt_array = np.zeros(frame_shape)
for row, col in confocal_pixel_list:
    confoc_pdt_array[row, col] = pdt[row, col]
sted_pdt_array = np.copy(pdt)

# 计算各动作所需时间
actions_required_time = {"confocal": np.sum(confoc_pdt_array), "sted": np.sum(sted_pdt_array)}
time_for_current_action = actions_required_time[action_selected]
time_spent_imaging = 0

# 初始像素列表设置
# first action is always a confocal
pixel_list = confocal_pixel_list
pixel_list_time_idx = 0

indices = {"flashes": t_stack_idx}


# ==================== 主采集循环 ===================
# start acquisition loop
print("Starting the experiment loop")
np.random.seed(flash_seed)
np.random.RandomState(flash_seed)
for t_step_idx in tqdm.trange(n_time_steps):
    # 累积时间银行
    microscope.time_bank += min_pdt
    # 获取下一个要成像的像素
    next_pixel_to_img = pixel_list[pixel_list_time_idx]

    # 根据当前动作类型处理像素
    if action_selected == "confocal":
        if microscope.time_bank - confoc_pdt_array[tuple(next_pixel_to_img)] >= 0:
            microscope.time_bank -= confoc_pdt_array[tuple(next_pixel_to_img)]
            pixel_list_time_idx += 1
            microscope.pixel_bank += 1
    elif action_selected == "sted":
        if microscope.time_bank - sted_pdt_array[tuple(next_pixel_to_img)] >= 0:
            microscope.time_bank -= sted_pdt_array[tuple(next_pixel_to_img)]
            pixel_list_time_idx += 1
            microscope.pixel_bank += 1
    # ici il va y avoir un elif pour XbyX sted

    # 检查是否到达闪烁步
    # verify if the current action is interrupted by a flash step
    if t_step_idx % n_tsteps_per_flash_step == 0:
        # 执行像素采集
        if microscope.pixel_bank >= 1:
            if action_selected == "confocal":
                confoc_acq, confoc_intensity, temporal_datamap, imaged_pixel_list = \
                    utils.action_execution_g(action_selected, frame_shape, confocal_starting_pixel, confoc_pxsize,
                                           temporal_datamap, frozen_datamap, microscope,
                                           confoc_pdt_array, p_ex, 0.0, confoc_intensity, bleach, indices)

            elif action_selected == "sted":
                sted_acq, sted_intensity, temporal_datamap, imaged_pixel_list = \
                    utils.action_execution_g(action_selected, frame_shape, sted_starting_pixel,
                                           temporal_datamap.pixelsize, temporal_datamap,
                                           frozen_datamap, microscope, sted_pdt_array, p_ex, p_sted, sted_intensity,
                                           bleach, indices)

            # shift the starting pixel
            if action_selected == "confocal":
                confocal_starting_pixel = imaged_pixel_list[-1]
                confocal_starting_pixel = utils.set_starting_pixel(confocal_starting_pixel, frame_shape, ratio=ratio)
            elif action_selected == "sted":
                sted_starting_pixel = imaged_pixel_list[-1]
                sted_starting_pixel = utils.set_starting_pixel(sted_starting_pixel, frame_shape)

            # empty the pixel bank after the acquisition
            imaged_pixels += microscope.pixel_bank
            microscope.empty_pixel_bank()
            pixel_list_time_idx = 0
            
            # # 检查动作是否完成
            if imaged_pixels == actions_required_pixels[action_selected]:
                action_completed = True

            # if not bleach:
            #     # temporal_datamap.list_dmaps[t_stack_idx] = np.copy(frozen_datamap)
            #     temporal_datamap.flash_tstack[t_stack_idx] = np.copy(frozen_datamap)
            # else:
            #     if t_stack_idx < temporal_datamap.flash_tstack.shape[0] - 1:
            #         temporal_datamap.update_whole_datamap(t_stack_idx)


        # get a copy of the datamap to add to a list to save later
        # j'ai l'impression que y'a gros des choses ici que je devrais faire dans la fonction get_signal_and...
        # 更新时间堆栈索引和数据图
        t_stack_idx += 1
        if t_stack_idx >= temporal_datamap.flash_tstack.shape[0]:
            t_stack_idx = temporal_datamap.flash_tstack.shape[0] - 1
        indices["flashes"] = t_stack_idx
        temporal_datamap["flashes"] = indices["flashes"]
        temporal_datamap.whole_datamap = temporal_datamap["base"] + temporal_datamap["flashes"]
        # 保存当前数据图状态
        roi_save_copy = np.copy(temporal_datamap.whole_datamap[temporal_datamap.roi])
        list_datamaps.append(roi_save_copy)
        idx_type[t_step_idx] = "datamap"

    # Verify how many pixels are needed to complete the acquisition
    # 检查动作完成状态并切换动作
    pixels_needed_to_complete_acq = pixels_for_current_action - imaged_pixels

    if microscope.pixel_bank == pixels_needed_to_complete_acq:
        # 执行最终采集
        if action_selected == "confocal":
            confoc_acq, confoc_intensity, temporal_datamap, imaged_pixel_list = \
                utils.action_execution_g(action_selected, frame_shape, confocal_starting_pixel, confoc_pxsize,
                                       temporal_datamap, frozen_datamap, microscope,
                                       confoc_pdt_array, p_ex, 0.0, confoc_intensity, bleach, indices)

        elif action_selected == "sted":
            sted_acq, sted_intensity, temporal_datamap, imaged_pixel_list = \
                utils.action_execution_g(action_selected, frame_shape, sted_starting_pixel, temporal_datamap.pixelsize,
                                       temporal_datamap, frozen_datamap, microscope,
                                       sted_pdt_array, p_ex, p_sted, sted_intensity, bleach, indices)

        # if bleach and t_stack_idx < len(temporal_datamap.list_dmaps) - 1:
        #     temporal_datamap.bleach_future(t_stack_idx)

        # shift the starting pixel
        if action_selected == "confocal":
            confocal_starting_pixel = imaged_pixel_list[-1]
            confocal_starting_pixel = utils.set_starting_pixel(confocal_starting_pixel, frame_shape, ratio=ratio)
        elif action_selected == "sted":
            sted_starting_pixel = imaged_pixel_list[-1]
            sted_starting_pixel = utils.set_starting_pixel(sted_starting_pixel, frame_shape)

        # empty the pixel bank after the acquisition
        imaged_pixels += microscope.pixel_bank
        microscope.empty_pixel_bank()
        pixel_list_time_idx = 0

        action_completed = True

    # 动作完成后的处理
    if action_completed:
        # add acquisition to be saved
        if action_selected == "confocal":
            list_confocals.append(np.copy(confoc_acq))
            idx_type[t_step_idx] = "confocal"
        elif action_selected == "sted":
            list_steds.append(np.copy(sted_acq))
            idx_type[t_step_idx] = "sted"

        # select the new action based off the previous
        # (so for now this is confocal -> sted -> confocal)
        action_completed = False
        if action_selected == "confocal":
            action_selected = "sted"
            pixel_list = utils.generate_raster_pixel_list(frame_shape[0] * frame_shape[1], [0, 0], frozen_datamap)
        elif action_selected == "sted":
            action_selected = "confocal"
            pixel_list = utils.generate_raster_pixel_list(frame_shape[0] * frame_shape[1], [0, 0], frozen_datamap)
            pixel_list = utils.pixel_list_filter(frozen_datamap, confocal_pixel_list, confoc_pxsize,
                                                 temporal_datamap.pixelsize, output_empty=True)

        pixel_list_time_idx = 0
        imaged_pixels = 0
        pixels_for_current_action = actions_required_pixels[action_selected]


# ==================== 数据保存 ====================
# make stacks for datamaps, confocals and steds, and save them

# 将列表转换为堆栈并保存
datamaps_stack = np.stack(list_datamaps)
confocals_stack = np.stack(list_confocals)
steds_stack = np.stack(list_steds)
np.save(save_path + "/datamaps", datamaps_stack)
np.save(save_path + "/confocals", confocals_stack)
np.save(save_path + "/steds", steds_stack)

# 创建FFmpeg脚本文件用于视频生成
# write the lines in the script file for video generation
ffmpeg_file_path = save_path + "/in.ffconcat"
file = open(ffmpeg_file_path, "a")
file.write("ffconcat version 1.0\n")
file.write("file 0.png\n")
file.write(f"duration 5\n")
file.close()

# 处理时间索引类型
keys_list = sorted(idx_type.keys())
for idx, key in enumerate(keys_list):
    if idx_type[key] == "datamap":
        list_datamaps.pop(0)
    elif idx_type[key] == "confocal":
        list_confocals.pop(0)
    elif idx_type[key] == "sted":
        list_steds.pop(0)
    else:
        print(f"FORBIDDEN UNKNOWN")

    # 写入FFmpeg脚本
    if key != keys_list[-1]:
        # make the calculations for times to write in script file
        duration = (keys_list[idx + 1] - key) * min_pdt * 10  # right?
        # write the lines to script file
        file = open(ffmpeg_file_path, "a")
        file.write(f"file {key + 1}.png\n")
        file.write(f"duration {duration}\n")
        file.close()
    else:
        for i in range(2):   # do it twice or else it skips the last frame
            duration = 10
            # write the lines to script file
            file = open(ffmpeg_file_path, "a")
            file.write(f"file {key + 1}.png\n")
            file.write(f"duration {duration}\n")
            file.close()

np.save(save_path + "/idx_type_dict", idx_type)
# so this should be everything for the experiment part of the script
