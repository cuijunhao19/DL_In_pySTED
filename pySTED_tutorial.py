import numpy as np
from matplotlib import pyplot as plt
import time

from pysted import base, utils, microscopes
from pysted import exp_data_gen as dg

"""
STED显微镜基础仿真教程
这个脚本演示了如何使用pySTED模拟共聚焦和STED采集
核心概念：仿真需要显微镜和样本，显微镜由激发光束、STED光束、探测器和荧光分子参数组成
"""

"""
This script will go over the basics of pySTED for simulation of confocal and STED acquisitions
on simulated samples.
In order to simulate an acquisition, we need a microscope and a sample. To build the STED microscope, we need an
excitation beam, a STED beam, a detector and the parameters of the fluorophores used in the sample. The class
code for the objects that make up the microscope and the sample are contained in pysted.base
Each object has parameters which can be tuned, which will affect the resulting acquisition
"""

print("Setting up the microscope...")

# ==================== 荧光分子参数配置 ====================
# 定义EGFP荧光蛋白的物理特性
# Fluorophore properties
egfp = {
    "lambda_": 535e-9,        # 发射波长 535nm
    "qy": 0.6,                # 量子产率 - 吸收光子后发射光子的概率
    "sigma_abs": {            # 吸收截面 - 不同波长下的吸收能力
        488: 0.08e-21,        # 488nm激发光的吸收截面
        575: 0.02e-21         # 575nm STED光的吸收截面
    },
    "sigma_ste": {            # STED截面 - STED光引发受激发射的能力
        575: 3.0e-22,
    },
    "tau": 3e-09,             # 荧光寿命 - 激发态平均存活时间
    "tau_vib": 1.0e-12,       # 振动驰豫时间
    "tau_tri": 1.2e-6,        # 三重态寿命
    "k1": 1.3e-15,            # 光漂白速率常数
    "b":1.4,                  # 漂白参数
    "triplet_dynamics_frac": 0, # 三重态动力学分数
}

# ==================== 显微镜系统配置 ====================
pixelsize = 20e-9  # 像素尺寸 20nm

# 创建光束和探测器对象
laser_ex = base.GaussianBeam(488e-9)  # 高斯激发光束，波长488nm
laser_sted = base.DonutBeam(575e-9, zero_residual=0, rate=40e6, 
                           tau=400e-12, anti_stoke=False)  # 环形STED光束
detector = base.Detector(noise=True, det_delay=750e-12, 
                        det_width=8e-9, background=0)  # 探测器，包含噪声模型
objective = base.Objective()  # 显微镜物镜
fluo = base.Fluorescence(**egfp)  # 荧光分子对象

# ==================== 采集参数范围定义 ====================
# 定义RL代理可以选择的参数范围
action_spaces = {
    "p_sted": {"low": 0., "high": 175e-3},   # STED功率范围: 0-175mW
    "p_ex": {"low": 0., "high": 150e-6},     # 激发功率范围: 0-150μW  
    "pdt": {"low": 10.0e-6, "high": 150.0e-6}, # 像素停留时间: 10-150μs
}

# STED采集参数示例
sted_params = {
    "pdt": action_spaces["pdt"]["low"] * 2,      # 20μs像素停留时间
    "p_ex": action_spaces["p_ex"]["high"] * 0.6, # 90μW激发功率
    "p_sted": action_spaces["p_sted"]["high"] * 0.6 # 105mW STED功率
}

# 共聚焦采集参数 (STED功率为0)
conf_params = {
    "pdt": action_spaces["pdt"]["low"],          # 10μs像素停留时间
    "p_ex": action_spaces["p_ex"]["high"] * 0.6, # 90μW激发功率
    "p_sted": 0.0   # STED功率为0 - 共聚焦模式
}


# ==================== 显微镜初始化 ====================
# generate the microscope from its constituent parts
# if load_cache is true, it will load the previously generated microscope. This can save time if a
# microscope was previsously generated and used the same pixelsize we are using now
# 创建显微镜对象，load_cache=True可以加速重复仿真
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)

# 缓存PSF计算，save_cache=True保存计算结果供后续使用
i_ex, i_sted, _ = microscope.cache(pixelsize, save_cache=True)

# 计算不同模式下的有效PSF
psf_conf = microscope.get_effective(pixelsize, action_spaces["p_ex"]["high"], 0.0)
psf_sted = microscope.get_effective(pixelsize, action_spaces["p_ex"]["high"], action_spaces["p_sted"]["high"] * 0.25)

# You can uncomment these lines to visualize the simulated excitation and STED beams, as well as the
# detection PSFs when using certain excitation / STED power combinations
# fig, axes = plt.subplots(2, 2)
#
# axes[0, 0].imshow(i_ex)
# axes[0, 0].set_title(f"Excitation beam")
#
# axes[0, 1].imshow(i_sted)
# axes[0, 1].set_title(f"STED beam")
#
# axes[1, 0].imshow(psf_conf)
# axes[1, 0].set_title(f"Detection PSF in confocal modality")
#
# axes[1, 1].imshow(psf_sted)
# axes[1, 1].set_title(f"Detection PSF in STED modality")
#
# plt.tight_layout()
# plt.show()

# ==================== 样本创建 ====================
# 创建突触样结构的样本
# we now need a sample on to which to do our acquisition, which we call the datamap
# I will show how to build a simple datamap, along with a more complex one which includes nanostructures and a
# temporal element
# First, we use the Synapse class in exp_data_gen to simulate a synapse-like structure and add nanostructures to it
# You could use any integer-valued array as a Datamap

# 第一个样本：基础突触结构
shroom1 = dg.Synapse(5, mode="mushroom", seed=42)      # 创建蘑菇状突触

# 添加纳米域 - 模拟蛋白质聚集区域
n_molecs_in_domain1, min_dist1 = 135, 50
shroom1.add_nanodomains(10, min_dist_nm=min_dist1, n_molecs_in_domain=n_molecs_in_domain1, valid_thickness=7, seed=42)

# create the Datamap and set its region of interest(创建数据图并设置感兴趣区域)
dmap = base.Datamap(shroom1.frame, pixelsize)
dmap.set_roi(i_ex, "max")

# 第二个样本：带时间动态的样本
shroom2 = dg.Synapse(5, mode="mushroom", seed=42)
n_molecs_in_domain2, min_dist2 = 0, 50
shroom2.add_nanodomains(10, min_dist_nm=min_dist2, n_molecs_in_domain=n_molecs_in_domain2, valid_thickness=7, seed=42)

# 创建时间动态数据图 - 模拟荧光闪烁
# create a temporal Datamap which will also contain information on the positions of nanodomains
# We create a temporal element by making the nanostructures flash
# We then set its temporal index to be at the flash peak
time_idx = 2
temp_dmap = base.TemporalSynapseDmap(shroom2.frame, pixelsize, shroom2)
temp_dmap.set_roi(i_ex, "max")
temp_dmap.create_t_stack_dmap(2000000)
temp_dmap.update_whole_datamap(time_idx)
temp_dmap.update_dicts({"flashes": time_idx})

# you can uncomment this code to see both datamaps, which should look similar
# fig, axes = plt.subplots(1, 2)
#
# axes[0].imshow(dmap.whole_datamap[dmap.roi])
# axes[0].set_title(f"Base Datamap")
#
# axes[1].imshow(temp_dmap.whole_datamap[temp_dmap.roi])
# axes[1].set_title(f"Datamap with temporal element")
#
# plt.show()

# uncomment this code to run through the flash
# for t in range(temp_dmap.flash_tstack.shape[0]):
#     temp_dmap.update_whole_datamap(t)
#     temp_dmap.update_dicts({"flashes": t})
#
#     plt.imshow(temp_dmap.whole_datamap[temp_dmap.roi])
#     plt.title(f"Time idx = {t}")
#     plt.show()

# ==================== 图像采集仿真 ====================
# 执行共聚焦和STED采集
# Now let's show a confocal acquisition and a STED acquisition on the datamaps
# The returns are :
# (1) The acquired image signal
# (2) The bleached datamaps
# (3) The acquired intensity. This is only useful when working in a temporal exeperiment setting, in which
#     an acquisition could be interrupted by the flash happening through it.
# get_signal_and_bleach返回: (采集信号, 漂白后数据图, 采集强度)
conf_acq, conf_bleached, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **conf_params,
                                                              bleach=True, update=False, seed=42)
conf_acq2, conf_bleached2, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **conf_params,
                                                              bleach=True, update=False, seed=42)
sted_acq, sted_bleached, _ = microscope.get_signal_and_bleach(temp_dmap, temp_dmap.pixelsize, **sted_params,
                                                              bleach=True, update=True, seed=42)
sted_acq2, sted_bleached2, _ = microscope.get_signal_and_bleach(temp_dmap, temp_dmap.pixelsize, **sted_params,
                                                              bleach=True, update=True, seed=42)

# ==================== 结果可视化 ====================
fig, axes = plt.subplots(2, 2)

# 共聚焦图像显示 - 使用相同的归一化
vmax = conf_acq.max()
axes[0,0].imshow(conf_acq, vmax=vmax)
axes[0,0].set_title(f"Confocal 1")

axes[0,1].imshow(conf_acq2, vmax=vmax)
axes[0,1].set_title(f"Confocal 2")

# STED图像显示 - 使用相同的归一化
vmax = sted_acq.max()
axes[1,0].imshow(sted_acq, vmax=vmax)
axes[1,0].set_title(f"STED 1")


axes[1,1].imshow(sted_acq2, vmax=vmax)
axes[1,1].set_title(f"STED 2")

plt.suptitle("The four images where acquired sequentially. \nSame normalization on each row")

plt.show()

# I have set the bleaching to false in these acquisitions for speed. You can set it to True to see its effects
# on the acquired signal and the datamaps. You can also of course modify other parameters to see their effects
# on the acquired images. :)
