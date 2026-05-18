import pykitti
import numpy as np
import cv2
import os
import sys
from tqdm import tqdm

# 1. 工程化路径配置 (动态获取根目录)
# 获取当前脚本所在目录作为项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 原始数据配置 (指向项目内的 data 文件夹)
basedir = os.path.join(PROJECT_ROOT, 'data', 'kitti')
date = '2011_09_26'
drive = '0001'

# 保存路径配置 (统一输出到项目内的 output 文件夹)
save_root = os.path.join(PROJECT_ROOT, 'output')
img_save_dir = os.path.join(save_root, 'labeled_images')
txt_save_dir = os.path.join(save_root, 'lidar_coords')

for d in [img_save_dir, txt_save_dir]:
    if not os.path.exists(d): os.makedirs(d)

# 2. 核心算法逻辑 (完全保持原样)
dataset = pykitti.raw(basedir, date, drive)
P_rect_02 = dataset.calib.P_rect_20
T_velo_to_cam = dataset.calib.T_cam2_velo


def project_points_with_z(pts3d, T, P):
    """投影并保留原始高度Z"""
    pts3d_h = np.hstack([pts3d[:, :3], np.ones((pts3d.shape[0], 1))])
    pts_cam = (T @ pts3d_h.T).T
    mask = pts_cam[:, 2] > 0
    pts_cam_filtered = pts_cam[mask]
    # 保留对应的原始3D点高度 (在雷达坐标系下的 z)
    z_values = pts3d[mask, 2]

    pts_2d = (P @ pts_cam_filtered.T).T
    u = pts_2d[:, 0] / pts_2d[:, 2]
    v = pts_2d[:, 1] / pts_2d[:, 2]
    return u, v, z_values


# 批量循环处理
print(f"开始批量处理 {len(dataset)} 帧 LiDAR 数据...")
for frame_idx in tqdm(range(len(dataset))):
    img = cv2.cvtColor(np.array(dataset.get_cam2(frame_idx)), cv2.COLOR_RGB2BGR)
    points = dataset.get_velo(frame_idx)

    # 模拟单线 (-4.1° 到 -3.9° 之间的扫描线)
    dist = np.linalg.norm(points[:, :3], axis=1)
    angles = np.arcsin(points[:, 2] / dist) * 180 / np.pi
    mask = (angles > -4.1) & (angles < -3.9)
    points_single = points[mask]

    if len(points_single) == 0: continue

    # 投影：获取像素坐标 u, v 和 深度 z
    u_all, v_all, z_all = project_points_with_z(points_single, T_velo_to_cam, P_rect_02)

    # 保存 TXT 数据 (用于后续融合)
    # 格式: u v z
    lidar_output = np.column_stack((u_all, v_all, z_all))
    txt_name = f"{frame_idx:010d}.txt"  # 保持 KITTI 的 10 位命名习惯
    np.savetxt(os.path.join(txt_save_dir, txt_name), lidar_output, fmt='%d %d %.4f')

    # 可视化并保存图片 (用于论文素材)
    for i in range(len(u_all)):
        if 0 <= u_all[i] < img.shape[1] and 0 <= v_all[i] < img.shape[0]:
            cv2.circle(img, (int(u_all[i]), int(v_all[i])), 2, (0, 255, 0), -1)

    img_name = f"{frame_idx:010d}.png"
    cv2.imwrite(os.path.join(img_save_dir, img_name), img)

print(f"LiDAR 预处理完成！\n提取的坐标数据和可视化图片已保存在: {save_root}\n")