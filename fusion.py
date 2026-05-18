import cv2
import numpy as np
import os
import sys

# 1. 工程化路径配置 (动态获取根目录)
# 获取当前脚本所在目录作为项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 统一整合后的相对路径配置
YOLOP_MASK_DIR = os.path.join(PROJECT_ROOT, 'output', 'masks')
LIDAR_TXT_DIR = os.path.join(PROJECT_ROOT, 'output', 'lidar_coords')
ORIGINAL_IMG_DIR = os.path.join(PROJECT_ROOT, 'data', 'kitti', '2011_09_26', '2011_09_26_drive_0001_sync', 'image_02', 'data')
SAVE_DIR = os.path.join(PROJECT_ROOT, 'output', 'fusion_results_advanced')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 绿色覆盖层的透明度 (0.0 到 1.0，0.4 比较合适)
ALPHA = 0.4

# 2. 核心融合算法逻辑 (完全保持原样)
def main():
    if not os.path.exists(LIDAR_TXT_DIR):
        print(f"错误: 找不到雷达坐标目录 {LIDAR_TXT_DIR}，请先运行 save_lidar_data.py")
        return

    txt_files = sorted([f for f in os.listdir(LIDAR_TXT_DIR) if f.endswith('.txt')])
    print(f"开始高级多模态后融合处理，共 {len(txt_files)} 帧...")

    for txt_name in txt_files:
        lidar_data = np.loadtxt(os.path.join(LIDAR_TXT_DIR, txt_name))

        img_name = txt_name.replace('.txt', '.png')
        mask_path = os.path.join(YOLOP_MASK_DIR, img_name)
        orig_img_path = os.path.join(ORIGINAL_IMG_DIR, img_name)

        if not os.path.exists(mask_path) or not os.path.exists(orig_img_path):
            continue

        img = cv2.imread(orig_img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]

        # 核心逻辑 1：渲染 YOLOP 半透明绿色路面
        # 创建一张纯绿色的图
        green_layer = np.zeros_like(img)
        green_layer[:, :, 1] = 255  # BGR 格式，G 通道设为 255

        # 只保留 Mask 白色区域的绿色
        road_overlay = cv2.bitwise_and(green_layer, green_layer, mask=mask)

        # 将绿色区域以 ALPHA 透明度叠加到原图上
        img = cv2.addWeighted(img, 1.0, road_overlay, ALPHA, 0)

        # 核心逻辑 2：提取最可能路缘点 (Best Curb)
        left_candidates = []
        right_candidates = []

        # 如果数据只有一行，转换形状以防迭代报错
        if lidar_data.ndim == 1:
            lidar_data = lidar_data.reshape(1, -1)

        # 遍历该帧激光点找所有跳变点
        for i in range(1, len(lidar_data)):
            u, v, z = lidar_data[i]
            u_p, v_p, z_p = lidar_data[i - 1]  # 把前一个点的 u_p 也取出来
            u, v, u_p, v_p = int(u), int(v), int(u_p), int(v_p)

            if not (0 <= v < h and 0 <= u < w):
                continue

            # 1. 计算高度差
            dz = abs(z - z_p)

            # 2. 读取 YOLOP 视觉判定
            is_road = (mask[v, u] == 255)

            # 3. 计算水平像素距离 (杜绝断层假跳变)
            du = abs(u - u_p)

            # 【三重安全锁】：
            # 条件1: 必须有高度跳变 (dz > 0.025)
            # 条件2: 两个点在画面上必须是紧挨着的 (比如横向距离 du < 30 像素)，防止跨越缺口计算
            # 条件3: 绝对不能是视觉认定的安全路面内部 (not is_road)

            if dz > 0.025 and du < 30 and not is_road:
                # 划分左右半场
                if u < w // 2:
                    left_candidates.append((u, v))
                else:
                    right_candidates.append((u, v))

        # 寻找并绘制最终决定点
        # 逻辑：左侧找最靠右的点 (u最大)，右侧找最靠左的点 (u最小)

        # 左侧路沿
        if left_candidates:
            best_left = max(left_candidates, key=lambda p: p[0])
            cv2.circle(img, best_left, 6, (0, 0, 255), -1)  # 标红实心圆
            cv2.circle(img, best_left, 12, (0, 0, 255), 2)  # 外面画个空心圈，更醒目

        # 右侧路沿
        if right_candidates:
            best_right = min(right_candidates, key=lambda p: p[0])
            cv2.circle(img, best_right, 6, (0, 0, 255), -1)
            cv2.circle(img, best_right, 12, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(SAVE_DIR, img_name), img)

    print(f" 决策级后融合处理成功")
    print(f"请在以下路径查看叠加了绿色可行驶区域与红色物理路缘的高级效果图:\n{SAVE_DIR}")


if __name__ == '__main__':
    main()