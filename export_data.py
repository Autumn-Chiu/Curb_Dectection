import argparse
import os, sys
from pathlib import Path
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm

# 1. 工程化路径配置 (动态获取根目录)
# 获取当前脚本所在目录作为项目根目录 (PROJECT_ROOT)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 将项目根目录加入环境变量，这样就能直接 from lib.xxx 导入 YOLOP 的核心库了
sys.path.append(PROJECT_ROOT)

from lib.config import cfg
from lib.utils.utils import select_device
from lib.models import get_net
from lib.dataset import LoadImages

# 图像预处理标准化
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def export_masks(opt):
    # 1. 设备与保存目录初始化
    device = select_device(logger=None, device=opt.device)
    half = device.type != 'cpu'

    # 输出掩码的保存路径设定为 output/masks
    mask_dir = os.path.join(opt.save_dir, 'masks')
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # 2. 加载模型
    print(f"正在加载 YOLOP 模型权重: {opt.weights}...")
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()
    model.eval()

    # 3. 使用官方的 LoadImages 类（完美解决变形和对齐问题）
    print(f"正在读取原始图像数据: {opt.source}")
    dataset = LoadImages(opt.source, img_size=opt.img_size)

    print(f"开始处理，将导出精确对齐的二值化 Mask 到: {mask_dir}")

    # 4. 推理与精确还原
    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):

        # 预处理
        img = transform(img).to(device)
        img = img.half() if half else img.float()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 模型推理
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = model(img)

        #  核心：官方的精确尺寸还原逻辑
        _, _, height, width = img.shape
        orig_h, orig_w = shapes[0]  # 直接从 dataloader 提取真实原图尺寸 (例如 375, 1242)

        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)

        # 1. 裁剪掉 Padding 的灰边部分
        da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]

        # 2. 提取类别索引
        _, da_seg_mask = torch.max(da_predict, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

        # 3. 强行缩放回精确的原图物理尺寸
        # 注意：必须使用 cv2.INTER_NEAREST (最近邻插值)，这样能保证边缘依旧是干净的 0 和 1，不会出现半透明灰度值
        da_seg_mask = cv2.resize(da_seg_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 4. 转换为 0 (背景) 和 255 (路面) 的二值图
        mask_to_save = da_seg_mask * 255

        # 保存图片
        mask_path = os.path.join(mask_dir, Path(path).name)
        cv2.imwrite(mask_path, mask_to_save)

    print("Mask 导出完成！现在您可以运行 save_lidar_data.py 进行雷达数据处理了。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 2. 参数默认路径修改为工程化的动态路径
    # 默认权重路径: data/weights/End-to-end.pth
    default_weights = os.path.join(PROJECT_ROOT, 'data', 'weights', 'End-to-end.pth')

    # 默认数据输入路径: 自动指向 kitti 数据集图片目录
    default_source = os.path.join(PROJECT_ROOT, 'data', 'kitti', '2011_09_26', '2011_09_26_drive_0001_sync', 'image_02',
                                  'data')

    # 默认结果保存路径: output/
    default_save_dir = os.path.join(PROJECT_ROOT, 'output')

    parser.add_argument('--weights', type=str, default=default_weights, help='模型权重路径')
    parser.add_argument('--source', type=str, default=default_source, help='KITTI 原始图像文件夹路径')
    parser.add_argument('--img-size', type=int, default=640, help='网络输入尺寸 (勿改)')
    parser.add_argument('--device', default='cpu', help='cuda device: 0 or cpu')
    parser.add_argument('--save-dir', type=str, default=default_save_dir, help='导出结果保存路径')

    opt = parser.parse_args()

    export_masks(opt)