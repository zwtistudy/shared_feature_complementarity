import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import torch.nn.functional as F

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np


# 共享特征语义空间的颜色映射表
# 用于将不同类别的语义特征可视化
Labscene_COLORMAP = [
    [0, 0, 0],        # 背景
    [0, 255, 0],      # 红色车辆
    [0, 0, 255],      # 白色车辆
    [0, 0, 255],      # 蓝色车辆
    [0, 0, 128],      # 目标物体
    [128, 0, 128],    # 黄色物体
    [0, 128, 128],    # 墙壁
    [0, 0, 0],        # 平面
    [64, 0, 0],       # 其他背景
]

# 语义特征ID映射表
# 用于将训练时的类别ID映射到实际语义空间中的ID
Labscene_IDMAP = [[7], [8], [11], [12], [13], [17], [19], [20], [21]]

# 共享语义空间中的类别定义
# 包含视觉和雷达观测数据中的共同语义类别
Labscene_Class = [
    "S_Obs",        # 静态障碍物
    "R_Car",        # 红色车辆
    "W_Car",        # 白色车辆
    "B_Car",        # 蓝色车辆
    "B_Target",     # 目标物体
    "Y_Object",     # 黄色物体
    "Wall",         # 墙壁
    "Plane",        # 平面
    "_background_", # 背景
]


# 将预测结果转换为可视化图像
# 用于在共享语义空间中展示特征提取结果
def label2image(pred, COLORMAP=Labscene_COLORMAP):
    colormap = np.array(COLORMAP, dtype="uint8")
    X = pred.astype("int32")
    return colormap[X, :]

# 将训练ID转换为实际语义空间ID
# 用于将模型输出映射到共享语义空间
def trainid2id(pred, IDMAP=Labscene_IDMAP):
    colormap = np.array(IDMAP, dtype="uint8")
    X = pred.astype("int32")
    return colormap[X, :]


# 为现实观测提取语义特征
def detect():
    # 从命令行参数中获取输入参数
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    weights = "./race.pt"       # 指定模型权重文件路径
    source = "./data/test_imgs" # 指定输入图像目录
    view_img = True             # 开启实时显示
    # 确定是否需要保存结果图像（当不指定--nosave且输入不是txt文件时保存）
    save_img = not opt.nosave and not source.endswith('.txt')

    # 创建输出目录
    # increment_path用于自动递增运行目录（避免覆盖已有结果）
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # 初始化设置
    set_logging()  # 设置日志
    device = select_device(opt.device)  # 选择计算设备（GPU/CPU）
    # 确定是否使用半精度（FP16）推理（仅CUDA设备支持）
    half = device.type != 'cpu'
    # 加载模型
    model = attempt_load(weights, map_location=device)  # 加载FP32模型
    stride = int(model.stride.max())  # 获取模型步长
    imgsz = check_img_size(imgsz, s=stride)  # 检查输入尺寸是否符合模型要求
    if half:
        model.half()  # 转换为FP16精度

    cudnn.benchmark = False  # 禁用cudnn基准测试（固定输入尺寸时建议关闭）
    # 创建数据加载器（处理图像/视频/摄像头输入）
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 获取类别名称和随机颜色（用于可视化）
    names = model.module.names if hasattr(model, 'module') else model.names  # 模型类别名称
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]  # 为每个类别生成随机颜色

    # 运行推理过程
    if device.type != 'cpu':
        # 在GPU设备上进行一次预热推理（初始化CUDA上下文）
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()  # 记录推理开始时间

    # 遍历数据集中的每个图像/视频帧
    for path, img, im0s, vid_cap in dataset:
        # 将numpy数组转换为torch张量并移动到指定设备
        img = torch.from_numpy(img).to(device)
        # 根据设置选择半精度(FP16)或单精度(FP32)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 将像素值从0-255归一化到0.0-1.0范围
        # 如果输入是3维(HWC)，则增加一个批次维度变为4维(NCHW)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理过程（禁用梯度计算以节省内存）
        with torch.no_grad():
            t1 = time_synchronized()  # 同步CUDA操作并记录开始时间
            out = model(img, augment=opt.augment)  # 执行模型推理
            pred = out[0][0]  # 获取目标检测结果
            seg = out[1]  # 获取语义分割结果
            # 应用非极大值抑制(NMS)过滤重叠的检测框
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                      classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()  # 记录推理结束时间

        # 处理检测结果（每张图像可能有多个检测框）
        for i, det in enumerate(pred):  # 遍历每张图像的检测结果
            # 获取当前图像路径、信息字符串、原始图像和帧号
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # 将路径转换为Path对象
            save_path = str(save_dir / p.name)  # 构建保存路径（如：runs/detect/exp/img.jpg）
            # 构建标签文件路径（如：runs/detect/exp/labels/img.png）
            s += '%gx%g ' % img.shape[2:]  # 添加图像尺寸到信息字符串（如：640x480）

            if len(det):  # 如果检测到目标
                # 将检测框坐标从推理尺寸缩放到原始图像尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 打印检测结果（统计每个类别的检测数量）
                for c in det[:, -1].unique():  # 遍历所有检测到的类别
                    n = (det[:, -1] == c).sum()  # 计算当前类别的检测数量
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到信息字符串（如：2 cars, 1 person）

                # 保存检测结果
                for *xyxy, conf, cls in reversed(det):  # 遍历每个检测框（从后往前）
                    if save_img or view_img:  # 如果需要显示或保存可视化结果
                        # 构建标签文本（类别名+置信度）
                        label = f'{names[int(cls)]} {conf:.2f}'
                        # 在图像上绘制检测框
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # 打印推理时间（包括推理+NMS处理时间）
            print(f'{s}Done. ({t2 - t1:.5f}s)')

            # 对分割结果进行双线性插值，使其尺寸与原始图像一致
            seg = F.interpolate(seg, (im0.shape[0], im0.shape[1]), mode='bilinear', align_corners=True)[0]

            # 将分割结果转换为彩色图像（使用预定义的颜色映射表）
            # seg.max(axis=0)[1]获取每个像素点概率最大的类别索引
            # [:, :, ::-1]将BGR格式转换为RGB格式
            mask = label2image(seg.max(axis=0)[1].cpu().numpy(), Labscene_COLORMAP)[:, :, ::-1]

            # 将分割掩码与原图按比例混合显示（透明度0.4）
            dst = cv2.addWeighted(mask, 0.4, im0, 0.6, 0)

            # 实时显示结果
            if view_img:
                # 显示原始图像（带检测框）
                cv2.imshow(str(p), im0)
                # 显示纯分割结果
                cv2.imshow("segmentation", mask)
                # 显示混合结果
                cv2.imshow("mix", dst)
                # 等待按键（0表示无限等待）
                cv2.waitKey(0)  # 1 millisecond

            # 保存结果图像（带分割掩码）
            if save_img:
                # 在原始文件名后添加"_mask"后缀保存分割结果
                # 如：img.jpg -> img_mask.jpg
                cv2.imwrite(save_path[:-4] + "_mask" + save_path[-4:], mask)

    # 检查是否需要保存文本或图像结果
    if save_img:
        # 打印结果保存路径信息
        print(f"Results saved to {save_dir}")

    # 打印整个检测过程的总耗时（从开始到结束）
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-as-video', action='store_true', help='save same size images as a video')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    # 使用torch.no_grad()上下文管理器，禁用梯度计算以节省内存
    with torch.no_grad():
        # 检查是否需要更新所有模型
        if opt.update:  # update all models (to fix SourceChangeWarning)
            # 遍历所有预定义的YOLOv5模型权重文件
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()  # 执行目标检测
                strip_optimizer(opt.weights)  # 去除优化器状态以减小文件大小
        else:
            # 如果不需要更新所有模型，则只执行一次检测
            detect()
