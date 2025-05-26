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
from utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
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


# 为现实观测提取语义特征。虚拟语义提取在Unity中做的
def detect(save_img=False):
    # 解析命令行参数，获取输入源、模型权重等配置
    source, weights, view_img, save_txt, imgsz = (
        opt.source,
        opt.weights,
        opt.view_img,
        opt.save_txt,
        opt.img_size,
    )
    # 判断是否需要保存推理结果图像
    save_img = not opt.nosave and not source.endswith(".txt")
    # 判断输入源是否为网络摄像头或视频流
    webcam = (
        source.isnumeric()
        or source.endswith(".txt")
        or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    )

    # 创建保存结果的目录
    save_dir = Path(
        increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    )
    (save_dir / "labels" if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )
    # 如果需要提交结果，创建专门的子目录
    if opt.submit:
        sub_dir = str(save_dir) + "/results/"
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

    # 初始化日志和硬件设备
    set_logging()
    device = select_device(opt.device)
    # 判断是否使用半精度浮点数（FP16）
    half = device.type != "cpu"

    # 加载预训练模型
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    # 检查输入图像尺寸是否符合要求
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()  # 将模型转换为FP16

    # 初始化第二阶段的分类器（当前未启用）
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        ).to(device).eval()

    # 初始化视频写入器
    vid_path, vid_writer, s_writer = None, None, None
    # 根据输入源类型选择数据加载方式
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # 启用CUDA加速
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        cudnn.benchmark = False
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # 如果提交结果或保存视频，强制启用CUDA加速
    if opt.submit or opt.save_as_video:
        cudnn.benchmark = True

    # 获取模型输出的类别名称和颜色
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # 运行一次推理以初始化模型
    if device.type != "cpu":
        model(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))
        )
    t0 = time.time()

    # 遍历数据集进行推理
    for path, img, im0s, vid_cap in dataset:
        # 将输入图像转换为张量并发送到设备
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0  # 将像素值归一化到[0,1]范围
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 执行推理
        with torch.no_grad():
            t1 = time_synchronized()
            # 模型输出包含检测结果和分割结果
            out = model(img, augment=opt.augment)
            pred = out[0][0]
            seg = out[1]
            # 应用非极大值抑制（NMS）过滤检测结果
            pred = non_max_suppression(
                pred,
                opt.conf_thres,
                opt.iou_thres,
                classes=opt.classes,
                agnostic=opt.agnostic_nms,
            )
            t2 = time_synchronized()

        # 处理检测结果
        for i, det in enumerate(pred):
            # 根据输入源类型获取当前帧信息
            if webcam:
                # 如果是网络摄像头输入，获取当前帧的路径、状态信息、图像和帧号
                p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(), dataset.count
            else:
                # 如果是文件输入，获取文件路径、空状态信息、原始图像和帧号（默认为0）
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            p = Path(p)
            # 构建保存路径
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )
            s += "%gx%g " % img.shape[2:]
            # 获取归一化参数
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # 将检测框坐标从模型输入尺寸缩放到原始图像尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 统计每个类别的检测数量
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # 保存检测结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        # 将检测框坐标转换为归一化的xywh格式
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or view_img:
                        # 在图像上绘制检测框
                        label = f"{names[int(cls)]} {conf:.2f}"
                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=colors[int(cls)],
                            line_thickness=3,
                        )

            # 打印推理时间
            print(f"{s}Done. ({t2 - t1:.5f}s)")
            # 对分割结果进行插值，使其与原始图像尺寸一致
            seg = F.interpolate(
                seg, (im0.shape[0], im0.shape[1]), mode="bilinear", align_corners=True
            )[0]
            # 将分割结果转换为可视化图像
            mask = label2image(seg.max(axis=0)[1].cpu().numpy(), Labscene_COLORMAP)[
                :, :, ::-1
            ]
            # 将分割结果与原始图像融合
            dst = cv2.addWeighted(mask, 0.4, im0, 0.6, 0)

            # 显示结果
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.imshow("segmentation", mask)
                cv2.imshow("mix", dst)
                cv2.waitKey(0)

            # 保存提交结果
            if opt.submit:
                sub_path = sub_dir + str(p.name)
                sub_path = sub_path[:-4] + "_pred.png"
                result = trainid2id(seg.max(axis=0)[1].cpu().numpy(), Labscene_IDMAP)
                cv2.imwrite(sub_path, result)

            # 保存结果图像
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path[:-4] + "_mask" + save_path[-4:], mask)
                else:
                    # 处理视频或流媒体输入
                    if vid_path != save_path:
                        vid_path = save_path  # 更新当前视频路径
                        # 如果已存在视频写入器，先释放资源
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        # 如果输入是视频文件，获取其帧率和分辨率
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频宽度
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频高度
                        else:
                            # 如果是流媒体输入，使用默认帧率和当前帧分辨率
                            fps, w, h = 30, dst.shape[1], dst.shape[0]
                            save_path += ".mp4"  # 为输出文件添加扩展名
                            save_path2 = save_path + "mask.mp4"  # 生成掩码视频路径
                            save_path3 = save_path + "origin.mp4"  # 生成原始视频路径

                        # 初始化视频写入器
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                        vid_writer2 = cv2.VideoWriter(
                            save_path2, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                        vid_writer3 = cv2.VideoWriter(
                            save_path3, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )

                    # 写入视频帧
                    vid_writer.write(dst)
                    vid_writer2.write(mask)
                    vid_writer3.write(im0)

            # 如果需要保存为单独的视频文件
            if opt.save_as_video:
                if not s_writer:
                    fps, w, h = 30, dst.shape[1], dst.shape[0]
                    s_writer = cv2.VideoWriter(
                        str(save_dir) + "out.mp4",
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )
                s_writer.write(dst)

    # 打印最终结果信息
    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ""
        )
        print(f"Results saved to {save_dir}{s}")
    if s_writer != None:
        s_writer.release()
    print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", nargs="+", type=str, default="./best.pt", help="model.pt path(s)"
    )
    parser.add_argument(
        "--source", type=str, default="0", help="source"
    )  # file/folder, 0 for webcam
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--save-as-video", action="store_true", help="save same size images as a video"
    )
    parser.add_argument(
        "--submit", action="store_true", help="get submit file in folder submit"
    )
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
