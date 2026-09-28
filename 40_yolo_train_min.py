from pathlib import Path

from ultralytics import YOLO

# 1. 加载预训练模型（在已有能力上微调，而不是从零训练）
model = YOLO("yolov8n.pt")

# 2. 训练
# data=coco8.yaml : 官方超小数据集，自动下载
# epochs=5        : 入门先跑通，轮数少
# imgsz=640       : 常用输入尺寸
# device=cpu      : 你当前是 CPU 环境
results = model.train(
    data="coco8.yaml",
    epochs=5,
    imgsz=640,
    device="cpu",
    project="yolo_runs",
    name="coco8_min",
)

print("\n训练完成")
print("运行目录通常在: runs/detect/yolo_runs/coco8_min/")


# 3. 加载刚训练出的权重做一次推理
best = Path("runs/detect/yolo_runs/coco8_min/weights/best.pt")
if best.exists():
    finetuned = YOLO(str(best))
    pred = finetuned(r"C:\AI-Learning\images\bus.jpg", conf=0.5)
    out = Path("yolo_out")
    out.mkdir(exist_ok=True)
    save_path = out / "bus_finetuned.jpg"
    pred[0].save(filename=str(save_path))
    print("微调模型预测已保存:", save_path.resolve())
else:
    print("未找到 best.pt，请检查 runs/detect/yolo_runs/coco8_min/weights/")
