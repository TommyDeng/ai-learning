from pathlib import Path

from ultralytics import YOLO

# 按你终端里的实际路径
weights = Path(r"C:\AI-Learning\runs\detect\yolo_runs\coco8_min\weights\best.pt")

if not weights.exists():
    # 有时只有 last.pt
    weights = Path(r"C:\AI-Learning\runs\detect\yolo_runs\coco8_min\weights\last.pt")

print("使用权重:", weights)
model = YOLO(str(weights))

results = model(r"C:\AI-Learning\images\bus.jpg", conf=0.5)
out = Path("yolo_out")
out.mkdir(exist_ok=True)
save_path = out / "bus_finetuned.jpg"
results[0].save(filename=str(save_path))

print("检测到目标数:", len(results[0].boxes))
print("结果图:", save_path.resolve())
