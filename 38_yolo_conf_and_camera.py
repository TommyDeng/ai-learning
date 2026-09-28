from pathlib import Path

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# ===== 1) 图片检测：过滤低置信度 =====
image_path = r"C:\AI-Learning\images\bus.jpg"  # 改成你的图片路径
results = model(image_path, conf=0.5)  # 只保留置信度 >= 0.5
result = results[0]

print("置信度>=0.5 的目标数:", len(result.boxes))
for i, box in enumerate(result.boxes):
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    print(f"{i + 1}. {result.names[cls_id]:12s}  conf={conf:.2f}")

out = Path("yolo_out")
out.mkdir(exist_ok=True)
result.save(filename=str(out / "result_conf05.jpg"))
print("已保存:", out / "result_conf05.jpg")


# ===== 2) 摄像头实时检测（可选）=====
# 若没有摄像头，或打开失败，这段会报错，可注释掉
print("\n尝试打开摄像头，按 q 退出...")
try:
    # source=0 默认摄像头；show=True 弹窗显示
    model.predict(source=0, conf=0.5, show=True, stream=False)
except Exception as e:
    print("摄像头未启用或不可用:", e)
