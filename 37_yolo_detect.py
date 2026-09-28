from pathlib import Path

from ultralytics import YOLO

# 1. 加载预训练模型（首次会自动下载，体积不大）
# n = nano，最小最快，适合入门
model = YOLO("yolov8n.pt")

# 2. 准备一张图片
# 方式A：用 Ultralytics 自带示例（无本地图时）
# 方式B：换成你自己的图片路径，例如 r"C:\AI-Learning\images\test.jpg"
# image_path = "https://ultralytics.com/images/bus.jpg"
image_path = r"C:\AI-Learning\images\bus.jpg"

# 3. 推理
results = model(image_path)

# 4. 解析结果
result = results[0]
print("检测到的目标数量:", len(result.boxes))

names = result.names  # 类别 id -> 名称
for i, box in enumerate(result.boxes):
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
    print(
        f"{i + 1}. {names[cls_id]:12s}  置信度={conf:.2f}  框={[round(v, 1) for v in xyxy]}"
    )

# 5. 保存画框后的图
out_dir = Path("yolo_out")
out_dir.mkdir(exist_ok=True)
save_path = out_dir / "result.jpg"
result.save(filename=str(save_path))
print("结果图已保存:", save_path.resolve())
