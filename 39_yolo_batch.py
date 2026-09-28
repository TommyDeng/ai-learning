from collections import Counter
from pathlib import Path

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

input_dir = Path("images")
output_dir = Path("yolo_out") / "batch"
output_dir.mkdir(parents=True, exist_ok=True)

# 支持的图片后缀
suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
images = [p for p in input_dir.iterdir() if p.suffix.lower() in suffixes]

if not images:
    print(f"未在 {input_dir.resolve()} 找到图片，请先放入几张图")
    raise SystemExit(1)

print(f"共找到 {len(images)} 张图片\n")

total_counter = Counter()

for img_path in images:
    results = model(str(img_path), conf=0.5, verbose=False)
    result = results[0]

    # 统计本图类别
    local_counter = Counter()
    for box in result.boxes:
        name = result.names[int(box.cls[0])]
        local_counter[name] += 1
        total_counter[name] += 1

    # 保存画框结果
    save_path = output_dir / f"{img_path.stem}_det.jpg"
    result.save(filename=str(save_path))

    summary = ", ".join(f"{k}×{v}" for k, v in local_counter.items()) or "无目标"
    print(f"{img_path.name:20s} -> {summary}")
    print(f"  保存至: {save_path}")

print("\n===== 全部图片汇总 =====")
if total_counter:
    for name, cnt in total_counter.most_common():
        print(f"{name:12s}: {cnt}")
else:
    print("没有检测到任何目标")

print(f"\n结果目录: {output_dir.resolve()}")
