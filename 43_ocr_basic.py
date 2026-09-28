import os
from pathlib import Path

# 必须在 import paddleocr 之前设置
os.environ["FLAGS_enable_pir_api"] = "0"

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="ch",
    enable_mkldnn=False,  # 关闭 oneDNN，避开这个 bug
)

image_path = r"C:\AI-Learning\images\ocr_test.png"  # 改成你的图片

if not Path(image_path).exists():
    raise SystemExit(f"图片不存在: {image_path}")

# 新版本推荐 predict
result = ocr.predict(image_path)

print("识别结果：\n")
lines = []

# 兼容解析不同返回结构
if isinstance(result, list):
    for item in result:
        if hasattr(item, "rec_texts"):
            texts = item.rec_texts or []
            scores = getattr(item, "rec_scores", None) or []
            for i, text in enumerate(texts, start=1):
                conf = float(scores[i - 1]) if i - 1 < len(scores) else 0.0
                lines.append(text)
                print(f"{i}. {text}  (conf={conf:.2f})")
        elif isinstance(item, dict):
            texts = item.get("rec_texts") or []
            scores = item.get("rec_scores") or []
            for i, text in enumerate(texts, start=1):
                conf = float(scores[i - 1]) if i - 1 < len(scores) else 0.0
                lines.append(text)
                print(f"{i}. {text}  (conf={conf:.2f})")
        else:
            # 旧结构兜底
            try:
                for i, row in enumerate(item, start=1):
                    text = row[1][0]
                    conf = float(row[1][1])
                    lines.append(text)
                    print(f"{i}. {text}  (conf={conf:.2f})")
            except Exception:
                print("无法解析的条目:", item)
else:
    print("未知返回:", type(result), result)

print("\n合并文本：")
print("\n".join(lines) if lines else "(空)")
