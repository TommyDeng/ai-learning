import os
from pathlib import Path

# 避开 Windows CPU oneDNN 兼容问题
os.environ["FLAGS_enable_pir_api"] = "0"

from fastapi import FastAPI, File, HTTPException, UploadFile
from paddleocr import PaddleOCR

app = FastAPI(title="OCR API")

ocr = PaddleOCR(
    lang="ch",
    enable_mkldnn=False,
)

UPLOAD_DIR = Path("ocr_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def parse_ocr_result(result) -> list[dict]:
    items = []
    if not result:
        return items

    # 新版本 predict 常返回 list[对象/dict]
    rows = result if isinstance(result, list) else [result]
    for row in rows:
        texts, scores = [], []
        if hasattr(row, "rec_texts"):
            texts = row.rec_texts or []
            scores = getattr(row, "rec_scores", None) or []
        elif isinstance(row, dict):
            texts = row.get("rec_texts") or []
            scores = row.get("rec_scores") or []
        else:
            # 旧结构: list of [box, (text, conf)]
            try:
                for line in row:
                    items.append(
                        {
                            "text": line[1][0],
                            "confidence": round(float(line[1][1]), 3),
                        }
                    )
            except Exception:
                pass
            continue

        for i, text in enumerate(texts):
            conf = float(scores[i]) if i < len(scores) else 0.0
            items.append({"text": text, "confidence": round(conf, 3)})
    return items


@app.get("/")
def root():
    return {"message": "OCR API running"}


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}:
        raise HTTPException(status_code=400, detail="仅支持图片文件")

    save_path = UPLOAD_DIR / file.filename
    save_path.write_bytes(await file.read())

    try:
        result = ocr.predict(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 失败: {e}")

    items = parse_ocr_result(result)
    full_text = "\n".join([x["text"] for x in items])

    return {
        "filename": file.filename,
        "count": len(items),
        "lines": items,
        "text": full_text,
    }
