from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO

app = FastAPI(title="YOLO Detect API")

model = YOLO("yolov8n.pt")

UPLOAD_DIR = Path("yolo_uploads")
RESULT_DIR = Path("yolo_out") / "api"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {"message": "YOLO API running", "endpoints": ["/detect", "/docs"]}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = 0.5,
    only_person: bool = False,
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail="仅支持图片文件")

    # 保存上传图
    save_in = UPLOAD_DIR / file.filename
    content = await file.read()
    save_in.write_bytes(content)

    # 推理
    results = model(str(save_in), conf=conf, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        name = result.names[cls_id]
        if only_person and name != "person":
            continue
        conf_score = float(box.conf[0])
        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
        detections.append(
            {
                "class": name,
                "confidence": round(conf_score, 3),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )

    # 保存画框图
    out_name = f"{save_in.stem}_det.jpg"
    out_path = RESULT_DIR / out_name
    result.save(filename=str(out_path))

    return {
        "filename": file.filename,
        "count": len(detections),
        "detections": detections,
        "result_image": f"/result/{out_name}",
    }


@app.get("/result/{filename}")
def get_result(filename: str):
    path = RESULT_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="结果图不存在")
    return FileResponse(path)
