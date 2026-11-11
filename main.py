"""API skrining Down Syndrome: YOLOv8n-cls + Grad-CAM.
Preprocessing meniru notebook: RGB, 224x224, skala 0-1, TETAPI input dibuat persegi
(crop di sekitar wajah) karena data training berupa foto wajah persegi.
Foto diproses di memori, tidak disimpan/di-log."""
import base64, hmac, os, threading, time, uuid
from contextlib import asynccontextmanager

import cv2, numpy as np, torch, torch.nn as nn
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
API_KEY = os.getenv("API_KEY", "")
ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
IMG_SIZE, MAX_BYTES = 224, 5 * 1024 * 1024
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "60"))  # heuristik; dihitung pada crop 224x224, kalibrasi ulang
FACE_SCALE = float(os.getenv("FACE_SCALE", "1.7"))        # sisi crop = FACE_SCALE x sisi kotak wajah Haar; KALIBRASI (lihat catatan)
REQUIRE_FACE = os.getenv("REQUIRE_FACE", "1") == "1"       # 1 = tolak foto tanpa wajah terdeteksi
S: dict = {}
LOCK = threading.Lock()  # hook Grad-CAM tidak thread-safe


class LogitWrapper(nn.Module):  # sama dengan notebook
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        out = self.m(x)
        return out[-1] if isinstance(out, (list, tuple)) else out


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))
    yolo = YOLO(MODEL_PATH)
    m = yolo.model.float().cpu().eval()
    for p in m.parameters():
        p.requires_grad_(True)
    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face.empty():
        raise RuntimeError("Haar cascade wajah gagal dimuat.")
    S.update(names=yolo.names, cam=LogitWrapper(m).eval(), layer=m.model[-1].conv, face=face)
    infer(np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8))  # warm-up
    yield


app = FastAPI(title="DS Screening API", lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["POST", "GET"], allow_headers=["*"])


def auth(x_api_key: str = Header(default="")):
    if API_KEY and not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "API key tidak valid.")


def infer(rgb: np.ndarray):
    x = rgb.astype(np.float32) / 255.0
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    with LOCK:
        with torch.no_grad():
            probs = torch.softmax(S["cam"](t), dim=1)[0]
        idx = int(probs.argmax())
        with GradCAM(model=S["cam"], target_layers=[S["layer"]]) as cam:
            heat = cam(input_tensor=t, targets=[ClassifierOutputTarget(idx)])[0]
    return probs.tolist(), idx, show_cam_on_image(x, heat, use_rgb=True)


def detect_faces(bgr: np.ndarray):
    """Kembalikan daftar kotak wajah (x, y, w, h) pada koordinat gambar asli, terbesar dulu."""
    h, w = bgr.shape[:2]
    s = min(1.0, 640 / max(h, w))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    min_side = max(40, int(0.12 * min(gray.shape[:2])))
    faces = S["face"].detectMultiScale(gray, 1.1, 5, minSize=(min_side, min_side))
    boxes = [(int(x / s), int(y / s), int(fw / s), int(fh / s)) for x, y, fw, fh in faces]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def crop_square(bgr: np.ndarray, box=None):
    """Crop persegi tanpa distorsi. Dengan box: berpusat di wajah (sedikit ke atas agar dahi/rambut ikut).
    Tanpa box: center-crop (setara classify_transforms Ultralytics)."""
    h, w = bgr.shape[:2]
    if box is not None:
        x, y, bw, bh = box
        cx, cy = x + bw / 2, y + bh / 2 - 0.10 * bh
        side = int(max(bw, bh) * FACE_SCALE)
    else:
        cx, cy, side = w / 2, h / 2, min(h, w)
    side = max(1, min(side, h, w))  # tidak melebihi gambar -> tidak perlu padding
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    return bgr[y0:y0 + side, x0:x0 + side]


def quality(crop_bgr: np.ndarray, n_faces: int):
    warns = []
    if crop_bgr.shape[0] < IMG_SIZE:
        warns.append({"code": "LOW_RESOLUTION", "message": "Resolusi area wajah rendah; hasil kurang andal. Gunakan foto wajah yang lebih jelas untuk evaluasi berikutnya."})
    gray = cv2.cvtColor(cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD:
        warns.append({"code": "BLURRY", "message": "Foto tampak buram; hasil kurang andal."})
    if n_faces == 0:
        warns.append({"code": "NO_FACE", "message": "Wajah tidak terdeteksi; foto di-crop di tengah dan hasil tidak andal."})
    elif n_faces > 1:
        warns.append({"code": "FACE_COUNT", "message": f"Terdeteksi {n_faces} wajah; model memakai wajah terbesar. Pastikan foto hanya memuat 1 wajah."})
    return warns


def b64jpg(rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


@app.get("/health")
def health():
    return {"status": "ok", "classes": list(S.get("names", {}).values())}


@app.post("/predict", dependencies=[Depends(auth)])
def predict(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Ukuran berkas melebihi 5 MB.")
    if not (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(415, "Format berkas tidak didukung. Gunakan JPEG atau PNG.")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(422, "Berkas gambar rusak atau tidak dapat dibaca.")

    faces = detect_faces(bgr)
    if not faces and REQUIRE_FACE:
        raise HTTPException(422, "Wajah tidak terdeteksi. Gunakan foto frontal, wajah jelas dan tidak terpotong.")
    crop = crop_square(bgr, faces[0] if faces else None)
    warns = quality(crop, len(faces))

    rgb = cv2.cvtColor(cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    probs, idx, overlay = infer(rgb)
    names = S["names"]  # jangan hardcode urutan: ambil dari model
    return {
        "request_id": uuid.uuid4().hex[:12],
        "model": "YOLOv8n-cls",
        "predicted_class": names[idx],
        "confidence": round(probs[idx] * 100, 1),
        "probabilities": {names[i]: round(p * 100, 1) for i, p in enumerate(probs)},
        "warnings": warns,
        "face_detected": bool(faces),
        "model_input_image": b64jpg(rgb),  # tampilkan ini sebagai "Citra asli", sejajar dengan Grad-CAM
        "gradcam_image": b64jpg(overlay),
        "processing_ms": int((time.perf_counter() - t0) * 1000),
    }