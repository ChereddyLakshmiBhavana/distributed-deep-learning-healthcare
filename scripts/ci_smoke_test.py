"""CI / local smoke test helper.
1. Waits for backend /health
2. POSTs a sample image to /predict/upload
3. Optionally queues an async job and polls its status
"""
import requests
import time
import base64
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

BASE = "http://localhost:5000"

def wait_for_health(timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE}/health", timeout=3)
            if r.status_code == 200:
                print("/health OK", r.json())
                return True
        except Exception:
            pass
        time.sleep(1)
    print("/health did not become ready")
    return False

def post_file(image_path, model="fast_resnet_model"):
    files = {"file": open(image_path, "rb")}
    data = {"model": model}
    r = requests.post(f"{BASE}/predict/upload", files=files, data=data)
    print("POST /predict/upload ->", r.status_code)
    try:
        print(r.json())
    except Exception:
        print(r.text)
    return r

def post_async(image_path, model="fast_resnet_model"):
    b = base64.b64encode(open(image_path, "rb").read()).decode()
    payload = {"image": b, "model": model}
    r = requests.post(f"{BASE}/predict/async", json=payload)
    print("POST /predict/async ->", r.status_code, r.text)
    if r.status_code not in (200, 202):
        return None
    jid = r.json().get("job_id")
    return jid

def poll_job(job_id, timeout=60):
    url = f"{BASE}/distributed/jobs/{job_id}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(url)
        s = r.json()
        print("job", job_id, s.get("status"))
        if s.get("status") == "completed":
            print("result:", s.get("result"))
            return s
        time.sleep(2)
    print("job did not complete in time")
    return None


def ensure_sample_image(sample_path):
    if sample_path.exists():
        return sample_path
    if Image is None:
        raise RuntimeError("Pillow is required to generate a fallback test image")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (224, 224), 128)
    image.save(sample_path)
    print("Created fallback sample image at", sample_path)
    return sample_path

def main():
    sample = Path("data/test_samples/pneumonia_001.jpg")
    sample = ensure_sample_image(sample)
    if not wait_for_health(60):
        raise SystemExit(1)
    post_file(sample)
    jid = post_async(sample)
    if jid:
        poll_job(jid)

if __name__ == "__main__":
    main()
