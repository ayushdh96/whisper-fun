import os, uuid, json, subprocess, sys
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# --- Load env ---
load_dotenv()
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
RESULT_DIR = os.getenv("RESULT_DIR", "results")
ALLOWED_EXT = set(os.getenv("ALLOWED_EXT", ".mp3,.wav,.m4a,.flac,.mp4,.mov,.mkv").split(","))
MODEL_SIZE = os.getenv("MODEL_SIZE", "small")
DEVICE = os.getenv("DEVICE", "auto")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "auto")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__)

# --- Faster-Whisper model lazy-load (so /healthz works even if torch/cuda isn’t ready) ---
_model = None
def get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        device_arg = "cpu" if DEVICE == "auto" else DEVICE
        compute_arg = "default" if COMPUTE_TYPE == "auto" else COMPUTE_TYPE
        _model = WhisperModel(MODEL_SIZE, device=device_arg, compute_type=compute_arg)
    return _model

def is_video(ext: str) -> bool:
    return ext.lower() in {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".m4v"}

def ffmpeg_extract_audio(in_path: str, out_path: str):
    # Mono, 16 kHz WAV → solid default for ASR
    cmd = ["ffmpeg", "-y", "-i", in_path, "-vn", "-ac", "1", "-ar", "16000", out_path]
    # capture stderr for debugging; if ffmpeg fails, raise
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr[:500]}")

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "asr-backend"}

@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    language = request.form.get("language") or None  # None = auto-detect
    task = request.form.get("task", "transcribe")    # or "translate"

    if not f:
        return jsonify({"error": "file is required"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"extension {ext} not allowed"}), 400

    job_id = str(uuid.uuid4())
    src_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    f.save(src_path)

    # If video, extract audio
    audio_path = src_path
    if is_video(ext):
        audio_path = os.path.join(UPLOAD_DIR, f"{job_id}.wav")
        try:
            ffmpeg_extract_audio(src_path, audio_path)
        except Exception as e:
            return jsonify({"error": "audio extraction failed", "detail": str(e)}), 500

    # Transcribe (synchronous MVP)
    try:
        model = get_model()
        segments, info = model.transcribe(audio_path, language=language, task=task)
        segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        full_text = " ".join(s["text"].strip() for s in segs)

        result = {
            "duration_sec": getattr(info, "duration", None),
            "language": getattr(info, "language", language),
            "segments": segs,
            "text": full_text,
            "meta": {
                "source_filename": f.filename,
                "audio_path": os.path.basename(audio_path),
                "model": MODEL_SIZE,
                "task": task,
            },
        }

        out_path = os.path.join(RESULT_DIR, f"{job_id}.json")
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False)

        return jsonify({"job_id": job_id, "status": "done", "result": result})
    except Exception as e:
        return jsonify({"error": "transcription failed", "detail": str(e)}), 500

def transcribe_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    audio_path = file_path
    if is_video(ext):
        audio_path = os.path.splitext(file_path)[0] + ".wav"
        try:
            ffmpeg_extract_audio(file_path, audio_path)
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return
    try:
        model = get_model()
        segments, info = model.transcribe(audio_path)
        segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
        full_text = " ".join(s["text"].strip() for s in segs)
        print("\nTranscription result:\n")
        print(full_text)
    except Exception as e:
        print(f"Transcription failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        transcribe_file(sys.argv[1])
    else:
        port = int(os.getenv("PORT", "5001"))
        app.run(host="0.0.0.0", port=port, debug=True)