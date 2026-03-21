import cv2
import numpy as np
import json
import os
import subprocess
import time
import tempfile
import shutil
from PIL import Image
import imagehash


# Settings for the compression algorithm
PHASH_SIMILARITY_THRESHOLD = 0.95
OPTICAL_FLOW_THRESHOLD     = 0.05
CONTEXT_FRAME_INTERVAL_SEC = 3.0
OUTPUT_FPS                 = 12
FACE_SCALE_FACTOR          = 1.1
FACE_MIN_NEIGHBORS         = 5
FACE_MIN_SIZE              = (30, 30)

HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def compute_phash(frame_bgr: np.ndarray) -> imagehash.ImageHash:
    # Convert frame from BGR to RGB then compute perceptual hash
    pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img)


def phash_similarity(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> float:
    # Compare two hashes and return how similar they are (1.0 = identical)
    max_bits = len(h1.hash.flatten())
    diff_bits = h1 - h2
    return 1.0 - (diff_bits / max_bits)


def compute_optical_flow_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    # Calculate how much movement happened between two frames
    if prev_gray is None:
        return 1.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(magnitude))


def detect_faces(frame_bgr: np.ndarray, face_cascade: cv2.CascadeClassifier) -> bool:
    # Check if any face is visible in the frame using Haar cascade
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=FACE_MIN_SIZE
    )
    return len(faces) > 0


def extract_intelligent_frames(video_path: str, segments_json_path: str) -> list:
    # Read the JSON log and pull out only the frames we decided to keep
    with open(segments_json_path, "r") as f:
        segments = json.load(f)

    kept_indices = set(seg["frame_index"] for seg in segments["kept_frames"])

    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in kept_indices:
            frames.append((idx, frame))
        idx += 1
    cap.release()
    return frames


def compress_video(input_path: str,
                   output_video_path: str = "compressed_output.mp4",
                   report_path: str = "compression_report.html",
                   segments_path: str = "segments_kept.json") -> dict:

    start_time = time.time()

    # Open the video and read basic properties
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / source_fps

    print(f"[INFO] Input : {input_path}")
    print(f"[INFO] Frames: {total_frames} @ {source_fps:.1f} fps  ({duration_sec:.1f} s)")
    print(f"[INFO] Resolution: {width}x{height}")

    # Load the Haar face detector
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError("Haar cascade not found. Check OpenCV installation.")

    kept_frames_meta = []
    temp_dir         = tempfile.mkdtemp(prefix="sentio_frames_")

    prev_gray        = None
    last_kept_hash   = None
    last_context_sec = -CONTEXT_FRAME_INTERVAL_SEC

    face_kept    = 0
    motion_drop  = 0
    hash_drop    = 0
    context_kept = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / source_fps
        curr_gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 1: drop frame if too similar to the last kept frame
        curr_hash = compute_phash(frame)
        if last_kept_hash is not None:
            sim = phash_similarity(curr_hash, last_kept_hash)
            if sim >= PHASH_SIMILARITY_THRESHOLD:
                hash_drop += 1
                prev_gray  = curr_gray
                frame_idx += 1
                continue

        # Step 2: measure how much motion is in this frame
        motion_score = compute_optical_flow_score(prev_gray, curr_gray)

        # Step 3: check if there is a face in this frame
        has_face = detect_faces(frame, face_cascade)

        # Step 4: check if we need a context frame to maintain continuity
        needs_context = (timestamp_sec - last_context_sec) >= CONTEXT_FRAME_INTERVAL_SEC

        # Decide whether to keep this frame and why
        keep_reason = None
        if has_face:
            keep_reason = "face_detected"
            face_kept  += 1
        elif motion_score < OPTICAL_FLOW_THRESHOLD and not needs_context:
            motion_drop += 1
            prev_gray    = curr_gray
            frame_idx   += 1
            continue
        elif needs_context:
            keep_reason   = "context_frame"
            context_kept += 1
        else:
            keep_reason = "motion"

        # Save the kept frame as a jpg to temp folder
        frame_file = os.path.join(temp_dir, f"frame_{frame_idx:07d}.jpg")
        cv2.imwrite(frame_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

        kept_frames_meta.append({
            "frame_index":   frame_idx,
            "timestamp_sec": round(timestamp_sec, 4),
            "keep_reason":   keep_reason,
            "motion_score":  round(float(motion_score), 6),
            "has_face":      bool(has_face),
            "phash":         str(curr_hash),
            "saved_file":    frame_file
        })

        last_kept_hash   = curr_hash
        last_context_sec = timestamp_sec
        prev_gray        = curr_gray
        frame_idx       += 1

    cap.release()

    kept_count    = len(kept_frames_meta)
    dropped_count = total_frames - kept_count
    print(f"[INFO] Kept {kept_count}/{total_frames} frames "
          f"(hash_drop={hash_drop}, motion_drop={motion_drop}, "
          f"face_kept={face_kept}, context_kept={context_kept})")

    # Step 5: use ffmpeg to combine saved frames into a compressed mp4
    print("[INFO] Re-encoding with ffmpeg...")
    frame_list_file = os.path.join(temp_dir, "frames.txt")
    with open(frame_list_file, "w") as f:
        for meta in kept_frames_meta:
            f.write(f"file '{meta['saved_file']}'\n")
            f.write(f"duration {1.0/OUTPUT_FPS:.6f}\n")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", frame_list_file,
        "-vf", f"fps={OUTPUT_FPS}",
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        output_video_path
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[WARN] ffmpeg stderr:", result.stderr[-800:])
    else:
        print(f"[INFO] Encoded → {output_video_path}")

    # Remove temporary frame images
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Calculate compression stats
    input_size_bytes  = os.path.getsize(input_path)
    output_size_bytes = os.path.getsize(output_video_path) if os.path.exists(output_video_path) else 0
    reduction_pct     = (1 - output_size_bytes / input_size_bytes) * 100 if input_size_bytes else 0
    elapsed           = time.time() - start_time
    processing_speed  = duration_sec / elapsed if elapsed > 0 else 0

    # Build and save the JSON segment log
    segments_payload = {
        "metadata": {
            "input_file":           input_path,
            "output_file":          output_video_path,
            "source_fps":           source_fps,
            "output_fps":           OUTPUT_FPS,
            "total_frames":         total_frames,
            "kept_frames_count":    kept_count,
            "dropped_frames_count": dropped_count,
            "duration_sec":         round(duration_sec, 2),
            "input_size_bytes":     input_size_bytes,
            "output_size_bytes":    output_size_bytes,
            "reduction_percent":    round(reduction_pct, 2),
            "processing_time_sec":  round(elapsed, 2),
            "processing_speed_x":   round(processing_speed, 2),
            "resolution":           f"{width}x{height}",
            "algorithm_params": {
                "phash_similarity_threshold": PHASH_SIMILARITY_THRESHOLD,
                "optical_flow_threshold":     OPTICAL_FLOW_THRESHOLD,
                "context_frame_interval_sec": CONTEXT_FRAME_INTERVAL_SEC
            }
        },
        "kept_frames": [
            {k: v for k, v in m.items() if k != "saved_file"}
            for m in kept_frames_meta
        ]
    }

    with open(segments_path, "w") as f:
        json.dump(segments_payload, f, indent=2)
    print(f"[INFO] Segment log → {segments_path}")

    # Generate the HTML report
    _generate_html_report(segments_payload, report_path, kept_frames_meta)
    print(f"[INFO] Report → {report_path}")

    print(f"\n{'='*55}")
    print(f"  Size reduction : {reduction_pct:.1f}%")
    print(f"  Frames kept    : {kept_count} / {total_frames}")
    print(f"  Processing time: {elapsed:.1f} s  ({processing_speed:.1f}x real-time)")
    print(f"{'='*55}\n")

    return segments_payload["metadata"]


def _generate_html_report(data: dict, report_path: str, kept_frames_meta: list):
    # Build a colour-coded timeline and stats table, save as offline HTML
    meta = data["metadata"]

    reason_colors = {
        "face_detected": "#4CAF50",
        "context_frame": "#2196F3",
        "motion":        "#FF9800",
    }

    timeline_items = ""
    for m in data["kept_frames"][:200]:
        color = reason_colors.get(m["keep_reason"], "#999")
        timeline_items += (
            f'<div class="tick" style="background:{color}" '
            f'title="t={m["timestamp_sec"]}s | {m["keep_reason"]} | '
            f'motion={m["motion_score"]:.3f}"></div>'
        )

    kept_pct = round(meta["kept_frames_count"] / meta["total_frames"] * 100, 1) if meta["total_frames"] else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sentio Mind - Compression Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Arial, sans-serif; background: #0f1117; color: #e0e0e0; padding: 24px; }}
  h1 {{ color: #00bcd4; font-size: 1.8rem; margin-bottom: 4px; }}
  h2 {{ color: #00bcd4; font-size: 1.1rem; margin: 24px 0 10px; border-left: 3px solid #00bcd4; padding-left: 10px; }}
  .subtitle {{ color: #888; font-size: 0.85rem; margin-bottom: 24px; }}
  .cards {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; }}
  .card {{ background: #1e2130; border-radius: 10px; padding: 20px 24px; flex: 1; min-width: 160px; text-align: center; }}
  .card .val {{ font-size: 2rem; font-weight: bold; color: #00e5ff; }}
  .card .lbl {{ font-size: 0.78rem; color: #aaa; margin-top: 4px; }}
  .bar-wrap {{ background: #1e2130; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; }}
  .bar-bg {{ background: #333; border-radius: 4px; height: 22px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 4px; background: linear-gradient(90deg, #00bcd4, #4caf50); display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; font-size: 0.8rem; font-weight: bold; color: #000; }}
  .legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 10px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.8rem; }}
  .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
  .timeline {{ display: flex; flex-wrap: wrap; gap: 2px; background: #1e2130; border-radius: 8px; padding: 14px; }}
  .tick {{ width: 6px; height: 20px; border-radius: 2px; opacity: 0.85; cursor: default; }}
  .algo-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
  .algo-table th, .algo-table td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #2a2d3a; font-size: 0.85rem; }}
  .algo-table th {{ background: #1a1d2e; color: #00bcd4; font-weight: 600; }}
  .algo-table tr:hover td {{ background: #1e2130; }}
  footer {{ margin-top: 32px; color: #555; font-size: 0.78rem; text-align: center; }}
  .green {{ color: #4caf50; }} .orange {{ color: #ff9800; }} .blue {{ color: #2196f3; }}
</style>
</head>
<body>
<h1>Sentio Mind - Smart Compression Report</h1>
<div class="subtitle">Input: {meta["input_file"]} &nbsp;|&nbsp; Generated by solution.py</div>

<div class="cards">
  <div class="card"><div class="val green">{meta["reduction_percent"]:.1f}%</div><div class="lbl">File Size Reduction</div></div>
  <div class="card"><div class="val">{meta["kept_frames_count"]}</div><div class="lbl">Frames Kept / {meta["total_frames"]}</div></div>
  <div class="card"><div class="val orange">{meta["processing_speed_x"]:.1f}x</div><div class="lbl">Real-time Speed</div></div>
  <div class="card"><div class="val blue">{meta["output_fps"]} fps</div><div class="lbl">Output Frame Rate</div></div>
  <div class="card"><div class="val">{meta["duration_sec"]:.0f}s</div><div class="lbl">Video Duration</div></div>
  <div class="card"><div class="val green">{meta["processing_time_sec"]:.1f}s</div><div class="lbl">Processing Time</div></div>
</div>

<h2>File Size Comparison</h2>
<div class="bar-wrap">
  <div style="font-size:0.82rem; color:#aaa; margin-bottom:8px;">
    Input: <b style="color:#fff">{meta["input_size_bytes"]//1024//1024} MB</b>
    &nbsp;to&nbsp;
    Output: <b style="color:#4caf50">{meta["output_size_bytes"]//1024//1024} MB</b>
  </div>
  <div style="margin-bottom:8px;">
    <div style="font-size:0.78rem;color:#aaa;margin-bottom:3px;">Input size</div>
    <div class="bar-bg"><div class="bar-fill" style="width:100%">100%</div></div>
  </div>
  <div>
    <div style="font-size:0.78rem;color:#aaa;margin-bottom:3px;">Output size</div>
    <div class="bar-bg"><div class="bar-fill" style="width:{100-meta['reduction_percent']:.1f}%">{100-meta['reduction_percent']:.1f}%</div></div>
  </div>
</div>

<h2>Algorithm Steps</h2>
<table class="algo-table">
  <tr><th>#</th><th>Step</th><th>Technique</th><th>Parameter</th><th>Purpose</th></tr>
  <tr><td>1</td><td>pHash Filter</td><td>Perceptual Hash</td><td>95% similar = drop</td><td>Remove duplicate static frames</td></tr>
  <tr><td>2</td><td>Motion Filter</td><td>Farneback Optical Flow</td><td>score &lt; 0.05 = drop</td><td>Remove empty idle scenes</td></tr>
  <tr><td>3</td><td>Face Detection</td><td>Haar Cascade</td><td>Any face = keep</td><td>Keep all frames with humans</td></tr>
  <tr><td>4</td><td>Context Frame</td><td>Time-based</td><td>Every 3 seconds</td><td>Scene continuity</td></tr>
  <tr><td>5</td><td>Re-encode</td><td>ffmpeg H.264</td><td>12 fps, CRF 23</td><td>Final compressed output</td></tr>
</table>

<h2>Frame Timeline</h2>
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#4CAF50"></div>Face Detected</div>
  <div class="legend-item"><div class="legend-dot" style="background:#2196F3"></div>Context Frame</div>
  <div class="legend-item"><div class="legend-dot" style="background:#FF9800"></div>Motion</div>
</div>
<div class="timeline">{timeline_items}</div>
<p style="font-size:0.75rem;color:#555;margin-top:6px;">Each bar is one kept frame. Hover to see details. Showing first 200 kept frames.</p>

<h2>Summary</h2>
<table class="algo-table">
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Input resolution</td><td>{meta["resolution"]}</td></tr>
  <tr><td>Source FPS to Output FPS</td><td>{meta["source_fps"]:.1f} to {meta["output_fps"]}</td></tr>
  <tr><td>Frames kept</td><td>{meta["kept_frames_count"]} / {meta["total_frames"]} ({kept_pct}%)</td></tr>
  <tr><td>Frames dropped</td><td>{meta["dropped_frames_count"]}</td></tr>
  <tr><td>Input file size</td><td>{meta["input_size_bytes"]/1024/1024:.2f} MB</td></tr>
  <tr><td>Output file size</td><td>{meta["output_size_bytes"]/1024/1024:.2f} MB</td></tr>
  <tr><td>File size reduction</td><td class="green"><b>{meta["reduction_percent"]:.2f}%</b></td></tr>
  <tr><td>Processing time</td><td>{meta["processing_time_sec"]} s</td></tr>
  <tr><td>Processing speed</td><td>{meta["processing_speed_x"]}x real-time</td></tr>
</table>

<footer>Sentio Mind · Smart Behavioral Video Compression · solution.py</footer>
</body>
</html>"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    import sys

    input_video = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\DELL\Downloads\files\Class_8_cctv_video_1.mov"

    if not os.path.exists(input_video):
        print(f"[ERROR] Input video not found: {input_video}")
        print("Usage: python solution.py <path_to_video>")
        sys.exit(1)

    stats = compress_video(
        input_path=input_video,
        output_video_path="compressed_output.mp4",
        report_path="compression_report.html",
        segments_path="segments_kept.json"
    )

    print("Done! Deliverables generated:")
    print("  -> compressed_output.mp4")
    print("  -> compression_report.html")
    print("  -> segments_kept.json")