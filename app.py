import os
import io
import math
import json
import zipfile
import tempfile
import traceback
from flask import Flask, request, jsonify, send_file, Blueprint
from flask_cors import CORS

try:
    import pyembroidery
except ImportError:
    pyembroidery = None

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None


MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

SUPPORTED_INPUT_FORMATS = [".pes", ".dst", ".jef", ".vp3", ".exp", ".hus", ".vip", ".xxx"]
SUPPORTED_OUTPUT_FORMATS = [".pes", ".dst", ".jef", ".vp3", ".exp", ".hus", ".vip", ".xxx", ".svg", ".png"]

MIME_TYPES = {
    ".pes": "application/octet-stream",
    ".dst": "application/octet-stream",
    ".jef": "application/octet-stream",
    ".vp3": "application/octet-stream",
    ".exp": "application/octet-stream",
    ".hus": "application/octet-stream",
    ".vip": "application/octet-stream",
    ".xxx": "application/octet-stream",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".zip": "application/zip",
}

# URL prefix — Flask routes are at root, Express proxy strips the path prefix
URL_PREFIX = ""

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

CORS(app, expose_headers=[
    "stitch_count", "color_count", "width_mm", "height_mm",
    "estimated_time", "sections_count", "layout"
])

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    response.headers.add(
        "Access-Control-Expose-Headers",
        "stitch_count,color_count,estimated_time,width_mm,height_mm,sections_count,layout"
    )
    return response

bp = Blueprint("bobbin", __name__, url_prefix=URL_PREFIX)


def get_design_info(pattern):
    stitch_count = 0
    jump_count = 0
    trim_count = 0
    color_changes = []
    current_color_idx = 0
    current_color_stitch_count = 0

    threads = pattern.threadlist if pattern.threadlist else []

    for stitch in pattern.stitches:
        cmd = stitch[2] & 0xF0
        if cmd == pyembroidery.STITCH:
            stitch_count += 1
            current_color_stitch_count += 1
        elif cmd == pyembroidery.JUMP:
            jump_count += 1
        elif cmd == pyembroidery.TRIM:
            trim_count += 1
        elif cmd == pyembroidery.COLOR_CHANGE:
            if current_color_idx < len(threads):
                t = threads[current_color_idx]
                c = getattr(t, 'color', 0x000000)
                color_hex = "#{:02X}{:02X}{:02X}".format(
                    (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)
            else:
                color_hex = "#000000"
            color_changes.append({"color": color_hex, "stitch_count": current_color_stitch_count})
            current_color_idx += 1
            current_color_stitch_count = 0
        elif cmd == pyembroidery.END:
            break

    if current_color_stitch_count > 0:
        if current_color_idx < len(threads):
            t = threads[current_color_idx]
            c = getattr(t, 'color', 0x000000)
            color_hex = "#{:02X}{:02X}{:02X}".format(
                (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)
        else:
            color_hex = "#000000"
        color_changes.append({"color": color_hex, "stitch_count": current_color_stitch_count})

    color_count = len(threads) if threads else max(1, len(color_changes))

    thread_colors = []
    for t in threads:
        c = getattr(t, 'color', 0x000000)
        thread_colors.append("#{:02X}{:02X}{:02X}".format(
            (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF))

    xs = [s[0] for s in pattern.stitches if (s[2] & 0xF0) in (pyembroidery.STITCH, pyembroidery.JUMP)]
    ys = [s[1] for s in pattern.stitches if (s[2] & 0xF0) in (pyembroidery.STITCH, pyembroidery.JUMP)]

    if xs and ys:
        width_mm = (max(xs) - min(xs)) / 10.0
        height_mm = (max(ys) - min(ys)) / 10.0
    else:
        width_mm = 0.0
        height_mm = 0.0

    estimated_time = stitch_count / 400.0

    return {
        "stitch_count": stitch_count,
        "color_count": color_count,
        "width_mm": round(width_mm, 2),
        "height_mm": round(height_mm, 2),
        "estimated_time_minutes": round(estimated_time, 2),
        "jump_count": jump_count,
        "trim_count": trim_count,
        "thread_colors": thread_colors,
        "color_changes": color_changes,
    }


def read_uploaded_file(file_storage, allowed_extensions=None):
    filename = file_storage.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    if allowed_extensions and ext not in allowed_extensions:
        return None, ext, "Unsupported file format: {}".format(ext)
    data = file_storage.read()
    if len(data) > MAX_FILE_SIZE:
        return None, ext, "File too large. Maximum size is 50MB."
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name, ext, None


# ─── HEALTH ──────────────────────────────────────────────────────────────────

@bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@bp.route("/test", methods=["GET"])
def test():
    """Browser-friendly connectivity test — returns server info and CORS headers."""
    import sys
    return jsonify({
        "status": "ok",
        "message": "Bobbin Embroidery API is reachable",
        "python": sys.version,
        "libraries": {
            "pyembroidery": pyembroidery is not None,
            "opencv":       cv2 is not None,
            "pillow":       Image is not None,
            "numpy":        np is not None,
        },
        "endpoints": [
            "GET  /health",
            "GET  /test",
            "GET  /formats",
            "POST /convert",
            "POST /digitize",
            "POST /preview",
            "POST /split",
            "POST /info",
            "POST /resize",
            "POST /rotate",
            "POST /merge",
        ],
        "cors": "enabled — all origins accepted",
    })


@bp.route("/formats", methods=["GET"])
def formats():
    return jsonify({
        "input_formats": SUPPORTED_INPUT_FORMATS,
        "output_formats": SUPPORTED_OUTPUT_FORMATS
    })


# ─── CONVERT ─────────────────────────────────────────────────────────────────

@bp.route("/convert", methods=["POST"])
def convert():
    if pyembroidery is None:
        return jsonify({"error": "pyembroidery not installed"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    target_format = request.form.get("target_format", "").lower().strip()
    if not target_format:
        return jsonify({"error": "target_format parameter is required"}), 400
    if not target_format.startswith("."):
        target_format = "." + target_format
    if target_format not in SUPPORTED_OUTPUT_FORMATS:
        return jsonify({"error": "Unsupported output format: {}".format(target_format)}), 415

    tmp_in, in_ext, err = read_uploaded_file(request.files["file"], SUPPORTED_INPUT_FORMATS)
    if err:
        status = 415 if "format" in err.lower() else 400
        return jsonify({"error": err}), status

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=target_format)
    tmp_out.close()

    try:
        pattern = pyembroidery.read(tmp_in)
        if pattern is None:
            return jsonify({"error": "Could not read embroidery file"}), 400

        info = get_design_info(pattern)
        pyembroidery.write(pattern, tmp_out.name)

        mime = MIME_TYPES.get(target_format, "application/octet-stream")
        response = send_file(tmp_out.name, mimetype=mime, as_attachment=True,
                             download_name="converted" + target_format)
        response.headers["stitch_count"] = str(info["stitch_count"])
        response.headers["color_count"] = str(info["color_count"])
        response.headers["width_mm"] = str(info["width_mm"])
        response.headers["height_mm"] = str(info["height_mm"])
        response.headers["estimated_time"] = str(info["estimated_time_minutes"])
        return response
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Processing error: {}".format(str(e))}), 500
    finally:
        _cleanup(tmp_in)


# ─── DIGITIZE ────────────────────────────────────────────────────────────────

def _rgb_distance(c1, c2):
    """Euclidean distance between two (R, G, B) tuples."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def _quantize_colors(pil_img, n_colors):
    """
    Reduce pil_img to at most n_colors using PIL palette quantization, then
    merge any palette entries within 30 RGB units of each other.
    Returns canonical_palette: {index → (r,g,b)}, and q_arr: numpy uint8 array
    of pixel → palette index.
    """
    rgb = pil_img.convert("RGB")

    # Quantize to a palette image
    try:
        quantized = rgb.quantize(colors=n_colors, dither=0)
    except TypeError:
        quantized = rgb.quantize(colors=n_colors)

    palette_raw = quantized.getpalette()  # flat [r,g,b, …], length varies by PIL version
    if not palette_raw:
        # Fallback: treat image as single color block
        avg = np.array(rgb).mean(axis=(0, 1)).astype(int)
        return {0: (int(avg[0]), int(avg[1]), int(avg[2]))}, np.zeros(
            (pil_img.height, pil_img.width), dtype=np.uint8)

    q_arr = np.array(quantized, dtype=np.uint8)

    # Determine which palette indices are actually used in the image
    used_indices = sorted(set(q_arr.flatten().tolist()))

    # Build palette only for used indices (capped at n_colors entries)
    palette = {}
    for idx in used_indices:
        if idx * 3 + 2 < len(palette_raw):
            palette[idx] = (palette_raw[idx*3], palette_raw[idx*3+1], palette_raw[idx*3+2])

    if not palette:
        return {0: (0, 0, 0)}, np.zeros((pil_img.height, pil_img.width), dtype=np.uint8)

    # Merge similar palette entries (within 30 RGB distance)
    # Build merge_map: old_idx → canonical_idx
    all_indices = list(palette.keys())
    merge_map = {idx: idx for idx in all_indices}

    for i, idx_i in enumerate(all_indices):
        for idx_j in all_indices[i+1:]:
            if merge_map[idx_j] == idx_j and _rgb_distance(palette[idx_i], palette[idx_j]) < 30:
                merge_map[idx_j] = merge_map[idx_i]

    # Remap pixels using a LUT (max palette index + 1 entries)
    max_idx = max(all_indices) + 1
    lut = np.arange(max_idx, dtype=np.uint8)
    for old, new in merge_map.items():
        if old < max_idx:
            lut[old] = new

    safe_arr = np.clip(q_arr, 0, max_idx - 1)
    q_arr = lut[safe_arr]

    # Collect surviving unique canonical entries
    canonical_palette = {merge_map[idx]: palette[idx]
                         for idx in all_indices if merge_map[idx] == idx}

    return canonical_palette, q_arr


def _detect_background(pil_img, q_arr):
    """
    Detect the background color index by sampling the four corner pixels
    of the quantized image.
    Returns the most common corner index (int), or None if all corners differ.
    """
    h, w = q_arr.shape
    corners = [
        q_arr[0, 0], q_arr[0, w - 1],
        q_arr[h - 1, 0], q_arr[h - 1, w - 1],
    ]
    from collections import Counter
    counts = Counter(corners)
    bg_idx, freq = counts.most_common(1)[0]
    return bg_idx if freq >= 2 else None


@bp.route("/digitize", methods=["POST"])
def digitize():
    print(f"DIGITIZE REQUEST: files={list(request.files.keys())}, form={dict(request.form)}", flush=True)

    if pyembroidery is None:
        return jsonify({"error": "pyembroidery not installed"}), 500
    if cv2 is None:
        return jsonify({"error": "opencv not installed"}), 500
    if Image is None:
        return jsonify({"error": "Pillow not installed"}), 500

    if "file" not in request.files:
        print("DIGITIZE ERROR: No file in request.files", flush=True)
        return jsonify({"error": "No file provided"}), 400

    output_format = request.form.get("output_format", "pes").lower().strip()
    if not output_format.startswith("."):
        output_format = "." + output_format
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        return jsonify({"error": "Unsupported output format: {}".format(output_format)}), 415

    stitch_type = request.form.get("stitch_type", "auto").lower().strip()
    hoop_width_mm  = float(request.form.get("hoop_width_mm", 101.6))
    hoop_height_mm = float(request.form.get("hoop_height_mm", 101.6))
    max_stitch_length = float(request.form.get("max_stitch_length", 4.0))   # updated default
    min_stitch_length = float(request.form.get("min_stitch_length", 1.5))
    density = float(request.form.get("density", 3.0))                        # updated default
    color_count_param = min(16, max(1, int(request.form.get("color_count", 8))))
    do_simplify = request.form.get("simplify", "true").lower() not in ("false", "0", "no")

    allowed_image_ext = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
    tmp_in, in_ext, err = read_uploaded_file(request.files["file"], allowed_image_ext)
    if err:
        status = 415 if "format" in err.lower() else 400
        return jsonify({"error": err}), status

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=output_format)
    tmp_out.close()

    try:
        # ── 1. Preprocess image ────────────────────────────────────────────────
        pil_img = Image.open(tmp_in)

        # Flatten any transparency onto a white background
        if pil_img.mode in ("RGBA", "LA", "PA"):
            if pil_img.mode == "PA":
                pil_img = pil_img.convert("RGBA")
            white = Image.new("RGB", pil_img.size, (255, 255, 255))
            white.paste(pil_img, mask=pil_img.split()[-1])
            pil_img = white
        elif pil_img.mode == "P" and "transparency" in pil_img.info:
            pil_img = pil_img.convert("RGBA")
            white = Image.new("RGB", pil_img.size, (255, 255, 255))
            white.paste(pil_img, mask=pil_img.split()[-1])
            pil_img = white
        else:
            pil_img = pil_img.convert("RGB")

        img_rgb = np.array(pil_img, dtype=np.uint8)
        img_h, img_w = img_rgb.shape[:2]

        # Bilateral filter — d=5 preserves fine stripe / line details better than d=9
        img_filtered = cv2.bilateralFilter(img_rgb, d=5, sigmaColor=100, sigmaSpace=100)

        # ── 2. K-means color clustering ────────────────────────────────────────
        k = min(16, max(2, color_count_param))
        pixels = np.float32(img_filtered.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _, labels_flat, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        centers   = np.uint8(centers)                              # k × 3 RGB
        labels_2d = labels_flat.flatten().reshape(img_h, img_w)   # pixel → cluster

        # ── 3. Background detection (edge-sampling) ────────────────────────────
        # Sample 20 evenly-spaced pixels along each of the 4 edges (80 total).
        # Any cluster appearing in > 30% of edge pixels is treated as background.
        from collections import Counter as _Counter
        _edge_samples = []
        _n = 20
        for _i in range(_n):
            _t = int(_i * (img_w - 1) / max(_n - 1, 1))
            _edge_samples.append(int(labels_2d[0,           _t]))        # top
            _edge_samples.append(int(labels_2d[img_h - 1,  _t]))        # bottom
        for _i in range(_n):
            _t = int(_i * (img_h - 1) / max(_n - 1, 1))
            _edge_samples.append(int(labels_2d[_t,          0]))         # left
            _edge_samples.append(int(labels_2d[_t, img_w - 1]))         # right

        _edge_total  = len(_edge_samples)
        _edge_counts = _Counter(_edge_samples)
        _threshold   = 0.30                                              # 30% of edge pixels
        bg_clusters  = {cidx for cidx, cnt in _edge_counts.items()
                        if cnt / _edge_total >= _threshold}

        # Also exclude any K-means centre that is light (cream/beige/ivory/off-white):
        # R > 220 AND G > 200 AND B > 180
        _light_clusters = {
            cidx for cidx in range(k)
            if int(centers[cidx][0]) > 220
            and int(centers[cidx][1]) > 200
            and int(centers[cidx][2]) > 180
        }
        skip_clusters = bg_clusters | _light_clusters

        # Keep the single legacy name for logging / backward compat
        bg_cluster = next(iter(bg_clusters)) if len(bg_clusters) == 1 else (
            _edge_counts.most_common(1)[0][0] if bg_clusters else None
        )
        print(
            f"DIGITIZE: k={k}  bg_clusters={bg_clusters}  light_clusters={_light_clusters}"
            f"  image={img_w}×{img_h}px",
            flush=True,
        )

        # ── 4. Hoop constants  (1 pyembroidery unit = 0.1 mm) ─────────────────
        hoop_w_u = hoop_width_mm  * 10
        hoop_h_u = hoop_height_mm * 10
        SATIN_ROW_U  = 5    # 0.5 mm satin row spacing  (~0.45 mm, nearest integer unit)
        TATAMI_ROW_U = 5    # 0.5 mm tatami row spacing
        STITCH_2MM   = 20   # 2 mm running-stitch spacing
        STITCH_4MM   = 40   # 4 mm tatami stitch length
        MIN_SHAPE_U  = 8    # 0.8 mm minimum shape bbox side — low enough to catch thin 'I'/'E'

        # ── 5. Contour collection with min-area retry ──────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        def _collect(min_area):
            out = {}
            for cidx in range(k):
                if cidx in skip_clusters:
                    continue
                mask = ((labels_2d == cidx).astype(np.uint8) * 255)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                raw, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                valid  = [c for c in raw if cv2.contourArea(c) >= min_area]
                if valid:
                    out[cidx] = valid
            return out

        def _bbox_px(cc):
            xs = [int(pt[0][0]) for cl in cc.values() for c in cl for pt in c]
            ys = [int(pt[0][1]) for cl in cc.values() for c in cl for pt in c]
            if not xs:
                return 0, 0, 0, 0, 0, 0
            return min(xs), max(xs), min(ys), max(ys), max(max(xs)-min(xs),1), max(max(ys)-min(ys),1)

        color_contours = None
        found_any      = False
        for min_area in [200, 50, 10, 8]:
            cc = _collect(min_area)
            if cc:
                found_any = True
                _, _, _, _, pw, ph = _bbox_px(cc)
                if pw >= 10 and ph >= 10:
                    color_contours = cc
                    print(f"DIGITIZE: accepted min_area={min_area} bbox={pw}×{ph}px", flush=True)
                    break
                print(f"DIGITIZE: bbox {pw}×{ph}px too small, retrying min_area={min_area}", flush=True)

        if not found_any:
            return jsonify({"error": (
                "No design elements detected — try an image with clearer edges "
                "and higher contrast"
            )}), 400
        if color_contours is None:
            return jsonify({"error": "Scaling error — design produced at incorrect size"}), 500

        # ── 6. Scale & centering ───────────────────────────────────────────────
        px_min_x, px_max_x, px_min_y, px_max_y, px_w, px_h = _bbox_px(color_contours)
        scale   = min(hoop_w_u / px_w, hoop_h_u / px_h) * 0.9
        off_x   = (hoop_w_u - px_w * scale) / 2
        off_y   = (hoop_h_u - px_h * scale) / 2

        def px_to_emb(px, py):
            return (
                int((px - px_min_x) * scale + off_x),
                int((py - px_min_y) * scale + off_y),
            )

        def px_w_to_mm(w_px):
            return w_px * scale / 10.0

        # ── 7. Stitch helpers ──────────────────────────────────────────────────

        def tie_on(pat, ex, ey):
            """3 locking stitches at thread start."""
            for _ in range(3):
                pat.add_stitch_absolute(pyembroidery.STITCH, ex + 5, ey)
                pat.add_stitch_absolute(pyembroidery.STITCH, ex, ey)

        def tie_off(pat, ex, ey):
            """3 locking stitches at thread end."""
            for _ in range(3):
                pat.add_stitch_absolute(pyembroidery.STITCH, ex + 5, ey)
                pat.add_stitch_absolute(pyembroidery.STITCH, ex, ey)

        def running_outline(pat, contour, stitch_u=STITCH_2MM):
            """Running stitches every stitch_u units around a closed contour."""
            pts  = [px_to_emb(int(p[0][0]), int(p[0][1])) for p in contour]
            if len(pts) < 2:
                return
            loop = pts + [pts[0]]   # close the path
            pat.add_stitch_absolute(pyembroidery.STITCH, loop[0][0], loop[0][1])
            prev = loop[0]
            for pt in loop[1:]:
                dx, dy = pt[0] - prev[0], pt[1] - prev[1]
                dist   = math.sqrt(dx * dx + dy * dy)
                if dist >= stitch_u:
                    steps = max(1, int(dist / stitch_u))
                    for i in range(1, steps + 1):
                        pat.add_stitch_absolute(
                            pyembroidery.STITCH,
                            prev[0] + int(dx * i / steps),
                            prev[1] + int(dy * i / steps),
                        )
                elif dist > 5:
                    pat.add_stitch_absolute(pyembroidery.STITCH, pt[0], pt[1])
                prev = pt

        def satin_along_path(pat, contour, cluster_mask_2d, stitch_u=STITCH_2MM):
            """Satin columns perpendicular to the contour path direction.
            For each stitch position along the path, casts perpendicular rays through
            the cluster mask to measure actual stroke width, then stitches across it.
            This makes text and thin outlines look like real embroidered letters."""
            pts = [px_to_emb(int(p[0][0]), int(p[0][1])) for p in contour]
            if len(pts) < 3:
                return
            loop   = pts + [pts[0]]
            toggle = True
            MAX_R  = 80   # max half-width to search: 80 units = 8 mm

            prev = loop[0]
            for pt in loop[1:]:
                dx, dy   = pt[0] - prev[0], pt[1] - prev[1]
                seg_len  = math.sqrt(dx*dx + dy*dy)
                if seg_len < 1:
                    prev = pt
                    continue

                tx, ty = dx / seg_len, dy / seg_len  # unit tangent
                nx, ny = -ty, tx                      # unit normal (left of tangent)

                steps = max(1, int(seg_len / stitch_u))
                for i in range(steps):
                    t_pos = i * stitch_u
                    if t_pos >= seg_len:
                        break
                    x0 = prev[0] + tx * t_pos
                    y0 = prev[1] + ty * t_pos

                    # Find stroke extent in +normal direction through cluster mask
                    pos_end = 0
                    for r in range(MAX_R):
                        px_c, py_c = emb_to_px_f(x0 + nx * r, y0 + ny * r)
                        ix, iy = int(round(px_c)), int(round(py_c))
                        if 0 <= ix < img_w and 0 <= iy < img_h and cluster_mask_2d[iy, ix] > 0:
                            pos_end = r
                        else:
                            break

                    # Find stroke extent in -normal direction through cluster mask
                    neg_end = 0
                    for r in range(MAX_R):
                        px_c, py_c = emb_to_px_f(x0 - nx * r, y0 - ny * r)
                        ix, iy = int(round(px_c)), int(round(py_c))
                        if 0 <= ix < img_w and 0 <= iy < img_h and cluster_mask_2d[iy, ix] > 0:
                            neg_end = r
                        else:
                            break

                    left_x  = int(x0 + nx * pos_end)
                    left_y  = int(y0 + ny * pos_end)
                    right_x = int(x0 - nx * neg_end)
                    right_y = int(y0 - ny * neg_end)

                    span = math.sqrt((left_x - right_x)**2 + (left_y - right_y)**2)
                    if span >= 5:   # minimum column width 0.5 mm
                        if toggle:
                            pat.add_stitch_absolute(pyembroidery.STITCH, left_x,  left_y)
                            pat.add_stitch_absolute(pyembroidery.STITCH, right_x, right_y)
                        else:
                            pat.add_stitch_absolute(pyembroidery.STITCH, right_x, right_y)
                            pat.add_stitch_absolute(pyembroidery.STITCH, left_x,  left_y)
                        toggle = not toggle

                prev = pt

        def emb_to_px_f(ex, ey):
            """Convert embroidery units to floating-point pixel coords for polygon tests."""
            return (
                (ex - off_x) / scale + px_min_x,
                (ey - off_y) / scale + px_min_y,
            )

        def satin_fill(pat, contour, xs_e, ys_e, row_u=SATIN_ROW_U):
            """Satin rows clipped to the actual contour via pointPolygonTest.
            Finds the true left/right edge of the shape at each row."""
            mn_x, mx_x = min(xs_e), max(xs_e)
            mn_y, mx_y = min(ys_e), max(ys_e)
            contour_f   = contour.astype(np.float32)
            toggle = True
            y_e = mn_y
            while y_e <= mx_y:
                _, py = emb_to_px_f(mn_x, y_e)
                # Scan left→right for first inside point
                left_x = None
                x_e = mn_x
                while x_e <= mx_x:
                    px_t, _ = emb_to_px_f(x_e, y_e)
                    if cv2.pointPolygonTest(contour_f, (float(px_t), float(py)), False) >= 0:
                        left_x = x_e
                        break
                    x_e += 1
                # Scan right→left for last inside point
                right_x = None
                x_e = mx_x
                while x_e >= mn_x:
                    px_t, _ = emb_to_px_f(x_e, y_e)
                    if cv2.pointPolygonTest(contour_f, (float(px_t), float(py)), False) >= 0:
                        right_x = x_e
                        break
                    x_e -= 1
                if left_x is not None and right_x is not None and right_x >= left_x:
                    if toggle:
                        pat.add_stitch_absolute(pyembroidery.STITCH, left_x,  y_e)
                        pat.add_stitch_absolute(pyembroidery.STITCH, right_x, y_e)
                    else:
                        pat.add_stitch_absolute(pyembroidery.STITCH, right_x, y_e)
                        pat.add_stitch_absolute(pyembroidery.STITCH, left_x,  y_e)
                y_e   += row_u
                toggle = not toggle

        def tatami_fill(pat, contour, row_u=TATAMI_ROW_U, stitch_u=STITCH_4MM, angle_deg=45):
            """Tatami fill at a given fill angle (default 45° — industry standard).
            Rotates the scan grid to the requested angle, tests each candidate point
            in the original coordinate system, then emits the rotated stitch."""
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < 1 or bh < 1:
                return
            contour_f = contour.astype(np.float32)

            # Get all contour vertices in embroidery coords
            emb_verts = np.array(
                [px_to_emb(int(p[0][0]), int(p[0][1])) for p in contour],
                dtype=np.float64,
            )
            cx_e = float(emb_verts[:, 0].mean())
            cy_e = float(emb_verts[:, 1].mean())

            # Rotate each vertex by -angle so the scan runs horizontally
            a  = math.radians(angle_deg)
            ca, sa = math.cos(a), math.sin(a)

            def rot_fwd(ex, ey):   # original → rotated frame
                dx, dy = ex - cx_e, ey - cy_e
                return cx_e + dx*ca + dy*sa, cy_e - dx*sa + dy*ca

            def rot_inv(rx, ry):   # rotated frame → original
                dx, dy = rx - cx_e, ry - cy_e
                return cx_e + dx*ca - dy*sa, cy_e + dx*sa + dy*ca

            rot_verts = np.array([rot_fwd(v[0], v[1]) for v in emb_verts])
            rx_min, rx_max = rot_verts[:, 0].min(), rot_verts[:, 0].max()
            ry_min, ry_max = rot_verts[:, 1].min(), rot_verts[:, 1].max()

            row = 0
            ry = ry_min
            while ry <= ry_max:
                row_offset = (stitch_u // 2) if row % 2 == 1 else 0
                rx = rx_min + row_offset
                while rx <= rx_max:
                    ex, ey = rot_inv(rx, ry)        # back to original emb coords
                    px_t, py_t = emb_to_px_f(ex, ey)
                    if cv2.pointPolygonTest(contour_f, (float(px_t), float(py_t)), False) >= 0:
                        pat.add_stitch_absolute(pyembroidery.STITCH, int(ex), int(ey))
                    rx += stitch_u
                ry += row_u
                row += 1

        # ── 8. Build pattern ───────────────────────────────────────────────────
        pattern      = pyembroidery.EmbPattern()
        first_thread = True
        any_stitches = False

        for cidx in sorted(color_contours.keys()):
            rgb = tuple(int(x) for x in centers[cidx])

            thread       = pyembroidery.EmbThread()
            thread.color = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
            thread.name  = "K#{:02X}{:02X}{:02X}".format(*rgb)
            pattern.add_thread(thread)

            if not first_thread:
                # COLOR_CHANGE signals a thread swap; TRIM cuts cleanly before the new color
                pattern.add_stitch_absolute(pyembroidery.COLOR_CHANGE, 0, 0)
                pattern.add_stitch_absolute(pyembroidery.TRIM, 0, 0)
            first_thread = False

            # Cluster mask — precomputed once per color, reused for fill-density + satin-along-path
            cluster_px_mask = (labels_2d == cidx).astype(np.uint8) * 255

            for contour in color_contours[cidx]:
                # Optional contour simplification
                if do_simplify:
                    simplified = cv2.approxPolyDP(contour, epsilon=1.5, closed=True)
                    # If simplification produced a degenerate non-convex shape with fewer
                    # than 6 points (e.g. building outline, awning), keep the original
                    # to avoid self-intersecting paths that create crossing stitch lines.
                    if len(simplified) < 6 and not cv2.isContourConvex(simplified):
                        pass   # keep original contour
                    else:
                        contour = simplified
                if len(contour) < 2:
                    continue

                emb_pts = [px_to_emb(int(p[0][0]), int(p[0][1])) for p in contour]
                if len(emb_pts) < 2:
                    continue

                xs_e = [p[0] for p in emb_pts]
                ys_e = [p[1] for p in emb_pts]
                _, _, c_w_px, c_h_px = cv2.boundingRect(contour)
                c_w_mm  = px_w_to_mm(c_w_px)

                # ── Min-shape filter: skip tiny noise shapes < 3mm × 3mm ──────
                c_w_emb = c_w_px * scale
                c_h_emb = c_h_px * scale
                if c_w_emb < MIN_SHAPE_U or c_h_emb < MIN_SHAPE_U:
                    continue

                # ── Fill-density guard ─────────────────────────────────────────
                # Count the actual cluster pixels that fall inside the contour
                # boundary, divided by the total enclosed pixel area of the contour.
                #
                #   Solid star / circle  → fill_density ~0.95  → tatami fill
                #   Building outline     → fill_density ~0.15  → running outline
                #   Text letters         → fill_density ~0.30  → running outline
                #
                # Threshold: > 0.4 → fill,  ≤ 0.4 → outline only
                c_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.drawContours(c_mask, [contour], -1, 255, thickness=cv2.FILLED)
                contour_px_area = int(np.count_nonzero(c_mask))
                if contour_px_area > 0:
                    inside_px     = int(np.count_nonzero(cv2.bitwise_and(c_mask, cluster_px_mask)))
                    fill_density  = inside_px / contour_px_area
                else:
                    fill_density = 1.0          # degenerate contour — default to fill

                is_hollow = fill_density <= 0.4
                print(
                    f"DIGITIZE: contour {c_w_emb/10:.1f}×{c_h_emb/10:.1f}mm "
                    f"fill_density={fill_density:.2f} → "
                    f"{'outline' if is_hollow else 'fill'}",
                    flush=True,
                )

                start = emb_pts[0]
                end   = emb_pts[-1]

                # Always TRIM before moving to this contour's start — eliminates jump lines
                pattern.add_stitch_absolute(pyembroidery.TRIM, start[0], start[1])

                tie_on(pattern, start[0], start[1])

                if is_hollow or stitch_type == "running" or (stitch_type == "auto" and c_w_mm < 3.0):
                    # ── Hollow / thin shapes: underlay running → satin columns along path ──
                    # Satin columns are stitched perpendicular to the contour direction,
                    # spanning the actual stroke width — makes text look like real embroidery.
                    running_outline(pattern, contour, stitch_u=STITCH_2MM)
                    pattern.add_stitch_absolute(pyembroidery.TRIM, start[0], start[1])
                    satin_along_path(pattern, contour, cluster_px_mask, stitch_u=STITCH_2MM)

                elif stitch_type == "satin" or (stitch_type == "auto" and c_w_mm < 8.0):
                    # ── Medium: underlay running → satin, 0.5 mm rows ─────────
                    running_outline(pattern, contour, stitch_u=STITCH_2MM)
                    pattern.add_stitch_absolute(pyembroidery.TRIM, start[0], start[1])
                    satin_fill(pattern, contour, xs_e, ys_e, row_u=SATIN_ROW_U)

                else:
                    # ── Large: underlay running → tatami fill, 0.5 mm rows ────
                    running_outline(pattern, contour, stitch_u=STITCH_2MM)
                    pattern.add_stitch_absolute(pyembroidery.TRIM, start[0], start[1])
                    tatami_fill(pattern, contour, row_u=TATAMI_ROW_U, stitch_u=STITCH_4MM)

                tie_off(pattern, end[0], end[1])
                any_stitches = True

        if not any_stitches:
            return jsonify({"error": "No stitches generated — try a different image"}), 400

        pattern.add_stitch_absolute(pyembroidery.END, 0, 0)
        pyembroidery.write(pattern, tmp_out.name)

        # ── 9. Verify output ───────────────────────────────────────────────────
        verify = pyembroidery.read(tmp_out.name)
        if verify is not None:
            v_s = [s for s in verify.stitches if s[2] == pyembroidery.STITCH]
            if v_s:
                vxs  = [s[0] for s in v_s]; vys = [s[1] for s in v_s]
                ow   = (max(vxs) - min(vxs)) / 10.0
                oh   = (max(vys) - min(vys)) / 10.0
                sc   = len(v_s)
                print(f"DIGITIZE VERIFY: {sc} stitches  {ow:.1f}mm × {oh:.1f}mm", flush=True)
                if sc < 500 or sc > 50000:
                    print(f"DIGITIZE WARN: stitch count {sc} outside 500–50000", flush=True)
                if not (10 <= ow <= 200 and 10 <= oh <= 200):
                    print(f"DIGITIZE WARN: dimensions {ow:.1f}×{oh:.1f}mm outside 10–200mm", flush=True)
                if ow < 1 or oh < 1:
                    return jsonify({"error": "Scaling error — design produced at incorrect size"}), 500

        info = get_design_info(pattern)
        mime = MIME_TYPES.get(output_format, "application/octet-stream")
        response = send_file(tmp_out.name, mimetype=mime, as_attachment=True,
                             download_name="digitized" + output_format)
        response.headers["stitch_count"]   = str(info["stitch_count"])
        response.headers["color_count"]    = str(info["color_count"])
        response.headers["width_mm"]       = str(info["width_mm"])
        response.headers["height_mm"]      = str(info["height_mm"])
        response.headers["estimated_time"] = str(info["estimated_time_minutes"])
        print(f"DIGITIZE OK: stitch_count={info['stitch_count']} color_count={info['color_count']} "
              f"width_mm={info['width_mm']} height_mm={info['height_mm']}", flush=True)
        return response
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"DIGITIZE ERROR: {error_details}", flush=True)
        return jsonify({
            "error": str(e),
            "traceback": error_details
        }), 500
    finally:
        _cleanup(tmp_in)


# ─── PREVIEW ─────────────────────────────────────────────────────────────────

@bp.route("/preview", methods=["POST"])
def preview():
    if pyembroidery is None:
        return jsonify({"error": "pyembroidery not installed"}), 500
    if Image is None:
        return jsonify({"error": "Pillow not installed"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    tmp_in, in_ext, err = read_uploaded_file(request.files["file"], SUPPORTED_INPUT_FORMATS)
    if err:
        status = 415 if "format" in err.lower() else 400
        return jsonify({"error": err}), status

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_out.close()

    try:
        pattern = pyembroidery.read(tmp_in)
        if pattern is None:
            return jsonify({"error": "Could not read embroidery file"}), 400

        xs = [s[0] for s in pattern.stitches if (s[2] & 0xF0) in (pyembroidery.STITCH, pyembroidery.JUMP)]
        ys = [s[1] for s in pattern.stitches if (s[2] & 0xF0) in (pyembroidery.STITCH, pyembroidery.JUMP)]

        if not xs or not ys:
            img = Image.new("RGB", (200, 200), "white")
            img.save(tmp_out.name, "PNG")
        else:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            w = max(max_x - min_x, 1)
            h = max(max_y - min_y, 1)

            target_size = 800
            scale = target_size / max(w, h)
            img_w = int(w * scale) + 40
            img_h = int(h * scale) + 40
            pad = 20

            img = Image.new("RGB", (img_w, img_h), "white")
            draw = ImageDraw.Draw(img)

            threads = pattern.threadlist or []

            def get_color(idx):
                if idx < len(threads):
                    t = threads[idx]
                    c = getattr(t, 'color', 0x000000)
                    return ((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)
                return (0, 0, 0)

            color_idx = 0
            prev_pt = None
            is_drawing = False

            for stitch in pattern.stitches:
                x, y, cmd = stitch[0], stitch[1], stitch[2]   # raw cmd — do NOT mask with 0xF0
                px = int((x - min_x) * scale) + pad            # STITCH=0,JUMP=1,TRIM=2 are all
                py = int((y - min_y) * scale) + pad            # < 16, so 0xF0 wipes them to 0

                if cmd == pyembroidery.STITCH:
                    if prev_pt is not None and is_drawing:
                        draw.line([prev_pt, (px, py)], fill=get_color(color_idx), width=1)
                    prev_pt = (px, py)
                    is_drawing = True
                elif cmd == pyembroidery.TRIM:
                    # TRIM: cut thread — reset anchor so no line is drawn to the next stitch
                    prev_pt = None
                    is_drawing = False
                elif cmd == pyembroidery.JUMP:
                    # JUMP: needle repositions without stitching — suppress the connecting line
                    prev_pt = None
                    is_drawing = False
                elif cmd == pyembroidery.COLOR_CHANGE:
                    color_idx += 1
                    prev_pt = None
                    is_drawing = False
                elif cmd == pyembroidery.END:
                    break

            img.save(tmp_out.name, "PNG")

        return send_file(tmp_out.name, mimetype="image/png", download_name="preview.png")
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Processing error: {}".format(str(e))}), 500
    finally:
        _cleanup(tmp_in)


# ─── SPLIT ───────────────────────────────────────────────────────────────────

@bp.route("/split", methods=["POST"])
def split():
    if pyembroidery is None:
        return jsonify({"error": "pyembroidery not installed"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    hoop_width_mm = float(request.form.get("hoop_width_mm", 101.6))
    hoop_height_mm = float(request.form.get("hoop_height_mm", 101.6))
    overlap_mm = float(request.form.get("overlap_mm", 3.0))

    tmp_in, in_ext, err = read_uploaded_file(request.files["file"], SUPPORTED_INPUT_FORMATS)
    if err:
        status = 415 if "format" in err.lower() else 400
        return jsonify({"error": err}), status

    base_name = os.path.splitext(request.files["file"].filename or "Design")[0]
    section_files = []

    try:
        pattern = pyembroidery.read(tmp_in)
        if pattern is None:
            return jsonify({"error": "Could not read embroidery file"}), 400

        stitches = pattern.stitches
        xs = [s[0] for s in stitches if (s[2] & 0xF0) in (pyembroidery.STITCH, pyembroidery.JUMP)]
        ys = [s[1] for s in stitches if (s[2] & 0xF0) in (pyembroidery.STITCH, pyembroidery.JUMP)]

        if not xs or not ys:
            return jsonify({"error": "Design has no stitches"}), 400

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        hoop_w_units = hoop_width_mm * 10
        hoop_h_units = hoop_height_mm * 10
        overlap_units = overlap_mm * 10

        cols = max(1, math.ceil((max_x - min_x) / hoop_w_units))
        rows = max(1, math.ceil((max_y - min_y) / hoop_h_units))

        threads = pattern.threadlist or []

        for row in range(rows):
            for col in range(cols):
                sect_min_x = min_x + col * hoop_w_units - overlap_units
                sect_max_x = sect_min_x + hoop_w_units + overlap_units * 2
                sect_min_y = min_y + row * hoop_h_units - overlap_units
                sect_max_y = sect_min_y + hoop_h_units + overlap_units * 2

                sect_pattern = pyembroidery.EmbPattern()
                for t in threads:
                    sect_pattern.add_thread(t)

                has_stitches = False
                for stitch in stitches:
                    x, y, cmd = stitch[0], stitch[1], stitch[2] & 0xF0
                    if cmd in (pyembroidery.STITCH, pyembroidery.JUMP):
                        if sect_min_x <= x <= sect_max_x and sect_min_y <= y <= sect_max_y:
                            nx = int(x - sect_min_x)
                            ny = int(y - sect_min_y)
                            sect_pattern.add_stitch_absolute(cmd, nx, ny)
                            has_stitches = True
                    elif cmd in (pyembroidery.TRIM, pyembroidery.COLOR_CHANGE):
                        sect_pattern.add_stitch_absolute(cmd, 0, 0)
                    elif cmd == pyembroidery.END:
                        break

                if not has_stitches:
                    continue

                sect_pattern.add_stitch_absolute(pyembroidery.END, 0, 0)

                part_num = row * cols + col + 1
                out_name = "{}_Part{}.pes".format(base_name, part_num)
                tmp_sect = tempfile.NamedTemporaryFile(delete=False, suffix=".pes")
                tmp_sect.close()
                pyembroidery.write(sect_pattern, tmp_sect.name)
                section_files.append((out_name, tmp_sect.name))

        if not section_files:
            return jsonify({"error": "Design already fits within the specified hoop size"}), 400

        zip_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        zip_tmp.close()
        with zipfile.ZipFile(zip_tmp.name, "w") as zf:
            for out_name, path in section_files:
                zf.write(path, out_name)

        sections_count = len(section_files)
        layout = "{}x{}".format(cols, rows)

        response = send_file(zip_tmp.name, mimetype="application/zip", as_attachment=True,
                             download_name="{}_split.zip".format(base_name))
        response.headers["sections_count"] = str(sections_count)
        response.headers["layout"] = layout
        return response
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Processing error: {}".format(str(e))}), 500
    finally:
        _cleanup(tmp_in)
        for _, path in section_files:
            _cleanup(path)


# ─── INFO ────────────────────────────────────────────────────────────────────

@bp.route("/info", methods=["POST"])
def info():
    if pyembroidery is None:
        return jsonify({"error": "pyembroidery not installed"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    tmp_in, in_ext, err = read_uploaded_file(request.files["file"], SUPPORTED_INPUT_FORMATS)
    if err:
        status = 415 if "format" in err.lower() else 400
        return jsonify({"error": err}), status

    try:
        pattern = pyembroidery.read(tmp_in)
        if pattern is None:
            return jsonify({"error": "Could not read embroidery file"}), 400

        data = get_design_info(pattern)
        data["format"] = in_ext.lstrip(".")
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Processing error: {}".format(str(e))}), 500
    finally:
        _cleanup(tmp_in)


# ─── RESIZE ──────────────────────────────────────────────────────────────────

@bp.route("/resize", methods=["POST"])
def resize():
    if pyembroidery is None:
        return jsonify({"error": "pyembroidery not installed"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    target_width_mm = request.form.get("target_width_mm")
    target_height_mm = request.form.get("target_height_mm")
    lock_aspect = request.form.get("lock_aspect_ratio", "true").lower() in ("true", "1", "yes")
    output_format = request.form.get("output_format", "").lower().strip()

    if not target_width_mm and not target_height_mm:
        return jsonify({"error": "target_width_mm or target_height_mm is required"}), 400

    tmp_in, in_ext, err = read_uploaded_file(request.files["file"], SUPPORTED_INPUT_FORMATS)
    if err:
        status = 415 if "format" in err.lower() else 400
        return jsonify({"error": err}), status

    if not output_format:
        output_format = in_ext
    if not output_format.startswith("."):
        output_format = "." + output_format
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        return jsonify({"error": "Unsupported output format"}), 415

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=output_format)
    tmp_out.close()

    try:
        pattern = pyembroidery.read(tmp_in)
        if pattern is None:
            return jsonify({"error": "Could not read embroidery file"}), 400

        xs = [s[0] for s in pattern.stitches if (s[2] & 0xF0) in (pyembroidery.STITCH, pyembroidery.JUMP)]
        ys = [s[1] for s in pattern.stitches if (s[2] & 0xF0) in (pyembroidery.STITCH, pyembroidery.JUMP)]

        if not xs or not ys:
            return jsonify({"error": "Design has no stitches"}), 400

        current_w_mm = (max(xs) - min(xs)) / 10.0
        current_h_mm = (max(ys) - min(ys)) / 10.0

        scale_x = float(target_width_mm) / max(current_w_mm, 0.001) if target_width_mm else None
        scale_y = float(target_height_mm) / max(current_h_mm, 0.001) if target_height_mm else None

        if lock_aspect:
            scale = min(v for v in [scale_x, scale_y] if v is not None)
            scale_x = scale_y = scale
        else:
            scale_x = scale_x or 1.0
            scale_y = scale_y or 1.0

        new_pattern = pyembroidery.EmbPattern()
        for t in (pattern.threadlist or []):
            new_pattern.add_thread(t)

        for stitch in pattern.stitches:
            x, y, cmd = stitch[0], stitch[1], stitch[2]
            new_pattern.add_stitch_absolute(cmd & 0xF0, int(x * scale_x), int(y * scale_y))

        pyembroidery.write(new_pattern, tmp_out.name)

        info = get_design_info(new_pattern)
        mime = MIME_TYPES.get(output_format, "application/octet-stream")
        response = send_file(tmp_out.name, mimetype=mime, as_attachment=True,
                             download_name="resized" + output_format)
        response.headers["stitch_count"] = str(info["stitch_count"])
        response.headers["color_count"] = str(info["color_count"])
        response.headers["width_mm"] = str(info["width_mm"])
        response.headers["height_mm"] = str(info["height_mm"])
        response.headers["estimated_time"] = str(info["estimated_time_minutes"])
        return response
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Processing error: {}".format(str(e))}), 500
    finally:
        _cleanup(tmp_in)


# ─── ROTATE ──────────────────────────────────────────────────────────────────

@bp.route("/rotate", methods=["POST"])
def rotate():
    if pyembroidery is None:
        return jsonify({"error": "pyembroidery not installed"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    degrees = float(request.form.get("degrees", 90))
    output_format = request.form.get("output_format", "").lower().strip()

    tmp_in, in_ext, err = read_uploaded_file(request.files["file"], SUPPORTED_INPUT_FORMATS)
    if err:
        status = 415 if "format" in err.lower() else 400
        return jsonify({"error": err}), status

    if not output_format:
        output_format = in_ext
    if not output_format.startswith("."):
        output_format = "." + output_format
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        return jsonify({"error": "Unsupported output format"}), 415

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=output_format)
    tmp_out.close()

    try:
        pattern = pyembroidery.read(tmp_in)
        if pattern is None:
            return jsonify({"error": "Could not read embroidery file"}), 400

        angle_rad = math.radians(degrees)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        new_pattern = pyembroidery.EmbPattern()
        for t in (pattern.threadlist or []):
            new_pattern.add_thread(t)

        for stitch in pattern.stitches:
            x, y, cmd = stitch[0], stitch[1], stitch[2]
            nx = int(x * cos_a - y * sin_a)
            ny = int(x * sin_a + y * cos_a)
            new_pattern.add_stitch_absolute(cmd & 0xF0, nx, ny)

        pyembroidery.write(new_pattern, tmp_out.name)

        info = get_design_info(new_pattern)
        mime = MIME_TYPES.get(output_format, "application/octet-stream")
        response = send_file(tmp_out.name, mimetype=mime, as_attachment=True,
                             download_name="rotated" + output_format)
        response.headers["stitch_count"] = str(info["stitch_count"])
        response.headers["color_count"] = str(info["color_count"])
        response.headers["width_mm"] = str(info["width_mm"])
        response.headers["height_mm"] = str(info["height_mm"])
        response.headers["estimated_time"] = str(info["estimated_time_minutes"])
        return response
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Processing error: {}".format(str(e))}), 500
    finally:
        _cleanup(tmp_in)


# ─── MERGE ───────────────────────────────────────────────────────────────────

@bp.route("/merge", methods=["POST"])
def merge():
    if pyembroidery is None:
        return jsonify({"error": "pyembroidery not installed"}), 500

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided. Send multiple files with field name 'files'"}), 400

    layout = request.form.get("layout", "side_by_side").lower().strip()
    spacing_mm = float(request.form.get("spacing_mm", 5.0))
    output_format = request.form.get("output_format", "pes").lower().strip()

    if not output_format.startswith("."):
        output_format = "." + output_format
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        return jsonify({"error": "Unsupported output format"}), 415

    spacing_units = int(spacing_mm * 10)
    tmp_files = []

    for f in files:
        tmp_in, in_ext, err = read_uploaded_file(f, SUPPORTED_INPUT_FORMATS)
        if err:
            return jsonify({"error": "File '{}': {}".format(f.filename, err)}), 415
        tmp_files.append(tmp_in)

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=output_format)
    tmp_out.close()

    try:
        patterns = []
        for tf in tmp_files:
            p = pyembroidery.read(tf)
            if p is None:
                return jsonify({"error": "Could not read one of the embroidery files"}), 400
            patterns.append(p)

        def get_bounds(p):
            xs = [s[0] for s in p.stitches if (s[2] & 0xF0) in (pyembroidery.STITCH, pyembroidery.JUMP)]
            ys = [s[1] for s in p.stitches if (s[2] & 0xF0) in (pyembroidery.STITCH, pyembroidery.JUMP)]
            if not xs:
                return 0, 0, 0, 0
            return min(xs), min(ys), max(xs), max(ys)

        merged = pyembroidery.EmbPattern()
        offsets = []
        cursor_x = 0
        cursor_y = 0

        for p in patterns:
            mnx, mny, mxx, mxy = get_bounds(p)
            w = mxx - mnx
            h = mxy - mny
            if layout == "side_by_side":
                offsets.append((cursor_x - mnx, -mny))
                cursor_x += w + spacing_units
            elif layout == "stacked":
                offsets.append((-mnx, cursor_y - mny))
                cursor_y += h + spacing_units
            else:  # overlay
                offsets.append((-mnx, -mny))

        for i, p in enumerate(patterns):
            ox, oy = offsets[i]
            for t in (p.threadlist or []):
                merged.add_thread(t)
            if i > 0:
                merged.add_stitch_absolute(pyembroidery.COLOR_CHANGE, 0, 0)

            first = True
            for stitch in p.stitches:
                x, y, cmd = stitch[0], stitch[1], stitch[2] & 0xF0
                if cmd == pyembroidery.END:
                    break
                nx, ny = x + ox, y + oy
                if first and cmd == pyembroidery.STITCH:
                    merged.add_stitch_absolute(pyembroidery.TRIM, nx, ny)
                    first = False
                merged.add_stitch_absolute(cmd, nx, ny)

        merged.add_stitch_absolute(pyembroidery.END, 0, 0)
        pyembroidery.write(merged, tmp_out.name)

        info = get_design_info(merged)
        mime = MIME_TYPES.get(output_format, "application/octet-stream")
        response = send_file(tmp_out.name, mimetype=mime, as_attachment=True,
                             download_name="merged" + output_format)
        response.headers["stitch_count"] = str(info["stitch_count"])
        response.headers["color_count"] = str(info["color_count"])
        response.headers["width_mm"] = str(info["width_mm"])
        response.headers["height_mm"] = str(info["height_mm"])
        response.headers["estimated_time"] = str(info["estimated_time_minutes"])
        return response
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Processing error: {}".format(str(e))}), 500
    finally:
        for tf in tmp_files:
            _cleanup(tf)


# ─── ERROR HANDLERS ──────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 50MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _cleanup(path):
    if path:
        try:
            os.unlink(path)
        except Exception:
            pass


# ─── REGISTER & RUN ──────────────────────────────────────────────────────────

app.register_blueprint(bp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
