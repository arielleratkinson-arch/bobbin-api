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
    color_count_param = min(16, max(1, int(request.form.get("color_count", 6))))
    do_simplify = request.form.get("simplify", "true").lower() not in ("false", "0", "no")

    allowed_image_ext = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
    tmp_in, in_ext, err = read_uploaded_file(request.files["file"], allowed_image_ext)
    if err:
        status = 415 if "format" in err.lower() else 400
        return jsonify({"error": err}), status

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=output_format)
    tmp_out.close()

    try:
        # ── 1. Load image & quantize to color_count colors ────────────────────
        pil_img_raw = Image.open(tmp_in)
        if pil_img_raw.mode == 'RGBA':
            background = Image.new('RGB', pil_img_raw.size, (255, 255, 255))
            background.paste(pil_img_raw, mask=pil_img_raw.split()[3])
            pil_img = background
        else:
            pil_img = pil_img_raw.convert('RGB')
            orig_w, orig_h = pil_img.size

        # Special case: color_count == 2 → black and white
        if color_count_param == 2:
            bw = pil_img.convert("L").point(lambda x: 0 if x < 128 else 255, "1")
            bw_rgb = bw.convert("RGB")
            palette_map = {0: (0, 0, 0), 255: (255, 255, 255)}
            q_arr = np.array(bw.convert("L"), dtype=np.uint8)
            q_arr = (q_arr > 128).astype(np.uint8) * 255
            canonical_palette = {0: (0, 0, 0), 255: (255, 255, 255)}
        else:
            canonical_palette, q_arr = _quantize_colors(pil_img, color_count_param)

        # ── 2. Detect and exclude background ─────────────────────────────────
        bg_idx = _detect_background(pil_img, q_arr)

        # ── 3. Hoop units and stitch settings ────────────────────────────────
        hoop_w_units = hoop_width_mm * 10   # pyembroidery units = 1/10 mm
        hoop_h_units = hoop_height_mm * 10

        max_stitch_units = max(1, int(max_stitch_length * 10))
        density_units    = max(1, int(10.0 / density))

        # ── 4. Pass 1 — collect valid contours per color ──────────────────────
        # Use an image-scale estimate only for the area/bbox filter checks.
        img_h, img_w = q_arr.shape[:2]
        scale_est = min(hoop_w_units / max(img_w, 1), hoop_h_units / max(img_h, 1))
        MIN_BBOX_PX = 50 / scale_est   # 5 mm in pixels (approx)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        color_contours = {}   # color_idx → list of (simplified) contours

        for color_idx in sorted(canonical_palette.keys()):
            if color_idx == bg_idx:
                continue

            mask = (q_arr == color_idx).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid = []
            for c in contours:
                if cv2.contourArea(c) < 50:
                    continue
                _, _, cw, ch = cv2.boundingRect(c)
                if cw < MIN_BBOX_PX or ch < MIN_BBOX_PX:
                    continue
                if do_simplify:
                    c = cv2.approxPolyDP(c, epsilon=2.0, closed=True)
                if len(c) >= 2:
                    valid.append(c)

            if valid:
                color_contours[color_idx] = valid

        if not color_contours:
            return jsonify({"error": "No stitchable regions found after filtering. Try a simpler image or adjust color_count."}), 400

        # ── 5. Compute content bounding box across ALL valid contours ─────────
        all_px = []
        all_py = []
        for contours_list in color_contours.values():
            for c in contours_list:
                for pt in c:
                    all_px.append(int(pt[0][0]))
                    all_py.append(int(pt[0][1]))

        px_min_x = min(all_px)
        px_max_x = max(all_px)
        px_min_y = min(all_py)
        px_max_y = max(all_py)
        pixel_width  = max(px_max_x - px_min_x, 1)
        pixel_height = max(px_max_y - px_min_y, 1)

        # Scale to fit hoop with 10% padding, preserving aspect ratio
        scale_x = hoop_w_units / pixel_width
        scale_y = hoop_h_units / pixel_height
        scale   = min(scale_x, scale_y) * 0.9

        # Centering offset so design sits in the middle of the hoop
        design_w_units  = pixel_width  * scale
        design_h_units  = pixel_height * scale
        offset_x = (hoop_w_units - design_w_units) / 2
        offset_y = (hoop_h_units - design_h_units) / 2

        def px_to_emb(px, py):
            """Convert pixel coordinates to pyembroidery units with centering."""
            return (
                int((px - px_min_x) * scale + offset_x),
                int((py - px_min_y) * scale + offset_y),
            )

        # ── 6. Build embroidery pattern ────────────────────────────────────────
        pattern = pyembroidery.EmbPattern()

        def add_tie_stitches(pat, ex, ey):
            for _ in range(3):
                pat.add_stitch_absolute(pyembroidery.STITCH, ex + 10, ey)
                pat.add_stitch_absolute(pyembroidery.STITCH, ex, ey)

        def stitch_contour_running(pat, contour, max_su):
            """Running stitches along a contour path in embroidery units."""
            pts = [px_to_emb(int(pt[0][0]), int(pt[0][1])) for pt in contour]
            if len(pts) < 2:
                return
            pat.add_stitch_absolute(pyembroidery.STITCH, pts[0][0], pts[0][1])
            prev = pts[0]
            for pt in pts[1:]:
                dx, dy = pt[0] - prev[0], pt[1] - prev[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist >= max_su:
                    steps = max(1, int(dist / max_su))
                    for i in range(1, steps + 1):
                        ix = prev[0] + int(dx * i / steps)
                        iy = prev[1] + int(dy * i / steps)
                        pat.add_stitch_absolute(pyembroidery.STITCH, ix, iy)
                else:
                    pat.add_stitch_absolute(pyembroidery.STITCH, pt[0], pt[1])
                prev = pt
            # close the contour
            pat.add_stitch_absolute(pyembroidery.STITCH, pts[0][0], pts[0][1])

        first_thread = True
        any_stitches = False

        for color_idx in sorted(color_contours.keys()):
            rgb = canonical_palette[color_idx]

            thread = pyembroidery.EmbThread()
            thread.color = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
            thread.name = "Color #{:02X}{:02X}{:02X}".format(*rgb)
            pattern.add_thread(thread)

            if not first_thread:
                pattern.add_stitch_absolute(pyembroidery.COLOR_CHANGE, 0, 0)
            first_thread = False

            color_first = True
            for contour in color_contours[color_idx]:
                emb_pts = [px_to_emb(int(pt[0][0]), int(pt[0][1])) for pt in contour]
                if len(emb_pts) < 2:
                    continue

                xs = [p[0] for p in emb_pts]
                ys = [p[1] for p in emb_pts]
                bbox_w_mm = (max(xs) - min(xs)) / 10.0

                use_running = stitch_type == "running" or (stitch_type == "auto" and bbox_w_mm < 2)
                use_satin   = stitch_type == "satin"   or (stitch_type == "auto" and 2 <= bbox_w_mm < 8)
                use_fill    = stitch_type == "fill"    or (stitch_type == "auto" and bbox_w_mm >= 8)

                first_pt = emb_pts[0]
                last_pt  = emb_pts[-1]

                if not color_first:
                    pattern.add_stitch_absolute(pyembroidery.TRIM, first_pt[0], first_pt[1])
                color_first = False

                add_tie_stitches(pattern, first_pt[0], first_pt[1])

                if use_running:
                    stitch_contour_running(pattern, contour, max_stitch_units)

                elif use_satin:
                    toggle = True
                    y = min(ys)
                    while y <= max(ys):
                        if toggle:
                            pattern.add_stitch_absolute(pyembroidery.STITCH, min(xs), y)
                            pattern.add_stitch_absolute(pyembroidery.STITCH, max(xs), y)
                        else:
                            pattern.add_stitch_absolute(pyembroidery.STITCH, max(xs), y)
                            pattern.add_stitch_absolute(pyembroidery.STITCH, min(xs), y)
                        y += density_units
                        toggle = not toggle

                elif use_fill:
                    row = 0
                    y = min(ys)
                    while y <= max(ys):
                        row_offset = (max_stitch_units // 2) if row % 2 == 1 else 0
                        x = min(xs) + row_offset
                        while x <= max(xs):
                            pattern.add_stitch_absolute(pyembroidery.STITCH, x, y)
                            x += max_stitch_units
                        y += density_units
                        row += 1

                add_tie_stitches(pattern, last_pt[0], last_pt[1])
                any_stitches = True

        if not any_stitches:
            return jsonify({"error": "No stitchable regions found after filtering. Try a simpler image or adjust color_count."}), 400

        pattern.add_stitch_absolute(pyembroidery.END, 0, 0)
        pyembroidery.write(pattern, tmp_out.name)

        # ── 7. Verify output dimensions ────────────────────────────────────────
        verify = pyembroidery.read(tmp_out.name)
        if verify is not None:
            vxs = [s[0] for s in verify.stitches if s[2] == pyembroidery.STITCH]
            vys = [s[1] for s in verify.stitches if s[2] == pyembroidery.STITCH]
            if vxs and vys:
                out_w = max(vxs) - min(vxs)
                out_h = max(vys) - min(vys)
                if out_w < 1 or out_h < 1:
                    return jsonify({"error": "Image too complex — try a simpler image with clearer edges"}), 400
        info = get_design_info(pattern)
        mime = MIME_TYPES.get(output_format, "application/octet-stream")
        response = send_file(tmp_out.name, mimetype=mime, as_attachment=True,
                             download_name="digitized" + output_format)
        response.headers["stitch_count"] = str(info["stitch_count"])
        response.headers["color_count"]  = str(info["color_count"])
        response.headers["width_mm"]     = str(info["width_mm"])
        response.headers["height_mm"]    = str(info["height_mm"])
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
                x, y, cmd = stitch[0], stitch[1], stitch[2] & 0xF0
                px = int((x - min_x) * scale) + pad
                py = int((y - min_y) * scale) + pad

                if cmd == pyembroidery.STITCH:
                    if prev_pt is not None and is_drawing:
                        draw.line([prev_pt, (px, py)], fill=get_color(color_idx), width=1)
                    prev_pt = (px, py)
                    is_drawing = True
                elif cmd in (pyembroidery.JUMP, pyembroidery.TRIM):
                    prev_pt = (px, py)
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
