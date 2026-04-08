import os
import io
import re
import math
import json
import shutil
import zipfile
import tempfile
import subprocess
import traceback
import xml.etree.ElementTree as _ET
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


# ─── POTRACE VECTORIZATION HELPERS ───────────────────────────────────────────

import shutil as _shutil

def _find_potrace():
    """Find a working potrace binary.  Prefer our self-compiled build (avoids nix SIGSEGV);
    fall back to any potrace on PATH if the local build doesn't exist."""
    _local_build = os.path.join(os.path.expanduser("~"), ".local", "bin", "potrace")
    if os.path.isfile(_local_build) and os.access(_local_build, os.X_OK):
        return _local_build
    # Try every PATH entry that has a potrace binary (skip nix-profile ones that may segfault)
    for _candidate in (_shutil.which("potrace"),):
        if _candidate and os.path.isfile(_candidate) and os.access(_candidate, os.X_OK):
            if "nix-profile" not in _candidate:   # nix-profile builds SIGSEGV in Flask subprocess
                return _candidate
    return _local_build  # best guess even if missing; _potrace_available() will return False


POTRACE_BIN = _find_potrace()


def _potrace_available():
    return bool(POTRACE_BIN) and os.path.isfile(POTRACE_BIN) and os.access(POTRACE_BIN, os.X_OK)


def _potrace_env():
    """Build an environment dict that includes the library paths potrace needs.

    When Flask/gunicorn is launched by Replit's workflow runner it may not inherit
    the nix-store LD_LIBRARY_PATH that the interactive shell has.  We detect the
    required dirs from `ldd` once and inject them so the subprocess doesn't SIGSEGV.
    """
    env = os.environ.copy()
    if not _potrace_available():
        return env
    try:
        ldd_out = subprocess.run(
            ["ldd", POTRACE_BIN], capture_output=True, text=True, timeout=5
        ).stdout
        extra_dirs = []
        for line in ldd_out.splitlines():
            # Lines look like:  libm.so.6 => /path/to/libm.so.6 (0xaddr)
            parts = line.split("=>")
            if len(parts) == 2:
                lib_path = parts[1].strip().split()[0]
                if lib_path.startswith("/"):
                    lib_dir = os.path.dirname(lib_path)
                    if lib_dir not in ("", "/lib64") and lib_dir not in extra_dirs:
                        extra_dirs.append(lib_dir)
        if extra_dirs:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(extra_dirs) + (":" + existing if existing else "")
    except Exception:
        pass
    return env


# Pre-compute potrace environment once at module load (avoids ldd overhead per request)
_POTRACE_ENV: dict = {}


def _write_pbm(mask_uint8, path):
    """Write a binary mask (255=foreground/black) as a P1 ASCII PBM file.
    P1 ASCII format is universally compatible with all potrace builds.
    Row format: '1' = black/foreground, '0' = white/background."""
    h, w = mask_uint8.shape
    with open(path, "w") as f:
        f.write(f"P1\n{w} {h}\n")
        for row in mask_uint8:
            f.write(" ".join("1" if int(px) >= 128 else "0" for px in row))
            f.write("\n")


def _potrace_svg_to_contours(svg_path, img_h, img_w, samples_per_curve=8):
    """Parse a potrace SVG file and return contours in OpenCV format.

    potrace SVG coordinate system:
      The <g> element carries  transform="translate(tx,ty) scale(sx,sy)"
      where sy is negative (flip).  A path coordinate (px, py) maps to
      image pixel:   x = px*sx + tx,   y = py*sy + ty
    Returns a list of numpy arrays shaped (N, 1, 2) int32.
    """

    def _is_num(tok):
        try:
            float(tok)
            return True
        except ValueError:
            return False

    def _cubic_bezier(p0, p1, p2, p3, t):
        mt = 1.0 - t
        return (
            mt**3 * p0[0] + 3*mt**2*t * p1[0] + 3*mt*t**2 * p2[0] + t**3 * p3[0],
            mt**3 * p0[1] + 3*mt**2*t * p1[1] + 3*mt*t**2 * p2[1] + t**3 * p3[1],
        )

    try:
        tree = _ET.parse(svg_path)
        root = tree.getroot()
    except Exception:
        return []

    # Namespace-agnostic element search
    def _find_all(elem, local_tag):
        return [ch for ch in elem.iter() if ch.tag.split("}")[-1] == local_tag]

    g_elems = _find_all(root, "g")
    transform = g_elems[0].get("transform", "") if g_elems else ""

    # Parse "translate(tx,ty) scale(sx,sy)"
    tm = re.search(r"translate\(([^,]+),([^)]+)\)", transform)
    sm = re.search(r"scale\(([^,]+),([^)]+)\)", transform)
    tx = float(tm.group(1)) if tm else 0.0
    ty = float(tm.group(2)) if tm else float(img_h)
    sx = float(sm.group(1)) if sm else 1.0
    sy = float(sm.group(2)) if sm else -1.0

    def to_px(ppx, ppy):
        """potrace path coord → clamped image pixel (int x, int y)."""
        rx = ppx * sx + tx
        ry = ppy * sy + ty
        return (
            int(round(max(0, min(img_w - 1, rx)))),
            int(round(max(0, min(img_h - 1, ry)))),
        )

    def parse_path_d(d):
        """Parse one SVG path 'd' string → list of (x, y) image-pixel tuples."""
        # Tokenise: command letters and signed numbers (incl. scientific notation)
        tokens = re.findall(
            r"[MCLZSmclzsm]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", d
        )
        pts = []
        i = 0
        n = len(tokens)
        cur = (0.0, 0.0)
        path_start = (0.0, 0.0)
        cmd = "M"

        while i < n:
            if tokens[i].isalpha():
                cmd = tokens[i]
                i += 1

            if cmd in ("M", "m"):
                abs_c = cmd == "M"
                while i + 1 < n and _is_num(tokens[i]):
                    x, y = float(tokens[i]), float(tokens[i + 1])
                    i += 2
                    cur = (x, y) if abs_c else (cur[0] + x, cur[1] + y)
                    pts.append(to_px(*cur))
                    path_start = cur
                # Subsequent coords after M/m are treated as L/l
                cmd = "L" if abs_c else "l"

            elif cmd in ("L", "l"):
                abs_c = cmd == "L"
                while i + 1 < n and _is_num(tokens[i]):
                    x, y = float(tokens[i]), float(tokens[i + 1])
                    i += 2
                    cur = (x, y) if abs_c else (cur[0] + x, cur[1] + y)
                    pts.append(to_px(*cur))

            elif cmd in ("C", "c"):
                abs_c = cmd == "C"
                while i + 5 < n and _is_num(tokens[i]):
                    coords = [float(tokens[i + j]) for j in range(6)]
                    i += 6
                    if abs_c:
                        c1 = (coords[0], coords[1])
                        c2 = (coords[2], coords[3])
                        ep = (coords[4], coords[5])
                    else:
                        c1 = (cur[0] + coords[0], cur[1] + coords[1])
                        c2 = (cur[0] + coords[2], cur[1] + coords[3])
                        ep = (cur[0] + coords[4], cur[1] + coords[5])
                    # Sample cubic bezier: cur → c1 → c2 → ep
                    for si in range(1, samples_per_curve + 1):
                        t_val = si / samples_per_curve
                        bx, by = _cubic_bezier(cur, c1, c2, ep, t_val)
                        pts.append(to_px(bx, by))
                    cur = ep

            elif cmd in ("Z", "z"):
                pts.append(to_px(*path_start))
                cur = path_start
                i  # Z has no operands — outer loop will advance cmd

            else:
                # Unknown command — skip one token to avoid infinite loop
                if i < n and not tokens[i].isalpha():
                    i += 1
                elif i < n:
                    i += 1  # will be re-read as cmd on next iteration

        return pts

    try:
        import numpy as _np
        contours = []
        for path_elem in _find_all(root, "path"):
            d = path_elem.get("d", "")
            if not d:
                continue
            pts = parse_path_d(d)
            if len(pts) >= 3:
                arr = _np.array([[p] for p in pts], dtype=_np.int32)
                contours.append(arr)
        return contours
    except Exception as exc:
        print(f"VECTORIZE: SVG parse error: {exc}", flush=True)
        return []


def _vectorize_mask(mask_uint8, tmp_dir, cidx=0):
    """Run potrace on a binary mask to get clean bezier-derived contours.

    Returns a list of numpy (N,1,2) int32 arrays (OpenCV contour format),
    or None if potrace is unavailable or produces no contours.
    """
    global _POTRACE_ENV
    if not _potrace_available():
        return None
    # Initialise the subprocess environment once (injects nix lib paths to avoid SIGSEGV)
    if not _POTRACE_ENV:
        _POTRACE_ENV = _potrace_env()
        nix_libs = _POTRACE_ENV.get("LD_LIBRARY_PATH", "(none)")
        print(f"VECTORIZE: potrace env LD_LIBRARY_PATH={nix_libs[:80]}", flush=True)
    try:
        import numpy as _np
        img_h, img_w = mask_uint8.shape
        pbm = os.path.join(tmp_dir, f"vmask_{cidx}.pbm")
        svg = os.path.join(tmp_dir, f"vmask_{cidx}.svg")
        _write_pbm(mask_uint8, pbm)
        result = subprocess.run(
            [POTRACE_BIN, "--svg", "-o", svg, pbm],
            capture_output=True, timeout=20, env=_POTRACE_ENV,
        )
        if result.returncode != 0 or not os.path.exists(svg):
            err = result.stderr.decode(errors="replace")[:120]
            print(f"VECTORIZE: potrace cidx={cidx} rc={result.returncode}: {err}", flush=True)
            return None
        contours = _potrace_svg_to_contours(svg, img_h, img_w)
        if contours:
            print(f"VECTORIZE: potrace cidx={cidx} → {len(contours)} smooth path(s)", flush=True)
            return contours
        return None
    except Exception as exc:
        print(f"VECTORIZE: exception cidx={cidx}: {exc}", flush=True)
        return None


# ─── END VECTORIZATION HELPERS ────────────────────────────────────────────────

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
        cmd = stitch[2]
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

    xs = [s[0] for s in pattern.stitches if s[2] in (pyembroidery.STITCH, pyembroidery.JUMP)]
    ys = [s[1] for s in pattern.stitches if s[2] in (pyembroidery.STITCH, pyembroidery.JUMP)]

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

    stitch_type     = request.form.get("stitch_type", "auto").lower().strip()
    hoop_width_mm   = float(request.form.get("hoop_width_mm",  101.6))
    hoop_height_mm  = float(request.form.get("hoop_height_mm", 101.6))
    color_count_param = min(16, max(1, int(request.form.get("color_count", 8))))
    do_simplify     = request.form.get("simplify",  "true").lower()  not in ("false", "0", "no")
    applique_mode   = request.form.get("applique",  "false").lower() not in ("false", "0", "no")

    allowed_image_ext = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]
    tmp_in, in_ext, err = read_uploaded_file(request.files["file"], allowed_image_ext)
    if err:
        status = 415 if "format" in err.lower() else 400
        return jsonify({"error": err}), status

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=output_format)
    tmp_out.close()

    # Scratch directory for potrace PBM/SVG scratch files
    vec_tmp = tempfile.mkdtemp(prefix="bobbin_vec_")

    try:
        # ── STEP 1: Image Loading ──────────────────────────────────────────────
        pil_img = Image.open(tmp_in)

        if pil_img.mode in ("RGBA", "LA", "PA"):
            if pil_img.mode == "PA":
                pil_img = pil_img.convert("RGBA")
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[-1])
            pil_img = bg
        elif pil_img.mode == "P" and "transparency" in pil_img.info:
            pil_img = pil_img.convert("RGBA")
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[-1])
            pil_img = bg
        else:
            pil_img = pil_img.convert("RGB")

        # ── STEP 2: Preprocessing ──────────────────────────────────────────────
        # Resize to max 1000px on longest side preserving aspect ratio; white-pad if needed
        MAX_DIM = 1000
        orig_w, orig_h = pil_img.size
        if max(orig_w, orig_h) > MAX_DIM:
            ratio   = min(MAX_DIM / orig_w, MAX_DIM / orig_h)
            new_w   = max(1, int(orig_w * ratio))
            new_h   = max(1, int(orig_h * ratio))
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
            canvas  = Image.new("RGB", (MAX_DIM, MAX_DIM), (255, 255, 255))
            canvas.paste(pil_img, ((MAX_DIM - new_w) // 2, (MAX_DIM - new_h) // 2))
            pil_img = canvas

        img_rgb      = np.array(pil_img, dtype=np.uint8)
        img_h, img_w = img_rgb.shape[:2]

        # Bilateral filter — d=5 preserves fine stripe / line details better than d=9
        img_filtered = cv2.bilateralFilter(img_rgb, d=5, sigmaColor=100, sigmaSpace=100)

        # ── STEP 3: Color Clustering ───────────────────────────────────────────
        k         = min(16, max(2, color_count_param))
        pixels    = np.float32(img_filtered.reshape(-1, 3))
        criteria  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _, labels_flat, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        centers   = np.uint8(centers)
        labels_2d = labels_flat.flatten().reshape(img_h, img_w)

        # Background: brightness > 200 OR appears in > 30% of edge pixels
        from collections import Counter as _Counter
        _n      = 20
        _edge_s = []
        for _i in range(_n):
            _tx = int(_i * (img_w - 1) / max(_n - 1, 1))
            _edge_s += [int(labels_2d[0, _tx]), int(labels_2d[img_h - 1, _tx])]
        for _i in range(_n):
            _ty = int(_i * (img_h - 1) / max(_n - 1, 1))
            _edge_s += [int(labels_2d[_ty, 0]), int(labels_2d[_ty, img_w - 1])]
        _edge_total  = len(_edge_s)
        _edge_counts = _Counter(_edge_s)
        _edge_bg     = {c for c, cnt in _edge_counts.items() if cnt / _edge_total >= 0.30}
        _light_bg    = {
            c for c in range(k)
            if (int(centers[c][0]) + int(centers[c][1]) + int(centers[c][2])) / 3 > 200
        }
        bg_clusters = _edge_bg | _light_bg
        print(f"DIGITIZE: k={k}  bg_clusters={bg_clusters}  image={img_w}×{img_h}px", flush=True)

        # ── STEP 4: Vectorization per Color Cluster ────────────────────────────
        hoop_w_u     = hoop_width_mm  * 10
        hoop_h_u     = hoop_height_mm * 10
        STITCH_2MM   = 20    # 2 mm running-stitch spacing
        STITCH_1_5MM = 15    # 1.5 mm — finer stitch for thin text strokes
        TATAMI_ROW_U = 5     # 0.5 mm tatami row spacing
        STITCH_4MM   = 40    # 4 mm tatami stitch length
        MIN_SHAPE_U  = 8     # 0.8 mm minimum shape bbox side

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # cluster_data: {cidx: {"rgb", "brightness", "contours": [(arr, fill_density, stitch_t)]}}
        cluster_data: dict        = {}
        vectorized_cidxs: set     = set()

        for cidx in range(k):
            if cidx in bg_clusters:
                continue
            rgb        = tuple(int(x) for x in centers[cidx])
            brightness = (rgb[0] + rgb[1] + rgb[2]) / 3.0
            if brightness > 200:
                continue

            mask = ((labels_2d == cidx).astype(np.uint8) * 255)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)

            # Vectorize with potrace; fall back to cv2.findContours
            raw_contours = None
            if _potrace_available():
                raw_contours = _vectorize_mask(mask, vec_tmp, cidx)
                if raw_contours is not None:
                    vectorized_cidxs.add(cidx)
            if raw_contours is None:
                raw_c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                raw_contours = list(raw_c)

            cluster_mask  = (labels_2d == cidx).astype(np.uint8) * 255
            contours_info = []
            for contour in raw_contours:
                if cv2.contourArea(contour) < 15:
                    continue
                c_mask = np.zeros((img_h, img_w), np.uint8)
                cv2.drawContours(c_mask, [contour], -1, 255, cv2.FILLED)
                c_area = int(np.count_nonzero(c_mask))
                if c_area > 0:
                    fill_density = (
                        int(np.count_nonzero(cv2.bitwise_and(c_mask, cluster_mask))) / c_area
                    )
                else:
                    fill_density = 1.0

                # Stitch type:
                #   brightness < 60  → ALWAYS running stitch (black, very dark colors)
                #   fill_density > 0.75 → tatami fill (solid colored shapes, incl. dark red)
                #   otherwise → running stitch outline
                if brightness < 60:
                    stitch_t = "running"
                elif fill_density > 0.75:
                    stitch_t = "tatami"
                else:
                    stitch_t = "running"

                contours_info.append((contour, fill_density, stitch_t))

            if contours_info:
                cluster_data[cidx] = {
                    "rgb":        rgb,
                    "brightness": brightness,
                    "contours":   contours_info,
                }

        if not cluster_data:
            return jsonify({"error": (
                "No design elements detected — try an image with clearer edges "
                "and higher contrast"
            )}), 400

        # ── STEP 5: Scaling and Centering ─────────────────────────────────────
        # Collect ALL contour points from ALL clusters for one unified bounding box.
        # pyembroidery origin (0,0) = hoop centre; +X right, +Y down.
        all_px = [int(p[0][0]) for d in cluster_data.values() for c, _, _ in d["contours"] for p in c]
        all_py = [int(p[0][1]) for d in cluster_data.values() for c, _, _ in d["contours"] for p in c]
        if not all_px:
            return jsonify({"error": "No contour points found after vectorization"}), 400

        px_min_x     = min(all_px);  px_max_x = max(all_px)
        px_min_y     = min(all_py);  px_max_y = max(all_py)
        pixel_width  = max(px_max_x - px_min_x, 1)
        pixel_height = max(px_max_y - px_min_y, 1)
        scale        = min(hoop_w_u / pixel_width, hoop_h_u / pixel_height) * 0.90
        design_w_u   = pixel_width  * scale
        design_h_u   = pixel_height * scale

        print(
            f"DIGITIZE SCALE: px_bbox={pixel_width}×{pixel_height}  scale={scale:.3f}"
            f"  design={design_w_u/10:.1f}×{design_h_u/10:.1f}mm"
            f"  hoop={hoop_w_u/10:.1f}×{hoop_h_u/10:.1f}mm",
            flush=True,
        )

        def px_to_emb(px, py):
            """Pixel → embroidery units, centered at hoop origin (0,0)."""
            return (
                int((px - px_min_x) * scale - design_w_u / 2),
                int((py - px_min_y) * scale - design_h_u / 2),
            )

        def emb_to_px_f(ex, ey):
            """Embroidery units → floating-point pixel coords (inverse of px_to_emb)."""
            return (
                (ex + design_w_u / 2) / scale + px_min_x,
                (ey + design_h_u / 2) / scale + px_min_y,
            )

        # ── STEP 6: Stitch Generation ──────────────────────────────────────────
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
            loop = pts + [pts[0]]
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

        def tatami_fill(pat, contour, row_u=TATAMI_ROW_U, stitch_u=STITCH_4MM, max_rows=50):
            """Scanline tatami fill with pointPolygonTest clipping.
            0.5 mm row spacing; odd rows offset by half a stitch length."""
            contour_f = contour.astype(np.float32)
            emb_verts = np.array(
                [px_to_emb(int(p[0][0]), int(p[0][1])) for p in contour],
                dtype=np.float64,
            )
            if len(emb_verts) < 3:
                return
            mn_x = float(emb_verts[:, 0].min());  mx_x = float(emb_verts[:, 0].max())
            mn_y = float(emb_verts[:, 1].min());  mx_y = float(emb_verts[:, 1].max())
            total_span      = mx_y - mn_y
            effective_row_u = max(row_u, total_span / max_rows) if total_span > 0 else row_u
            row = 0
            y_e = mn_y
            while y_e <= mx_y and row < max_rows:
                x_start = mn_x + (stitch_u / 2 if row % 2 == 1 else 0)
                x_e     = x_start
                while x_e <= mx_x:
                    px_t, py_t = emb_to_px_f(x_e, y_e)
                    if cv2.pointPolygonTest(contour_f, (float(px_t), float(py_t)), False) >= 0:
                        pat.add_stitch_absolute(pyembroidery.STITCH, int(x_e), int(y_e))
                    x_e += stitch_u
                y_e += effective_row_u
                row += 1

        # Sort colors darkest to lightest (dark outlines sewn first, fills on top)
        sorted_cidxs = sorted(cluster_data.keys(), key=lambda c: cluster_data[c]["brightness"])

        def _build_pattern(tatami_row_u=TATAMI_ROW_U):
            pat          = pyembroidery.EmbPattern()
            first_thread = True
            has_stitches = False

            for cidx in sorted_cidxs:
                info = cluster_data[cidx]
                rgb  = info["rgb"]

                thread       = pyembroidery.EmbThread()
                thread.color = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]
                thread.name  = "K#{:02X}{:02X}{:02X}".format(*rgb)
                pat.add_thread(thread)

                if not first_thread:
                    pat.add_stitch_absolute(pyembroidery.COLOR_CHANGE, 0, 0)
                first_thread = False

                for contour, fill_density, stitch_t in info["contours"]:
                    # Simplify non-potrace contours to reduce noise
                    if do_simplify and cidx not in vectorized_cidxs:
                        simplified = cv2.approxPolyDP(contour, epsilon=1.5, closed=True)
                        if len(simplified) >= 6 or cv2.isContourConvex(simplified):
                            contour = simplified

                    emb_pts = [px_to_emb(int(p[0][0]), int(p[0][1])) for p in contour]
                    if len(emb_pts) < 2:
                        continue

                    _, _, c_w_px, c_h_px = cv2.boundingRect(contour)
                    c_w_emb = c_w_px * scale
                    c_h_emb = c_h_px * scale
                    if c_w_emb < MIN_SHAPE_U or c_h_emb < MIN_SHAPE_U:
                        continue

                    c_w_mm = c_w_emb / 10.0
                    c_h_mm = c_h_emb / 10.0

                    # Thin-text detection: bbox < 4mm in either dimension → finer spacing
                    is_thin          = c_w_mm < 4.0 or c_h_mm < 4.0
                    outline_stitch_u = STITCH_1_5MM if is_thin else STITCH_2MM

                    # Appliqué detection: large hollow shape (> 40mm × 40mm, fd < 0.3)
                    is_applique = (
                        applique_mode
                        and fill_density < 0.3
                        and c_w_mm > 40.0
                        and c_h_mm > 40.0
                    )

                    print(
                        f"DIGITIZE: contour {c_w_mm:.1f}×{c_h_mm:.1f}mm "
                        f"bright={info['brightness']:.0f} fd={fill_density:.2f} type={stitch_t}",
                        flush=True,
                    )

                    start = emb_pts[0]
                    end   = emb_pts[-1]

                    if is_applique:
                        # 3-pass appliqué: placement line → tack-down (2mm inset) → border
                        pat.add_stitch_absolute(pyembroidery.TRIM, start[0], start[1])
                        tie_on(pat, start[0], start[1])
                        running_outline(pat, contour, stitch_u=STITCH_2MM)
                        tie_off(pat, end[0], end[1])
                        # Pass 2: erode mask by 2mm to get inner tack-down contour
                        _ap_mask = np.zeros((img_h, img_w), np.uint8)
                        cv2.drawContours(_ap_mask, [contour], -1, 255, cv2.FILLED)
                        _kern_td = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (max(1, int(20 / scale)), max(1, int(20 / scale)))
                        )
                        _inner = cv2.erode(_ap_mask, _kern_td, iterations=1)
                        _ic, _ = cv2.findContours(_inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if _ic:
                            _td     = max(_ic, key=cv2.contourArea)
                            _td_pts = [px_to_emb(int(p[0][0]), int(p[0][1])) for p in _td]
                            if len(_td_pts) >= 2:
                                pat.add_stitch_absolute(pyembroidery.TRIM, _td_pts[0][0], _td_pts[0][1])
                                tie_on(pat, _td_pts[0][0], _td_pts[0][1])
                                running_outline(pat, _td, stitch_u=STITCH_2MM)
                                tie_off(pat, _td_pts[-1][0], _td_pts[-1][1])
                        # Pass 3: satin border along original contour
                        pat.add_stitch_absolute(pyembroidery.TRIM, start[0], start[1])
                        tie_on(pat, start[0], start[1])
                        running_outline(pat, contour, stitch_u=STITCH_2MM)
                        tie_off(pat, end[0], end[1])
                        print(f"DIGITIZE: appliqué 3-pass {c_w_mm:.1f}×{c_h_mm:.1f}mm", flush=True)

                    elif stitch_t == "tatami":
                        # Running underlay + tatami scanline fill
                        pat.add_stitch_absolute(pyembroidery.TRIM, start[0], start[1])
                        tie_on(pat, start[0], start[1])
                        running_outline(pat, contour, stitch_u=outline_stitch_u)
                        pat.add_stitch_absolute(pyembroidery.TRIM, start[0], start[1])
                        tatami_fill(pat, contour, row_u=tatami_row_u)
                        tie_off(pat, end[0], end[1])

                    else:
                        # Running stitch outline (dark shapes, hollow shapes)
                        pat.add_stitch_absolute(pyembroidery.TRIM, start[0], start[1])
                        tie_on(pat, start[0], start[1])
                        running_outline(pat, contour, stitch_u=outline_stitch_u)
                        tie_off(pat, end[0], end[1])

                    has_stitches = True

            return pat, has_stitches

        pattern, any_stitches = _build_pattern()
        if not any_stitches:
            return jsonify({"error": "No stitches generated — try a different image"}), 400

        # Stitch density cap: max 5,000 stitches; widen tatami rows up to 6 times
        _row_u = TATAMI_ROW_U
        for _cap_pass in range(6):
            _sc = sum(1 for s in pattern.stitches if s[2] == pyembroidery.STITCH)
            if _sc <= 5000:
                break
            _row_u = max(_row_u + 1, round(_row_u * 1.2))
            print(
                f"DIGITIZE: density cap — {_sc} > 5000; "
                f"pass {_cap_pass + 1}/6 → row_u={_row_u} ({_row_u/10:.1f}mm)",
                flush=True,
            )
            pattern, any_stitches = _build_pattern(tatami_row_u=_row_u)
            if not any_stitches:
                return jsonify({"error": "No stitches generated after density adjustment"}), 400

        pattern.add_stitch_absolute(pyembroidery.END, 0, 0)
        pyembroidery.write(pattern, tmp_out.name)

        # ── STEP 7: Verification ───────────────────────────────────────────────
        verify = pyembroidery.read(tmp_out.name)
        if verify is None:
            return jsonify({"error": "Could not read back output file"}), 500

        v_s = [s for s in verify.stitches if s[2] == pyembroidery.STITCH]
        if not v_s:
            return jsonify({"error": "Output file contains no stitches"}), 400

        vxs = [s[0] for s in v_s];  vys = [s[1] for s in v_s]
        ow  = (max(vxs) - min(vxs)) / 10.0
        oh  = (max(vys) - min(vys)) / 10.0
        sc  = len(v_s)
        print(f"DIGITIZE VERIFY: {sc} stitches  {ow:.1f}mm × {oh:.1f}mm", flush=True)

        if sc < 100 or sc > 50000:
            return jsonify({"error": f"Stitch count {sc} is outside valid range 100–50000"}), 400
        if ow < 5 or oh < 5:
            return jsonify({"error": f"Design too small: {ow:.1f}×{oh:.1f}mm (minimum 5mm per side)"}), 400

        info_d   = get_design_info(pattern)
        mime     = MIME_TYPES.get(output_format, "application/octet-stream")
        response = send_file(tmp_out.name, mimetype=mime, as_attachment=True,
                             download_name="digitized" + output_format)
        response.headers["stitch_count"]   = str(info_d["stitch_count"])
        response.headers["color_count"]    = str(info_d["color_count"])
        response.headers["width_mm"]       = str(info_d["width_mm"])
        response.headers["height_mm"]      = str(info_d["height_mm"])
        response.headers["estimated_time"] = str(info_d["estimated_time_minutes"])
        print(
            f"DIGITIZE OK: stitch_count={info_d['stitch_count']} color_count={info_d['color_count']}"
            f" width_mm={info_d['width_mm']} height_mm={info_d['height_mm']}",
            flush=True,
        )
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
        shutil.rmtree(vec_tmp, ignore_errors=True)


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

        xs = [s[0] for s in pattern.stitches if s[2] in (pyembroidery.STITCH, pyembroidery.JUMP)]
        ys = [s[1] for s in pattern.stitches if s[2] in (pyembroidery.STITCH, pyembroidery.JUMP)]

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
        xs = [s[0] for s in stitches if s[2] in (pyembroidery.STITCH, pyembroidery.JUMP)]
        ys = [s[1] for s in stitches if s[2] in (pyembroidery.STITCH, pyembroidery.JUMP)]

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
                    x, y, cmd = stitch[0], stitch[1], stitch[2]
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

        xs = [s[0] for s in pattern.stitches if s[2] in (pyembroidery.STITCH, pyembroidery.JUMP)]
        ys = [s[1] for s in pattern.stitches if s[2] in (pyembroidery.STITCH, pyembroidery.JUMP)]

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
            new_pattern.add_stitch_absolute(cmd, int(x * scale_x), int(y * scale_y))

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
            new_pattern.add_stitch_absolute(cmd, nx, ny)

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
            xs = [s[0] for s in p.stitches if s[2] in (pyembroidery.STITCH, pyembroidery.JUMP)]
            ys = [s[1] for s in p.stitches if s[2] in (pyembroidery.STITCH, pyembroidery.JUMP)]
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
                x, y, cmd = stitch[0], stitch[1], stitch[2]
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
