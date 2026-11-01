#!/usr/bin/env python3
"""refinedocr - PDF, CBZ and image OCR to Markdown/TXT/ODT.

Uses local Tesseract by default, or an OCR model on a local llama-server
(OpenAI-compatible API). PyMuPDF renders PDF pages (and CBZ comic archives)
to images; each page is sent to Tesseract or to the model with the
model-specific prompt ("OCR markdown" by default).

Examples:
    python src/refinedocr.py samples/scan.pdf
    python src/refinedocr.py samples/scan.pdf -o scan.odt
    python src/refinedocr.py photo.jpg --format txt
    python src/refinedocr.py samples/sample.cbz -o comic.md
    python src/refinedocr.py samples/scan.pdf --pages 1-10 --dpi 150 ^
        --engine glm --server http://10.0.2.2:8080
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# PyMuPDF 2.x auto-imports its optional ONNX layout analyzer at import time.
# We only use rendering/text extraction, and PyInstaller does not bundle the
# ONNX resources. A None entry makes that import raise ImportError, which
# PyMuPDF catches and skips. It also avoids loading the analyzer in normal runs.
sys.modules["pymupdf.layout"] = None

try:
    import pymupdf
except ImportError:  # pragma: no cover - dependency check
    sys.exit("PyMuPDF is required: pip install pymupdf")

DEFAULT_SERVER = "http://127.0.0.1:8080"
DEFAULT_MODEL = "GLM-OCR"
DEFAULT_PROMPT = "OCR markdown"
DEFAULT_DPI = 200
# Tesseract is local and cheap, so it keeps more pixels than the slow
# CPU-bound vision prefill of the server engine.
DEFAULT_MAX_SIDE = 1200
DEFAULT_MAX_SIDE_GLM = 600
DEFAULT_MAX_TOKENS = 3072
DEFAULT_TIMEOUT = 180
DEFAULT_ENGINE = "tesseract"
DEFAULT_ENGINE_LABEL = "OCR model (server)"
DEFAULT_TESSERACT_LANG = "rus+eng"
SETTINGS_FILENAME = "refinedocr.ini"
# A page this much wider than tall is treated as a two-page scan in the
# "auto" spreads mode. Dendy guide spreads measure ~1.15-1.25.
AUTO_SPREAD_MIN_ASPECT = 1.1

TESSERACT_MISSING = (
    "Tesseract not found. Put tesseract.exe and tessdata/ in a 'tesseract' "
    "folder next to the app/project, install Tesseract-OCR, or set "
    "TESSERACT_CMD / tesseract_path in refinedocr.ini. Put *.traineddata "
    "files in tesseract/tessdata/ next to the app/project (or set "
    "TESSDATA_PREFIX)."
)

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
# Comic-book ZIP archives: JPEG/PNG pages inside a zip (stored or deflated).
# PyMuPDF opens them natively, ignores non-image entries, and skips images it
# cannot decode (notably WebP/AVIF).
COMIC_SUFFIXES = {".cbz"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES | COMIC_SUFFIXES
OUTPUT_FORMATS = ("md", "txt", "odt", "docx", "html")

# The OCR server is local. Ignore HTTP(S)_PROXY / Windows proxy settings:
# a VPN or proxy tool intercepting 127.0.0.1 is a common cause of
# "server is listening but unreachable".
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class OCRCancelled(Exception):
    """Raised when a GUI/worker cancellation is requested mid-stream."""


def normalize_server_url(value: str) -> str:
    """Return a base URL: scheme present, no trailing slash, no /v1 suffix."""
    url = (value or "").strip().rstrip("/")
    if not url:
        return DEFAULT_SERVER
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    for suffix in ("/v1/chat/completions", "/v1"):
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
            break
    return url


def log(message: str) -> None:
    stream = sys.stderr or sys.stdout
    if stream is None:  # windowed exe: no console attached
        return
    print(message, file=stream, flush=True)


def extract_text_layer(page) -> str:
    """Read a PDF page's text layer, preserving block/paragraph boundaries."""
    blocks = page.get_text("blocks")
    return "\n\n".join(block[4].strip() for block in blocks if block[4].strip())


def parse_pages(spec: str | None, total: int) -> list[int]:
    """Parse 1-based page ranges: None -> all, '1-3,5' -> [1,2,3,5]."""
    if not spec:
        return list(range(1, total + 1))
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                first, last = part.split("-", 1)
                pages.extend(range(int(first), int(last) + 1))
            else:
                pages.append(int(part))
        except ValueError:
            raise SystemExit(f"Invalid --pages value: {part!r}")
    unique = list(dict.fromkeys(pages))
    selected = [p for p in unique if 1 <= p <= total]
    if not selected:
        raise SystemExit(f"--pages selected no pages (document has {total}).")
    return selected


def native_image_scale(page) -> float:
    """Scale that renders an image page at its source pixel resolution.

    PyMuPDF maps image documents to points using the embedded DPI (e.g. a
    300 dpi scan gets a page rect 4x smaller than its pixels). A scale of 1.0
    would therefore throw away 94% of the pixels; this restores 1:1.
    """
    try:
        info = page.get_image_info()
    except Exception:
        return 1.0
    if not info:
        return 1.0
    width = info[0].get("width") or 0
    height = info[0].get("height") or 0
    rect = page.rect
    if width <= 0 or height <= 0 or rect.width <= 0 or rect.height <= 0:
        return 1.0
    return max(width / rect.width, height / rect.height)


def page_scale(
    page, is_pdf: bool, dpi: int, max_side: int, clip=None
) -> float:
    """Render scale: DPI for PDFs, source pixels for images, capped by max_side.

    With a ``clip`` (GUI region selection) the cap applies to the crop, so a
    small region keeps full DPI instead of inheriting the whole-page limit.
    """
    rect = page.rect if clip is None else clip
    scale = (dpi / 72.0) if is_pdf else native_image_scale(page)
    longest = max(rect.width, rect.height)
    if max_side > 0 and longest > 0:
        scale = min(scale, max_side / longest)
    return max(scale, 0.01)


def render_page_png(
    page, is_pdf: bool, dpi: int, max_side: int, clip=None
) -> bytes:
    """PNG of the whole page, or of ``clip`` (display-space page rect)."""
    scale = page_scale(page, is_pdf, dpi, max_side, clip)
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        clip=clip,
        colorspace=pymupdf.csRGB,
        alpha=False,
    )
    return pix.tobytes("png")


def png_size(data: bytes) -> tuple[int, int]:
    """Return (width, height) from a PNG header without decoding it."""
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    return 0, 0


def spread_clips(page, mode: str, fraction: float = 0.5) -> list:
    """Left/right clip rects for a landscape two-page spread, in reading order.

    ``mode`` is "ltr" (left page first, western books), "rtl" (right page
    first, manga), or "auto" (split only clearly landscape pages, LTR);
    ``fraction`` is the split position from the left edge. Returns [] for
    portrait pages, for "off", and for "auto" pages that are not wide enough
    to be a two-page scan.
    """
    if mode not in ("auto", "ltr", "rtl"):
        return []
    rect = page.rect
    if rect.width <= rect.height:
        return []
    if mode == "auto" and rect.width / rect.height < AUTO_SPREAD_MIN_ASPECT:
        return []
    fraction = min(max(fraction, 0.1), 0.9)
    middle = rect.x0 + rect.width * fraction
    left = pymupdf.Rect(rect.x0, rect.y0, middle, rect.y1)
    right = pymupdf.Rect(middle, rect.y0, rect.x1, rect.y1)
    return [left, right] if mode != "rtl" else [right, left]


def app_dir() -> Path:
    """Folder next to the app: the exe when frozen, else this source file."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def project_root() -> Path:
    """Project folder when running from source, else the app/exe folder.

    Source lives in ``src/``; portable data (``tesseract/``, ``tessdata/``,
    ``refinedocr.ini``) stays in the project root next to it. A flat source
    layout (refinedocr.py in the root) still returns that folder.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    base = Path(__file__).resolve().parent
    if base.name.lower() == "src":
        return base.parent
    return base


def _app_dirs() -> list[Path]:
    """Portable locations, most specific first (no duplicates)."""
    dirs = [app_dir(), project_root()]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(Path(meipass))
    dirs.append(Path(__file__).resolve().parent)
    unique: list[Path] = []
    for directory in dirs:
        if directory not in unique:
            unique.append(directory)
    return unique


def find_tesseract() -> str | None:
    """Locate a portable or installed Tesseract binary.

    Portable layout next to the source/app/exe wins over PATH, so a project or
    app folder can ship its own ``tesseract/`` (tesseract.exe + DLLs +
    tessdata).
    """
    env = os.environ.get("TESSERACT_CMD", "").strip()
    if env and Path(env).is_file():
        return env
    for base in _app_dirs():
        for directory in (base / "tesseract", base):
            for name in ("tesseract.exe", "tesseract"):
                candidate = directory / name
                if candidate.is_file():
                    return str(candidate)
            nested = directory / "Tesseract-OCR" / "tesseract.exe"
            if nested.is_file():
                return str(nested)
    found = shutil.which("tesseract")
    if found:
        return found
    for directory in (
        Path(r"C:\Program Files\Tesseract-OCR"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR"),
    ):
        for name in ("tesseract.exe", "tesseract"):
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    return None


def find_tessdata(exe: str | None = None) -> Path | None:
    """Find a folder containing *.traineddata next to the app/root or via env."""
    candidates: list[Path] = []
    env = os.environ.get("TESSDATA_PREFIX", "").strip()
    if env:
        candidates += [Path(env), Path(env) / "tessdata"]
    if exe:
        binary = Path(exe)
        candidates += [binary.parent / "tessdata", binary.parent.parent / "tessdata"]
    for base in _app_dirs():
        candidates += [base / "tesseract" / "tessdata", base / "tessdata"]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.traineddata")):
            return candidate
    return None


def list_tesseract_languages(
    tessdata: str | Path | None = None,
    exe: str | None = None,
) -> list[str]:
    """Return the language codes available as *.traineddata, sorted."""
    data_dir = Path(tessdata) if tessdata else find_tessdata(exe)
    if data_dir is None or not data_dir.is_dir():
        return []
    suffix = ".traineddata"
    return sorted(
        entry.name[: -len(suffix)]
        for entry in data_dir.glob("*" + suffix)
        if entry.is_file()
    )


def resolve_tesseract_lang(
    lang: str,
    tessdata: str | Path | None = None,
    exe: str | None = None,
) -> tuple[str, str]:
    """Validate a Tesseract language string against the available data.

    Returns ``(language, warning)``. A missing language falls back to
    ``rus+eng`` (or the subset that exists) so a typo does not break a run.
    When no tessdata is discoverable, the value is passed through untouched
    (Tesseract may have its own default language path).
    """
    requested = (lang or "").strip() or DEFAULT_TESSERACT_LANG
    available = set(list_tesseract_languages(tessdata, exe))
    if not available:
        return requested, ""
    parts = [part.strip() for part in requested.split("+") if part.strip()]
    missing = [part for part in parts if part not in available]
    if not missing:
        return requested, ""
    fallback = "+".join(
        part for part in ("rus", "eng") if part in available
    )
    if fallback:
        return fallback, (
            f"Tesseract languages {requested!r} not found "
            f"({', '.join(missing)} missing); using {fallback!r}."
        )
    return requested, (
        f"Tesseract languages {requested!r} not found; available: "
        f"{', '.join(sorted(available))}."
    )


def tesseract_ocr(
    png: bytes,
    lang: str = DEFAULT_TESSERACT_LANG,
    exe: str | None = None,
    tessdata: str | None = None,
    psm: int = 3,
) -> str:
    """OCR a PNG with an external Tesseract binary; returns plain text."""
    executable = exe or find_tesseract()
    if not executable:
        raise RuntimeError(TESSERACT_MISSING)
    data_dir = Path(tessdata) if tessdata else find_tessdata(executable)
    lang, warning = resolve_tesseract_lang(lang, data_dir)
    if warning:
        log(f"WARNING: {warning}")
    # Feed the PNG through stdin instead of a temp file: no per-page disk
    # write/read (noticeable when TEMP or the source is on a slow drive).
    command = [
        executable, "stdin", "stdout", "-l", lang or "eng",
        "--psm", str(psm),
    ]
    if data_dir is not None:
        command += ["--tessdata-dir", str(data_dir)]
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
    try:
        proc = subprocess.run(
            command,
            input=png,
            capture_output=True,
            timeout=300,
            **kwargs,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(TESSERACT_MISSING) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Tesseract timed out after 300 s") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or b"").decode(
            "utf-8", "replace"
        ).strip()[:500]
        raise RuntimeError(
            f"Tesseract failed: {detail or 'unknown error'}"
        )
    return proc.stdout.decode("utf-8", "replace").strip()


def ocr_png(
    server: str,
    model: str,
    png: bytes,
    prompt: str,
    max_tokens: int,
    timeout: int,
    label: str = "",
    on_chunk=None,
    should_stop=None,
) -> tuple[str, dict, str]:
    """Stream one page from llama-server; return (text, usage, finish_reason).

    Streaming matters: a non-streaming client that gives up leaves the server
    generating on its slot with nobody reading, which can stall the server.
    ``timeout`` is the maximum silence between SSE events, not total time.
    """
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    }
    url = normalize_server_url(server) + "/v1/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    chunks: list[str] = []
    char_count = 0
    usage: dict = {}
    finish_reason = ""
    started = time.time()
    last_report = started
    interactive = bool(label) and sys.stderr is not None and sys.stderr.isatty()

    try:
        with _OPENER.open(request, timeout=timeout) as response:
            for raw_line in response:
                if should_stop is not None and should_stop():
                    raise OCRCancelled("cancelled by user")
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in event:
                    raise RuntimeError(f"server error: {event['error']}")
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if piece:
                        chunks.append(piece)
                        char_count += len(piece)
                        if on_chunk is not None:
                            on_chunk(piece)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                if interactive:
                    now = time.time()
                    if now - last_report >= 1.0:
                        last_report = now
                        sys.stderr.write(
                            f"\r{label}: OCR {char_count} chars, "
                            f"{now - started:.0f}s"
                        )
                        sys.stderr.flush()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"server returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach {server}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"no data from {server} for {timeout}s "
            f"(raise --timeout or lower --max-side)"
        ) from exc
    finally:
        if interactive and char_count:
            sys.stderr.write("\n")
            sys.stderr.flush()

    return "".join(chunks), usage, finish_reason


def server_status(server: str, timeout: int = 5) -> tuple[bool, str]:
    """Check /health and return (ready, human-readable explanation)."""
    base = normalize_server_url(server)
    url = base + "/health"
    try:
        with _OPENER.open(url, timeout=timeout) as response:
            body = response.read(8192)
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            return False, (
                f"{base} is reachable but still loading the model "
                f"(HTTP 503); wait for it to finish"
            )
        return False, f"{base} answered HTTP {exc.code} for /health"
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        hint = ""
        if "localhost" in base:
            hint = " (try http://127.0.0.1:8080 if localhost resolves to IPv6)"
        return False, f"cannot connect to {base}: {reason}{hint}"
    except TimeoutError:
        return False, f"{base} did not answer /health within {timeout}s"
    except Exception as exc:  # noqa: BLE001 - diagnostic message
        return False, f"cannot connect to {base}: {exc}"

    try:
        payload = json.loads(body.decode("utf-8", "replace"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return True, f"{base} answered /health (non-JSON response)"
    if payload.get("status") == "ok":
        return True, f"{base} is ready"
    return False, f"{base} answered /health with: {payload}"


def check_server(server: str, timeout: int = 5) -> bool:
    """Return True when llama-server answers /health with status ok."""
    return server_status(server, timeout)[0]


def clean_markdown(text: str) -> str:
    """Drop a wrapping ```markdown fence if the model added one."""
    stripped = text.strip()
    match = re.fullmatch(r"```(?:markdown|md)?[ \t]*\n(.*)\n```", stripped, re.S)
    if match:
        return match.group(1).strip()
    return stripped


def markdown_to_plain(markdown: str) -> str:
    """Convert Markdown to plain text, preferring Pandoc."""
    try:
        import pypandoc

        return pypandoc.convert_text(markdown, "plain", format="markdown").strip() + "\n"
    except Exception:
        pass
    text = re.sub(r"```.*?```", "", markdown, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.M)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    text = re.sub(r"^\s*[-*+]\s+", "- ", text, flags=re.M)
    return text.strip() + "\n"


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def make_odt_aoo_compatible(path: Path) -> None:
    """Relabel a pandoc ODT package from ODF 1.3 to 1.2.

    Pandoc 3.x declares ODF 1.3, which Apache OpenOffice (1.2 only) flags as
    "unsupported version" but then opens anyway after upgrading. The writer
    only emits 1.2-era elements, so the version label is the whole difference.
    The mimetype entry must stay first and uncompressed.
    """
    import io
    import zipfile

    with zipfile.ZipFile(path) as source:
        entries = [
            (info.filename, source.read(info.filename))
            for info in source.infolist()
        ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as target:
        for name, data in entries:
            if name.endswith(".xml"):
                data = (
                    data.decode("utf-8")
                    .replace('office:version="1.3"', 'office:version="1.2"')
                    .replace(
                        'manifest:version="1.3"', 'manifest:version="1.2"'
                    )
                    .encode("utf-8")
                )
            stored = name == "mimetype"
            target.writestr(
                name,
                data,
                compress_type=(
                    zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
                ),
            )
    path.write_bytes(buffer.getvalue())


def convert_with_pandoc(markdown: str, fmt: str, path: Path) -> None:
    """Convert Markdown to odt/docx/html via pypandoc or the pandoc CLI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp{path.suffix}")
    try:
        try:
            import pypandoc

            pypandoc.convert_text(
                markdown,
                fmt,
                format="markdown",
                outputfile=str(tmp_path),
                extra_args=["--standalone"],
            )
        except Exception as primary_error:
            pandoc = shutil.which("pandoc")
            if pandoc is None:
                raise RuntimeError(
                    f"cannot write {fmt}: install Pandoc "
                    f"(pip install pypandoc_binary). {primary_error}"
                ) from primary_error
            with tempfile.NamedTemporaryFile(
                "w", suffix=".md", encoding="utf-8", delete=False
            ) as handle:
                handle.write(markdown)
                source = handle.name
            try:
                subprocess.run(
                    [pandoc, source, "-f", "markdown", "-t", fmt,
                     "-o", str(tmp_path)],
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"pandoc conversion to {fmt} failed") from exc
            finally:
                os.unlink(source)
        if fmt == "odt":
            make_odt_aoo_compatible(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def save_output(markdown: str, fmt: str, path: Path) -> None:
    if fmt == "md":
        write_atomic(path, markdown if markdown.endswith("\n") else markdown + "\n")
    elif fmt == "txt":
        write_atomic(path, markdown_to_plain(markdown))
    elif fmt in ("odt", "docx", "html"):
        convert_with_pandoc(markdown, fmt, path)
    else:  # pragma: no cover - argparse validates
        raise SystemExit(f"Unsupported format: {fmt}")


def process_file(source: Path, out_path: Path, fmt: str, args) -> None:
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise SystemExit(
            f"{source}: unsupported file type {suffix!r} "
            f"(supported: {', '.join(sorted(SUPPORTED_SUFFIXES))})"
        )
    if not source.is_file():
        raise SystemExit(f"{source}: no such file")

    try:
        document = pymupdf.open(source)
    except Exception as exc:
        raise SystemExit(f"{source}: cannot open: {exc}") from exc

    with document:
        if document.needs_pass:
            raise SystemExit(f"{source}: password-protected PDFs are not supported")
        total = document.page_count
        if total == 0:
            if suffix in COMIC_SUFFIXES:
                raise SystemExit(
                    f"{source}: no readable image pages in the archive "
                    f"(WebP/AVIF entries are not supported)"
                )
            raise SystemExit(f"{source}: document has no pages")
        is_pdf = suffix in PDF_SUFFIXES
        pages = parse_pages(args.pages, total)
        spread_mode = getattr(args, "spreads", "off")
        texts: list[str] = []
        ocr_pages = 0

        for index, page_number in enumerate(pages, start=1):
            page = document.load_page(page_number - 1)
            prefix = f"[{index}/{len(pages)}] page {page_number}"
            if page.rect.is_empty:
                # A CBZ entry in a format MuPDF cannot decode (e.g. BMP)
                # becomes a 0x0 page; skip it instead of failing the file.
                log(f"{prefix}: unreadable page, skipped")
                continue

            clips = spread_clips(
                page, spread_mode, getattr(args, "split_at", 0.5)
            )
            if clips:
                log(f"{prefix}: spread {spread_mode.upper()} (2 halves)")
            else:
                text = extract_text_layer(page) if is_pdf else ""
                if text and not args.force_ocr:
                    texts.append(text)
                    log(f"{prefix}: text layer ({len(text)} chars)")
                    continue
                clips = [None]

            results: list[str] = []
            prompt_total = 0
            completion_total = 0
            started = time.time()
            for clip_index, clip in enumerate(clips, 1):
                png = render_page_png(
                    page, is_pdf, args.dpi, args.max_side, clip=clip
                )
                width, height = png_size(png)
                where = f"half {clip_index} " if len(clips) > 1 else ""
                engine = getattr(args, "engine", "glm")
                how = "tesseract OCR" if engine == "tesseract" else "OCR"
                log(
                    f"{prefix}: {where}{how} ({width}x{height} px, "
                    f"{len(png) // 1024} KiB)..."
                )
                if engine == "tesseract":
                    result = tesseract_ocr(
                        png,
                        getattr(
                            args, "tesseract_lang", DEFAULT_TESSERACT_LANG
                        ),
                        getattr(args, "tesseract", "") or None,
                    )
                    usage, finish_reason = {}, "stop"
                else:
                    content, usage, finish_reason = ocr_png(
                        args.server,
                        args.model,
                        png,
                        args.prompt,
                        args.max_tokens,
                        args.timeout,
                        label=prefix,
                    )
                    result = clean_markdown(content)
                if not result:
                    if len(clips) > 1:
                        log(f"{prefix}: half {clip_index} returned no text")
                        continue
                    raise SystemExit(
                        f"{source}: model returned no text for page {page_number}"
                    )
                results.append(result)
                prompt_total += usage.get("prompt_tokens") or 0
                completion_total += usage.get("completion_tokens") or 0
                if finish_reason == "length":
                    log(
                        f"{prefix}: WARNING output hit --max-tokens "
                        f"({args.max_tokens}); text may be truncated"
                    )
            page_text = "\n\n".join(results)
            if not page_text:
                raise SystemExit(
                    f"{source}: nothing recognized for page {page_number}"
                )
            texts.append(page_text)
            ocr_pages += 1
            log(
                f"{prefix}: {len(page_text)} chars, "
                f"{prompt_total}+{completion_total} tokens, "
                f"{time.time() - started:.1f}s"
            )

        markdown = "\n\n".join(part for part in texts if part.strip())
        if not markdown:
            raise SystemExit(f"{source}: nothing recognized")

    if str(out_path) == "-":
        if fmt not in ("md", "txt"):
            raise SystemExit("stdout output supports only md/txt")
        sys.stdout.write(markdown if fmt == "md" else markdown_to_plain(markdown))
        return

    save_output(markdown, fmt, out_path)
    log(f"saved {out_path} ({fmt}, {len(pages)} page(s), {ocr_pages} OCRed)")


def infer_format(out: str | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if out and out != "-":
        suffix = Path(out).suffix.lower().lstrip(".")
        if suffix in OUTPUT_FORMATS:
            return suffix
    return "md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refinedocr",
        description=(
            "OCR PDFs, CBZ comics and images to Markdown/TXT/ODT using "
            "local Tesseract or an OCR model on llama-server."
        ),
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="PDF, CBZ or image file(s)"
    )
    parser.add_argument("-o", "--output", help="output file (or - for stdout)")
    parser.add_argument(
        "-f", "--format", choices=OUTPUT_FORMATS,
        help="output format (default: from -o suffix, else md)",
    )
    parser.add_argument(
        "--server", default=DEFAULT_SERVER,
        help=f"llama-server base URL (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"model name sent to the API (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help=f"OCR prompt (default: {DEFAULT_PROMPT!r})",
    )
    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI,
        help=f"render DPI for PDF pages only (default: {DEFAULT_DPI}); "
             f"images and CBZ ignore it",
    )
    parser.add_argument(
        "--max-side", type=int, default=None,
        help=f"cap rendered long side in pixels for all inputs, 0 = no cap "
             f"(default: {DEFAULT_MAX_SIDE} for tesseract, "
             f"{DEFAULT_MAX_SIDE_GLM} for the server engine); this is the "
             f"resolution control for images and CBZ",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"max output tokens per page (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--pages", help="page selection for PDFs, e.g. 1-5,8 (default: all)",
    )
    parser.add_argument(
        "--force-ocr", action="store_true",
        help="OCR every PDF page even when a text layer exists",
    )
    parser.add_argument(
        "--spreads", choices=("off", "auto", "ltr", "rtl"), default="off",
        help="split landscape two-page scans into halves: auto = split pages "
             f"at least {AUTO_SPREAD_MIN_ASPECT:g}x wider than tall (LTR), "
             "ltr = books, rtl = manga; default: off",
    )
    parser.add_argument(
        "--split-at", type=float, default=0.5, metavar="FRACTION",
        help="split position from the left edge for --spreads (default: 0.5)",
    )
    parser.add_argument(
        "--engine", choices=("glm", "tesseract"), default=DEFAULT_ENGINE,
        help=f"OCR engine (default: {DEFAULT_ENGINE}); tesseract uses a "
             f"local binary and needs no server, glm/OCR model needs "
             f"llama-server",
    )
    parser.add_argument(
        "--tesseract", default="", metavar="PATH",
        help="tesseract executable (default: auto-detect / TESSERACT_CMD)",
    )
    parser.add_argument(
        "--tesseract-lang", default=DEFAULT_TESSERACT_LANG,
        help=f"tesseract languages (default: {DEFAULT_TESSERACT_LANG})",
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"max silence between stream chunks in seconds (default: {DEFAULT_TIMEOUT})",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    fmt = infer_format(args.output, args.format)
    if args.max_side is None:
        args.max_side = (
            DEFAULT_MAX_SIDE
            if args.engine == "tesseract"
            else DEFAULT_MAX_SIDE_GLM
        )

    if args.engine == "tesseract":
        executable = args.tesseract or find_tesseract()
        if not executable:
            raise SystemExit(TESSERACT_MISSING)
        data_dir = find_tessdata(executable)
        if data_dir is not None:
            log(f"Tesseract: {executable} (tessdata {data_dir})")
        else:
            log(
                f"Tesseract: {executable} (no tessdata folder found; "
                f"using Tesseract's default language path)"
            )
        lang, warning = resolve_tesseract_lang(
            args.tesseract_lang, exe=executable
        )
        if warning:
            log(f"WARNING: {warning}")
        args.tesseract_lang = lang

    if args.output and len(args.inputs) > 1:
        parser.error("--output can only be used with a single input file")
    if str(args.output) == "-" and fmt not in ("md", "txt"):
        parser.error("stdout output supports only --format md/txt")

    for source in args.inputs:
        if args.output:
            out_path = Path(args.output) if args.output != "-" else Path("-")
        else:
            out_path = source.with_suffix(f".{fmt}")
        try:
            process_file(source, out_path, fmt, args)
        except RuntimeError as exc:
            raise SystemExit(f"{source}: {exc}") from exc


if __name__ == "__main__":
    main()
