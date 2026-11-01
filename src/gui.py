#!/usr/bin/env python3
"""refinedocr GUI - PySide6 desktop front end for Tesseract or OCR models.

Reuses the OCR core from refinedocr.py. Workflow:
  add PDFs/images -> Start OCR -> watch pages stream -> review each page
  side by side -> open the saved Markdown/TXT/ODT.

Run:
    python src/gui.py
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# A --windowed (no console) exe sets sys.stdout/sys.stderr to None; libraries
# that call .isatty() or write on them crash. Route them to devnull instead.
if sys.stdout is None or sys.stderr is None:
    _null_stream = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = _null_stream
    if sys.stderr is None:
        sys.stderr = _null_stream

# Skip PyMuPDF 2.x's ONNX layout analyzer (unused; its data is not bundled).
sys.modules["pymupdf.layout"] = None
import pymupdf
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QRectF,
    QSettings,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QDesktopServices,
    QFontDatabase,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

import refinedocr as core

APP_VERSION = "0.8"
# Keep previews light: the previous 1600 px/150 dpi previews made zooming
# expensive on large pages. Zoom stays digital, but the UI stays responsive.
PREVIEW_DPI = 100
PREVIEW_MAX_SIDE = 1200


@dataclass
class OcrSettings:
    server: str = core.DEFAULT_SERVER
    model: str = core.DEFAULT_MODEL
    prompt: str = core.DEFAULT_PROMPT
    dpi: int = core.DEFAULT_DPI
    max_side: int = core.DEFAULT_MAX_SIDE
    max_tokens: int = core.DEFAULT_MAX_TOKENS
    timeout: int = core.DEFAULT_TIMEOUT
    out_format: str = "md"
    output_dir: str = ""
    force_ocr: bool = False
    spreads: str = "off"
    engine: str = core.DEFAULT_ENGINE
    tesseract_lang: str = core.DEFAULT_TESSERACT_LANG
    tesseract_path: str = ""


@dataclass
class FileJob:
    path: Path
    pages: int = 0
    status: str = "Pending"
    page_pngs: list = field(default_factory=list)
    page_texts: list[str] = field(default_factory=list)
    regions: dict[int, list[list[float]]] = field(default_factory=dict)
    splits: dict[int, float] = field(default_factory=dict)
    selected_pages: set[int] = field(default_factory=set)
    output_path: str = ""
    error: str = ""


class OCRWorker(QObject):
    """Runs the whole batch off the GUI thread."""

    log = Signal(str)
    file_started = Signal(int, str)
    file_pages = Signal(int, int)
    page_started = Signal(int, int, int, object)  # file, page, total, png bytes
    page_chunk = Signal(int, int, str)
    page_done = Signal(int, int, str)
    file_done = Signal(int, str)
    failed = Signal(int, str)
    finished = Signal()

    def __init__(self, jobs: list[FileJob], settings: OcrSettings):
        super().__init__()
        self.jobs = jobs
        self.settings = settings
        self.stop = threading.Event()

    def run(self) -> None:
        if self.settings.engine == "tesseract":
            executable = (
                self.settings.tesseract_path.strip()
                or core.find_tesseract()
            )
            if not executable:
                self.failed.emit(-1, core.TESSERACT_MISSING)
                self.finished.emit()
                return
            data_dir = core.find_tessdata(executable)
            if data_dir is not None:
                self.log.emit(
                    f"Tesseract: {executable} (tessdata {data_dir})"
                )
            else:
                self.log.emit(
                    f"Tesseract: {executable} (no tessdata folder found; "
                    f"using Tesseract's default language path)"
                )
            lang, warning = core.resolve_tesseract_lang(
                self.settings.tesseract_lang, exe=executable
            )
            if warning:
                self.log.emit(f"WARNING: {warning}")
            self.settings.tesseract_lang = lang
        else:
            ready, message = core.server_status(
                self.settings.server, timeout=5
            )
            if not ready:
                self.failed.emit(-1, message)
                self.finished.emit()
                return
        for index, job in enumerate(self.jobs):
            if self.stop.is_set():
                self.log.emit("Cancelled.")
                break
            if not job.selected_pages:
                job.status = "Skipped"
                self.log.emit(f"{job.path.name}: no pages checked, skipped")
                continue
            try:
                self._process(index, job)
            except core.OCRCancelled:
                self.log.emit("Cancelled.")
                break
            except Exception as exc:  # noqa: BLE001 - report per file, keep batch going
                job.status = "Error"
                job.error = str(exc)
                self.failed.emit(index, str(exc))
        self.finished.emit()

    def _process(self, index: int, job: FileJob) -> None:
        job.status = "Working"
        self.file_started.emit(index, job.path.name)
        try:
            document = pymupdf.open(job.path)
        except Exception as exc:
            raise RuntimeError(f"cannot open: {exc}") from exc

        with document:
            if document.needs_pass:
                raise RuntimeError("password-protected PDFs are not supported")
            total = document.page_count
            if total < 1:
                raise RuntimeError("document has no pages")
            job.pages = total
            if len(job.page_pngs) != total:
                job.page_pngs = [None] * total
            if len(job.page_texts) != total:
                job.page_texts = [""] * total
            selected = sorted(
                p for p in job.selected_pages if 1 <= p <= total
            )
            if not selected:
                job.status = "Skipped"
                return
            for page_number in selected:
                job.page_texts[page_number - 1] = ""
            self.file_pages.emit(index, total)
            is_pdf = job.path.suffix.lower() in core.PDF_SUFFIXES

            for page_number in selected:
                if self.stop.is_set():
                    raise core.OCRCancelled("cancelled by user")
                page = document.load_page(page_number - 1)
                if page.rect.is_empty:
                    # A CBZ entry in a format MuPDF cannot decode (e.g. BMP)
                    # becomes a 0x0 page; skip it instead of failing the file.
                    self.log.emit(
                        f"{job.path.name}: page {page_number} unreadable, "
                        f"skipped"
                    )
                    self.page_started.emit(index, page_number, total, b"")
                    self.page_done.emit(index, page_number, "")
                    continue
                preview = core.render_page_png(
                    page, is_pdf, min(self.settings.dpi, 120), 1200
                )
                self.page_started.emit(index, page_number, total, preview)

                regions = job.regions.get(page_number, [])
                clips = []
                if regions:
                    clips = [
                        self._clip_rect(page, region) for region in regions
                    ]
                    self.log.emit(
                        f"{job.path.name}: page {page_number} OCR "
                        f"({len(clips)} region(s))..."
                    )
                else:
                    spread = core.spread_clips(
                        page,
                        self.settings.spreads,
                        job.splits.get(page_number, 0.5),
                    )
                    if spread:
                        clips = spread
                        self.log.emit(
                            f"{job.path.name}: page {page_number} "
                            f"spread {self.settings.spreads.upper()} "
                            f"(2 halves)..."
                        )
                if not clips:
                    text = core.extract_text_layer(page) if is_pdf else ""
                    if text and not self.settings.force_ocr:
                        job.page_texts[page_number - 1] = text
                        self.log.emit(
                            f"{job.path.name}: page {page_number} text layer"
                        )
                        self.page_done.emit(index, page_number, text)
                        continue
                    clips = [None]

                started = time.time()
                pieces: list[str] = []
                prompt_tokens = 0
                completion_tokens = 0
                truncated = False
                for clip_index, clip in enumerate(clips, 1):
                    if self.stop.is_set():
                        raise core.OCRCancelled("cancelled by user")
                    if clip_index > 1:
                        self.page_chunk.emit(index, page_number, "\n\n")
                    render_started = time.time()
                    png = core.render_page_png(
                        page,
                        is_pdf,
                        self.settings.dpi,
                        self.settings.max_side,
                        clip=clip,
                    )
                    render_s = time.time() - render_started
                    width, height = core.png_size(png)
                    if len(clips) == 1:
                        where = ""
                    elif regions:
                        where = f"region {clip_index} "
                    else:
                        where = f"half {clip_index} "
                    how = (
                        "tesseract OCR"
                        if self.settings.engine == "tesseract"
                        else "OCR"
                    )
                    self.log.emit(
                        f"{job.path.name}: page {page_number} {where}{how} "
                        f"({width}x{height} px, {len(png) // 1024} KiB, "
                        f"render {render_s * 1000:.0f} ms)..."
                    )
                    if self.settings.engine == "tesseract":
                        tess_started = time.time()
                        result = core.tesseract_ocr(
                            png,
                            self.settings.tesseract_lang,
                            self.settings.tesseract_path.strip() or None,
                        )
                        tess_s = time.time() - tess_started
                        usage, finish_reason = {}, "stop"
                        if result:
                            self.page_chunk.emit(index, page_number, result)
                        self.log.emit(
                            f"{job.path.name}: page {page_number} "
                            f"tesseract {tess_s:.2f}s"
                        )
                    else:
                        content, usage, finish_reason = core.ocr_png(
                            self.settings.server,
                            self.settings.model,
                            png,
                            self.settings.prompt,
                            self.settings.max_tokens,
                            self.settings.timeout,
                            label="",
                            on_chunk=lambda piece, i=index, p=page_number: self.page_chunk.emit(
                                i, p, piece
                            ),
                            should_stop=self.stop.is_set,
                        )
                        result = core.clean_markdown(content)
                    if not result:
                        if len(clips) > 1:
                            kind = "region" if regions else "half"
                            self.log.emit(
                                f"{job.path.name}: page {page_number} "
                                f"{kind} {clip_index} returned no text"
                            )
                            continue
                        if not regions:
                            raise RuntimeError(
                                f"page {page_number}: model returned no text"
                            )
                        # A single selected region: fall through to the
                        # shared "nothing recognized" error below.
                    pieces.append(result)
                    prompt_tokens += usage.get("prompt_tokens") or 0
                    completion_tokens += usage.get("completion_tokens") or 0
                    truncated = truncated or finish_reason == "length"
                result = "\n\n".join(pieces)
                if not result:
                    if regions:
                        detail = "selected regions"
                    elif len(clips) > 1:
                        detail = "page halves"
                    else:
                        detail = "page"
                    raise RuntimeError(
                        f"page {page_number}: nothing recognized in the "
                        f"{detail}"
                    )
                if truncated:
                    self.log.emit(
                        f"{job.path.name}: page {page_number} WARNING "
                        f"output hit --max-tokens and may be truncated"
                    )
                elapsed = time.time() - started
                job.page_texts[page_number - 1] = result
                self.log.emit(
                    f"{job.path.name}: page {page_number} {len(result)} chars, "
                    f"{prompt_tokens}+{completion_tokens} tokens, "
                    f"{elapsed:.1f}s"
                )
                self.page_done.emit(index, page_number, result)

            markdown = "\n\n".join(
                job.page_texts[p - 1]
                for p in selected
                if p - 1 < len(job.page_texts)
                and job.page_texts[p - 1].strip()
            )
            if not markdown:
                raise RuntimeError("nothing recognized")
            output_path = self._output_path(job)
            core.save_output(markdown, self.settings.out_format, output_path)
            job.status = "Done"
            job.output_path = str(output_path)
            self.file_done.emit(index, str(output_path))

    @staticmethod
    def _clip_rect(page, region):
        """Map a normalised preview region to a page-space clip rectangle.

        PyMuPDF's get_pixmap(clip=...) uses the displayed (rotated) page
        coordinates, which is what the preview shows; verified on rotated
        pages.
        """
        x0, y0, x1, y1 = region
        rect = page.rect
        return pymupdf.Rect(
            rect.x0 + x0 * rect.width,
            rect.y0 + y0 * rect.height,
            rect.x0 + x1 * rect.width,
            rect.y0 + y1 * rect.height,
        ).normalize()

    def _output_path(self, job: FileJob) -> Path:
        fmt = self.settings.out_format
        if self.settings.output_dir.strip():
            directory = Path(self.settings.output_dir.strip())
            directory.mkdir(parents=True, exist_ok=True)
            return directory / f"{job.path.stem}.{fmt}"
        return job.path.with_suffix(f".{fmt}")


class AskWorker(QObject):
    """One-shot question about a page image, sent to the server model."""

    chunk = Signal(str)
    info = Signal(str)
    done = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        settings: OcrSettings,
        path: Path,
        page_number: int,
        is_pdf: bool,
        prompt: str,
    ):
        super().__init__()
        self.settings = settings
        self.path = path
        self.page_number = page_number
        self.is_pdf = is_pdf
        self.prompt = prompt
        self.stop = threading.Event()

    def run(self) -> None:
        ready, message = core.server_status(self.settings.server, timeout=5)
        if not ready:
            self.failed.emit(message)
            self.finished.emit()
            return
        try:
            started = time.time()
            with pymupdf.open(self.path) as document:
                page = document.load_page(self.page_number - 1)
                png = core.render_page_png(
                    page,
                    self.is_pdf,
                    self.settings.dpi,
                    self.settings.max_side,
                )
            render_s = time.time() - started
            width, height = core.png_size(png)
            self.info.emit(
                f"Ask: {self.path.name} page {self.page_number} "
                f"({width}x{height} px, {len(png) // 1024} KiB, "
                f"render {render_s * 1000:.0f} ms)..."
            )
            content, _usage, _finish = core.ocr_png(
                self.settings.server,
                self.settings.model,
                png,
                self.prompt,
                self.settings.max_tokens,
                self.settings.timeout,
                label="ask",
                on_chunk=self.chunk.emit,
                should_stop=self.stop.is_set,
            )
            self.done.emit(content.strip())
        except Exception as exc:  # noqa: BLE001 - show it in the Ask tab
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class AutoscrollIndicator(QWidget):
    """Fixed overlay on the scroll viewport (does not scroll with the image)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedSize(44, 44)
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self.hide()

    def paintEvent(self, event):  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        center = QPointF(self.width() / 2, self.height() / 2)
        radius = 18
        painter.setPen(QPen(QColor("#1976d2"), 2))
        painter.setBrush(QColor(255, 255, 255, 225))
        painter.drawEllipse(center, radius, radius)
        painter.setBrush(QColor("#1976d2"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, 3, 3)
        painter.setPen(QPen(QColor("#1976d2"), 2))
        for sign in (-1, 1):
            tip = QPointF(center.x(), center.y() + sign * (radius - 5))
            painter.drawLine(
                QPointF(center.x(), center.y() + sign * 6), tip
            )
            painter.drawLine(
                tip, QPointF(center.x() - 3, tip.y() - sign * 4)
            )
            painter.drawLine(
                tip, QPointF(center.x() + 3, tip.y() - sign * 4)
            )
        for sign in (-1, 1):
            tip = QPointF(center.x() + sign * (radius - 5), center.y())
            painter.drawLine(
                QPointF(center.x() + sign * 6, center.y()), tip
            )
            painter.drawLine(
                tip, QPointF(tip.x() - sign * 4, center.y() - 3)
            )
            painter.drawLine(
                tip, QPointF(tip.x() - sign * 4, center.y() + 3)
            )


class ImageView(QWidget):
    """Page preview with FineReader-style OCR region selection.

    Up to ``MAX_REGIONS`` rectangles can be drawn on a page; further drags are
    ignored. They are stored normalised (0..1 of the page), so they survive
    preview/OCR render sizes.

    Interaction: Ctrl+drag on empty space creates a region, dragging inside
    moves it, dragging one of the eight handles resizes it, and right-click or
    Delete removes a region.
    """

    MAX_REGIONS = 5
    HANDLE = 7  # half-size of a handle square, in widget pixels
    MIN_REGION = 12  # smaller drags are discarded, in widget pixels
    AUTOSCROLL_DEADZONE = 10
    AUTOSCROLL_MAX_STEP = 60
    AUTOSCROLL_INTERVAL = 30  # ms

    regions_changed = Signal()
    message = Signal(str)
    zoom_requested = Signal(float)
    split_changed = Signal(float)
    pan_requested = Signal(int, int)

    _HANDLE_CURSORS = {
        "tl": Qt.CursorShape.SizeFDiagCursor,
        "br": Qt.CursorShape.SizeFDiagCursor,
        "tr": Qt.CursorShape.SizeBDiagCursor,
        "bl": Qt.CursorShape.SizeBDiagCursor,
        "tc": Qt.CursorShape.SizeVerCursor,
        "bc": Qt.CursorShape.SizeVerCursor,
        "ml": Qt.CursorShape.SizeHorCursor,
        "mr": Qt.CursorShape.SizeHorCursor,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source: QPixmap | None = None
        self._empty_text = "No page selected"
        self._regions: list[list[float]] = []
        self._selected = -1
        self._mode = ""  # "", "create", "move" or "resize"
        self._handle = ""
        self._drag_start = QPointF()
        self._drag_orig: list[float] = []
        self._editable = True
        self._spread = False
        self._auto_spread = False
        self._split = 0.5
        self._scale = 1.0
        self._autoscroll_active = False
        self._autoscroll_anchor_global = QPointF()
        self._autoscroll_indicator: AutoscrollIndicator | None = None
        self._autoscroll_timer = QTimer(self)
        self._autoscroll_timer.setInterval(self.AUTOSCROLL_INTERVAL)
        self._autoscroll_timer.timeout.connect(self._autoscroll_tick)
        self.setMinimumSize(1, 1)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    # ------------------------------------------------------------- public
    def set_source(self, pixmap: QPixmap | None) -> None:
        self.stop_autoscroll()
        self._source = None if pixmap is None or pixmap.isNull() else pixmap
        self.update()

    def set_empty_text(self, text: str) -> None:
        self._empty_text = text
        if self._source is None:
            self.update()

    def set_regions(self, regions) -> None:
        self._regions = [list(region) for region in regions]
        self._selected = -1
        self._mode = ""
        self.update()

    def regions(self) -> list[list[float]]:
        return [list(region) for region in self._regions]

    def clear_regions(self) -> None:
        if self._regions:
            self._regions.clear()
            self._selected = -1
            self._emit_changed()

    def set_editable(self, editable: bool) -> None:
        self._editable = editable
        if not editable:
            self._mode = ""
            self.unsetCursor()

    def set_spread(
        self, enabled: bool, fraction: float = 0.5, auto: bool = False
    ) -> None:
        """Show a movable split marker on landscape (or auto-wide) pages."""
        self._spread = enabled
        self._auto_spread = auto
        self._split = min(max(fraction, 0.1), 0.9)
        self.update()

    def _split_active(self) -> bool:
        if self._spread is False or self._source is None:
            return False
        width, height = self._source.width(), self._source.height()
        if width <= height:
            return False
        if self._auto_spread and width / height < core.AUTO_SPREAD_MIN_ASPECT:
            return False
        return True

    def _split_x(self) -> int:
        return int(round(self.width() * self._split))

    def _set_split_from_x(self, x: float) -> None:
        self._split = min(max(x / max(self.width(), 1), 0.1), 0.9)
        self.split_changed.emit(self._split)
        self.update()

    def source_size(self):
        return self._source.size() if self._source is not None else None

    def display_scale(self) -> float:
        return self._scale

    def set_display_scale(self, scale: float) -> None:
        """Resize the widget to the image scaled by ``scale`` (px per page pt)."""
        if self._source is None or scale <= 0:
            return
        self.stop_autoscroll()
        self._scale = scale
        self.resize(
            max(1, int(self._source.width() * scale)),
            max(1, int(self._source.height() * scale)),
        )
        self.update()

    # -------------------------------------------------------- autoscroll
    def start_autoscroll(self, pos: QPointF) -> None:
        """Start Firefox-style middle-click autoscroll at ``pos``."""
        parent = self.parentWidget()
        self._autoscroll_active = True
        self._autoscroll_anchor_global = QPointF(
            self.mapToGlobal(pos.toPoint())
        )
        if parent is not None:
            if (
                self._autoscroll_indicator is None
                or self._autoscroll_indicator.parent() is not parent
            ):
                self._autoscroll_indicator = AutoscrollIndicator(parent)
            point = self.mapTo(parent, pos.toPoint())
            self._autoscroll_indicator.move(
                point.x() - self._autoscroll_indicator.width() // 2,
                point.y() - self._autoscroll_indicator.height() // 2,
            )
            self._autoscroll_indicator.show()
            self._autoscroll_indicator.raise_()
        self._autoscroll_timer.start()

    def stop_autoscroll(self) -> None:
        if not self._autoscroll_active:
            return
        self._autoscroll_active = False
        self._autoscroll_timer.stop()
        if self._autoscroll_indicator is not None:
            self._autoscroll_indicator.hide()

    @staticmethod
    def _autoscroll_axis(value: float) -> int:
        if abs(value) <= ImageView.AUTOSCROLL_DEADZONE:
            return 0
        step = int(
            (abs(value) - ImageView.AUTOSCROLL_DEADZONE) / 4
        ) + 1
        step = min(step, ImageView.AUTOSCROLL_MAX_STEP)
        return step if value > 0 else -step

    def _autoscroll_tick(self) -> None:
        if not self._autoscroll_active:
            return
        delta = QPointF(QCursor.pos()) - self._autoscroll_anchor_global
        dx = self._autoscroll_axis(delta.x())
        dy = self._autoscroll_axis(delta.y())
        if dx or dy:
            self.pan_requested.emit(-dx, -dy)

    # ----------------------------------------------------------- painting
    def paintEvent(self, event):  # noqa: N802 - Qt override
        painter = QPainter(self)
        exposed = QRectF(event.rect())
        # At high zoom the widget itself is huge. Never fill or scale more
        # than the part that is actually visible in the scroll viewport.
        viewport = self.parentWidget()
        if viewport is not None:
            visible = QRectF(
                -self.pos().x(),
                -self.pos().y(),
                viewport.width(),
                viewport.height(),
            )
            exposed = exposed.intersected(visible)
        if exposed.isEmpty():
            return
        painter.fillRect(exposed, self.palette().window())
        image_rect = self._image_rect()
        if self._source is None or image_rect.isEmpty():
            painter.setPen(self.palette().placeholderText().color())
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty_text
            )
            return
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # Scale only the visible part. Scaling the whole pixmap into a huge
        # zoomed widget made each paint/scroll take seconds on Windows.
        scale = self.width() / max(self._source.width(), 1)
        source_rect = QRectF(
            exposed.x() / scale,
            exposed.y() / scale,
            exposed.width() / scale,
            exposed.height() / scale,
        ).intersected(QRectF(self._source.rect()))
        if not source_rect.isEmpty():
            target = QRectF(
                source_rect.x() * scale,
                source_rect.y() * scale,
                source_rect.width() * scale,
                source_rect.height() * scale,
            )
            painter.drawPixmap(target, self._source, source_rect)
        if self._split_active():
            painter.setPen(
                QPen(QColor("#f57c00"), 2, Qt.PenStyle.DashLine)
            )
            split_x = self._split_x()
            painter.drawLine(split_x, 0, split_x, self.height())
        for index, region in enumerate(self._regions):
            rect = self._region_rect(region)
            selected = index == self._selected
            pen = QPen(QColor("#1976d2"))
            pen.setWidth(2 if selected else 1)
            painter.setPen(pen)
            painter.setBrush(QColor(25, 118, 210, 35))
            painter.drawRect(rect)
            badge = QRectF(rect.left(), rect.top(), 18, 18)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#1976d2"))
            painter.drawRect(badge)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(
                badge, Qt.AlignmentFlag.AlignCenter, str(index + 1)
            )
            if selected:
                painter.setPen(QPen(QColor("#1976d2"), 1))
                painter.setBrush(QColor("#ffffff"))
                for _, handle in self._handles(rect):
                    painter.drawRect(handle)

    # ----------------------------------------------------------- geometry
    def _image_rect(self) -> QRectF:
        # The widget is sized to the scaled image by the owner; the scroll
        # area centers/scrolls it.
        if self._source is None:
            return QRectF()
        return QRectF(0, 0, self.width(), self.height())

    def _region_rect(self, region) -> QRectF:
        image = self._image_rect()
        x0, y0, x1, y1 = region
        return QRectF(
            image.left() + x0 * image.width(),
            image.top() + y0 * image.height(),
            (x1 - x0) * image.width(),
            (y1 - y0) * image.height(),
        )

    @staticmethod
    def _handles(rect: QRectF) -> list[tuple[str, QRectF]]:
        h = ImageView.HANDLE
        left, top = rect.left(), rect.top()
        right, bottom = rect.right(), rect.bottom()
        cx, cy = rect.center().x(), rect.center().y()
        points = {
            "tl": (left, top),
            "tc": (cx, top),
            "tr": (right, top),
            "ml": (left, cy),
            "mr": (right, cy),
            "bl": (left, bottom),
            "bc": (cx, bottom),
            "br": (right, bottom),
        }
        return [
            (name, QRectF(x - h, y - h, 2 * h, 2 * h))
            for name, (x, y) in points.items()
        ]

    def _hit(self, pos: QPointF) -> tuple[int, str]:
        """Return (region index, part); part is a handle name or "move"."""
        for index in range(len(self._regions) - 1, -1, -1):
            rect = self._region_rect(self._regions[index])
            if index == self._selected:
                for name, handle in self._handles(rect):
                    if handle.contains(pos):
                        return index, name
            if rect.contains(pos):
                return index, "move"
        return -1, ""

    @staticmethod
    def _clamp01(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    def _normalise(self, rect: QRectF) -> list[float]:
        image = self._image_rect()
        x0 = self._clamp01((rect.left() - image.left()) / image.width())
        y0 = self._clamp01((rect.top() - image.top()) / image.height())
        x1 = self._clamp01((rect.right() - image.left()) / image.width())
        y1 = self._clamp01((rect.bottom() - image.top()) / image.height())
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

    # ------------------------------------------------------------- mouse
    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if (
            event.button() == Qt.MouseButton.MiddleButton
            and self._source is not None
        ):
            # Firefox-style autoscroll: click once, move the cursor to scroll.
            if self._autoscroll_active:
                self.stop_autoscroll()
            else:
                self.start_autoscroll(event.position())
            event.accept()
            return
        if self._autoscroll_active:
            # Any other click stops autoscroll (and is consumed, like Firefox).
            self.stop_autoscroll()
            event.accept()
            return
        if not self._editable or self._source is None:
            super().mousePressEvent(event)
            return
        pos = event.position()
        if event.button() == Qt.MouseButton.RightButton:
            index, _ = self._hit(pos)
            if index >= 0:
                del self._regions[index]
                self._selected = -1
                self._emit_changed()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self._image_rect().contains(pos):
            return
        ctrl = bool(
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        )
        if (
            not ctrl
            and self._split_active()
            and abs(pos.x() - self._split_x()) <= 6
        ):
            self._mode = "split"
            self._selected = -1
            self.update()
            return
        index, part = self._hit(pos)
        if ctrl and index < 0:
            if len(self._regions) >= self.MAX_REGIONS:
                self.message.emit(
                    f"Region limit reached ({self.MAX_REGIONS} per page); "
                    f"remove one to add another."
                )
                return
            index = len(self._regions)
            self._regions.append([0.0, 0.0, 0.0, 0.0])
            self._selected = index
            self._mode = "create"
            self._drag_start = pos
            self._drag_orig = list(self._regions[index])
            self._emit_changed()
            return
        if index < 0:
            self._selected = -1
            self.update()
            return
        self._selected = index
        self._drag_start = pos
        self._drag_orig = list(self._regions[index])
        self._handle = part if part in self._HANDLE_CURSORS else ""
        self._mode = "resize" if self._handle else "move"
        self.update()

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt override
        if not self._editable or self._source is None:
            return
        pos = event.position()
        if self._mode == "split":
            self._set_split_from_x(pos.x())
            return
        if self._mode:
            self._drag_to(pos)
            return
        if self._split_active() and abs(pos.x() - self._split_x()) <= 6:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            return
        index, part = self._hit(pos)
        ctrl = bool(
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        )
        if index < 0:
            if ctrl and self._image_rect().contains(pos):
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.unsetCursor()
        elif part == "move":
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(self._HANDLE_CURSORS[part])

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt override
        if not self._mode:
            return
        if self._mode == "split":
            self._mode = ""
            self.split_changed.emit(self._split)
            return
        index = self._selected
        if 0 <= index < len(self._regions):
            rect = self._region_rect(self._regions[index])
            if rect.width() < self.MIN_REGION or rect.height() < self.MIN_REGION:
                if self._mode == "create":
                    del self._regions[index]
                    self._selected = -1
                else:
                    self._regions[index] = list(self._drag_orig)
        self._mode = ""
        self._handle = ""
        self._emit_changed()

    def wheelEvent(self, event):  # noqa: N802 - Qt override
        self.stop_autoscroll()
        if (
            self._source is not None
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            delta = event.angleDelta().y()
            if delta:
                self.zoom_requested.emit(1.25 if delta > 0 else 1 / 1.25)
                event.accept()
                return
        super().wheelEvent(event)

    def keyPressEvent(self, event):  # noqa: N802 - Qt override
        if (
            self._editable
            and self._selected >= 0
            and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
        ):
            del self._regions[self._selected]
            self._selected = -1
            self._emit_changed()
            return
        super().keyPressEvent(event)

    def _drag_to(self, pos: QPointF) -> None:
        image = self._image_rect()
        if image.isEmpty() or not (0 <= self._selected < len(self._regions)):
            return
        pos = QPointF(
            min(max(pos.x(), image.left()), image.right()),
            min(max(pos.y(), image.top()), image.bottom()),
        )
        if self._mode == "create":
            self._regions[self._selected] = self._normalise(
                QRectF(self._drag_start, pos).normalized()
            )
        elif self._mode == "move":
            dx = (pos.x() - self._drag_start.x()) / image.width()
            dy = (pos.y() - self._drag_start.y()) / image.height()
            x0, y0, x1, y1 = self._drag_orig
            dx = min(max(dx, -x0), 1.0 - x1)
            dy = min(max(dy, -y0), 1.0 - y1)
            self._regions[self._selected] = [
                x0 + dx, y0 + dy, x1 + dx, y1 + dy
            ]
        elif self._mode == "resize":
            nx = (pos.x() - image.left()) / image.width()
            ny = (pos.y() - image.top()) / image.height()
            x0, y0, x1, y1 = self._drag_orig
            if self._handle in ("tl", "ml", "bl"):
                x0 = nx
            if self._handle in ("tr", "mr", "br"):
                x1 = nx
            if self._handle in ("tl", "tc", "tr"):
                y0 = ny
            if self._handle in ("bl", "bc", "br"):
                y1 = ny
            self._regions[self._selected] = [
                min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
            ]
        self.update()

    def _emit_changed(self) -> None:
        self.regions_changed.emit()
        self.update()


class FileDropList(QListWidget):
    """File list that accepts dropped documents."""

    pathsDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(True)

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # noqa: N802
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ]
        if paths:
            self.pathsDropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class MainWindow(QMainWindow):
    run_finished = Signal()

    STATUS_COLORS = {
        "Pending": QColor("#8a8a8a"),
        "Working": QColor("#1e88e5"),
        "Done": QColor("#2e7d32"),
        "Error": QColor("#c62828"),
        "Cancelled": QColor("#a1887f"),
        "Skipped": QColor("#a1887f"),
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"refinedocr v{APP_VERSION}")
        self.resize(1280, 820)
        self.setMinimumSize(960, 620)

        self.jobs: list[FileJob] = []
        self.current_file: int | None = None
        self.current_page = 1
        self.stream_buf: dict[tuple[int, int], str] = {}
        self.active_pages: set[tuple[int, int]] = set()
        self.thread: QThread | None = None
        self.worker: OCRWorker | None = None
        self.ask_thread: QThread | None = None
        self.ask_worker: AskWorker | None = None
        self.ask_running = False
        self.ask_buf = ""
        self.ask_prompt_text = ""
        self.pages_done = 0
        self.total_pages = 0
        self.cancelled = False
        self.server_error = False
        self.running = False
        self.preview_zoom = 1.0
        self.ocr_wait_started: float | None = None
        self.ocr_wait_key: tuple[int, int] | None = None
        self._loading_settings = True
        self._last_engine = core.DEFAULT_ENGINE
        self._engine_max_sides = {
            "glm": core.DEFAULT_MAX_SIDE_GLM,
            "tesseract": core.DEFAULT_MAX_SIDE,
        }
        self._tesseract_path = ""
        self.settings_path: Path | None = None
        self.settings_store = self._make_settings_store()

        self._build_toolbar()
        self._build_ui()
        if self.settings_path is not None:
            self._log(f"Settings: {self.settings_path}")
        else:
            self._log(
                "Settings: app folder is not writable; using the system store."
            )
        self._load_settings()
        self._set_running(False)
        self._clear_view()

    # ------------------------------------------------------------- UI setup
    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main", self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(toolbar)

        self.act_add = QAction("Open files…", self)
        self.act_add.setShortcut("Ctrl+O")
        self.act_add.setToolTip(
            "Open documents as a new project (the current list is cleared). "
            "Drag files onto the window to add them to the current list."
        )
        self.act_add.triggered.connect(self.choose_files)
        toolbar.addAction(self.act_add)

        self.act_remove = QAction("Remove", self)
        self.act_remove.triggered.connect(self.remove_selected)
        toolbar.addAction(self.act_remove)

        self.act_clear = QAction("Clear", self)
        self.act_clear.triggered.connect(self.clear_files)
        toolbar.addAction(self.act_clear)

        toolbar.addSeparator()
        self.act_start = QAction("Start OCR", self)
        self.act_start.setShortcut("Ctrl+Return")
        self.act_start.triggered.connect(self.start_ocr)
        toolbar.addAction(self.act_start)

        self.act_cancel = QAction("Cancel", self)
        self.act_cancel.setShortcut("Esc")
        self.act_cancel.triggered.connect(self.cancel_ocr)
        toolbar.addAction(self.act_cancel)

        toolbar.addSeparator()
        self.act_save_as = QAction("Save as…", self)
        self.act_save_as.setShortcut("Ctrl+Shift+S")
        self.act_save_as.setToolTip(
            "Save the recognized text of checked pages as md/txt/odt "
            "without running OCR again"
        )
        self.act_save_as.triggered.connect(self.save_as)
        toolbar.addAction(self.act_save_as)

        self.act_open = QAction("Open output folder", self)
        self.act_open.triggered.connect(self.open_output_dir)
        toolbar.addAction(self.act_open)

    def _build_ui(self) -> None:
        self._build_settings_panel()
        self._build_main_panel()
        self._build_bottom_panel()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self.settings_panel)
        layout.addWidget(self.main_splitter, 1)
        layout.addWidget(self.bottom_panel)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

        self.render_timer = QTimer(self)
        self.render_timer.setInterval(500)
        self.render_timer.timeout.connect(self._refresh_rendered)

    def _build_settings_panel(self) -> None:
        panel = QWidget()
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(6)

        self.server_edit = QLineEdit(core.DEFAULT_SERVER)
        self.server_edit.setMinimumWidth(160)
        self.server_edit.setPlaceholderText("http://127.0.0.1:8080")
        self.server_edit.setToolTip(
            "Full base URL including port, e.g. http://127.0.0.1:8090.\n"
            "From the VM use http://10.0.2.2:8090"
        )
        self.format_combo = QComboBox()
        self.format_combo.addItems(["md", "txt", "odt"])
        self.format_combo.setToolTip(
            "odt needs Pandoc; pypandoc_binary bundles it"
        )
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(core.DEFAULT_DPI)
        self.dpi_spin.setSuffix(" dpi")
        self.dpi_spin.setToolTip(
            "Render resolution for PDF pages only. Images and CBZ are "
            "unaffected; use Max side to limit their resolution."
        )
        self.test_btn = QPushButton("Test")
        self.test_btn.setToolTip("Check the server's /health endpoint")
        self.test_btn.clicked.connect(self.test_server)
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Tesseract (local)", "tesseract")
        self.engine_combo.addItem(core.DEFAULT_ENGINE_LABEL, "glm")
        self.engine_combo.setToolTip(
            "OCR engine. Tesseract runs locally and needs no server; "
            "OCR model (server) uses any OpenAI-compatible multimodal "
            "endpoint (GLM-OCR, Qwen, Gemma, ...).\n"
            "Tesseract is much better for Russian/European print."
        )
        self.tesseract_lang_combo = QComboBox()
        self.tesseract_lang_combo.setEditable(True)
        self.tesseract_lang_combo.setMinimumWidth(90)
        languages = core.list_tesseract_languages()
        choices = []
        if "rus" in languages and "eng" in languages:
            # Russian game text is full of English names; make the common
            # combination a first-class choice, not just a typed value.
            choices.append("rus+eng")
        choices.extend(languages)
        if choices:
            self.tesseract_lang_combo.addItems(choices)
        self.tesseract_lang_combo.setCurrentText(core.DEFAULT_TESSERACT_LANG)
        self.tesseract_lang_combo.setToolTip(
            "Tesseract language(s) found in tessdata, e.g. rus, eng, jpn. "
            "Combine with + (rus+eng). A missing language falls back to "
            "rus+eng."
        )
        self.prompt_edit = QLineEdit(core.DEFAULT_PROMPT)
        self.prompt_edit.setToolTip(
            "Used by Start OCR and by Send prompt in the Ask tab.\n"
            "OCR prompts: OCR | OCR markdown | OCR HTML table.\n"
            "For Ask, type a question about the current page image."
        )
        self.prompt_edit.returnPressed.connect(self.send_prompt)
        self.max_side_spin = QSpinBox()
        self.max_side_spin.setRange(0, 6000)
        self.max_side_spin.setSingleStep(100)
        self.max_side_spin.setValue(core.DEFAULT_MAX_SIDE)
        self.max_side_spin.setSuffix(" px")
        self.max_side_spin.setToolTip(
            "Caps the rendered long side for every input. This is the "
            "resolution control for images and CBZ.\n"
            "0 = no limit: full-resolution photos are sent as-is and can "
            "take minutes.\n"
            "Keep image tokens + output under the server's per-slot "
            "context (8192)."
        )
        self.force_check = QCheckBox("Force OCR")
        self.force_check.setToolTip(
            "OCRs the page image even when the PDF has a text layer.\n"
            "Enable this for tables/layout: the text layer loses "
            "row/column structure."
        )
        self.spreads_combo = QComboBox()
        for label, value in (
            ("Off", "off"),
            ("Auto (LTR)", "auto"),
            ("LTR (books)", "ltr"),
            ("RTL (manga)", "rtl"),
        ):
            self.spreads_combo.addItem(label, value)
        self.spreads_combo.setToolTip(
            "Split landscape two-page scans into left/right halves.\n"
            f"Auto: split pages at least "
            f"{core.AUTO_SPREAD_MIN_ASPECT:g}x wider than tall, LTR order.\n"
            "LTR: left page first (western books); RTL: right page first "
            "(manga). Portrait pages are not split.\n"
            "Drag the dashed line in the preview to adjust the split for the "
            "current page. Pages with selected regions use the regions."
        )

        # Row 1: server/OCR settings. Prompt lives in the Ask tab.
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        row1.addWidget(QLabel("Server URL"))
        row1.addWidget(self.server_edit, 1)
        row1.addWidget(self.test_btn)
        row1.addWidget(QLabel("Format"))
        row1.addWidget(self.format_combo)
        row1.addWidget(QLabel("DPI"))
        row1.addWidget(self.dpi_spin)
        row1.addWidget(QLabel("Max side"))
        row1.addWidget(self.max_side_spin)
        row1.addWidget(QLabel("Language"))
        row1.addWidget(self.tesseract_lang_combo)
        row1.addWidget(QLabel("Engine"))
        row1.addWidget(self.engine_combo)
        row1.addWidget(QLabel("Spreads"))
        row1.addWidget(self.spreads_combo)
        row1.addWidget(self.force_check)

        # Row 2: output folder.
        row2 = QHBoxLayout()
        self.outdir_edit = QLineEdit()
        self.outdir_edit.setPlaceholderText("same folder as input")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.choose_output_dir)
        row2.addWidget(QLabel("Output folder"))
        row2.addWidget(self.outdir_edit, 1)
        row2.addWidget(browse)

        outer.addLayout(row1)
        outer.addLayout(row2)
        self.engine_combo.currentIndexChanged.connect(self.on_engine_changed)
        self.settings_panel = panel

    def _build_main_panel(self) -> None:
        self.files_list = FileDropList()
        self.files_list.pathsDropped.connect(self.add_paths)
        self.files_list.currentRowChanged.connect(self.select_file)

        self.pages_list = QListWidget()
        self.pages_list.currentRowChanged.connect(self.select_page)
        self.pages_list.itemChanged.connect(self.on_page_check_changed)
        self.pages_list.setToolTip(
            "Click a row to preview the page; check its box to OCR it "
            "and include it in the output."
        )

        pages_row = QHBoxLayout()
        self.pages_label = QLabel("Pages")
        self.pages_label.setToolTip(
            "Checked pages are OCR'd and included in the output."
        )
        self.btn_pages_all = QPushButton("All")
        self.btn_pages_none = QPushButton("None")
        self.btn_pages_invert = QPushButton("Invert")
        for button, mode in (
            (self.btn_pages_all, "all"),
            (self.btn_pages_none, "none"),
            (self.btn_pages_invert, "invert"),
        ):
            button.setMaximumWidth(56)
            button.setToolTip("Check/uncheck pages")
            button.clicked.connect(
                lambda checked=False, m=mode: self.select_pages(m)
            )
            pages_row.addWidget(button)
        pages_row.addWidget(self.pages_label, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Files"))
        left_layout.addWidget(self.files_list, 2)
        left_layout.addLayout(pages_row)
        left_layout.addWidget(self.pages_list, 1)

        self.image_view = ImageView()
        self.image_view.regions_changed.connect(self.on_regions_changed)
        self.image_view.message.connect(self._log)
        self.image_view.zoom_requested.connect(self._zoom_preview)
        self.image_view.pan_requested.connect(self._pan_preview)
        self.image_view.split_changed.connect(self.on_split_changed)

        # Keyboard zoom for mice without a wheel.
        self.zoom_shortcuts = []
        for sequence, factor in (
            ("Ctrl++", 1.25),
            ("Ctrl+=", 1.25),
            ("Ctrl+-", 1 / 1.25),
        ):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(
                lambda f=factor: self._zoom_preview(f)
            )
            self.zoom_shortcuts.append(shortcut)
        self.spreads_combo.currentIndexChanged.connect(
            self.on_spreads_changed
        )
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setWidget(self.image_view)
        self.scroll.viewport().installEventFilter(self)

        self.nav_label = QLabel("Page – / –")
        self.prev_btn = QPushButton("‹ Prev")
        self.next_btn = QPushButton("Next ›")
        self.prev_btn.clicked.connect(lambda: self.step_page(-1))
        self.next_btn.clicked.connect(lambda: self.step_page(1))
        self.btn_clear_regions = QPushButton("Clear regions")
        self.btn_clear_regions.setToolTip(
            "Remove all OCR regions selected on this page"
        )
        self.btn_clear_regions.clicked.connect(self.clear_page_regions)
        nav = QHBoxLayout()
        nav.addWidget(self.prev_btn)
        nav.addStretch(1)
        nav.addWidget(self.nav_label)
        nav.addStretch(1)
        nav.addWidget(self.next_btn)
        nav.addSpacing(8)
        nav.addWidget(self.btn_clear_regions)

        # Keep the hint single-line and the count a constant width, so
        # dragging a region does not reflow the preview.
        self.regions_count = QLabel(f"0/{ImageView.MAX_REGIONS} regions")
        self.regions_count.setToolTip(
            "OCR regions selected on this page (max "
            f"{ImageView.MAX_REGIONS})"
        )
        self.regions_count.setStyleSheet("color: #8a8a8a;")
        self.regions_hint = QLabel(
            "Ctrl+drag add · drag move · right-click remove · "
            "Ctrl+wheel / Ctrl+± zoom · middle-click auto-scroll"
        )
        self.regions_hint.setStyleSheet("color: #8a8a8a;")
        hint_row = QHBoxLayout()
        hint_row.setContentsMargins(0, 0, 0, 0)
        hint_row.addWidget(self.regions_count)
        hint_row.addSpacing(8)
        hint_row.addWidget(self.regions_hint, 1)

        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addLayout(nav)
        preview_layout.addLayout(hint_row)
        preview_layout.addWidget(self.scroll, 1)

        fixed_font = QFontDatabase.systemFont(
            QFontDatabase.SystemFont.FixedFont
        )
        self.raw_edit = QPlainTextEdit()
        self.raw_edit.setReadOnly(True)
        self.raw_edit.setFont(fixed_font)
        self.page_browser = QTextBrowser()
        self.doc_browser = QTextBrowser()
        self.ask_browser = QTextBrowser()
        self.ask_browser.setPlaceholderText(
            "Ask a question about the current page image."
        )
        self.ask_btn = QPushButton("Send prompt")
        self.ask_btn.clicked.connect(self.send_prompt)
        self.ask_stop_btn = QPushButton("Stop")
        self.ask_stop_btn.setEnabled(False)
        self.ask_stop_btn.clicked.connect(self.stop_prompt)
        self.ask_hint = QLabel()
        self.ask_hint.setStyleSheet("color: #8a8a8a;")
        self.ask_hint.setWordWrap(True)
        ask_tab = QWidget()
        ask_layout = QVBoxLayout(ask_tab)
        ask_layout.setContentsMargins(0, 0, 0, 0)
        ask_row = QHBoxLayout()
        ask_row.addWidget(QLabel("Prompt"))
        ask_row.addWidget(self.prompt_edit, 1)
        ask_row.addWidget(self.ask_btn)
        ask_row.addWidget(self.ask_stop_btn)
        ask_layout.addLayout(ask_row)
        ask_layout.addWidget(self.ask_hint)
        ask_layout.addWidget(self.ask_browser, 1)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.page_browser, "Page view")
        self.tabs.addTab(self.raw_edit, "Page Markdown")
        self.tabs.addTab(self.doc_browser, "Whole document")
        self.tabs.addTab(ask_tab, "Ask")
        self.tabs.setTabToolTip(
            0, "Current page's text rendered as Markdown."
        )
        self.tabs.setTabToolTip(
            1, "Raw Markdown source of the current page; "
               "updates live while OCR runs."
        )
        self.tabs.setTabToolTip(
            2, "All processed pages of the current file, concatenated."
        )
        self.tabs.setTabToolTip(
            3, "Ask the server model about the current page image "
               "(requires Engine = OCR model (server))."
        )
        self.doc_label = QLabel("No document")
        self.doc_label.setStyleSheet("font-weight: bold;")

        text_panel = QWidget()
        text_layout = QVBoxLayout(text_panel)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.addWidget(self.doc_label)
        text_layout.addWidget(self.tabs, 1)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(preview)
        splitter.addWidget(text_panel)
        splitter.setSizes([250, 560, 470])
        self.main_splitter = splitter

    def _build_bottom_panel(self) -> None:
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.status_label = QLabel("Ready")
        self.terminal_check = QCheckBox("Terminal")
        self.terminal_check.setToolTip("Show or hide the log console below")
        self.terminal_check.setChecked(True)
        self.terminal_check.toggled.connect(self._toggle_terminal)

        row = QHBoxLayout()
        row.addWidget(self.progress, 1)
        row.addWidget(self.status_label)
        row.addWidget(self.terminal_check)

        fixed_font = QFontDatabase.systemFont(
            QFontDatabase.SystemFont.FixedFont
        )
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(2000)
        self.log_edit.setMaximumHeight(120)
        self.log_edit.setFont(fixed_font)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addLayout(row)
        bottom_layout.addWidget(self.log_edit)
        self.bottom_panel = bottom

    def _toggle_terminal(self, checked: bool) -> None:
        self.log_edit.setVisible(checked)

    # ------------------------------------------------------------ settings
    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    @staticmethod
    def _as_int(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _make_settings_store(self) -> QSettings:
        """Portable refinedocr.ini at the project root/app, registry fallback."""
        base = core.project_root()
        path = base / core.SETTINGS_FILENAME
        if not os.access(base, os.W_OK):
            return QSettings("refinedocr", "refinedocr")
        existed = path.exists()
        store = QSettings(str(path), QSettings.Format.IniFormat)
        if not existed:
            # One-time migration from the old registry/system location.
            legacy = QSettings("refinedocr", "refinedocr")
            for key in legacy.allKeys():
                store.setValue(key, legacy.value(key))
            store.sync()
        self.settings_path = path
        return store

    def _load_settings(self) -> None:
        store = self.settings_store
        self._loading_settings = True
        try:
            self.server_edit.setText(
                core.normalize_server_url(
                    str(store.value("server", core.DEFAULT_SERVER))
                )
            )
            self.format_combo.setCurrentText(str(store.value("format", "md")))
            self.dpi_spin.setValue(
                self._as_int(
                    store.value("dpi", core.DEFAULT_DPI), core.DEFAULT_DPI
                )
            )
            legacy_max = self._as_int(
                store.value("max_side", core.DEFAULT_MAX_SIDE),
                core.DEFAULT_MAX_SIDE,
            )
            # One-time migration: the old default (1600 px) is too slow on
            # CPU-bound vision prefill; move saved defaults to 1200 px.
            if legacy_max == 1600 and not self._as_bool(
                store.value("max_side_1200_migrated", False)
            ):
                legacy_max = core.DEFAULT_MAX_SIDE
                store.setValue("max_side", legacy_max)
                store.setValue("max_side_1200_migrated", True)
            self._engine_max_sides["glm"] = self._as_int(
                store.value("max_side_glm", core.DEFAULT_MAX_SIDE_GLM),
                core.DEFAULT_MAX_SIDE_GLM,
            )
            self._engine_max_sides["tesseract"] = self._as_int(
                store.value("max_side_tesseract", legacy_max), legacy_max
            )

            stored_engine = str(store.value("engine", core.DEFAULT_ENGINE))
            # The old build only had GLM as a default; move once to the new
            # local default so an upgrade actually uses Tesseract.
            if stored_engine == "glm" and not self._as_bool(
                store.value("engine_tesseract_migrated", False)
            ):
                stored_engine = core.DEFAULT_ENGINE
                store.setValue("engine", stored_engine)
            store.setValue("engine_tesseract_migrated", True)
            engine_index = self.engine_combo.findData(stored_engine)
            if engine_index < 0:
                engine_index = self.engine_combo.findData(
                    core.DEFAULT_ENGINE
                )
            self.engine_combo.setCurrentIndex(max(engine_index, 0))
            self._last_engine = (
                self.engine_combo.currentData() or core.DEFAULT_ENGINE
            )
            self.max_side_spin.setValue(
                self._engine_max_sides[self._last_engine]
            )

            self.tesseract_lang_combo.setCurrentText(
                str(
                    store.value(
                        "tesseract_lang", core.DEFAULT_TESSERACT_LANG
                    )
                )
            )
            self._tesseract_path = str(store.value("tesseract_path", ""))
            self.prompt_edit.setText(
                str(store.value("prompt", core.DEFAULT_PROMPT))
            )
            self.outdir_edit.setText(str(store.value("output_dir", "")))
            self.force_check.setChecked(
                self._as_bool(store.value("force_ocr", False))
            )
            self.terminal_check.setChecked(
                self._as_bool(store.value("show_terminal", True))
            )
            spread_index = self.spreads_combo.findData(
                str(store.value("spreads", "off"))
            )
            self.spreads_combo.setCurrentIndex(
                spread_index if spread_index >= 0 else 0
            )
        finally:
            self._loading_settings = False
        self.on_engine_changed()

    def _save_settings(self) -> None:
        store = self.settings_store
        engine = self.engine_combo.currentData() or core.DEFAULT_ENGINE
        self._engine_max_sides[engine] = self.max_side_spin.value()
        store.setValue(
            "server", self.server_edit.text().strip() or core.DEFAULT_SERVER
        )
        store.setValue("format", self.format_combo.currentText())
        store.setValue("dpi", self.dpi_spin.value())
        store.setValue("max_side", self.max_side_spin.value())
        store.setValue(
            "max_side_glm", self._engine_max_sides["glm"]
        )
        store.setValue(
            "max_side_tesseract", self._engine_max_sides["tesseract"]
        )
        store.setValue(
            "prompt", self.prompt_edit.text().strip() or core.DEFAULT_PROMPT
        )
        store.setValue("output_dir", self.outdir_edit.text().strip())
        store.setValue("force_ocr", self.force_check.isChecked())
        store.setValue("show_terminal", self.terminal_check.isChecked())
        store.setValue("spreads", self.spreads_combo.currentData() or "off")
        store.setValue("engine", engine)
        store.setValue(
            "tesseract_lang",
            self.tesseract_lang_combo.currentText().strip()
            or core.DEFAULT_TESSERACT_LANG,
        )
        store.setValue("tesseract_path", self._tesseract_path)

    def _current_settings(self) -> OcrSettings:
        return OcrSettings(
            server=core.normalize_server_url(self.server_edit.text()),
            prompt=self.prompt_edit.text().strip() or core.DEFAULT_PROMPT,
            dpi=self.dpi_spin.value(),
            max_side=self.max_side_spin.value(),
            out_format=self.format_combo.currentText(),
            output_dir=self.outdir_edit.text().strip(),
            force_ocr=self.force_check.isChecked(),
            spreads=self.spreads_combo.currentData() or "off",
            engine=self.engine_combo.currentData() or core.DEFAULT_ENGINE,
            tesseract_lang=(
                self.tesseract_lang_combo.currentText().strip()
                or core.DEFAULT_TESSERACT_LANG
            ),
            tesseract_path=self._tesseract_path,
        )

    def on_engine_changed(self, *_args) -> None:
        engine = self.engine_combo.currentData() or core.DEFAULT_ENGINE
        if not self._loading_settings and engine != self._last_engine:
            self._engine_max_sides[self._last_engine] = (
                self.max_side_spin.value()
            )
            self.max_side_spin.setValue(
                self._engine_max_sides.get(engine, core.DEFAULT_MAX_SIDE)
            )
        self._last_engine = engine
        glm = engine == "glm"
        editable = not self.running
        self.server_edit.setEnabled(glm and editable)
        self.test_btn.setEnabled(glm and editable)
        self.tesseract_lang_combo.setEnabled(not glm and editable)
        self._update_ask_controls()

    # -------------------------------------------------------------- files
    def choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open documents",
            "",
            "Documents (*.pdf *.cbz *.png *.jpg *.jpeg *.webp *.bmp "
            "*.tif *.tiff);;All files (*)",
        )
        self.open_files(paths)

    def open_files(self, paths: list[str]) -> None:
        """Open a dialog selection as a new project; drag-drop appends."""
        if not paths:
            return
        if self.jobs:
            self._log("New project: cleared the previous files.")
            self.clear_files()
        self.add_paths(paths)

    def add_paths(self, paths: list[str]) -> None:
        added = False
        for raw in paths:
            path = Path(raw)
            if path.suffix.lower() not in core.SUPPORTED_SUFFIXES:
                self._log(f"Skipped unsupported file: {path}")
                continue
            if any(job.path == path for job in self.jobs):
                continue
            pages = self._page_count(path)
            if pages < 1:
                if path.suffix.lower() in core.COMIC_SUFFIXES:
                    self._log(
                        f"Skipped {path.name}: no readable image pages in "
                        f"the archive (WebP/AVIF entries are not supported)"
                    )
                else:
                    self._log(f"Skipped unreadable file: {path}")
                continue
            job = FileJob(path=path, pages=pages)
            job.page_pngs = [None] * pages
            job.page_texts = [""] * pages
            # Nothing is checked by default: OCR only what the user selects.
            job.selected_pages = set()
            self.jobs.append(job)
            added = True
        if added:
            self.rebuild_files_list()
            self.files_list.setCurrentRow(len(self.jobs) - 1)
            self._log(f"Added {len(self.jobs)} file(s).")
            self.statusBar().showMessage(f"{len(self.jobs)} file(s) queued")

    def _page_count(self, path: Path) -> int:
        try:
            with pymupdf.open(path) as document:
                if document.needs_pass:
                    return 0
                return document.page_count
        except Exception:
            return 0

    def _file_item(self, job: FileJob, index: int) -> QListWidgetItem:
        item = QListWidgetItem(
            f"{job.path.name}  ·  {job.pages} pp  ·  {job.status}"
        )
        item.setData(Qt.ItemDataRole.UserRole, index)
        item.setToolTip(str(job.path))
        item.setForeground(QBrush(self.STATUS_COLORS.get(
            job.status, QColor("#000000")
        )))
        return item

    def rebuild_files_list(self) -> None:
        row = self.current_file
        self.files_list.blockSignals(True)
        self.files_list.clear()
        for index, job in enumerate(self.jobs):
            self.files_list.addItem(self._file_item(job, index))
        self.files_list.blockSignals(False)
        if self.jobs:
            row = 0 if row is None or row >= len(self.jobs) else row
            self.files_list.setCurrentRow(row)
        else:
            self.current_file = None
            self.pages_list.clear()
            self._clear_view()

    def update_file_item(self, index: int) -> None:
        if not (0 <= index < len(self.jobs)):
            return
        item = self.files_list.item(index)
        if item is None:
            return
        job = self.jobs[index]
        item.setText(f"{job.path.name}  ·  {job.pages} pp  ·  {job.status}")
        item.setForeground(QBrush(self.STATUS_COLORS.get(
            job.status, QColor("#000000")
        )))

    def remove_selected(self) -> None:
        if self.thread is not None:
            return
        row = self.files_list.currentRow()
        if row < 0 or row >= len(self.jobs):
            return
        del self.jobs[row]
        self.current_file = None
        self.rebuild_files_list()

    def clear_files(self) -> None:
        if self.thread is not None:
            return
        self.jobs.clear()
        self.current_file = None
        self.rebuild_files_list()
        self._clear_view()
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------ selection
    def select_file(self, row: int) -> None:
        if row < 0 or row >= len(self.jobs):
            self.current_file = None
            self.pages_list.clear()
            self._clear_view()
            return
        self.current_file = row
        job = self.jobs[row]
        # Never leave the previous document's text visible while switching.
        self.raw_edit.clear()
        self.page_browser.clear()
        self.doc_browser.clear()
        self.doc_label.setText(job.path.name)
        self._populate_pages(job)
        if job.pages:
            page_row = min(max(self.current_page, 1), job.pages) - 1
            self.pages_list.setCurrentRow(page_row)
            self.select_page(page_row)

    def _populate_pages(self, job: FileJob) -> None:
        self.pages_list.blockSignals(True)
        self.pages_list.clear()
        for number in range(1, job.pages + 1):
            done = (
                number - 1 < len(job.page_texts)
                and bool(job.page_texts[number - 1])
            )
            item = QListWidgetItem(
                f"Page {number}" + ("  ✓" if done else "")
            )
            item.setData(Qt.ItemDataRole.UserRole, number)
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                Qt.CheckState.Checked
                if number in job.selected_pages
                else Qt.CheckState.Unchecked
            )
            item.setToolTip(
                "Checked: OCR this page and include it in the output. "
                "✓ means it is already recognized."
            )
            self.pages_list.addItem(item)
        self.pages_list.blockSignals(False)
        self._update_pages_label()

    def _update_pages_label(self) -> None:
        if self.current_file is None:
            self.pages_label.setText("Pages")
            return
        job = self.jobs[self.current_file]
        checked = len(
            [p for p in job.selected_pages if 1 <= p <= job.pages]
        )
        self.pages_label.setText(f"Pages — {checked}/{job.pages} checked")

    def on_page_check_changed(self, item: QListWidgetItem) -> None:
        if self.current_file is None:
            return
        page = item.data(Qt.ItemDataRole.UserRole)
        if page is None:
            return
        job = self.jobs[self.current_file]
        if item.checkState() == Qt.CheckState.Checked:
            job.selected_pages.add(int(page))
        else:
            job.selected_pages.discard(int(page))
        self._update_pages_label()
        self.doc_browser.setMarkdown(self._document_markdown(job))

    def select_pages(self, mode: str) -> None:
        if self.current_file is None:
            return
        job = self.jobs[self.current_file]
        all_pages = set(range(1, job.pages + 1))
        if mode == "all":
            job.selected_pages = all_pages
        elif mode == "none":
            job.selected_pages = set()
        elif mode == "invert":
            job.selected_pages = all_pages - job.selected_pages
        self._populate_pages(job)
        if job.pages:
            row = min(max(self.current_page, 1), job.pages) - 1
            self.pages_list.setCurrentRow(row)
        self.doc_browser.setMarkdown(self._document_markdown(job))

    def select_page(self, row: int) -> None:
        if self.current_file is None:
            return
        job = self.jobs[self.current_file]
        if row < 0 or row >= job.pages:
            return
        self.current_page = row + 1
        self.show_page()

    def step_page(self, delta: int) -> None:
        if self.current_file is None:
            return
        job = self.jobs[self.current_file]
        target = self.current_page + delta
        if 1 <= target <= job.pages:
            self.pages_list.setCurrentRow(target - 1)

    # -------------------------------------------------------------- regions
    def on_regions_changed(self) -> None:
        if self.current_file is None:
            return
        job = self.jobs[self.current_file]
        regions = self.image_view.regions()
        if regions:
            job.regions[self.current_page] = regions
        else:
            job.regions.pop(self.current_page, None)
        self._update_regions_hint()

    def clear_page_regions(self) -> None:
        if self.current_file is None:
            return
        self.jobs[self.current_file].regions.pop(self.current_page, None)
        self.image_view.set_regions([])
        self._update_regions_hint()

    def _update_regions_hint(self) -> None:
        count = len(self.image_view.regions())
        self.regions_count.setText(
            f"{count}/{self.image_view.MAX_REGIONS} regions"
        )
        self.btn_clear_regions.setEnabled(count > 0 and not self.running)

    def on_spreads_changed(self, *_args) -> None:
        mode = self.spreads_combo.currentData() or "off"
        fraction = 0.5
        if self.current_file is not None:
            fraction = self.jobs[self.current_file].splits.get(
                self.current_page, 0.5
            )
        self.image_view.set_spread(
            mode != "off", fraction, auto=(mode == "auto")
        )

    def on_split_changed(self, fraction: float) -> None:
        if self.current_file is None:
            return
        self.jobs[self.current_file].splits[self.current_page] = fraction

    def _sync_spreads(self, job: FileJob, page_number: int) -> None:
        mode = self.spreads_combo.currentData() or "off"
        self.image_view.set_spread(
            mode != "off",
            job.splits.get(page_number, 0.5),
            auto=(mode == "auto"),
        )

    # ---------------------------------------------------------------- zoom
    PREVIEW_MAX_ZOOM = 8.0

    def _zoom_preview(self, factor: float) -> None:
        if self.image_view.source_size() is None:
            return
        new_zoom = min(
            max(self.preview_zoom * factor, 1.0), self.PREVIEW_MAX_ZOOM
        )
        if abs(new_zoom - self.preview_zoom) < 1e-6:
            return
        self.preview_zoom = new_zoom
        self._apply_preview_zoom(keep_center=True)

    def _reset_preview_zoom(self) -> None:
        self.preview_zoom = 1.0
        self._apply_preview_zoom()
        self.scroll.horizontalScrollBar().setValue(0)
        self.scroll.verticalScrollBar().setValue(0)

    def _pan_preview(self, dx: int, dy: int) -> None:
        """Middle-drag: move the zoomed page under the cursor."""
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        hbar.setValue(hbar.value() - dx)
        vbar.setValue(vbar.value() - dy)

    def _apply_preview_zoom(self, keep_center: bool = False) -> None:
        viewport = self.scroll.viewport().size()
        source = self.image_view.source_size()
        if source is None:
            if viewport.width() > 0 and viewport.height() > 0:
                self.image_view.resize(viewport)
            return
        if viewport.width() <= 0 or viewport.height() <= 0:
            return
        fit = min(
            viewport.width() / source.width(),
            viewport.height() / source.height(),
        )
        scale = fit * self.preview_zoom
        old_scale = self.image_view.display_scale()
        hbar = self.scroll.horizontalScrollBar()
        vbar = self.scroll.verticalScrollBar()
        center = None
        if keep_center:
            center = (
                hbar.value() + viewport.width() / 2,
                vbar.value() + viewport.height() / 2,
            )
        self.image_view.set_display_scale(scale)
        if center is not None and old_scale > 0:
            ratio = scale / old_scale
            hbar.setValue(round(center[0] * ratio - viewport.width() / 2))
            vbar.setValue(round(center[1] * ratio - viewport.height() / 2))

    def eventFilter(self, obj, event):  # noqa: N802 - Qt override
        if obj is self.scroll.viewport() and event.type() == QEvent.Type.Resize:
            self._apply_preview_zoom()
        return super().eventFilter(obj, event)

    # ---------------------------------------------------------------- view
    def _preview_page(self, job: FileJob, page_number: int):
        """Render/cache a page preview and return (png_bytes, text_layer)."""
        index = page_number - 1
        if not (0 <= index < job.pages):
            return None, ""
        cached = job.page_pngs[index] if index < len(job.page_pngs) else None
        processed = job.page_texts[index] if index < len(job.page_texts) else ""
        text_layer = ""
        is_pdf = job.path.suffix.lower() in core.PDF_SUFFIXES
        if cached is not None and (not is_pdf or processed):
            # Already rendered, and there is no PDF text layer still to read:
            # do not touch the source file again (important for slow USB disks).
            return cached, ""

        try:
            with pymupdf.open(job.path) as document:
                page = document.load_page(index)
                if cached is None:
                    QApplication.setOverrideCursor(
                        Qt.CursorShape.WaitCursor
                    )
                    started = time.time()
                    try:
                        cached = core.render_page_png(
                            page, is_pdf, PREVIEW_DPI, PREVIEW_MAX_SIDE
                        )
                    finally:
                        QApplication.restoreOverrideCursor()
                    elapsed = time.time() - started
                    if elapsed >= 0.2:
                        width, height = core.png_size(cached)
                        self._log(
                            f"Preview page {page_number}: {width}x{height} "
                            f"rendered in {elapsed:.2f}s"
                        )
                    if index < len(job.page_pngs):
                        job.page_pngs[index] = cached
                if is_pdf and not processed:
                    text_layer = core.extract_text_layer(page)
        except Exception as exc:
            self._log(f"Preview failed for page {page_number}: {exc}")
        return cached, text_layer

    def _show_image_bytes(self, data, page_number: int) -> None:
        if data:
            pixmap = QPixmap()
            if pixmap.loadFromData(data, "PNG"):
                self.image_view.set_source(pixmap)
                return
        self.image_view.set_source(None)
        self.image_view.set_empty_text(
            f"Page {page_number}: preview unavailable"
            if page_number
            else "No page selected"
        )

    def _show_image(self, job: FileJob, page_number: int) -> None:
        data = None
        if 0 <= page_number - 1 < len(job.page_pngs):
            data = job.page_pngs[page_number - 1]
        self._show_image_bytes(data, page_number)
        self.image_view.set_regions(job.regions.get(page_number, []))
        self._sync_spreads(job, page_number)
        self._update_regions_hint()
        self._reset_preview_zoom()

    def show_page(self) -> None:
        if self.current_file is None:
            self._clear_view()
            return
        job = self.jobs[self.current_file]
        page_number = self.current_page
        if not (1 <= page_number <= job.pages):
            self._clear_view()
            return

        self.nav_label.setText(f"Page {page_number} / {job.pages}")
        self.doc_label.setText(
            f"{job.path.name} — page {page_number}/{job.pages}"
        )
        self.prev_btn.setEnabled(page_number > 1)
        self.next_btn.setEnabled(page_number < job.pages)
        png, text_layer = self._preview_page(job, page_number)
        self._show_image_bytes(png, page_number)
        self.image_view.set_regions(job.regions.get(page_number, []))
        self._sync_spreads(job, page_number)
        self._update_regions_hint()
        self._reset_preview_zoom()

        text = ""
        if page_number - 1 < len(job.page_texts):
            text = job.page_texts[page_number - 1]
        if not text:
            text = self.stream_buf.get(
                (self.current_file, page_number), ""
            )
        if not text:
            text = text_layer
        self.raw_edit.setPlainText(text)
        if text:
            self.page_browser.setMarkdown(text)
            if text_layer and not (
                page_number - 1 < len(job.page_texts)
                and job.page_texts[page_number - 1]
            ):
                self.doc_label.setText(
                    f"{job.path.name} — page {page_number}/{job.pages} "
                    f"(text layer preview, not OCR)"
                )
                self.statusBar().showMessage(
                    f"Page {page_number}: text layer preview (not OCR)",
                    4000,
                )
        else:
            self.page_browser.setPlainText("(not recognized yet)")
        self.doc_browser.setMarkdown(self._document_markdown(job))
        self._update_ask_controls()

    def _document_markdown(self, job: FileJob) -> str:
        """Markdown for checked pages only; unchecked pages are excluded."""
        if not job.selected_pages:
            return ""
        return "\n\n".join(
            job.page_texts[page - 1]
            for page in sorted(job.selected_pages)
            if 0 <= page - 1 < len(job.page_texts)
            and job.page_texts[page - 1].strip()
        )

    def _clear_view(self) -> None:
        self.image_view.set_source(None)
        self.image_view.set_empty_text("No page selected")
        self.image_view.set_regions([])
        self._update_regions_hint()
        self._reset_preview_zoom()
        self.doc_label.setText("No document")
        self.pages_label.setText("Pages")
        self.raw_edit.clear()
        self.page_browser.clear()
        self.doc_browser.clear()
        self.nav_label.setText("Page – / –")
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self._update_ask_controls()

    def _refresh_rendered(self) -> None:
        if self.current_file is None:
            return
        key = (self.current_file, self.current_page)
        if key not in self.active_pages:
            return
        text = self.stream_buf.get(key, "")
        if text:
            self.page_browser.setMarkdown(text)
            scrollbar = self.page_browser.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            return
        if (
            not self.cancelled
            and self.ocr_wait_started is not None
            and self.ocr_wait_key == key
        ):
            elapsed = int(time.time() - self.ocr_wait_started)
            self.page_browser.setPlainText(
                f"Waiting for the model… {elapsed}s\n\n"
                f"(image prefill can take minutes at high resolution)"
            )
            job = self.jobs[self.current_file]
            self.status_label.setText(
                f"{job.path.name} — page {self.current_page}/{job.pages}: "
                f"waiting for the model ({elapsed}s)…"
            )

    # ------------------------------------------------------------- running
    def _set_running(self, running: bool) -> None:
        self.act_add.setEnabled(not running)
        self.act_remove.setEnabled(not running)
        self.act_clear.setEnabled(not running)
        self.act_start.setEnabled(not running)
        self.act_cancel.setEnabled(running)
        self.act_save_as.setEnabled(not running)
        self.pages_list.setEnabled(not running)
        self.btn_pages_all.setEnabled(not running)
        self.btn_pages_none.setEnabled(not running)
        self.btn_pages_invert.setEnabled(not running)
        self.running = running
        self.engine_combo.setEnabled(not running)
        self.on_engine_changed()
        self.image_view.set_editable(not running)
        self._update_regions_hint()

    def start_ocr(self) -> None:
        if self.thread is not None:
            return
        if self.ask_running:
            self._log(
                "Wait for the Ask request to finish (or press Stop) before "
                "starting OCR."
            )
            return
        if not self.jobs:
            QMessageBox.information(
                self, "Nothing to do", "Add a PDF or image first."
            )
            return
        selected_total = sum(
            len([p for p in job.selected_pages if 1 <= p <= job.pages])
            for job in self.jobs
        )
        if selected_total == 0:
            QMessageBox.information(
                self,
                "No pages selected",
                "Check at least one page in the Pages list.",
            )
            return
        settings = self._current_settings()
        self.server_edit.setText(settings.server)

        self._save_settings()
        for job in self.jobs:
            job.status = "Pending"
            job.error = ""
            job.output_path = ""
            if len(job.page_pngs) != job.pages:
                job.page_pngs = [None] * job.pages
            if len(job.page_texts) != job.pages:
                job.page_texts = [""] * job.pages
            for page in job.selected_pages:
                if 1 <= page <= job.pages:
                    job.page_texts[page - 1] = ""
        self.rebuild_files_list()
        self.stream_buf.clear()
        self.active_pages.clear()
        self.pages_done = 0
        self.total_pages = selected_total
        self.progress.setRange(0, max(self.total_pages, 1))
        self.progress.setValue(0)
        self.log_edit.clear()
        self.cancelled = False
        self.server_error = False
        self._log(f"Starting OCR ({selected_total} page(s))...")
        self._log_run_settings(settings)
        self._set_running(True)
        self.render_timer.start()

        self.thread = QThread(self)
        self.worker = OCRWorker(self.jobs, settings)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self._log)
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_pages.connect(self.on_file_pages)
        self.worker.page_started.connect(self.on_page_started)
        self.worker.page_chunk.connect(self.on_page_chunk)
        self.worker.page_done.connect(self.on_page_done)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _log_run_settings(self, settings: OcrSettings) -> None:
        cap = (
            "unlimited"
            if settings.max_side <= 0
            else f"{settings.max_side} px"
        )
        if settings.engine == "tesseract":
            self._log(
                f"Settings: engine tesseract, lang "
                f"{settings.tesseract_lang!r}, dpi {settings.dpi} (PDFs), "
                f"max side {cap}"
            )
            return
        self._log(
            f"Settings: dpi {settings.dpi} (PDFs only), max side {cap}, "
            f"prompt {settings.prompt!r}"
        )
        raster = core.IMAGE_SUFFIXES | core.COMIC_SUFFIXES
        if settings.max_side <= 0 and any(
            job.selected_pages and job.path.suffix.lower() in raster
            for job in self.jobs
        ):
            self._log(
                "WARNING: max side is 0 (unlimited). Image/CBZ pages are "
                "sent at full resolution, which can take minutes. Set Max "
                f"side to e.g. {core.DEFAULT_MAX_SIDE} px."
            )

    def cancel_ocr(self) -> None:
        if self.worker is None:
            return
        self.cancelled = True
        self.worker.stop.set()
        self.act_cancel.setEnabled(False)
        self.status_label.setText("Cancelling…")
        self._log("Cancellation requested.")

    def _thread_finished(self) -> None:
        self.thread = None
        self.worker = None
        self.render_timer.stop()
        self.active_pages.clear()
        self.ocr_wait_started = None
        self.ocr_wait_key = None
        self._set_running(False)
        for index, job in enumerate(self.jobs):
            if job.status == "Working":
                job.status = "Cancelled" if self.cancelled else "Error"
            self.update_file_item(index)
        if self.cancelled:
            self.status_label.setText("Cancelled")
        elif self.server_error:
            self.status_label.setText("Server error")
        elif (
            self.total_pages > 0
            and self.pages_done >= self.total_pages
            and all(job.status in ("Done", "Skipped") for job in self.jobs)
        ):
            self.progress.setValue(self.total_pages)
            self.status_label.setText("Done")
        elif any(job.status == "Error" for job in self.jobs):
            self.status_label.setText("Finished with errors")
        else:
            self.status_label.setText("Finished")
        self.run_finished.emit()

    # ------------------------------------------------------------- Ask tab
    def _update_ask_controls(self) -> None:
        engine = self.engine_combo.currentData() or core.DEFAULT_ENGINE
        glm = engine == "glm"
        job = None
        has_page = False
        if (
            self.current_file is not None
            and 0 <= self.current_file < len(self.jobs)
        ):
            job = self.jobs[self.current_file]
            has_page = 1 <= self.current_page <= job.pages
        can_send = (
            glm and has_page and not self.running and not self.ask_running
        )
        self.ask_btn.setEnabled(can_send)
        self.ask_stop_btn.setEnabled(self.ask_running)
        self.prompt_edit.setEnabled(
            glm and not self.running and not self.ask_running
        )
        if not glm:
            self.ask_hint.setText(
                "Ask uses the server model. Select Engine = OCR model "
                "(server) to enable it."
            )
        elif self.ask_running:
            self.ask_hint.setText(
                "Waiting for the server model… (Stop cancels the request)"
            )
        elif has_page:
            self.ask_hint.setText(
                f"Asks about page {self.current_page} of {job.path.name}. "
                f"The Prompt is also used by Start OCR."
            )
        else:
            self.ask_hint.setText("Select a page to ask about.")

    def send_prompt(self) -> None:
        if self.ask_running or self.running:
            return
        engine = self.engine_combo.currentData() or core.DEFAULT_ENGINE
        if engine != "glm":
            self._log("Ask needs Engine = OCR model (server).")
            return
        if self.current_file is None:
            return
        job = self.jobs[self.current_file]
        page_number = self.current_page
        if not (1 <= page_number <= job.pages):
            return
        prompt = self.prompt_edit.text().strip()
        if not prompt:
            self._log("Enter a prompt/question first.")
            return
        settings = self._current_settings()
        self._save_settings()
        is_pdf = job.path.suffix.lower() in core.PDF_SUFFIXES
        self.ask_prompt_text = prompt
        self.ask_buf = ""
        self.ask_browser.setPlainText(f"You: {prompt}\n\nModel:\n")
        self.ask_running = True
        self._update_ask_controls()
        self.ask_thread = QThread(self)
        self.ask_worker = AskWorker(
            settings, job.path, page_number, is_pdf, prompt
        )
        self.ask_worker.moveToThread(self.ask_thread)
        self.ask_thread.started.connect(self.ask_worker.run)
        self.ask_worker.info.connect(self._log)
        self.ask_worker.chunk.connect(self.on_ask_chunk)
        self.ask_worker.done.connect(self.on_ask_done)
        self.ask_worker.failed.connect(self.on_ask_failed)
        self.ask_worker.finished.connect(self.ask_thread.quit)
        self.ask_worker.finished.connect(self.ask_worker.deleteLater)
        self.ask_thread.finished.connect(self._ask_thread_finished)
        self.ask_thread.finished.connect(self.ask_thread.deleteLater)
        self.ask_thread.start()

    def on_ask_chunk(self, piece: str) -> None:
        self.ask_buf += piece
        self.ask_browser.setPlainText(
            f"You: {self.ask_prompt_text}\n\nModel:\n{self.ask_buf}"
        )
        bar = self.ask_browser.verticalScrollBar()
        bar.setValue(bar.maximum())

    def on_ask_done(self, text: str) -> None:
        self.ask_buf = text or self.ask_buf
        body = self.ask_buf.strip() or "(no response)"
        self.ask_browser.setPlainText(
            f"You: {self.ask_prompt_text}\n\nModel:\n{body}"
        )
        self._log(f"Ask finished ({len(self.ask_buf)} chars)")

    def on_ask_failed(self, message: str) -> None:
        self.ask_browser.setPlainText(
            f"You: {self.ask_prompt_text}\n\nError: {message}"
        )
        self._log(f"ERROR Ask: {message}")

    def stop_prompt(self) -> None:
        if self.ask_worker is None:
            return
        self.ask_worker.stop.set()
        self.ask_stop_btn.setEnabled(False)
        self._log("Ask cancellation requested.")

    def _ask_thread_finished(self) -> None:
        self.ask_thread = None
        self.ask_worker = None
        self.ask_running = False
        self._update_ask_controls()

    # ------------------------------------------------------ worker signals
    def on_file_started(self, index: int, name: str) -> None:
        if not (0 <= index < len(self.jobs)):
            return
        self.jobs[index].status = "Working"
        self.update_file_item(index)
        self._log(f"=== {name} ===")
        if self.current_file != index:
            self.files_list.setCurrentRow(index)

    def on_file_pages(self, index: int, total: int) -> None:
        if not (0 <= index < len(self.jobs)):
            return
        job = self.jobs[index]
        job.pages = total
        if len(job.page_pngs) != total:
            job.page_pngs = [None] * total
        if len(job.page_texts) != total:
            job.page_texts = [""] * total
        if self.current_file == index:
            self._populate_pages(job)
        self.update_file_item(index)

    def on_page_started(
        self, index: int, page: int, total: int, png: bytes
    ) -> None:
        if not (0 <= index < len(self.jobs)):
            return
        job = self.jobs[index]
        if len(job.page_pngs) < total:
            job.page_pngs.extend([None] * (total - len(job.page_pngs)))
        job.page_pngs[page - 1] = png
        if self.current_file != index:
            return
        self.current_page = page
        self.pages_list.blockSignals(True)
        self.pages_list.setCurrentRow(page - 1)
        self.pages_list.blockSignals(False)
        self.nav_label.setText(f"Page {page} / {total}")
        self.prev_btn.setEnabled(page > 1)
        self.next_btn.setEnabled(page < total)
        self._show_image(job, page)
        self.stream_buf[(index, page)] = ""
        self.active_pages.add((index, page))
        self.ocr_wait_started = time.time()
        self.ocr_wait_key = (index, page)
        self.raw_edit.clear()
        self.page_browser.setPlainText("Recognizing…")
        self.status_label.setText(f"{job.path.name} — page {page}/{total}")

    def on_page_chunk(self, index: int, page: int, delta: str) -> None:
        key = (index, page)
        if delta and self.ocr_wait_key == key:
            self.ocr_wait_started = None
            self.ocr_wait_key = None
        self.stream_buf[key] = self.stream_buf.get(key, "") + delta
        if self.current_file == index and self.current_page == page:
            cursor = self.raw_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.raw_edit.setTextCursor(cursor)
            self.raw_edit.insertPlainText(delta)

    def on_page_done(self, index: int, page: int, text: str) -> None:
        if not (0 <= index < len(self.jobs)):
            return
        job = self.jobs[index]
        if page - 1 < len(job.page_texts):
            job.page_texts[page - 1] = text
        self.stream_buf[(index, page)] = text
        self.active_pages.discard((index, page))
        if self.ocr_wait_key == (index, page):
            self.ocr_wait_started = None
            self.ocr_wait_key = None
        if self.current_file == index:
            item = self.pages_list.item(page - 1)
            if item is not None:
                item.setText(f"Page {page}  ✓")
            if self.current_page == page:
                self.raw_edit.setPlainText(text)
                self.page_browser.setMarkdown(text)
            self.doc_browser.setMarkdown(self._document_markdown(job))
        self.pages_done += 1
        self.progress.setValue(min(self.pages_done, self.total_pages))

    def on_file_done(self, index: int, output_path: str) -> None:
        if not (0 <= index < len(self.jobs)):
            return
        job = self.jobs[index]
        job.status = "Done"
        job.output_path = output_path
        self.update_file_item(index)
        self._log(f"Saved: {output_path}")
        self.statusBar().showMessage(f"Saved {output_path}", 10000)
        if self.current_file == index:
            self.doc_browser.setMarkdown(self._document_markdown(job))

    def on_failed(self, index: int, message: str) -> None:
        if index < 0:
            self.server_error = True
            self._log(f"ERROR: {message}")
            self.status_label.setText("Server error")
            self.statusBar().showMessage(message, 15000)
            return
        if 0 <= index < len(self.jobs):
            job = self.jobs[index]
            job.status = "Error"
            job.error = message
            self.update_file_item(index)
            self._log(f"ERROR {job.path.name}: {message}")

    # -------------------------------------------------------------- misc
    def _log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log_edit.appendPlainText(f"[{stamp}] {message}")

    def test_server(self) -> None:
        ok, message = core.server_status(self.server_edit.text(), timeout=3)
        normalized = core.normalize_server_url(self.server_edit.text())
        self.server_edit.setText(normalized)
        self._log(("OK: " if ok else "ERROR: ") + message)
        self.statusBar().showMessage(message, 10000)
        self.status_label.setText("Server ready" if ok else "Server error")
        if not ok:
            QMessageBox.warning(self, "Server unreachable", message)

    def _text_layer_markdown(self, job: FileJob) -> str:
        """Read the PDF text layer for the checked pages (no server call)."""
        if job.path.suffix.lower() not in core.PDF_SUFFIXES:
            return ""
        pages = sorted(p for p in job.selected_pages if 1 <= p <= job.pages)
        if not pages:
            return ""
        parts: list[str] = []
        try:
            with pymupdf.open(job.path) as document:
                for number in pages:
                    text = core.extract_text_layer(
                        document.load_page(number - 1)
                    ).strip()
                    if text:
                        parts.append(text)
        except Exception as exc:
            self._log(f"Text layer extraction failed: {exc}")
            return ""
        return "\n\n".join(parts)

    def save_as(self) -> None:
        """Save already recognized text in another format without re-OCR."""
        if self.current_file is None:
            QMessageBox.information(
                self, "Save as", "Select a document first."
            )
            return
        job = self.jobs[self.current_file]
        markdown = self._document_markdown(job)
        if not markdown.strip() and not self.force_check.isChecked():
            # Force OCR off and nothing recognized yet: a PDF's own text layer
            # is exactly what Start OCR would use, and needs no server run.
            markdown = self._text_layer_markdown(job)
            if markdown.strip():
                self._log("No OCR result yet; exporting the PDF text layer.")
        if not markdown.strip():
            QMessageBox.information(
                self,
                "Save as",
                "This document has no recognized text yet. Run OCR first.",
            )
            return

        default_format = self.format_combo.currentText()
        if default_format not in ("md", "txt", "odt"):
            default_format = "md"
        default_path = job.path.with_suffix(f".{default_format}")
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save recognized text",
            str(default_path),
            "Markdown (*.md);;Plain text (*.txt);;OpenDocument (*.odt)",
        )
        if not path:
            return

        suffix = Path(path).suffix.lower().lstrip(".")
        if suffix in ("md", "txt", "odt"):
            fmt = suffix
        else:
            if "OpenDocument" in selected_filter:
                fmt = "odt"
            elif "Plain text" in selected_filter:
                fmt = "txt"
            else:
                fmt = "md"
            path = str(Path(path).with_suffix(f".{fmt}"))

        try:
            core.save_output(markdown, fmt, Path(path))
        except Exception as exc:
            self._log(f"ERROR saving {path}: {exc}")
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._log(f"Saved ({fmt}): {path}")
        self.statusBar().showMessage(f"Saved {path}", 10000)

    def choose_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Output folder")
        if folder:
            self.outdir_edit.setText(folder)

    def open_output_dir(self) -> None:
        folder = self.outdir_edit.text().strip()
        if not folder:
            if self.current_file is not None:
                folder = str(self.jobs[self.current_file].path.parent)
            elif self.jobs:
                folder = str(self.jobs[0].path.parent)
        if folder and Path(folder).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
        else:
            QMessageBox.information(
                self, "Output folder", "No output folder yet."
            )

    def closeEvent(self, event):  # noqa: N802 - Qt override
        if self.thread is not None:
            QMessageBox.information(
                self,
                "OCR running",
                "Cancel the current OCR job before closing.",
            )
            event.ignore()
            return
        if self.ask_thread is not None:
            QMessageBox.information(
                self,
                "Ask running",
                "Stop the Ask request before closing.",
            )
            event.ignore()
            return
        self._save_settings()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("refinedocr")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
