"""把 PDF / 图片 转成发给模型的页面图片（base64），PDF 另外抽取文字层。"""
import base64
import io
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageOps

from .config import Config


@dataclass
class Document:
    path: str
    pages: List[str] = field(default_factory=list)       # data URL 列表
    page_sizes: List[tuple] = field(default_factory=list)
    text: Optional[str] = None                            # PDF 文字层（扫描件为 None）


def _fit(img: Image.Image, cfg: Config, max_side: Optional[int] = None) -> Image.Image:
    if img.mode not in ("RGB", "L"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        rgba = img.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    elif img.mode == "L":
        img = img.convert("RGB")
    w, h = img.size
    long_side = max(w, h)
    max_side = max_side or cfg.max_side
    if long_side > max_side:
        s = max_side / long_side
    elif long_side < cfg.min_side:
        s = cfg.min_side / long_side
    else:
        s = 1.0
    if s != 1.0:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return img


def _to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    data, mime = buf.getvalue(), "image/png"
    if len(data) > 3_000_000:  # 太大就转 JPEG，避免超过接口限制
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        data, mime = buf.getvalue(), "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode()


# pdfium 不是线程安全的：多个线程同时打开/渲染 PDF 会让进程直接崩溃（Windows 上没有任何报错）。
# 所有 pdfium 操作都必须在这把全局锁里进行。
_PDFIUM_LOCK = threading.Lock()


def _load_pdf(path: Path, cfg: Config) -> Document:
    with _PDFIUM_LOCK:
        return _load_pdf_locked(path, cfg)


def _load_pdf_locked(path: Path, cfg: Config) -> Document:
    import pypdfium2 as pdfium

    doc = Document(str(path))
    pdf = pdfium.PdfDocument(str(path))
    texts = []
    try:
        raw = [pdf[i].get_textpage().get_text_range().strip() for i in range(len(pdf))]
        has_text = sum(len(t) for t in raw) > 50 * len(raw)
        scale = cfg.pdf_scale if has_text else cfg.scan_scale
        max_side = cfg.max_side if has_text else cfg.scan_max_side
        for i in range(len(pdf)):
            img = _fit(pdf[i].render(scale=scale).to_pil(), cfg, max_side)
            doc.pages.append(_to_data_url(img))
            doc.page_sizes.append(img.size)
            texts.append(f"----- 第{i + 1}页 -----\n{raw[i]}")
    finally:
        pdf.close()
    if has_text:
        doc.text = "\n".join(texts)[: cfg.pdf_text_chars]
    return doc


def _load_image(path: Path, cfg: Config) -> Document:
    doc = Document(str(path))
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im.load()
        img = _fit(im, cfg)
    doc.pages.append(_to_data_url(img))
    doc.page_sizes.append(img.size)
    return doc


_cache: Dict[str, Document] = {}
_lock = threading.Lock()
_file_locks: Dict[str, threading.Lock] = {}


def load_document(path: str, cfg: Config) -> Document:
    """同一文件只渲染一次（多线程安全）。"""
    with _lock:
        if path in _cache:
            return _cache[path]
        flock = _file_locks.setdefault(path, threading.Lock())
    with flock:
        if path in _cache:
            return _cache[path]
        p = Path(path)
        doc = _load_pdf(p, cfg) if p.suffix.lower() == ".pdf" else _load_image(p, cfg)
        with _lock:
            _cache[path] = doc
        return doc
