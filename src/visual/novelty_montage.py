from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import unicodedata
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.index import build_run_index
from src.utils.io import load_run_config

# ---------- path + filename helpers ----------


def _index_images_recursive(root: Path, exts=(".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    if not root.exists():
        return mapping
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            mapping[p.name.lower()] = p
    return mapping


_WINDOWS_FORBIDDEN = set('<>:"/\\|?*')
_WINDOWS_RESERVED = {
    "CON","PRN","AUX","NUL",
    *(f"COM{i}" for i in range(1,10)),
    *(f"LPT{i}" for i in range(1,10)),
}


def _safe_slug(name: str, max_len: int = 120) -> str:
    if name is None:
        return "unnamed"
    s = unicodedata.normalize("NFKC", str(name))
    s = "".join(("_" if ch in _WINDOWS_FORBIDDEN else ch) for ch in s)
    s = re.sub(r"[\x00-\x1f]", "_", s)
    s = re.sub(r"[ \t]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._ ")
    if not s:
        s = "unnamed"
    if s.upper() in _WINDOWS_RESERVED:
        s = f"_{s}"
    if len(s) > max_len:
        s = s[:max_len].rstrip("._ ")
    return s

# ---------- drawing ----------


def _make_contact_sheet(
    images: List[Tuple[str, Optional[Path]]],
    out_path: Path,
    title: str,
    thumb_size: Tuple[int, int] = (256, 256),
    thumbs_per_row: int = 8,
    margin: int = 12,
    bg=(255, 255, 255),
    text_color=(20, 20, 20),
):
    if not images:
        return
    w_thumb, h_thumb = thumb_size
    title_h = 40
    caption_h = 28
    rows = (len(images) + thumbs_per_row - 1) // thumbs_per_row
    sheet_w = margin + thumbs_per_row * (w_thumb + margin)
    sheet_h = title_h + margin + rows * (h_thumb + caption_h + margin)

    canvas = Image.new("RGB", (sheet_w, sheet_h), bg)
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("arial.ttf", 20)
        font_cap = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_cap = ImageFont.load_default()

    draw.text((margin, (title_h - 20) // 2), title, fill=text_color, font=font_title)

    y0 = title_h
    for i, (cap, pth) in enumerate(images):
        r = i // thumbs_per_row
        c = i % thumbs_per_row
        x = margin + c * (w_thumb + margin)
        y = y0 + margin + r * (h_thumb + caption_h + margin)
        if pth and pth.exists():
            try:
                im = Image.open(pth).convert("RGB").resize((w_thumb, h_thumb))
            except Exception:
                im = Image.new("RGB", (w_thumb, h_thumb), (200, 200, 200))
        else:
            im = Image.new("RGB", (w_thumb, h_thumb), (200, 200, 200))
        canvas.paste(im, (x, y))
        cap_txt = cap if len(cap) <= 40 else (cap[:37] + "…")
        draw.text((x, y + h_thumb + 2), cap_txt, fill=text_color, font=font_cap)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")

# ---------- main ----------


def make_novelty_montage_per_run(
    parent_dir: str,
    out_dir: str,
    top_n: int = 200,
    thumb_size: Tuple[int, int] = (256, 256),
    thumbs_per_row: int = 8,
):
    """
    For each run:
      - read <run>/novelty/novelty_inbox.csv
      - resolve abs paths from CSV or input_path scan
      - create PNG montage sorted by score (desc)
    """
    runs = build_run_index(parent_dir)
    if runs.empty:
        print(f"[WARN] No runs found under: {parent_dir}")
        return

    for _, r in runs.iterrows():
        run_id = r["run_id"]
        inbox_csv = Path(out_dir) / run_id / "novelty" / "novelty_inbox.csv"
        if not inbox_csv.exists():
            print(f"[WARN] run {run_id}: missing {inbox_csv}")
            continue

        try:
            df = pd.read_csv(inbox_csv)
        except Exception as e:
            print(f"[WARN] run {run_id}: cannot read {inbox_csv}: {e}")
            continue
        if df.empty:
            print(f"[WARN] run {run_id}: empty novelty_inbox.csv")
            continue

        # sort by score desc and trim
        cols_needed = ["image_name","abs_path","score","pred1_label","pred1_conf","pred2_label","pred2_conf"]
        for c in cols_needed:
            if c not in df.columns:
                df[c] = ""  # fill missing
        df = df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

        # scan input_path in case abs_path is missing
        cfg = load_run_config(r["run_cfg"])
        fmap = _index_images_recursive(Path(cfg["input_path"])) if cfg.get("input_path") else {}

        tiles: List[Tuple[str, Optional[Path]]] = []
        for i, row in df.iterrows():
            name = str(row["image_name"])
            ap = str(row.get("abs_path", "") or "")
            pth = Path(ap) if ap and Path(ap).exists() else fmap.get(name.lower())
            # caption: "#rank score=0.73 detritus (0.92-0.10)"
            try:
                c1 = float(row.get("pred1_conf", 0.0))
                c2 = float(row.get("pred2_conf", 0.0))
                lab = str(row.get("pred1_label", ""))
                cap = f"#{i+1} s={row.get('score', 0):.2f} {lab} ({c1:.2f}-{c2:.2f})"
            except Exception:
                cap = f"#{i+1} s={row.get('score', 0)} {row.get('pred1_label','')}"
            tiles.append((cap, pth))

        safe_run = _safe_slug(run_id)
        out_path = Path(out_dir) / run_id / "novelty" / f"novelty_montage_{safe_run}.png"
        title = f"{run_id} — Top-{len(tiles)} novelty"
        _make_contact_sheet(
            images=tiles,
            out_path=out_path,
            title=title,
            thumb_size=thumb_size,
            thumbs_per_row=thumbs_per_row,
        )
        print(f"[INFO] wrote novelty montage: {out_path}")
