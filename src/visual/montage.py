from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from PIL import Image, ImageDraw, ImageFont

from src.index import build_run_index
from src.utils.io import load_run_config
from src.utils.paths import _safe_slug


def _index_images_recursive(root: Path, exts=(".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")) -> Dict[str, Path]:
    """
    Build a {lower_filename: full_path} map by scanning input_path recursively once.
    This lets us find medoid files by filename regardless of subfolders.
    """
    root = Path(root)
    mapping: Dict[str, Path] = {}
    if not root.exists():
        return mapping
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            mapping[p.name.lower()] = p
    return mapping


def _load_prototypes_json(proto_path: Path) -> List[Dict]:
    if not proto_path.exists():
        return []
    with open(proto_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            data = []
    if not isinstance(data, list):
        return []
    return data


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
    """
    images: list of (caption, path or None). Missing paths will be rendered as grey boxes.
    """
    if len(images) == 0:
        return

    w_thumb, h_thumb = thumb_size
    rows = (len(images) + thumbs_per_row - 1) // thumbs_per_row

    # Reserve space for title (top) and small per-thumb captions
    title_h = 40
    caption_h = 24

    sheet_w = margin + thumbs_per_row * (w_thumb + margin)
    sheet_h = title_h + margin + rows * (h_thumb + caption_h + margin)

    sheet = Image.new("RGB", (sheet_w, sheet_h), bg)
    draw = ImageDraw.Draw(sheet)

    # Try default font; Pillow will fallback if truetype not found
    try:
        font_title = ImageFont.truetype("arial.ttf", 20)
        font_cap = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_cap = ImageFont.load_default()

    # Title
    draw.text((margin, (title_h - 20) // 2), title, fill=text_color, font=font_title)

    # Paste thumbs
    y0 = title_h
    for i, (cap, pth) in enumerate(images):
        r = i // thumbs_per_row
        c = i % thumbs_per_row
        x = margin + c * (w_thumb + margin)
        y = y0 + margin + r * (h_thumb + caption_h + margin)

        if pth and pth.exists():
            try:
                im = Image.open(pth).convert("RGB")
                im = im.resize((w_thumb, h_thumb))
            except Exception:
                im = Image.new("RGB", (w_thumb, h_thumb), (200, 200, 200))
        else:
            im = Image.new("RGB", (w_thumb, h_thumb), (200, 200, 200))

        sheet.paste(im, (x, y))
        # caption (truncate nicely)
        cap_txt = cap if len(cap) <= 28 else (cap[:25] + "…")
        draw.text((x, y + h_thumb + 2), cap_txt, fill=text_color, font=font_cap)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, format="PNG")


def make_prototype_montages(
    parent_dir: str,
    out_dir: str,
    mode: str = "per_class",              # 'per_class' or 'per_run'
    thumb_size: Tuple[int, int] = (256, 256),
    thumbs_per_row: int = 8,
):
    """
    For each run:
      - reads <run>/prototypes/prototypes.json
      - gets input_path from config.yaml
      - builds a filename→path map by scanning input_path recursively
      - makes one PNG per group (class) containing the group's medoids
    """
    runs = build_run_index(parent_dir)
    if runs.empty:
        print(f"[WARN] No runs found under: {parent_dir}")
        return

    for _, r in runs.iterrows():
        run_id = r["run_id"]
        cfg = load_run_config(r["run_cfg"])
        input_path = cfg.get("input_path")
        if not input_path:
            print(f"[WARN] run {run_id}: no input_path in {r['run_cfg']}; skipping montage.")
            continue

        proto_path = Path(out_dir) / run_id / "prototypes" / "prototypes.json"
        if not proto_path.exists():
            # try inside the same out_dir passed to run_prototypes (common case)
            alt = Path(out_dir) / run_id / "prototypes.json"
            if alt.exists():
                proto_path = alt
        protos = _load_prototypes_json(proto_path)
        if not protos:
            print(f"[WARN] run {run_id}: no prototypes found at {proto_path}")
            continue

        # Scan original images once per run
        file_map = _index_images_recursive(Path(input_path))

        # Group prototypes
        groups: Dict[str, List[Dict]] = {}
        for rec in protos:
            g = str(rec.get("group", "__ALL__"))
            groups.setdefault(g, []).append(rec)

        # For each group, build montage (ordered by medoid_rank)
        for g, items in groups.items():
            items = sorted(items, key=lambda z: int(z.get("medoid_rank", 0)))
            tiles: List[Tuple[str, Optional[Path]]] = []
            for it in items:
                name = str(it.get("medoid_name", ""))
                cap = f"#{it.get('medoid_rank', 0)}  {name}"
                pth = file_map.get(name.lower())  # match by filename only
                tiles.append((cap, pth))

            safe_group = _safe_slug(g)
            out_path = Path(out_dir) / run_id / "prototypes" / f"montage_{safe_group}.png"
            title = f"{run_id} — {('Class: ' + g) if g != '__ALL__' else 'Run prototypes'}"
            _make_contact_sheet(
                images=tiles,
                out_path=out_path,
                title=title,
                thumb_size=thumb_size,
                thumbs_per_row=thumbs_per_row,
            )
            print(f"[INFO] wrote montage: {out_path}")
