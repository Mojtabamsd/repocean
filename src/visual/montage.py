from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json, re, unicodedata, os

from PIL import Image, ImageDraw, ImageFont

from src.index import build_run_index
from src.utils.io import load_run_config
from src.utils.paths import _safe_slug


def _norm_rel_key(p: Path, root: Path) -> str:
    """lowercased relative path with forward slashes."""
    try:
        rel = p.relative_to(root)
    except Exception:
        rel = p
    return str(rel).replace("\\", "/").lower()


def _index_images_recursive(root: Path, exts=(".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff")) -> dict:
    """
    Build a rich index:
      - 'by_rel': normalized relative path -> Path (forward slashes, lower)
      - 'by_name': basename (lower) -> Path (first seen)
      - 'by_stem': stem (lower) -> [Path, ...]  (to tolerate extension mismatch)
    """
    root = Path(root)
    index = {"by_rel": {}, "by_name": {}, "by_stem": {}}
    if not root.exists():
        return index

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue

        rel_key = _norm_rel_key(p, root)               # e.g. "export__.../11/file.bmp"
        name_key = p.name.lower()                      # e.g. "file.bmp"
        stem_key = p.stem.lower()                      # e.g. "file"

        index["by_rel"][rel_key] = p
        index["by_rel"][rel_key.replace("/", "\\")] = p  # allow backslash style lookups too

        # keep first seen for name collisions (common on UVP exports)
        index["by_name"].setdefault(name_key, p)

        index["by_stem"].setdefault(stem_key, []).append(p)

    return index


def _load_prototypes_json(proto_path: Path) -> List[Dict]:
    if not proto_path.exists():
        return []
    with open(proto_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            data = []
    return data if isinstance(data, list) else []


def _resolve_image_path(
    medoid_name: str,
    input_root: Path,
    index: dict,
) -> Optional[Path]:
    """
    Resolve a medoid image name that might include nested folders and/or a different extension.
    Resolution order:
      1) exact normalized relative path match
      2) basename match
      3) stem match (best effort; try to disambiguate with path fragments)
    """
    if not medoid_name:
        return None

    # normalize incoming key
    key_rel = medoid_name.replace("\\", "/").lstrip("./").lower()
    base = os.path.basename(key_rel).lower()
    stem = os.path.splitext(base)[0].lower()

    # 1) exact rel-path match
    p = index["by_rel"].get(key_rel)
    if p:
        return p

    # also try the windows-style variant if input had forward slashes
    p = index["by_rel"].get(key_rel.replace("/", "\\"))
    if p:
        return p

    # 2) basename match
    p = index["by_name"].get(base)
    if p:
        return p

    # 3) stem match (tolerate extension mismatch)
    candidates = index["by_stem"].get(stem, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # disambiguate using any directory fragments present in medoid_name
    parts = [q for q in key_rel.split("/") if q and "." not in q]  # folder tokens
    if parts:
        # score candidates by how many fragments appear in their rel path
        best = None
        best_score = -1
        for c in candidates:
            c_rel = _norm_rel_key(c, input_root)
            score = sum(1 for t in parts if t in c_rel)
            if score > best_score:
                best, best_score = c, score
        if best is not None:
            return best

    # fallback: first candidate
    return candidates[0]


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
    if len(images) == 0:
        return

    w_thumb, h_thumb = thumb_size
    rows = (len(images) + thumbs_per_row - 1) // thumbs_per_row
    title_h = 40
    caption_h = 24

    sheet_w = margin + thumbs_per_row * (w_thumb + margin)
    sheet_h = title_h + margin + rows * (h_thumb + caption_h + margin)

    sheet = Image.new("RGB", (sheet_w, sheet_h), bg)
    draw = ImageDraw.Draw(sheet)

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
                im = Image.open(pth).convert("RGB")
                im = im.resize((w_thumb, h_thumb))
            except Exception:
                im = Image.new("RGB", (w_thumb, h_thumb), (200, 200, 200))
        else:
            im = Image.new("RGB", (w_thumb, h_thumb), (200, 200, 200))

        sheet.paste(im, (x, y))
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
        input_root = Path(input_path)

        proto_path = Path(out_dir) / run_id / "prototypes" / "prototypes.json"
        if not proto_path.exists():
            alt = Path(out_dir) / run_id / "prototypes.json"
            if alt.exists():
                proto_path = alt
        protos = _load_prototypes_json(proto_path)
        if not protos:
            print(f"[WARN] run {run_id}: no prototypes found at {proto_path}")
            continue

        # Index original images once per run (rich index)
        idx = _index_images_recursive(input_root)

        # Group prototypes
        groups: Dict[str, List[Dict]] = {}
        for rec in protos:
            g = str(rec.get("group", "__ALL__"))
            groups.setdefault(g, []).append(rec)

        for g, items in groups.items():
            items = sorted(items, key=lambda z: int(z.get("medoid_rank", 0)))
            tiles: List[Tuple[str, Optional[Path]]] = []
            for it in items:
                name = str(it.get("medoid_name", ""))  # may include nested path, different ext
                cap = f"#{it.get('medoid_rank', 0)}  {os.path.basename(name)}"
                pth = _resolve_image_path(name, input_root, idx)
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
