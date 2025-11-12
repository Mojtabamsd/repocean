import re
import unicodedata

_WINDOWS_FORBIDDEN = set('<>:"/\\|?*')
_WINDOWS_RESERVED = {
    "CON","PRN","AUX","NUL",
    *(f"COM{i}" for i in range(1,10)),
    *(f"LPT{i}" for i in range(1,10)),
}


def _safe_slug(name: str, max_len: int = 120) -> str:
    """
    Make a Windows-safe filename fragment:
    - strip/normalize unicode
    - replace forbidden chars with '_'
    - collapse whitespace/underscore runs
    - avoid reserved device names
    """
    if name is None:
        return "unnamed"
    # normalize unicode
    s = unicodedata.normalize("NFKC", str(name))
    s = "".join(("_" if ch in _WINDOWS_FORBIDDEN else ch) for ch in s)
    s = re.sub(r"[\x00-\x1f]", "_", s)
    s = re.sub(r"[ \t]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("._ ")
    if not s:
        s = "unnamed"
    if s.upper() in _WINDOWS_RESERVED:
        s = f"_{s}"
    if len(s) > max_len:
        s = s[:max_len].rstrip("._ ")
    return s




