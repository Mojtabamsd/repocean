MAX_LABEL_LEN = 10


def _shorten_for_vis(s: str) -> str:
    """
    Shorten any string for plotting / table display only.

    - First run through `shorten_label` (your central logic)
    - Then hard cap to MAX_LABEL_LEN with an ellipsis if still too long.
    """
    if s is None:
        return ""
    s2 = shorten_label(str(s))
    if len(s2) <= MAX_LABEL_LEN:
        return s2
    return s2[: MAX_LABEL_LEN - 1] + "…"


def shorten_label(label: str, max_len: int = 20) -> str:
    if len(label) <= max_len:
        return label
    keep = (max_len - 3) // 2
    return label[:keep] + "..." + label[-keep:]