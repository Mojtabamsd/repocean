

def shorten_label(label: str, max_len: int = 20) -> str:
    if len(label) <= max_len:
        return label
    keep = (max_len - 3) // 2
    return label[:keep] + "..." + label[-keep:]
