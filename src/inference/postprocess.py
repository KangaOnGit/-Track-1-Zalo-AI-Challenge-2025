def safe_clip_box(x1, y1, x2, y2, W, H):
    xi1 = max(0, min(int(round(x1)), W - 1))
    yi1 = max(0, min(int(round(y1)), H - 1))
    xi2 = max(0, min(int(round(x2)), W - 1))
    yi2 = max(0, min(int(round(y2)), H - 1))
    # ensure proper order
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return xi1, yi1, xi2, yi2