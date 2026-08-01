from __future__ import annotations


def hook_strength(index: int, duration: float, motion: float, rms: float,
                  has_face: bool, rms_max: float) -> str:
    """Heuristik kekuatan hook per scene dari sinyal (bukan ML).
    index 0 = pembuka (paling penting utk hook)."""
    score = 0.0
    if index == 0:
        score += 2.5
    elif index <= 2:
        score += 1.0

    # motion tinggi menarik perhatian
    score += min(motion / 3.0, 2.0)

    # energi audio relatif (loud opening / drop)
    if rms_max > 0:
        score += min((rms / rms_max) * 2.0, 2.0)

    # wajah bicara = hook talking-head
    if has_face:
        score += 1.0

    # scene pendek & padat cenderung nge-hook
    if duration <= 2.5:
        score += 1.0
    elif duration >= 12:
        score -= 1.0

    if score >= 4.5:
        return "strong_hook"
    if score >= 2.5:
        return "medium_hook"
    return "weak_hook"
