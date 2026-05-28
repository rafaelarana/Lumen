"""Deterministic typo injection for typo-robustness evaluation.

WANDS queries are clean, so they can't show whether the FTS branch tolerates
misspellings. This module injects realistic keyboard typos at a controlled rate
with a fixed seed, so the "typo" query set is reproducible across runs (a
prerequisite for comparing before/after a ``pg_trgm`` change).

Four edit types, mimicking how people mistype:
- **substitute** a character with a physically adjacent keyboard key,
- **transpose** two adjacent characters,
- **delete** a character,
- **insert** an adjacent key next to a character.
"""
from __future__ import annotations

import random

# Adjacency on a QWERTY keyboard (lowercase). Used so substitutions/insertions
# resemble real fat-finger errors rather than random noise.
_ADJACENT = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}


def _adjacent(ch: str, rng: random.Random) -> str:
    """A keyboard-adjacent replacement for ``ch`` (falls back to itself)."""
    neighbors = _ADJACENT.get(ch.lower())
    if not neighbors:
        return ch
    repl = rng.choice(neighbors)
    return repl.upper() if ch.isupper() else repl


def _typo_word(word: str, rng: random.Random) -> str:
    """Apply one random edit to a single word (>= 4 chars to stay readable)."""
    if len(word) < 4:
        return word
    op = rng.choice(("substitute", "transpose", "delete", "insert"))
    i = rng.randrange(len(word))
    chars = list(word)
    if op == "substitute":
        chars[i] = _adjacent(chars[i], rng)
    elif op == "transpose":
        # Clamp so there is always a right neighbour to swap with.
        i = min(i, len(word) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif op == "delete":
        del chars[i]
    elif op == "insert":
        chars.insert(i, _adjacent(chars[i], rng))
    return "".join(chars)


def inject(text: str, *, rate: float, rng: random.Random) -> str:
    """Return ``text`` with each eligible word typo'd with probability ``rate``.

    At least one word is always typo'd when the query has an eligible word and
    ``rate > 0``, so short queries still exercise the fuzzy path.
    """
    words = text.split()
    eligible = [i for i, w in enumerate(words) if len(w) >= 4]
    if not eligible or rate <= 0:
        return text

    touched = False
    for i in eligible:
        if rng.random() < rate:
            words[i] = _typo_word(words[i], rng)
            touched = True
    if not touched:
        i = rng.choice(eligible)
        original = words[i]
        for _ in range(5):  # retry until the edit actually changes the word
            candidate = _typo_word(original, rng)
            if candidate != original:
                words[i] = candidate
                break
    return " ".join(words)


def typo_queries(texts: list[str], *, rate: float, seed: int) -> list[str]:
    """Inject typos into a list of queries deterministically (seeded)."""
    rng = random.Random(seed)
    return [inject(t, rate=rate, rng=rng) for t in texts]


if __name__ == "__main__":
    samples = [
        "comfy reading chair for small living room",
        "mid-century modern accent chair",
        "salon chair",
        "stainless steel kitchen faucet",
    ]
    for orig, typ in zip(samples, typo_queries(samples, rate=0.4, seed=42)):
        print(f"{orig!r:55} -> {typ!r}")
