"""Paired answer accounting and problem-clustered inference."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import zlib
import numpy as np
from scipy.stats import binomtest


def problem_seed(identifier: str, seed_base: int = 0) -> int:
    # The paper does not publish its hash function. This is stable across processes.
    return int.from_bytes(hashlib.sha256(f'{seed_base}:{identifier}'.encode()).digest()[:8], 'big') % 2**31


def last_boxed(text: str) -> str | None:
    start = text.rfind('\\boxed{')
    if start < 0:
        return None
    start += len('\\boxed{')
    depth = 1
    for i in range(start, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}': depth -= 1
        if depth == 0:
            return text[start:i]
    return None


def math_equal(prediction: str, reference: str) -> bool:
    # No silent exact-string fallback: missing Math-Verify is an installation error.
    from math_verify import parse, verify
    return bool(verify(parse('\\boxed{' + reference + '}'), parse('\\boxed{' + prediction + '}')))


@dataclass(frozen=True)
class Answer:
    problem_id: str
    sample: int
    seed: int
    text: str
    tokens: int
    cut: bool
    correct: bool
    looping: bool

    def to_dict(self):
        return asdict(self)


def grade(problem_id, sample, seed, text, tokens, token_limit, reference,
          *, finish_reason=None, comparator=math_equal) -> Answer:
    cut = finish_reason == 'length' if finish_reason is not None else tokens >= token_limit
    # Thinking is enabled in all paper experiments, sometimes by an open template tag.
    closed = '</think>' in text and text.rfind('</think>') > text.rfind('<think>')
    final = text.rsplit('</think>', 1)[-1] if closed else ''
    boxed = last_boxed(final)
    correct = bool(closed and boxed is not None and comparator(boxed, reference))
    tail = text[-6000:].encode('utf-8')
    looping = bool(cut and tail and len(zlib.compress(tail)) / len(tail) < .15)
    return Answer(str(problem_id), sample, seed, text, tokens, cut, correct, looping)


def _paired_arrays(base, edited, field):
    def index(rows):
        out = {}
        for r in rows:
            key = (r.problem_id, r.sample, r.seed)
            if key in out:
                raise ValueError('Duplicate answer pair')
            out[key] = r
        return out
    a, b = index(base), index(edited)
    if not a or a.keys() != b.keys():
        raise ValueError('Nonempty, exactly matched problem/sample/seed pairs required')
    ids = sorted({key[0] for key in a})
    differences = []
    for pid in ids:
        keys = [k for k in a if k[0] == pid]
        differences.append(np.mean([int(getattr(b[k], field)) - int(getattr(a[k], field)) for k in keys]))
    keys = sorted(a)
    return np.array(differences), np.array([getattr(a[k], field) for k in keys], dtype=int), np.array([getattr(b[k], field) for k in keys], dtype=int)


def paired_report(base: list[Answer], edited: list[Answer]) -> dict:
    report = {}
    for field in ('correct', 'cut'):
        d, a, b = _paired_arrays(base, edited, field)
        delta = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None
        report[field] = dict(base=float(a.mean()), edited=float(b.mean()), delta=delta,
                             se=se, ci95=None if se is None else [delta-1.96*se, delta+1.96*se])
    _, a, b = _paired_arrays(base, edited, 'correct')
    up, down = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    report.update(wrong_to_right=up, right_to_wrong=down,
                  sign_p=float(binomtest(up, up+down, .5, alternative='greater').pvalue) if up+down else 1.)
    for name, rows in [('base', base), ('edited', edited)]:
        cuts = sum(r.cut for r in rows)
        report[name] = dict(mean_tokens=float(np.mean([r.tokens for r in rows])),
                            looping_share=float(np.mean([r.looping for r in rows])),
                            looping_among_cut=sum(r.looping for r in rows)/cuts if cuts else 0.)
    return report


def permutation_test(base, edited, *, flips=100_000, seed=0) -> float:
    if flips < 1:
        raise ValueError('flips must be positive')
    d, _, _ = _paired_arrays(base, edited, 'correct')
    rng = np.random.default_rng(seed)
    observed = d.sum()
    exceed = 0
    for offset in range(0, flips, 1024):
        signs = rng.integers(0, 2, (min(1024, flips-offset), len(d))) * 2 - 1
        exceed += int(np.count_nonzero(signs @ d >= observed - 1e-12))
    # Finite Monte Carlo correction avoids reporting an impossible p=0.
    return (exceed + 1) / (flips + 1)
