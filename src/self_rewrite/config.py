"""Explicit experiment presets; session turn limits are not shown to the editor."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class SessionConfig:
    size: str = '27B'
    protocol: int = 1
    repair: bool = False
    max_tokens: int = 16384
    diagnosis_count: int = 256
    hours: float = 3.5
    clock_includes_diagnosis: bool = False
    passes: int = 20
    code_runs: int = 200
    turns: int = 160
    editor_tokens: int = 12000
    context_reset_tokens: int = 90000
    code_timeout: int = 600
    output_chars: int = 10000
    max_read_texts: int = 256
    max_read_tokens: int = 8192
    max_generation_answers: int = 512
    reserved_count: int = 0
    check_count: int = 0
    check_samples: int = 2
    temperature: float = .6
    top_k: int = 20
    top_p: float = .95
    turn_seed: int = 0

    def __post_init__(self):
        if self.size not in {'2B','4B','9B','27B'} or self.protocol not in {1,2}:
            raise ValueError('Unsupported size or protocol')
        if self.max_tokens < 16 or self.passes < 1 or self.hours <= 0:
            raise ValueError('Invalid session limits')

    def to_dict(self):
        return asdict(self)


def preset(size='27B', *, repair=False, short=False, protocol=1, turn_seed=0):
    values = dict(size=size, repair=repair, protocol=protocol, turn_seed=turn_seed)
    if repair:
        if protocol != 1:
            raise ValueError('Paper does not report protocol 2 repair sessions')
        values.update(diagnosis_count=64, hours=2.5)
    elif protocol == 2:
        if size != '27B':
            raise ValueError('Paper reports protocol 2 only at 27B')
        values.update(max_tokens=4096, hours=3., max_generation_answers=128,
                      reserved_count=300, check_count=256)
    elif short:
        if size != '27B':
            raise ValueError('Short first-protocol session is reported only at 27B')
        values.update(max_tokens=4096, hours=2.5)
    else:
        values.update(clock_includes_diagnosis=size in {'2B','27B'})
    return SessionConfig(**values)
