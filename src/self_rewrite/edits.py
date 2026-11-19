"""CPU tensor operations used by the editing interface (no gradients)."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch


def unit(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    n = x.norm()
    if not torch.isfinite(n) or n <= 0:
        raise ValueError("Direction must be finite and nonzero")
    return x / n


def mean_difference(finished: torch.Tensor, unfinished: torch.Tensor) -> torch.Tensor:
    if finished.ndim != 2 or unfinished.ndim != 2 or not len(finished) or not len(unfinished):
        raise ValueError("Expected two nonempty [examples, hidden] matrices")
    return unit(unfinished.float().mean(0) - finished.float().mean(0))


@torch.no_grad()
def remove_direction(w: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    u = unit(direction)
    w = w.detach().float().cpu()
    if w.ndim != 2 or u.ndim != 1 or w.shape[0] != u.numel():
        raise ValueError("Direction must match the output dimension")
    return w - float(alpha) * torch.outer(u, u @ w)


@torch.no_grad()
def damage_rows(w: torch.Tensor, start: int, *, count: int = 8,
                strength: float = 200, seed: int = 0) -> torch.Tensor:
    if w.ndim != 2 or start < 0 or count < 1 or start + count > w.shape[0]:
        raise ValueError("Invalid damage block")
    if strength <= 0:
        raise ValueError("Damage strength must be positive")
    out = w.detach().float().cpu().clone()
    noise = torch.randn((count, w.shape[1]), generator=torch.Generator().manual_seed(seed))
    out[start:start+count] = noise * (strength * out.std())
    return out.to(torch.bfloat16)


@torch.no_grad()
def clip_outliers(w: torch.Tensor, multiplier: float = 15) -> torch.Tensor:
    if multiplier <= 0:
        raise ValueError("Multiplier must be positive")
    x = w.detach().float().cpu()
    threshold = multiplier * x.abs().median()
    return x.clamp(-threshold, threshold)


def protected(name: str) -> bool:
    # Layer-local input/post-attention norms remain editable.
    parts = name.split('.')
    return ('embed_tokens' in parts or 'lm_head' in parts or
            (len(parts) >= 2 and parts[-2] == 'norm' and 'layers' not in parts))


def is_bf16_rescaling(base, replacement):
    """Does any scalar map the baseline to these bf16 rounding bins?

    Least-squares reconstruction is insufficient: rounding can move its fitted
    scalar across a bin boundary even when the edit was an exact global scale.
    Intersect per-element scale intervals, then verify the intersection's midpoint.
    Chunking avoids full-size float64 copies of large output matrices.
    """
    b=base.detach().cpu().reshape(-1)
    y=replacement.detach().cpu().to(torch.bfloat16).reshape(-1)
    lower,upper=-float('inf'),float('inf')
    any_nonzero=False
    for start in range(0,b.numel(),1_000_000):
        source=b[start:start+1_000_000].double()
        target=y[start:start+1_000_000]
        nonzero=source!=0
        if torch.any((~nonzero)&(target!=0)): return False
        if not torch.any(nonzero): continue
        any_nonzero=True
        source=source[nonzero];target=target[nonzero]
        value=target.double()
        low=(torch.nextafter(target,torch.full_like(target,-float('inf'))).double()+value)/2
        high=(torch.nextafter(target,torch.full_like(target,float('inf'))).double()+value)/2
        a=low/source;c=high/source
        lower=max(lower,float(torch.minimum(a,c).max()))
        upper=min(upper,float(torch.maximum(a,c).min()))
        if lower>upper: return False
    if not any_nonzero: return True
    scale=(lower+upper)/2
    import math
    if not math.isfinite(scale):
        scale=lower if lower!=-float('inf') else upper
    for start in range(0,b.numel(),1_000_000):
        expected=(b[start:start+1_000_000].double()*scale).to(torch.bfloat16)
        if not torch.equal(expected,y[start:start+1_000_000]): return False
    return True


class EditStore:
    """Read-only baseline callback plus validated, atomic bf16 replacements.

    In repair sessions the callback MUST resolve the damaged baseline, never clean
    weights. Baseline tensors are cloned to prevent accidental in-place changes.
    """
    def __init__(self, root: str | Path, names, load_original):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.names = tuple(names)
        self._loader = load_original

    def _path(self, name):
        if name not in self.names:
            raise KeyError(name)
        return self.root / (hashlib.sha256(name.encode()).hexdigest() + '.pt')

    def load_original(self, name):
        self._path(name)
        return self._loader(name).detach().cpu().clone()

    def load(self, name):
        path = self._path(name)
        return torch.load(path, map_location='cpu', weights_only=True) if path.exists() else self.load_original(name)

    @torch.no_grad()
    def save_edit(self, name, tensor):
        path = self._path(name)
        if protected(name):
            raise ValueError('Protected tensor: ' + name)
        base = self.load_original(name).float()
        x = tensor.detach().cpu().float()
        if x.shape != base.shape or not torch.isfinite(x).all():
            raise ValueError('Edit must preserve shape and contain finite entries')
        # Validate the persisted representation too (bf16 can overflow).
        x = x.to(torch.bfloat16)
        if not torch.isfinite(x).all():
            raise ValueError('Edit overflows bf16')
        scalar = is_bf16_rescaling(base,x)
        if scalar:
            raise ValueError('Whole-tensor scalar rescaling (including identity) is forbidden')
        temp = path.with_suffix('.tmp')
        torch.save(x, temp)
        temp.replace(path)

    def drop_edit(self, name):
        self._path(name).unlink(missing_ok=True)

    def list_edits(self):
        return [name for name in self.names if self._path(name).exists()]

    def clear(self):
        for name in self.list_edits():
            self.drop_edit(name)

    def _candidate_path(self, label):
        import re
        if not isinstance(label,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',label):
            raise ValueError('Candidate labels must use letters, digits, hyphens or underscores')
        return self.root.parent/'candidates'/label

    def save_candidate(self,label):
        """Immutable snapshot. Reusing a label is forbidden."""
        import shutil
        target=self._candidate_path(label)
        if target.exists(): raise ValueError('Candidate already exists')
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name('.'+label+'.tmp')
        if temporary.exists(): shutil.rmtree(temporary)
        temporary.mkdir()
        for name in self.list_edits(): shutil.copy2(self._path(name),temporary/self._path(name).name)
        temporary.rename(target)
        return label

    def candidate(self,label):
        path=self._candidate_path(label)
        if not path.is_dir() or path.is_symlink(): raise ValueError('Unknown candidate')
        return EditStore(path,self.names,self._loader)

    def list_candidates(self):
        parent=self.root.parent/'candidates'
        return sorted(p.name for p in parent.iterdir() if p.is_dir() and not p.name.startswith('.')) if parent.exists() else []

    def restore_candidate(self,label):
        import shutil
        source=self.candidate(label)
        self.clear()
        for name in source.list_edits(): shutil.copy2(source._path(name),self._path(name))
        return self.list_edits()
