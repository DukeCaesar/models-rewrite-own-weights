"""Explicit backend choice; never silently substitute a serving engine."""
from .backend import TransformersBackend


def load_backend(checkpoint,edits,config,*,backend='transformers',device='cuda'):
    if backend=='transformers': return TransformersBackend.from_checkpoint(checkpoint,edits,config,device=device)
    if backend=='vllm':
        from .vllm_backend import VLLMBackend
        return VLLMBackend.from_checkpoint(checkpoint,edits,config,device=device)
    raise ValueError('Unknown serving backend')
