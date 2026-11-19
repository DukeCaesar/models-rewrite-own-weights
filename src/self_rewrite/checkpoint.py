"""Lazy CPU access to Hugging Face safetensors checkpoints."""
from __future__ import annotations
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch


class Checkpoint:
    def __init__(self, root, damage=None):
        self.root = Path(root).resolve()
        index = self.root / 'model.safetensors.index.json'
        if index.exists():
            self.files = json.loads(index.read_text())['weight_map']
        else:
            self.files = {}
            for file in sorted(self.root.glob('*.safetensors')):
                with safe_open(file, framework='pt', device='cpu') as handle:
                    for key in handle.keys():
                        if key in self.files:
                            raise ValueError('Duplicate checkpoint tensor: ' + key)
                        self.files[key] = file.name
        if not self.files:
            raise FileNotFoundError('No safetensors checkpoint found')
        self.damage = Path(damage).resolve() if damage else None
        self.damage_names = set()
        if self.damage:
            with safe_open(self.damage, framework='pt', device='cpu') as handle:
                self.damage_names = set(handle.keys())
            if not self.damage_names <= self.files.keys():
                raise ValueError('Damage contains unknown tensors')

    def load(self, name):
        file = self.damage if name in self.damage_names else self.root / self.files[name]
        with safe_open(file, framework='pt', device='cpu') as handle:
            return handle.get_tensor(name)

    def shapes(self):
        result = {}
        for name, file in self.files.items():
            with safe_open(self.root / file, framework='pt', device='cpu') as handle:
                result[name] = list(handle.get_slice(name).get_shape())
        return result

    def output_matrices(self, layer):
        marker = f'.layers.{int(layer)}.'
        suffixes = ('mlp.down_proj.weight', 'self_attn.o_proj.weight', 'linear_attn.out_proj.weight')
        return [k for k in self.files if marker in '.' + k and k.endswith(suffixes)
                and 'mtp' not in k.split('.') and 'visual' not in k.split('.') and 'vision' not in k.split('.')]

    def export(self, destination, edits):
        """Write a standalone checkpoint, shard by shard, with damage then edits.

        Copies model configuration/tokenizer assets; never copies Python code.
        Large checkpoints require space for a full additional model copy.
        """
        import shutil
        destination = Path(destination).resolve()
        if destination == self.root or destination.exists() and any(destination.iterdir()):
            raise ValueError('Export requires an empty destination separate from source')
        destination.mkdir(parents=True, exist_ok=True)
        edited = set(edits.list_edits())
        total = 0
        for file in sorted(set(self.files.values())):
            tensors = {}
            for key, shard in self.files.items():
                if shard == file:
                    tensor = edits.load(key) if key in edited else self.load(key)
                    tensors[key] = tensor.contiguous()
                    total += tensor.numel() * tensor.element_size()
            target = destination / file
            target.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, target, metadata={'format': 'pt'})
        for file in self.root.iterdir():
            if file.is_file() and file.suffix in {'.json', '.txt', '.model', '.jinja'} and file.name != 'model.safetensors.index.json':
                shutil.copy2(file, destination / file.name)
        (destination / 'model.safetensors.index.json').write_text(json.dumps(
            {'metadata': {'total_size': total}, 'weight_map': self.files}, indent=2))
