"""Persistent CPU Python worker with an explicit numerical capability interface.

Code has no filesystem, model, process, or network API. Host capabilities mediate
all weight and artifact access. Linux workers additionally install a seccomp filter by default. The filter
prohibits filesystem opens, new processes and network access after bootstrap.
Messages from the worker use data-only serialization, not Python pickle.
"""
from __future__ import annotations
import ast
import contextlib
import io
import multiprocessing as mp
import time
from .wire import DataConnection
from pathlib import Path


class Refused(ValueError):
    pass


FORBIDDEN = {'backward','autograd','optim','requires_grad','requires_grad_',
             'enable_grad','set_grad_enabled','cuda','mps','xpu','compile',
             'eval','exec','getattr','setattr','delattr','globals','locals','vars',
             'open','input','breakpoint','memoryview','type','object','super',
             'load','save','from_file','frombuffer','storage','untyped_storage',
             'data_ptr','set_','register_hook','register_forward_hook'}


def check_code(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
            if any(m not in {'torch','numpy','math','json'} for m in modules):
                raise Refused('Only numerical imports are available')
        if isinstance(node,ast.Attribute) and (node.attr.startswith('_') or node.attr in FORBIDDEN):
            raise Refused('Unavailable attribute: '+node.attr)
        if isinstance(node,ast.Name) and (node.id.startswith('_') or node.id in FORBIDDEN):
            raise Refused('Unavailable name: '+node.id)
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and any(x in node.value.lower() for x in ('cuda','/dev/','__')):
            raise Refused('Unavailable device or introspection')
    return compile(tree,'<editor>','exec')


class LimitedOutput(io.StringIO):
    def __init__(self,limit):
        super().__init__(); self.limit=limit
    def write(self,s):
        remaining=max(0,self.limit-self.tell())
        super().write(s[:remaining])
        return len(s)


def _main(connection, metadata, output_limit, sealed):
    connection=DataConnection(connection)
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import builtins
    import types
    import torch
    import numpy as np
    import math
    import json
    from .edits import unit, mean_difference, remove_direction
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch_api = types.SimpleNamespace(**{name:getattr(torch,name) for name in (
        'tensor','as_tensor','zeros','ones','empty','zeros_like','ones_like','full','full_like',
        'randn','randn_like','rand','arange','linspace','eye','cat','stack','split','chunk',
        'mean','median','std','sum','norm','abs','sqrt','exp','log','clamp','where','isfinite',
        'isnan','isinf','outer','matmul','einsum','diag','sort','argsort','topk','quantile',
        'maximum','minimum','allclose','equal','float32','float64','bfloat16','float16',
        'int64','int32','bool','manual_seed')})
    torch_api.linalg=types.SimpleNamespace(**{n:getattr(torch.linalg,n) for n in ('norm','svd','eigh','qr','vector_norm')})
    numpy_api = types.SimpleNamespace(**{n:getattr(np,n) for n in (
        'array','asarray','zeros','ones','arange','linspace','mean','median','std','sum',
        'abs','sqrt','exp','log','clip','where','isfinite','stack','concatenate','argsort',
        'float32','float64','int64')})
    modules={'torch':torch_api,'numpy':numpy_api,'math':math,'json':types.SimpleNamespace(loads=json.loads,dumps=json.dumps)}
    def importer(name,*args,**kwargs):
        if name not in modules: raise Refused('Import refused')
        return modules[name]
    def rpc(operation,*args):
        connection.send(('rpc',operation,args))
        ok,value=connection.recv()
        if not ok: raise RuntimeError(value)
        return value
    safe_builtins={n:getattr(builtins,n) for n in (
        'print','range','len','enumerate','zip','list','dict','tuple','set','sorted',
        'reversed','min','max','sum','abs','round','int','float','str','bool','all','any',
        'isinstance','Exception','ValueError','RuntimeError')}
    safe_builtins['__import__']=importer
    namespace={'__builtins__':safe_builtins,'torch':torch_api,'np':numpy_api,
               'math':math,'json':modules['json'],'unit':unit,'mean_difference':mean_difference,
               'remove_direction':remove_direction,**metadata}
    namespace['tokenizer']=types.SimpleNamespace(encode=lambda text:rpc('tokenize',text),
                                                decode=lambda ids:rpc('decode_tokens',ids))
    for name in ('load_original','load_current','save_edit','drop_edit','list_edits',
                 'output_matrices','load_artifact','read_notes','write_notes',
                 'save_candidate','restore_candidate','list_candidates'):
        namespace[name]=(lambda operation: lambda *args: rpc(operation,*args))(name)
    if sealed:
        try:
            from .isolation import seal_worker
            # Force common lazy numerical imports before denying filesystem access.
            torch.tensor([1.]).numpy(); torch.linalg.svd(torch.eye(2)); torch.manual_seed(0)
            import base64
            seal_worker()
        except Exception as exc:
            connection.send(('startup_error',str(exc))); return
    connection.send(('ready',{'sealed':sealed}))
    while True:
        code=connection.recv()
        if code is None: break
        output=LimitedOutput(output_limit)
        try:
            compiled=check_code(code)
            with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
                exec(compiled,namespace,namespace)
            result={'ok':True,'output':output.getvalue()}
        except BaseException as exc:
            result={'ok':False,'output':output.getvalue(),'error':f'{type(exc).__name__}: {exc}'[:output_limit]}
        connection.send(('result',result))


class CodeWorker:
    def __init__(self, checkpoint, edits, practice, work, *, timeout=600, output_limit=10000, sealed=True, tokenizer=None):
        self.checkpoint,self.edits=checkpoint,edits
        self.tokenizer=tokenizer
        self.work=Path(work).resolve()
        self.work.mkdir(parents=True,exist_ok=True)
        self.timeout,self.output_limit=timeout,output_limit
        self.metadata={'tensor_shapes':checkpoint.shapes(),'practice':[p.__dict__ for p in practice]}
        self.sealed=sealed
        self.process=None
        self._start()

    def _start(self):
        context=mp.get_context('spawn')
        self.connection,child=context.Pipe()
        self.process=context.Process(target=_main,args=(child,self.metadata,self.output_limit,self.sealed),daemon=True)
        self.process.start(); child.close()
        self.connection=DataConnection(self.connection)
        if not self.connection.poll(30):
            self.close(); raise RuntimeError('Worker bootstrap timed out')
        message=self.connection.recv()
        if message[0]!='ready':
            self.close(); raise RuntimeError('Worker isolation failed: '+str(message))

    def _rpc(self,operation,args):
        if operation in {'tokenize','decode_tokens'}:
            if self.tokenizer is None: raise ValueError('Tokenizer not configured')
            if operation=='tokenize':
                if len(args)!=1 or not isinstance(args[0],str): raise ValueError('Expected text')
                return self.tokenizer.encode(args[0],add_special_tokens=False)
            if len(args)!=1 or not isinstance(args[0],list) or any(type(i) is not int or i<0 for i in args[0]):
                raise ValueError('Expected nonnegative token IDs')
            return self.tokenizer.decode(args[0],skip_special_tokens=False)
        if operation=='load_original': return self.edits.load_original(*args)
        if operation=='load_current': return self.edits.load(*args)
        if operation=='save_edit': return self.edits.save_edit(*args)
        if operation=='drop_edit': return self.edits.drop_edit(*args)
        if operation=='list_edits': return self.edits.list_edits()
        if operation=='save_candidate': return self.edits.save_candidate(*args)
        if operation=='restore_candidate': return self.edits.restore_candidate(*args)
        if operation=='list_candidates': return self.edits.list_candidates()
        if operation=='output_matrices': return self.checkpoint.output_matrices(*args)
        if operation=='read_notes': return (self.work/'notes.md').read_text() if (self.work/'notes.md').exists() else ''
        if operation=='write_notes':
            if len(args)!=1 or not isinstance(args[0],str): raise ValueError('Expected notes text')
            (self.work/'notes.md').write_text(args[0]); return None
        if operation=='load_artifact':
            import re,json,torch
            name=args[0]
            if not isinstance(name,str) or not re.fullmatch(r'(census\.jsonl|gen_\d+\.jsonl|read_\d+\.pt)',name):
                raise ValueError('Only census and forward artifacts are readable')
            path=self.work/name
            if path.is_symlink(): raise ValueError('Symlink refused')
            if name.endswith('.pt'): return torch.load(path,map_location='cpu',weights_only=True)
            return [json.loads(line) for line in path.read_text().splitlines()]
        raise ValueError('Unknown capability')

    def run(self,code):
        # Validate before sending, including syntax errors; failed code still counts.
        try: check_code(code)
        except (Refused,SyntaxError) as exc: return {'ok':False,'error':str(exc)}
        if not self.process.is_alive(): self.close(); self._start()
        self.connection.send(code)
        deadline=time.monotonic()+self.timeout
        while time.monotonic()<deadline:
            if not self.connection.poll(min(.1,max(0,deadline-time.monotonic()))):
                if not self.process.is_alive():
                    return {'ok':False,'error':'Worker exited; variables lost; files retained'}
                continue
            message=self.connection.recv()
            if message[0]=='result': return message[1]
            if message[0]=='rpc':
                try: self.connection.send((True,self._rpc(message[1],message[2])))
                except Exception as exc: self.connection.send((False,f'{type(exc).__name__}: {exc}'))
        self.close()
        return {'ok':False,'error':'Code timed out; variables lost; files retained'}

    def close(self):
        if self.process is not None:
            if self.process.is_alive(): self.process.terminate()
            self.process.join(timeout=5)
            self.connection.close()
