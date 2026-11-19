"""Data-only worker IPC: never unpickle messages received from editor code."""
import base64
import json
import math


def encode(value):
    import torch
    import numpy as np
    if isinstance(value,torch.Tensor):
        value=value.detach().cpu().contiguous()
        raw=value.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {'$tensor':True,'dtype':str(value.dtype).split('.')[-1],
                'shape':list(value.shape),'data':base64.b64encode(raw).decode('ascii')}
    if isinstance(value,np.ndarray): return encode(torch.from_numpy(value.copy()))
    if isinstance(value,(list,tuple)): return [encode(x) for x in value]
    if isinstance(value,dict):
        if not all(isinstance(k,str) for k in value): raise ValueError('Wire keys must be strings')
        return {k:encode(v) for k,v in value.items()}
    if value is None or type(value) in (str,int,float,bool): return value
    raise TypeError('Unsupported wire value: '+type(value).__name__)


def decode(value):
    import torch
    if isinstance(value,dict) and value.get('$tensor') is True:
        types={name:getattr(torch,name) for name in ('float16','float32','float64','bfloat16','int8','uint8','int16','int32','int64','bool')}
        if value.get('dtype') not in types: raise ValueError('Invalid wire dtype')
        shape=value.get('shape')
        if not isinstance(shape,list) or len(shape)>32 or any(type(n) is not int or n<0 for n in shape):
            raise ValueError('Invalid wire shape')
        dtype=types[value['dtype']]
        raw=bytearray(base64.b64decode(value['data'],validate=True))
        if len(raw)!=math.prod(shape)*torch.empty((),dtype=dtype).element_size(): raise ValueError('Wire tensor size mismatch')
        if not raw: return torch.empty(shape,dtype=dtype)
        return torch.frombuffer(raw,dtype=dtype).reshape(shape).clone()
    if isinstance(value,dict): return {k:decode(v) for k,v in value.items()}
    if isinstance(value,list): return [decode(v) for v in value]
    return value


class DataConnection:
    def __init__(self,connection): self.connection=connection
    def send(self,value): self.connection.send_bytes(json.dumps(encode(value)).encode())
    def recv(self): return decode(json.loads(self.connection.recv_bytes()))
    def poll(self,timeout=0): return self.connection.poll(timeout)
    def close(self): return self.connection.close()
