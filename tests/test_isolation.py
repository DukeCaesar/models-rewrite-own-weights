import multiprocessing as mp
import os
import socket
import subprocess
import pytest
from self_rewrite.isolation import seal_worker
from self_rewrite.wire import encode,decode


def isolation_probe(connection):
    try:
        seal_worker()
        results={}
        for name,action in [('file',lambda:open('/etc/passwd')),
                            ('network',lambda:socket.socket()),
                            ('process',lambda:subprocess.run(['/bin/true']))]:
            try: action();results[name]=False
            except PermissionError: results[name]=True
        connection.send_bytes(str(results).encode())
    except Exception as exc: connection.send_bytes(('ERROR: '+repr(exc)).encode())


def test_os_denies_file_network_process_without_python_policy():
    ctx=mp.get_context('spawn');parent,child=ctx.Pipe()
    process=ctx.Process(target=isolation_probe,args=(child,));process.start();child.close()
    try:
        assert parent.poll(20)
        result=parent.recv_bytes().decode()
        assert result=="{'file': True, 'network': True, 'process': True}",result
    finally:
        process.join(5)
        if process.is_alive(): process.terminate();process.join()
        parent.close()


def test_data_wire_preserves_bf16_and_rejects_invalid_payload():
    import torch
    source={'tensor':torch.arange(6).reshape(2,3).bfloat16(),'scalar':torch.tensor(2.),'empty':torch.empty(0)}
    result=decode(encode(source))
    for key in source: assert torch.equal(source[key],result[key])
    with pytest.raises(ValueError): decode({'$tensor':True,'dtype':'float32','shape':[100],'data':''})
    with pytest.raises(TypeError): encode(lambda:None)
