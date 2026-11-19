"""Linux seccomp isolation installed after trusted numerical-library bootstrap.

The worker receives no checkpoint paths. Once sealed it cannot open files, create
processes/threads, access the network, mount filesystems, or attach to processes.
All useful I/O uses an existing pipe and host-validated capabilities. Failure to
install the filter is fatal; there is no silent fallback for sealed sessions.
"""
import ctypes
import ctypes.util
import errno
import sys


def seal_worker():
    if sys.platform!='linux': raise RuntimeError('Sealed workers require Linux/libseccomp')
    library=ctypes.util.find_library('seccomp')
    if not library: raise RuntimeError('libseccomp is unavailable')
    sec=ctypes.CDLL(library,use_errno=True)
    sec.seccomp_init.argtypes=[ctypes.c_uint32];sec.seccomp_init.restype=ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];sec.seccomp_syscall_resolve_name.restype=ctypes.c_int
    sec.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    sec.seccomp_load.argtypes=[ctypes.c_void_p]
    sec.seccomp_release.argtypes=[ctypes.c_void_p]
    context=sec.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not context: raise RuntimeError('seccomp initialization failed')
    denied='''open openat openat2 creat open_by_handle_at name_to_handle_at
    socket socketpair connect bind listen accept accept4 sendto sendmsg sendmmsg
    recvfrom recvmsg recvmmsg shutdown
    fork vfork clone clone3 execve execveat ptrace process_vm_readv process_vm_writev
    pidfd_open pidfd_getfd pidfd_send_signal kill tkill tgkill
    mount umount2 pivot_root chroot unshare setns bpf io_uring_setup
    init_module finit_module delete_module kexec_load reboot swapon swapoff
    link linkat symlink symlinkat unlink unlinkat rename renameat renameat2
    mkdir mkdirat rmdir mknod mknodat chmod fchmod fchmodat chown fchown lchown fchownat
    truncate ftruncate ioctl perf_event_open userfaultfd'''.split()
    try:
        for name in denied:
            number=sec.seccomp_syscall_resolve_name(name.encode())
            if number>=0 and sec.seccomp_rule_add(context,0x00050000|errno.EPERM,number,0)!=0:
                raise RuntimeError('Could not prohibit syscall '+name)
        if sec.seccomp_load(context)!=0: raise RuntimeError('Could not install seccomp filter')
    finally: sec.seccomp_release(context)
