"""Deny sampler filesystem reads below eval/test using Linux Landlock.

This protects held-out source images and answer-generation code during inference.
Directory enumeration remains available so Python can discover public packages;
listing a private directory does not grant access to any file contents.
It does not constrain procedures, objectives, output writes, or public resources.
"""
import ctypes
import os
from pathlib import Path

READ_FILE=1<<2
READ_DIR=1<<3

class Ruleset(ctypes.Structure):
    _fields_=[('handled_access_fs',ctypes.c_uint64)]
class Beneath(ctypes.Structure):
    _pack_=1
    _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int)]

def restrict_test_reads(own_folder=None):
    libc=ctypes.CDLL(None,use_errno=True)
    def checked(result):
        if result<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
        return result
    excluded=Path(__file__).resolve().parent
    ruleset=Ruleset(READ_FILE|READ_DIR)
    fd=checked(libc.syscall(444,ctypes.byref(ruleset),ctypes.sizeof(ruleset),0))
    def grant(path,mask):
        pfd=os.open(path,os.O_PATH|os.O_CLOEXEC)
        try:
            rule=Beneath(mask,pfd)
            checked(libc.syscall(445,fd,1,ctypes.byref(rule),0))
        finally:os.close(pfd)
    def allow(path):
        if path==excluded:return
        if path in excluded.parents:
            for child in path.iterdir():allow(child)
            return
        if path.is_symlink():return
        grant(path,READ_FILE|READ_DIR if path.is_dir() else READ_FILE)
    try:
        # Import finders enumerate /home/user before looking inside method/.
        # Child-directory file grants alone do not permit that enumeration.
        grant(Path('/'),READ_DIR)
        allow(Path('/'))
        if own_folder is not None:allow(Path(own_folder).resolve())
        checked(libc.prctl(38,1,0,0,0))
        checked(libc.syscall(446,fd,0))
    finally:os.close(fd)
