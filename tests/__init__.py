"""测试包入口。

本机 Windows 沙箱对 `mode=0o700` 新建目录会生成当前进程无法访问的 ACL，
`tempfile.TemporaryDirectory()` 因此无法在其中创建子目录（WinError 5）。
这里在导入测试前把 `tempfile.mkdtemp` 换成默认权限的等价实现，使测试可在受限环境运行；
仅影响本机测试进程，不改变被测代码。
"""
import os
import tempfile
import uuid


if os.name == 'nt':
    def _mkdtemp_default_acl(suffix=None, prefix=None, dir=None):
        base = dir or tempfile.gettempdir()
        path = os.path.join(base, (prefix or 'tmp') + uuid.uuid4().hex + (suffix or ''))
        os.makedirs(path)
        return path

    tempfile.mkdtemp = _mkdtemp_default_acl
