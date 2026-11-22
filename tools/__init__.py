"""工具包入口。

本机 Windows 沙箱对 `mode=0o700` 新建目录会生成当前进程无法访问的 ACL，
`tempfile.TemporaryDirectory()` 因此无法在其中创建子目录（WinError 5）。
工具在受限环境运行时改用默认权限临时目录；不改变工具逻辑。
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
