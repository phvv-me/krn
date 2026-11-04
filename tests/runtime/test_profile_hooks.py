"""Each profiling hook reads the active session once, so a session cleared mid-launch cannot fail the launch."""

from unittest.mock import MagicMock, patch

import pytest

from krn import Blocks, T
from krn.compiler.api import CPUDevice
from krn.runtime import runtime
from tests.profile.test_launches import twice


class CountingHook:
    """Stands in for `profile_launches`, counting every read of its `session`."""

    def __init__(self, session: MagicMock | None) -> None:
        self.current, self.reads = session, 0

    @property
    def session(self) -> MagicMock | None:
        self.reads += 1
        return self.current


def test_cpu_enqueue_and_drain_read_the_hook_once_each(device: CPUDevice, monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = device.empty(T.f32, 32), device.empty(T.f32, 32)
    call = device.launcher(twice, Blocks(2, threads=16)).bind(x, y)._call
    session = MagicMock()
    hook = CountingHook(session)
    stream = device.stream()
    monkeypatch.setattr(runtime, "profile_launches", hook)
    with patch.object(stream.pool, "submit"):
        stream.enqueue(call)
        assert hook.reads == 1
        stream._drain()
    assert hook.reads == 2
    session.cpu_enqueue.assert_called_once_with((call,), stream)
    session.cpu_run.assert_called_once_with(call, stream)


def test_metal_enqueue_reads_the_hook_once(monkeypatch: pytest.MonkeyPatch) -> None:
    metal = pytest.importorskip("krn.runtime.metal", reason="pyobjc Metal is not installed")
    session = MagicMock()
    hook = CountingHook(session)
    monkeypatch.setattr(metal, "profile_launches", hook)
    call = metal.MetalCall(program=MagicMock(), geometry=(1, 1, 1, 32), buffers=(), scalars=b"")
    stream = MagicMock()
    call.enqueue(stream)
    assert hook.reads == 1
    session.metal_enqueue.assert_called_once_with(call, stream, stream.queue.commandBuffer())


def test_cuda_enqueue_reads_the_hook_once(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = pytest.importorskip("krn.runtime.cuda_backend", reason="cuda-core and cuda-bindings are not installed")
    session = MagicMock()
    hook = CountingHook(session)
    monkeypatch.setattr(backend, "profile_launches", hook)
    call = backend.CUDACall(MagicMock(), (1, 1, 1, 32), (), ())
    stream = MagicMock()
    call.enqueue(stream)
    assert hook.reads == 1
    session.cuda_run.assert_called_once_with(call, stream)
