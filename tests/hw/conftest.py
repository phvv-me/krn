from collections.abc import Callable
from types import SimpleNamespace

import pytest

from krn.hw.devices import nvidia
from krn.hw.host import CPU, OS, Limits, Memory, PythonRuntime, Volume
from krn.hw.system import System


@pytest.fixture
def inventory() -> System:
    return System(
        os=OS(system="Linux", release="1", distribution="Linux", distribution_version="1", architecture="x86_64"),
        cpu=CPU(name="CPU", architecture="x86_64", logical_cores=8),
        memory=Memory(total=1024, page_size=4096),
        volumes=(Volume(path="/", total=100, free=50, used_fraction=0.5),),
        python=PythonRuntime(
            version="3.14", implementation="CPython", free_threaded=False, gil_enabled=True, executable="/python"
        ),
        limits=Limits(usable_cpus=8),
    )


@pytest.fixture
def cuda_devices(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], None]:
    """Expose this many devices through the CUDA runtime, which applies CUDA_VISIBLE_DEVICES; NVML lists them all."""

    def expose(count: int) -> None:
        driver = SimpleNamespace(
            cuInit=lambda flags: (0,),
            cuDeviceGetCount=lambda: (0, count),
            CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        )
        monkeypatch.setattr(nvidia, "driver", driver)

    return expose
