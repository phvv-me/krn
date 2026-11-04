import runpy
import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize(
    "module,dependency", [("apple", "Metal"), ("amd", "hip"), ("intel", "dpctl"), ("nvidia", "cuda")]
)
def test_optional_imports(module: str, dependency: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    monkeypatch.delitem(sys.modules, f"krn.hw.devices.{module}")
    monkeypatch.setitem(sys.modules, dependency, None)
    runpy.run_module(f"krn.hw.devices.{module}")
    fake = ModuleType(dependency)
    if dependency == "hip":
        fake.__dict__["hip"] = ModuleType("hip.hip")
        smi = ModuleType("amdsmi")
        smi.__dict__["AmdSmiException"] = RuntimeError
        monkeypatch.setitem(sys.modules, "amdsmi", smi)
    elif dependency == "cuda":
        fake.__dict__["core"] = ModuleType("cuda.core")
        monkeypatch.setitem(sys.modules, "cuda.core", ModuleType("cuda.core"))
        sys.modules["cuda.core"].__dict__["system"] = ModuleType("cuda.core.system")
        bindings = ModuleType("cuda.bindings")
        bindings.__dict__["nvml"] = ModuleType("cuda.bindings.nvml")
        monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, dependency, fake)
    runpy.run_module(f"krn.hw.devices.{module}")
