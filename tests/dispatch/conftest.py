import pytest

from krn.facts.targets import Target


@pytest.fixture(scope="module")
def metal_device():
    metal = pytest.importorskip("krn.runtime.metal")
    try:
        return metal.MetalDevice(Target.of("apple_m4"))
    except metal.MetalError as error:
        pytest.skip(str(error))
