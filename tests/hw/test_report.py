from krn.hw.host import CPU, OS, Limits, Memory, PythonRuntime
from krn.hw.report import report
from krn.hw.system import System


def test_report() -> None:
    machine = System(
        os=OS(system="Linux", release="1", distribution="Linux", distribution_version="1", architecture="x86_64"),
        cpu=CPU(name="CPU", architecture="x86_64", logical_cores=8),
        memory=Memory(total=1024, page_size=4096),
        volumes=(),
        python=PythonRuntime(
            version="3.14", implementation="CPython", free_threaded=False, gil_enabled=True, executable="/python"
        ),
        limits=Limits(usable_cpus=8),
    )
    text = report(machine)
    assert "pip install" in text and "CPU:" in text and "Memory:" in text and "python-build" in text
