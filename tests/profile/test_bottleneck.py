"""`BottleneckReport` ranks kernels and regions by work and counts overlapping work once."""

from krn.profile import BottleneckReport, KernelTrace, MemcpyTrace, RegionWindow


def test_a_report_ranks_by_work_while_the_busy_time_counts_overlap_once() -> None:
    kernels = [
        KernelTrace(name="a", start_ns=0, end_ns=10),
        KernelTrace(name="b", start_ns=5, end_ns=25),
        KernelTrace(name="a", start_ns=30, end_ns=36),
    ]
    copies = [MemcpyTrace(start_ns=8, end_ns=12, bytes_moved=64)]
    windows = [RegionWindow(name="step", start_ns=0, end_ns=26, wall_ns=40)]

    report = BottleneckReport.from_traces(windows, kernels, copies, top=1)

    assert (
        (report.device_busy_ns, report.total_kernel_ns, report.total_memcpy_ns),
        [(kernel.name, kernel.calls, kernel.total_ns) for kernel in report.hot_kernels],
        [(region.name, region.kernel_count, region.wall_ns) for region in report.hot_regions],
    ) == ((31, 36, 4), [("b", 1, 20)], [("step", 2, 40)])
