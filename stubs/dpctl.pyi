class SyclDevice:
    has_aspect_fp64: bool
    sub_group_sizes: list[int]
    name: str
    vendor: str
    global_mem_size: int
    local_mem_size: int
    max_compute_units: int
    max_work_group_size: int
    max_clock_frequency: int

def get_devices(*, device_type: str) -> list[SyclDevice]: ...
