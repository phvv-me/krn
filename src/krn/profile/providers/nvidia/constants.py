from ...enums import Activity

# The ActivityKind members kept as typed records, as integers: `int(cupti.ActivityKind.X)`.
CONCURRENT_KERNEL = 10
MEMCPY = 1

# Each Activity flag -> its CUPTI ActivityKind enum-member name.
CUPTI_KIND = {
    Activity.KERNEL: "CONCURRENT_KERNEL",
    Activity.MEMCPY: "MEMCPY",
    Activity.MEMSET: "MEMSET",
    Activity.SYNC: "SYNCHRONIZATION",
    Activity.OVERHEAD: "OVERHEAD",
    Activity.MEMORY: "MEMORY",
    Activity.JIT: "JIT",
    Activity.RUNTIME: "RUNTIME",
    Activity.DRIVER: "DRIVER",
    Activity.MEMORY_POOL: "MEMORY_POOL",
}
