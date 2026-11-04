"""Linux kernel files not covered by the host libraries."""

from pathlib import Path


def read_text(path: str | Path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def read_int(path: str | Path) -> int | None:
    text = read_text(path)
    try:
        return int(text, 0) if text is not None else None
    except ValueError:
        return None


def cpu_caches(root: str | Path = "/sys/devices/system/cpu/cpu0/cache") -> tuple[tuple[str, int], ...]:
    result = []
    for path in paths(root, "index[0-9]*"):
        level, kind, size = read_int(path / "level"), read_text(path / "type"), read_text(path / "size")
        if level is None or kind is None or size is None:
            continue
        multiplier = {"K": 1024, "M": 1024**2}.get(size[-1:], 1)
        number = size[:-1] if multiplier != 1 else size
        if number.isdigit():
            result.append((f"l{level}-{kind.lower()}", int(number) * multiplier))
    return tuple(result)


def paths(root: str | Path, pattern: str) -> tuple[Path, ...]:
    return tuple(sorted(Path(root).glob(pattern)))


def pci_id(path: Path) -> str:
    return path.resolve().name


def cgroup_root() -> Path:
    membership = read_text("/proc/self/cgroup") or ""
    relative = next((line[3:] for line in membership.splitlines() if line.startswith("0::")), "/")
    return Path("/sys/fs/cgroup") / relative.lstrip("/")


def cgroup_limits(root: Path | None = None, mount: Path = Path("/sys/fs/cgroup")) -> tuple[float | None, int | None]:
    groups = [cgroup_root() if root is None else root]
    if groups[0].is_relative_to(mount):
        while groups[-1] != mount:
            groups.append(groups[-1].parent)
    quotas, memory = [], []
    for group in groups:
        quota = (read_text(group / "cpu.max") or "max 100000").split()
        if quota[0] != "max":
            quotas.append(float(quota[0]) / float(quota[1]))
        limit = read_int(group / "memory.max")
        if limit is not None:
            memory.append(limit)
    return min(quotas, default=None), min(memory, default=None)


def drm_processes(bus_id: str, root: str | Path = "/proc") -> tuple[tuple[int, int | None], ...]:
    """Visible DRM clients only; permissions can hide other users' processes."""
    clients: set[tuple[int, str]] = set()
    memory: dict[int, int | None] = {}
    for path in paths(root, "[0-9]*/fdinfo/*"):
        text = read_text(path) or ""
        fields = {
            key.strip(): value.strip()
            for line in text.splitlines()
            if ":" in line
            for key, value in [line.split(":", 1)]
        }
        if fields.get("drm-pdev") != bus_id or "drm-client-id" not in fields:
            continue
        pid = int(path.parent.parent.name)
        client = (pid, fields["drm-client-id"])
        if client in clients:
            continue
        clients.add(client)
        regions = [value.split() for key, value in fields.items() if key.startswith("drm-resident-")]
        values = [
            int(value[0]) * {"KiB": 1024, "MiB": 1024**2}.get(value[1] if len(value) > 1 else "", 1)
            for value in regions
            if value and value[0].isdigit()
        ]
        memory[pid] = (memory.get(pid) or 0) + sum(values) if values else memory.get(pid)
    return tuple(sorted(memory.items()))
