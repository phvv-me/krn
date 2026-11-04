"""CPU shared-object cache, preserving the research cache policy."""

import hashlib
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

CACHE = Path(tempfile.gettempdir()) / "krn-cpu"


def compile_c(source: str, clang: tuple[str, ...], cache_dir: Path | None = None) -> tuple[Path, float]:
    """The shared object of `source` and the seconds clang took (0 when a previous build is cached)."""
    root = cache_dir or CACHE
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((source + " ".join(clang)).encode()).hexdigest()[:16]
    library = root / f"{digest}.so"
    if library.exists():
        return library, 0.0
    started = time.perf_counter()
    scratch = library.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    subprocess.run([*clang, "-o", str(scratch), "-"], input=source, text=True, check=True, timeout=300)
    scratch.replace(library)  # atomic, so a thread or process that sees the library sees all of it
    return library, time.perf_counter() - started
