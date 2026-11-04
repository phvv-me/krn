"""The retained generator's range reader, independent of NVIDIA generated outputs."""

import io
import logging
import os
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request

import pytest

import experiments.gen_cuda_bindings as generator

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("fault", ("", "ignored-range", "truncated", "unsafe-path"))
def test_range_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fault: str) -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        wheel.writestr("unwanted.bin", os.urandom(1 << 18))
        wheel.writestr("nvidia/cu13/include/header.h", b"typedef int handle;\n")
        wheel.writestr("nvidia_test-1.0.dist-info/METADATA", b"Name: nvidia-test\nVersion: 1.0\n")
        if fault == "unsafe-path":
            wheel.writestr("../include/escape.h", b"invalid")
    payload = archive.getvalue()
    ranges: list[tuple[int, int]] = []

    class Response(io.BytesIO):
        def __init__(self, data: bytes, status: int = 200, headers: dict[str, str] | None = None) -> None:
            super().__init__(data)
            self.status, self.headers = status, headers or {}

    def open_url(url: str | Request, *, timeout: int) -> Response:
        assert timeout == 30
        if isinstance(url, str):
            assert url == "https://pypi.org/pypi/nvidia-test/1.0/json"
            return Response(b'{"urls": [{"filename": "test-manylinux_x86_64.whl", "url": "https://test/wheel"}]}')
        assert url.full_url == "https://test/wheel"
        range_header = url.get_header("Range") or ""
        assert range_header.startswith("bytes=")
        start, end = map(int, range_header.removeprefix("bytes=").split("-"))
        ranges.append((start, end))
        data = payload[start : end + 1]
        if fault == "truncated" and end > start:
            data = data[:-1]
        return Response(
            data,
            200 if fault == "ignored-range" else 206,
            {"Content-Range": f"bytes {start}-{end}/{len(payload)}"},
        )

    monkeypatch.setattr(generator, "urlopen", open_url)
    monkeypatch.setattr(generator, "WHEELS", {"nvidia-test": "1.0"})
    if fault:
        with pytest.raises(ValueError if fault == "unsafe-path" else OSError):
            generator.download_headers(tmp_path)
        assert not (tmp_path / "nvidia-test").exists()
        assert not list(tmp_path.iterdir())
        return
    generator.download_headers(tmp_path)
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()) == [
        "nvidia-test/tree/nvidia/cu13/include/header.h",
        "nvidia-test/tree/nvidia_test-1.0.dist-info/METADATA",
    ]
    assert (tmp_path / "nvidia-test/tree/nvidia/cu13/include/header.h").read_bytes() == b"typedef int handle;\n"
    assert sum(end - start + 1 for start, end in ranges) < len(payload) // 4
    before = len(ranges)
    generator.download_headers(tmp_path)
    assert len(ranges) == before


def test_http_range_zip(tmp_path: Path) -> None:
    archive = tmp_path / "wheel.whl"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        wheel.writestr("unwanted.bin", os.urandom(1 << 18))
        wheel.writestr("nvidia/cu13/include/header.h", b"typedef int handle;\n")
    payload = archive.read_bytes()
    transferred: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            range_header = self.headers.get("Range", "")
            assert range_header.startswith("bytes=")
            start, end = map(int, range_header.removeprefix("bytes=").split("-"))
            content = payload[start : end + 1]
            transferred.append(len(content))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: str) -> None:
            logger.debug(format, *args)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except OSError as error:
        pytest.skip(f"loopback server is unavailable: {error}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with generator.RangeFile(f"http://127.0.0.1:{server.server_port}/wheel.whl") as ranged:
            for whence, offset in ((io.SEEK_SET, 4), (io.SEEK_CUR, 8), (io.SEEK_END, -32)):
                expected = offset + (
                    ranged.tell() if whence == io.SEEK_CUR else len(payload) if whence == io.SEEK_END else 0
                )
                assert ranged.seek(offset, whence) == expected
                assert ranged.read(7) == payload[expected : expected + 7]
            with zipfile.ZipFile(ranged) as wheel:
                assert wheel.read("nvidia/cu13/include/header.h") == b"typedef int handle;\n"
            assert ranged.seek(0, io.SEEK_END) == len(payload) and ranged.read(1) == b""
            with pytest.raises(ValueError):
                ranged.seek(-1)
        assert sum(transferred) < len(payload) // 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
