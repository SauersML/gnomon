"""Serve one directory over HTTP with byte-range support, for scorer parity tests.

Python's stock file server answers every GET with the whole file. The remote
scorer needs 206 responses with exact Content-Range headers, and a HEAD that
reports the object length, so this handler implements just that.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import sys


class RangeHandler(BaseHTTPRequestHandler):
    root = Path(".")
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def _target(self):
        path = (self.root / self.path.lstrip("/")).resolve()
        if self.root.resolve() not in path.parents or not path.is_file():
            self.send_error(404)
            return None
        return path

    def do_HEAD(self):
        if path := self._target():
            self.send_response(200)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

    def do_GET(self):
        path = self._target()
        if path is None:
            return
        size = path.stat().st_size
        header = self.headers.get("Range")
        if header is None:
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with path.open("rb") as handle:
                self.wfile.write(handle.read())
            return
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", header)
        if match is None or int(match[1]) > int(match[2]) or int(match[2]) >= size:
            self.send_error(416)
            return
        start, end = int(match[1]), int(match[2])
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            self.wfile.write(handle.read(end - start + 1))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 512


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: range_http_server.py DIRECTORY PORT")
    RangeHandler.root = Path(sys.argv[1])
    Server(("127.0.0.1", int(sys.argv[2])), RangeHandler).serve_forever()


if __name__ == "__main__":
    main()
