"""Bounded read-only comparison of demand faults and buffered file reads on MSI."""
import mmap
import os
import resource
import sys
import time

path = sys.argv[1]
length = 256 << 20
window = 4 << 20
with open(path, 'rb') as source:
    mapped = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
    for method, start in [('mmap', 4 << 30), ('pread', 5 << 30)]:
        assert start + length + 2*window < len(mapped)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        began = time.monotonic()
        checksum = 0
        for offset in range(start, start + length, window):
            if method == 'pread':
                block = os.pread(source.fileno(), window, offset)
            else:
                block = mapped[offset:offset+window]
            checksum ^= block[0]
        current = resource.getrusage(resource.RUSAGE_SELF)
        print(method, 'seconds', time.monotonic()-began,
              'major_faults', current.ru_majflt-usage.ru_majflt,
              'rss_kib', current.ru_maxrss, 'checksum', checksum, flush=True)
