#!/usr/bin/env bash
# Run the exported GitHub Rust cache in MSI's Ubuntu userspace. The host C
# compiler uses a disk-backed sysroot; cached Ubuntu binaries retain their libc.
set -euo pipefail

repo=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
sysroot="$repo/.validation/toolchain/ubuntu24"
if [[ ${0##*/} == cc ]]; then
    # Startup objects, linker scripts, libc_nonshared, and runtime libraries
    # must come from the same libc. A libc.so symlink alone still leaves GCC
    # selecting the host Scrt1.o and librt, which depend on private host symbols.
    exec /host/usr/bin/gcc --sysroot="$sysroot" \
        -B"$sysroot/usr/lib/x86_64-linux-gnu/" -B/host/usr/bin/ \
        -isystem "$sysroot/usr/include/x86_64-linux-gnu" \
        -L"$sysroot/usr/lib/x86_64-linux-gnu" -L"$sysroot/lib/x86_64-linux-gnu" \
        "$@"
fi
if [[ ${0##*/} == rustc-workspace ]]; then
    # Iterate on workspace code without repeating whole-program optimization
    # of the already optimized dependency graph. Release builds are unaffected.
    exec "$@" -C lto=off -C opt-level=1 -C codegen-units=64 \
        -C "incremental=$repo/.validation/incremental"
fi

project=/projects/standard/hsiehph/sauer354
image="$project/lean-ubuntu24.sif"
mkdir -p "$repo/.validation/bin" "$repo/.validation/tmp" "$repo/.validation/incremental"

if [[ ! -f "$sysroot/.complete" ]]; then
    packages="$repo/.validation/toolchain/packages"
    mkdir -p "$packages" "$sysroot"
    # Match the immutable Ubuntu image's libc version. Keep downloads and
    # extraction on shared disk; this setup performs no compilation.
    for package in \
        glibc/libc6_2.39-0ubuntu8.8_amd64.deb \
        glibc/libc6-dev_2.39-0ubuntu8.8_amd64.deb \
        linux/linux-libc-dev_6.8.0-31.31_amd64.deb; do
        archive="$packages/${package##*/}"
        source_package=${package%%/*}
        if [[ ! -s "$archive" ]]; then
            curl --fail --location --silent --show-error --max-time 60 \
                "https://archive.ubuntu.com/ubuntu/pool/main/${source_package:0:1}/$package" \
                --output "$archive.download"
            mv "$archive.download" "$archive"
        fi
        apptainer exec --bind /projects:/projects "$image" dpkg-deb --extract "$archive" "$sysroot"
    done
    # Ubuntu packages use /usr-merge, while libc.so's linker script still
    # names /lib paths interpreted relative to the sysroot.
    if [[ ! -e "$sysroot/lib" ]]; then ln -s usr/lib "$sysroot/lib"; fi
    if [[ ! -e "$sysroot/lib64" ]]; then ln -s usr/lib64 "$sysroot/lib64"; fi
    # GCC's own compiler support may come from the host, but cached C++
    # dependencies must resolve against the Ubuntu runtime versions.
    apptainer exec --bind /projects:/projects "$image" sh -c '
        cp -L /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 "$1/usr/lib/x86_64-linux-gnu/"
        cp -L /usr/lib/x86_64-linux-gnu/libstdc++.so.6 "$1/usr/lib/x86_64-linux-gnu/"
    ' sh "$sysroot"
    ln -sf libgcc_s.so.1 "$sysroot/usr/lib/x86_64-linux-gnu/libgcc_s.so"
    ln -sf libstdc++.so.6 "$sysroot/usr/lib/x86_64-linux-gnu/libstdc++.so"
    test -f "$sysroot/usr/lib/x86_64-linux-gnu/Scrt1.o"
    test -f "$sysroot/usr/lib/x86_64-linux-gnu/libc_nonshared.a"
    test -f "$sysroot/usr/include/linux/types.h"
    touch "$sysroot/.complete"
fi
for wrapper in cc rustc-workspace; do
    wrapper_path="$repo/.validation/bin/$wrapper"
    if [[ ! -L "$wrapper_path" ]] || [[ $(readlink "$wrapper_path") != "$repo/scripts/msi-cached-rust.sh" ]]; then
        ln -sf "$repo/scripts/msi-cached-rust.sh" "$wrapper_path"
    fi
done

if [[ ${1:-} == --setup-toolchain ]]; then
    exit 0
fi

compiler_binds=""
for library in /usr/lib64/libmpfr.so.4 /usr/lib64/libmpc.so.3 /usr/lib64/libgmp.so.10 /usr/lib64/libopcodes-*.so /usr/lib64/libbfd-*.so; do
    compiler_binds+=",$library:/usr/lib/x86_64-linux-gnu/${library##*/}"
done

# Run an already built test or CLI without spending its runtime budget on
# Cargo or linking. Separate compilation and execution caps give useful signal
# even when the first workspace link takes most of the build budget.
if [[ ${1:-} == --run ]]; then
    shift
    if [[ $# == 0 ]]; then
        echo "--run requires a program and optional arguments" >&2
        exit 2
    fi
    command=("$@")
else
    command=(cargo +nightly-2026-08-31
        --config 'profile.release.lto="thin"'
        --config profile.release.codegen-units=16 "$@")
fi

exec timeout 180 taskset -c 12-15 apptainer exec \
    --bind "/projects:/projects,/usr:/host/usr,/lib64:/host/lib64,$repo:/home/runner/work/gnomon/gnomon,$project/.cargo:/home/runner/.cargo,/etc/pki/tls/cert.pem:/etc/ssl/certs/ca-certificates.crt$compiler_binds" \
    --pwd /home/runner/work/gnomon/gnomon "$image" \
    env -i HOME="$project" RUSTUP_HOME="$project/.rustup" \
    CARGO_HOME=/home/runner/.cargo \
    PATH="$repo/.validation/bin:$project/.cargo/bin:/usr/bin:/bin:/host/usr/bin" \
    TMPDIR="$repo/.validation/tmp" \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    RUSTFLAGS="-C target-cpu=native" \
    RUSTC_WORKSPACE_WRAPPER="$repo/.validation/bin/rustc-workspace" \
    "${command[@]}"
