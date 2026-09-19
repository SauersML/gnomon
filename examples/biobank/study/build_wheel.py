"""Build gamfit from a pinned gam commit, verify it, stage the AoU wheelhouse and publish a venv.

Runs on a Linux build host, never inside AoU:

1. checks out the pinned commit in an existing gam clone (``--ref main`` re-pins to GitHub main);
2. builds gam-pyffi into a warm cargo target, so a re-pin recompiles only the gam crates that changed,
   with one of three profiles (``fast`` is gam's release-dev: opt-level 3, no cross-crate LTO, 16
   codegen units, for accuracy re-pins, where only the predictions matter):
   - ``release-pypi``, gam's PyPI profile (opt-level 3, fat LTO, one codegen unit, stripped), which is
     what the AoU wheelhouse ships;
   - ``quick``, gam's release profile (opt-level 3, thin LTO) with 16 codegen units and incremental
     compilation, for development re-pins that take minutes; it keeps symbols for profiling;
   either for baseline x86-64 or, with ``--cpu`` (e.g. x86-64-v3, hardware FMA), for a newer CPU, in which
   case the wheel's ``import gamfit`` refuses a CPU without those features by name;
3. verifies the extension (no overflow-check panic strings, glibc symbol versions within the wheel's
   manylinux tag, no debug sections or symbol table when stripped) and the profile's flags on the final
   rustc call;
4. proves the wheel plus the binary dependency wheels resolve offline for the runtime's CPython and
   glibc and, for ``release-pypi``, stages them as the wheelhouse tar the task installs with --no-index;
5. with ``--venv-root``, publishes ``venv-<sha>-quick`` or ``venv-<sha>-release`` there: the pin's
   gamfit alone, layered through a .pth over one base venv of the wheelhouse's dependencies, which is
   built once per dependency set; the ``venv`` symlink then moves to it once the engine and a smoke fit
   check out.

The printed ``engine_sha256`` (of ``gamfit/_rust.abi3.so``) identifies the build: every gam commit
builds a wheel with the same version and file name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

GAM_URL = "https://github.com/SauersML/gam.git"
# The AoU runtime image is python:3.12-slim-trixie (CPython 3.12, glibc 2.41). manylinux_2_28 wheels
# run there and on a glibc 2.28 build host, so what is staged here is also what the build host installs.
PYTHON = "3.12"
POLICY = "manylinux_2_28"
GLIBC_MAX = (2, 28)
# pip widens only the legacy manylinux2014/2010 tags, so --platform manylinux_2_28 alone rejects a wheel
# tagged manylinux_2_17 (grpcio): list every tag up to the policy.
PLATFORMS = [f"manylinux_2_{minor}_x86_64" for minor in range(GLIBC_MAX[1], 4, -1)] + [
    "manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64"]
TARGET = ["--only-binary=:all:", *(arg for platform in PLATFORMS for arg in ("--platform", platform)),
          "--implementation", "cp", "--python-version", PYTHON, "--abi", "cp312", "--abi", "abi3", "--abi", "none"]
PROFILES = {
    "release-pypi": {"cargo": "release-pypi", "env": {}, "suffix": "-release", "stripped": True,
                     "flags": ("opt-level=3", "codegen-units=1", "strip=symbols"), "lto": {"lto", "lto=fat"}},
    "quick": {"cargo": "release", "env": {"CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "16"}, "suffix": "-quick",
              "stripped": False, "flags": ("opt-level=3", "codegen-units=16"), "lto": {"lto=thin"}},
    # gam's own iteration profile: no cross-crate LTO, so a re-pin skips relinking every crate through
    # ThinLTO, which was 62% of a quick re-pin (gam-pyffi 279.6 s of 454.6 s at 40b5044e4f).
    "fast": {"cargo": "release-dev", "env": {}, "suffix": "-fast", "stripped": False,
             "flags": ("opt-level=3", "codegen-units=16"), "lto": {None, "lto=off"}},
}
# panic must unwind in both, so gam's GPU probe can fall back to the CPU where no driver loads (gam#2972).
FORBIDDEN_FLAGS = ("debug-assertions=on", "overflow-checks=on", "panic=abort")
# /proc/cpuinfo's name for each x86 target feature rustc can enable beyond baseline x86-64 (--cpu).
CPU_FLAGS = {"avx": "avx", "avx2": "avx2", "bmi1": "bmi1", "bmi2": "bmi2", "cmpxchg16b": "cx16", "f16c": "f16c",
             "fma": "fma", "lahfsahf": "lahf_lm", "lzcnt": "abm", "movbe": "movbe", "popcnt": "popcnt",
             "sse3": "pni", "sse4.1": "sse4_1", "sse4.2": "sse4_2", "ssse3": "ssse3", "xsave": "xsave"}
# Only overflow-checked arithmetic panics with these; division and remainder are checked in every
# profile, so their messages are reported but prove nothing.
OVERFLOW_CHECKS = [f"attempt to {op} with overflow".encode()
                   for op in ("add", "subtract", "multiply", "negate", "shift left", "shift right")]


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs).stdout


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1, default=str) + "\n")


def checkout(gam, ref):
    """Detach the clone at the pinned commit; refuse a tree that would not build that commit."""
    dirty = run(["git", "-C", gam, "status", "--porcelain", "--untracked-files=no"]).strip()
    if dirty:
        raise SystemExit(f"gam clone has local changes, refusing to build:\n{dirty}")
    known = subprocess.run(["git", "-C", gam, "cat-file", "-e", f"{ref}^{{commit}}"],
                           capture_output=True).returncode == 0
    if ref == "main" or not known:
        run(["git", "-C", gam, "fetch", "-q", GAM_URL, "main"])
    sha = run(["git", "-C", gam, "rev-parse", "--verify",
               f"{'FETCH_HEAD' if ref == 'main' else ref}^{{commit}}"]).strip()
    run(["git", "-C", gam, "checkout", "-q", "--detach", sha])
    return sha


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(line):
    return normalized(re.split(r"[\s<>=!~;\[@]", line, maxsplit=1)[0])


def requirements(path):
    lines = [line.split("#", 1)[0].strip() for line in Path(path).read_text().splitlines()]
    lines = [line for line in lines if line]
    if "gamfit" not in {requirement_name(line) for line in lines}:
        raise SystemExit(f"{path} does not require gamfit")
    return lines


def resolver(root):
    """The pip every resolution runs: the interpreter's own ensurepip pip, in a venv of it under ``root``.

    ``sys.executable -m pip`` imports whatever pip the interpreter can find, and on a build host whose Python ships
    none that is a user-site pip nothing here creates or records: a PYTHONUSERBASE elsewhere hides it and the build
    fails at the resolution, after the compile (gnomon#2404). ensurepip's pip comes with the interpreter, so this
    venv resolves the same under any user site, and its version is recorded with the build.
    """
    venv = root / f".resolver-{PYTHON}"
    python = venv / "bin" / "python"
    try:
        if not python.exists():
            # Assembled beside its final path and renamed, so two builds sharing ``root`` never see half a venv.
            root.mkdir(parents=True, exist_ok=True)
            partial = root / f".resolver-{PYTHON}.partial-{os.getpid()}"
            subprocess.run([sys.executable, "-m", "venv", "--clear", partial], check=True)
            try:
                partial.rename(venv)
            except OSError:
                shutil.rmtree(partial)
                if not python.exists():
                    raise
        pip = run([python, "-m", "pip", "--version"]).split()
        version = run([python, "-c", "import platform; print(platform.python_version())"]).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"the resolver venv {venv} cannot run pip ({error}); refusing before the build") from error
    return python, {"python": version, "pip": pip[1]}


def download_dependencies(lines, deps, pip_python):
    """Fetch the non-gamfit wheels for the runtime platform once; a re-pin reuses them."""
    wanted = sorted(line for line in lines if requirement_name(line) != "gamfit")
    stamp = hashlib.sha256("\n".join(wanted + TARGET).encode()).hexdigest()
    marker = deps / "REQUIREMENTS.sha256"
    if marker.is_file() and marker.read_text().strip() == stamp:
        return stamp
    if deps.exists():
        shutil.rmtree(deps)
    deps.mkdir(parents=True)
    listing = deps / "requirements.txt"
    listing.write_text("\n".join(wanted) + "\n")
    subprocess.run([pip_python, "-m", "pip", "download", "--disable-pip-version-check",
                    "--no-cache-dir", *TARGET, "--dest", deps, "-r", listing], check=True)
    listing.unlink()
    marker.write_text(stamp + "\n")
    return stamp


def build(gam, target, out, maturin, jobs, timeout, profile, cpu=None, extra=()):
    """Build the wheel into ``out`` and return it with the build log's evidence."""
    env = {key: value for key, value in os.environ.items()
           if key not in {"RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "LIBRARY_PATH",
                          "OPENSSL_LIB_DIR", "C_INCLUDE_PATH"}}
    # The warm targets were built with CARGO_INCREMENTAL=1, and cargo hashes it into every unit:
    # without it the whole dependency graph recompiles.
    env.update(CARGO_INCREMENTAL="1", CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS=str(jobs),
               CARGO_TERM_VERBOSE="true", CARGO_TERM_COLOR="never", **PROFILES[profile]["env"])
    if cpu:
        env["RUSTFLAGS"] = f"-C target-cpu={cpu}"
    out.mkdir(parents=True, exist_ok=True)
    for stale in [*out.glob("gamfit-*.whl"), *out.glob("wheelhouse-*.tar")]:
        stale.unlink()
    log = out / "build.log"
    start = time.monotonic()
    with log.open("w") as handle:
        child = subprocess.Popen([maturin, "build", "--profile", PROFILES[profile]["cargo"], "--locked", "--offline",
                                  "--timings", "--compatibility", POLICY, "--interpreter", sys.executable, "--out", out,
                                  *extra],
                                 cwd=gam, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise SystemExit(f"maturin build exceeded {timeout} s; the target keeps the units it finished")
    seconds = round(time.monotonic() - start, 1)
    text = log.read_text(errors="replace")
    if code:
        print(text[-4000:], file=sys.stderr)
        raise SystemExit(f"maturin build failed with exit {code} after {seconds} s")
    [wheel] = out.glob("gamfit-*.whl")
    slowest = slowest_units(target, out)
    if extra:
        return wheel, {"seconds": seconds, "slowest_units": slowest}
    return wheel, {"seconds": seconds, "slowest_units": slowest,
                   **compile_evidence(text, out / "compile.json", profile, cpu)}


def slowest_units(target, out):
    """Where a re-pin's time went, from cargo's --timings report (kept beside the build log)."""
    report = target / "cargo-timings" / "cargo-timing.html"
    if not report.is_file():
        return None
    shutil.copy2(report, out / "cargo-timing.html")
    found = re.search(r"const UNIT_DATA = (\[.*?\]);\n", report.read_text(errors="replace"), re.S)
    try:
        units = json.loads(found.group(1))
        return [f"{unit['name']} {unit.get('target', '').strip()} {unit['duration']:.1f}s"
                for unit in sorted(units, key=lambda unit: -unit["duration"])[:6]]
    except (AttributeError, ValueError, KeyError, TypeError):
        return "unparsed: see cargo-timing.html"


def compile_evidence(text, saved, profile, cpu):
    """The final rustc call must carry the profile; the other calls show how warm the target was.

    Rebuilding a pin whose extension is already built leaves gam_pyffi fresh, with no rustc call to
    read; the evidence saved with that extension is reused, and main checks it names the same bytes.
    """
    calls = [line for line in text.splitlines() if "Running `" in line and "--crate-name " in line]
    evidence = {"rustc_calls": len(calls), "fresh_units": len(re.findall(r"^\s+Fresh ", text, re.M)),
                "rebuilt": sorted({re.search(r"--crate-name (\S+)", line).group(1) for line in calls})}
    final = [line for line in calls if "--crate-name gam_pyffi " in line]
    if not final:
        if not saved.is_file():
            raise SystemExit("gam_pyffi was fresh and no saved evidence exists for its flags")
        previous = json.loads(saved.read_text())
        return {**evidence, "final_rustc_flags": previous["final_rustc_flags"],
                "flags_engine_sha256": previous["engine_sha256"]}
    [final] = final
    flags = re.findall(r"-C (\S+)", final)
    expected = PROFILES[profile]
    missing = [flag for flag in (*expected["flags"], *([f"target-cpu={cpu}"] if cpu else [])) if flag not in flags]
    if not cpu and any(flag.startswith("target-cpu=") for flag in flags):
        missing.append("baseline x86-64 (saw a target-cpu)")
    lto = [flag for flag in flags if flag == "lto" or flag.startswith("lto=")] or [None]
    if not set(lto) <= expected["lto"]:
        missing.append(f"lto in {sorted(map(str, expected['lto']))} (saw {lto})")
    forbidden = [flag for flag in FORBIDDEN_FLAGS if flag in flags]
    if missing or forbidden:
        raise SystemExit(f"gam_pyffi rustc flags are not the {profile} profile: missing {missing}, "
                         f"forbidden {forbidden}")
    return {**evidence, "final_rustc_flags": sorted(
        flag for flag in flags if not flag.startswith(("metadata=", "extra-filename=", "incremental=", "link-arg=")))}


def extract_extension(wheel, work):
    with zipfile.ZipFile(wheel) as archive:
        [member] = [name for name in archive.namelist() if re.fullmatch(r"gamfit/_rust[^/]*\.so", name)]
        extension = work / Path(member).name
        extension.write_bytes(archive.read(member))
    return member, extension


def inspect_extension(wheel, work, profile):
    """Read the shipped extension itself: sections, strings, glibc versions and dynamic needs."""
    member, extension = extract_extension(wheel, work)
    data = extension.read_bytes()
    sections = re.findall(r"^\s*\[\s*\d+\]\s+(\S+)", run(["readelf", "-S", "-W", extension]), re.M)
    versions = {tuple(map(int, match.split(".")))
                for match in re.findall(r"GLIBC_(\d+(?:\.\d+)+)", run(["readelf", "--dyn-syms", "-W", extension]))}
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", run(["readelf", "-d", "-W", extension]))
    report = {"member": member, "engine_sha256": sha256(extension), "bytes": len(data),
              "debug_sections": sorted(name for name in sections if name.startswith((".debug", ".zdebug"))),
              "symtab": ".symtab" in sections,
              "overflow_check_panics": sum(data.count(message) for message in OVERFLOW_CHECKS),
              "division_panics": data.count(b"to divide with overflow") + data.count(b"remainder with overflow"),
              "glibc_max": ".".join(map(str, max(versions))) if versions else None, "needed": needed}
    extension.unlink()
    problems = ["overflow_check_panics"] if report["overflow_check_panics"] else []
    if PROFILES[profile]["stripped"]:
        problems += [key for key in ("debug_sections", "symtab") if report[key]]
    if versions and max(versions) > GLIBC_MAX:
        problems.append("glibc_max")
    if problems:
        raise SystemExit(f"extension check failed on {problems}: {json.dumps(report)}")
    return report


def text_sha256(extension, work):
    section = work / "text.bin"
    run(["objcopy", "-O", "binary", "--only-section=.text", extension, section])
    digest = sha256(section)
    section.unlink()
    return digest


def unstripped_twin(gam, target, out, maturin, jobs, timeout, wheel, cpu):
    """Relink the release-pypi build with its symbol table kept, for perf and gdb.

    The profile sets strip = true, which maturin's --strip does not control; ``-C strip=none`` reaches
    only the final gam-pyffi rustc call, so the twin is the same fat-LTO code, which the .text digests
    prove. Rebuilding the shipped wheel afterwards relinks gam-pyffi once more.
    """
    twin = out / "unstripped"
    built, compiled = build(gam, target, twin, maturin, jobs, timeout, "release-pypi", cpu, ("--", "-C", "strip=none"))
    _, extension = extract_extension(built, twin)
    built.unlink()
    _, shipped = extract_extension(wheel, twin)
    report = {"path": str(extension), "seconds": compiled["seconds"],
              "symtab": ".symtab" in run(["readelf", "-S", "-W", extension]),
              "text_sha256": text_sha256(extension, twin), "shipped_text_sha256": text_sha256(shipped, twin)}
    shipped.unlink()
    report["same_code"] = report["text_sha256"] == report["shipped_text_sha256"]
    return report


def resolve(wheel, deps, lines, out, provenance, archive, pip_python):
    """Prove the wheels resolve for the runtime platform with no index; optionally write the tar."""
    house = out / "wheelhouse"
    if house.exists():
        shutil.rmtree(house)
    house.mkdir()
    for path in [wheel, *deps.glob("*.whl")]:
        shutil.copy2(path, house / path.name)
    listing = out / "requirements.txt"
    listing.write_text("\n".join(lines) + "\n")
    report = out / "resolution.json"
    with tempfile.TemporaryDirectory(dir=out) as scratch:
        subprocess.run([pip_python, "-m", "pip", "install", "--disable-pip-version-check", "--dry-run",
                        "--ignore-installed", "--no-index", "--find-links", house, *TARGET, "--target",
                        scratch, "--report", report, "-q", "-r", listing], check=True)
    installed = {normalized(item["metadata"]["name"]): item["metadata"]["version"]
                 for item in json.loads(report.read_text())["install"]}
    unused = sorted(path.name for path in house.glob("*.whl")
                    if normalized(path.name.split("-", 1)[0]) not in installed)
    provenance.update(resolved=dict(sorted(installed.items())), unused_wheels=unused,
                      wheels={path.name: sha256(path) for path in sorted(house.glob("*.whl"))})
    final = None
    if archive:
        write_json(house / "PROVENANCE.json", provenance)
        staged = out / "wheelhouse.tar"
        with tarfile.open(staged, "w") as handle:
            for path in sorted(house.iterdir()):
                handle.add(path, arcname=path.name)
        final = out / f"wheelhouse-{sha256(staged)[:8]}.tar"
        staged.rename(final)
    shutil.rmtree(house)
    return final


def ensure_base(root, deps, stamp, lines, resolved, tools):
    """One venv of the runtime dependencies per dependency set, shared by every pin's venv.

    Installed from the wheelhouse's own dependency wheels with the task's pip flags; build-host tools
    come from the index under constraints that pin every runtime package to its wheelhouse version.
    """
    base = root / f"venv-base-{stamp[:8]}"
    python = base / "bin" / "python"
    marker = base / "BASE.json"
    if not marker.is_file():
        if base.exists():
            shutil.rmtree(base)
        subprocess.run([sys.executable, "-m", "venv", base], check=True)
        listing = base / "requirements.txt"
        listing.write_text("\n".join(line for line in lines if requirement_name(line) != "gamfit") + "\n")
        subprocess.run([python, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir",
                        "--no-index", "--only-binary=:all:", "--find-links", deps, "-r", listing], check=True)
        write_json(marker, {"dependencies": stamp})
    constraints = base / "runtime-constraints.txt"
    constraints.write_text("".join(f"{name}=={version}\n" for name, version in resolved.items() if name != "gamfit"))
    if tools:
        subprocess.run([python, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "-q",
                        "-c", constraints, *tools], check=True)
    return base


SMOKE = r"""
import json, sys, time
from pathlib import Path
import numpy as np
import gamfit
rng = np.random.default_rng(20260918)
n = 3000
pcs = rng.normal(size=(n, 2))
age, sex, z = rng.uniform(20, 80, n), rng.integers(0, 2, n).astype(float), rng.standard_t(5, n)
eta = -1 + 0.02 * (age - 50) + 0.2 * sex + 0.3 * pcs[:, 0] + (0.5 + 0.2 * pcs[:, 1]) * z
data = {"y": (rng.normal(size=n) < eta).astype(float), "z": z, "age": age, "sex": sex,
        "PC1": pcs[:, 0], "PC2": pcs[:, 1]}
surface = "duchon(PC1, PC2, centers=8)"
start = time.monotonic()
model = gamfit.fit(data, formula=f"y ~ s(age, k=6) + sex + {surface}", family="bernoulli-marginal-slope",
                   link="probit", z_column="z", slope_formula=f"1 + {surface}",
                   config={"latent_measure": "global-empirical"})
seconds = time.monotonic() - start
risk = np.asarray(model.predict(data), dtype=float)
model.save(sys.argv[1])
law = (json.loads(Path(sys.argv[1]).read_text())["payload"].get("latent_measure") or {}).get("kind")
ok = risk.shape == (n,) and bool(np.isfinite(risk).all()) and law == "global-empirical"
print(json.dumps({"ok": ok, "fit_seconds": round(seconds, 2), "risk_range": [float(risk.min()), float(risk.max())],
                  "latent_measure": law, "version": gamfit.build_info().get("version")}))
sys.exit(0 if ok else 1)
"""

LISTING = ("import importlib.metadata as m, importlib.util as u, json; "
           "print(json.dumps({'origin': u.find_spec('gamfit._rust').origin, "
           "'packages': {d.metadata['Name']: d.version for d in m.distributions()}}))")


def publish(root, base, wheel, name, provenance, jobs, latest):
    """Write ``root/name`` as this pin's gamfit over the base venv, check it, and move ``venv`` to it.

    A published venv may be in use, so it is never rewritten: a new one is assembled beside it, checked,
    and renamed into place, and an existing one is only re-checked.
    """
    venv = root / name
    fresh = not venv.exists()
    if fresh:
        venv = root / f".{name}.partial"
        if venv.exists():
            shutil.rmtree(venv)
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", venv], check=True)
        site = venv / "lib" / f"python{PYTHON}" / "site-packages"
        base_site = base / "lib" / f"python{PYTHON}" / "site-packages"
        # addsitedir, not a bare path line, so the base's own .pth files are processed too.
        (site / "_study_base.pth").write_text(f"import site; site.addsitedir({str(base_site)!r})\n")
        subprocess.run([base / "bin" / "python", "-m", "pip", "install", "--disable-pip-version-check",
                        "--no-cache-dir", "--no-index", "--no-deps", "--target", site, wheel], check=True)
    result = check_venv(root, venv, base, provenance, jobs)
    if fresh and result["ok"]:
        venv = venv.rename(root / name)
    result["path"] = str(venv)
    (venv / "VENV.md").write_text(venv_notes(venv, base, provenance, result.pop("packages"), result))
    # The driver reads sys.prefix/PROVENANCE.json and believes gam_commit only while its engine_sha256 is
    # the installed engine's.
    write_json(venv / "PROVENANCE.json", {key: value for key, value in provenance.items() if key != "venv"})
    if result["ok"] and latest:
        link, staging = root / "venv", root / ".venv-next"
        if link.exists() and not link.is_symlink():
            raise SystemExit(f"{link} is a directory, not the published symlink; move it aside first")
        if staging.is_symlink():
            staging.unlink()
        staging.symlink_to(name)
        os.replace(staging, link)
        result["linked"] = str(link)
    return result


def check_venv(root, venv, base, provenance, jobs):
    """The venv imports this build's engine from its own site-packages, the base's versions, and fits."""
    site = (venv / "lib" / f"python{PYTHON}" / "site-packages").resolve()
    python = venv / "bin" / "python"
    # Run outside any gam checkout: its source gamfit/ would shadow the installed package.
    listed = json.loads(run([python, "-c", LISTING], cwd=root))
    packages = {normalized(key): value for key, value in listed["packages"].items()}
    # Resolve before comparing: a Red Hat venv lists site-packages through its lib64 -> lib symlink.
    result = {"base": str(base), "packages": packages,
              "libcuda_on_node": run(["/sbin/ldconfig", "-p"]).count("libcuda.so"),
              "engine_matches": (Path(listed["origin"]).resolve().is_relative_to(site)
                                 and sha256(listed["origin"]) == provenance["engine_sha256"]),
              "differs_from_resolution": {key: packages.get(key) for key, value in provenance["resolved"].items()
                                          if packages.get(key) != value}}
    with tempfile.TemporaryDirectory(dir=root) as scratch:
        try:
            smoke = subprocess.run([python, "-c", SMOKE, Path(scratch) / "smoke.gamfit"], cwd=scratch, text=True,
                                   capture_output=True, timeout=600, env=dict(os.environ, RAYON_NUM_THREADS=str(jobs)))
        except subprocess.TimeoutExpired:
            smoke = subprocess.CompletedProcess([], 124, "", "smoke fit exceeded 600 s")
    printed = smoke.stdout.strip().splitlines()
    result["smoke"] = json.loads(printed[-1]) if printed and printed[-1].startswith("{") else None
    if smoke.returncode:
        result["smoke_stderr"] = smoke.stderr[-2000:]
    guard_ok = True
    if provenance.get("cpu_guard"):
        # The guard ran on this CPU when gamfit imported above; here it must refuse a baseline one by name.
        result["guard_on_baseline"] = json.loads(run([python, "-c", GUARD_PROBE], cwd=root))
        guard_ok = (result["guard_on_baseline"]["import_error"]
                    and result["guard_on_baseline"]["missing"] == sorted(provenance["cpu_guard"]["required"]))
    result["ok"] = (result["engine_matches"] and not result["differs_from_resolution"] and guard_ok
                    and smoke.returncode == 0 and bool((result["smoke"] or {}).get("ok")))
    return result


GUARD_PROBE = r"""
import json
import gamfit._cpu_guard as guard
try:
    guard.check("fpu sse sse2")
except guard.UnsupportedCPUError as error:
    print(json.dumps({"import_error": isinstance(error, ImportError), "missing": error.missing, "message": str(error)}))
else:
    print(json.dumps({"import_error": False, "missing": [], "message": "baseline CPU accepted"}))
"""


def venv_notes(venv, base, provenance, packages, result):
    listing = "\n".join(f"- {name} {version}" for name, version in sorted(packages.items()))
    note = f"**{provenance['note']}**\n\n" if provenance.get("note") else ""
    guard = provenance.get("cpu_guard")
    cpu_line = (f"compiled for {guard['cpu']}; `import gamfit` raises gamfit._cpu_guard.UnsupportedCPUError on a CPU "
                f"lacking any of {', '.join(guard['required'])}" if guard else "baseline x86-64")
    return f"""# Study venv {venv.name}

{note}`{venv}/bin/python` (CPython {PYTHON}): gamfit from gam `{provenance['gam_commit']}` built with the
`{provenance['profile']}` profile, over `{base}`, which holds the AoU wheelhouse's other packages
at the versions the wheelhouse resolves to. Written by `examples/biobank/study/build_wheel.py`.

- CPU: {cpu_line}
- engine_sha256 (`gamfit/_rust.abi3.so`): `{provenance['engine_sha256']}`
- final rustc flags: {' '.join(provenance['build']['final_rustc_flags'])}
- wheelhouse: {provenance.get('wheelhouse') or 'none (only release-pypi builds are staged for AoU)'}
- installed engine is the verified build: {result['engine_matches']}
- smoke fit (Bernoulli marginal slope, global-empirical, n=3000): {json.dumps(result.get('smoke'))}

There is no pip in this venv: every pin gets its own directory, and the `venv` symlink next to it moves to
the newest. Run fits from outside any gam checkout, whose source `gamfit/` would shadow this package.

## Packages
{listing}
"""


def cpu_features(cpu):
    """The target features ``-C target-cpu=cpu`` adds to baseline x86-64, with their /proc/cpuinfo flags."""
    def enabled(*flags):
        return {line.split('"')[1] for line in run(["rustc", "--print", "cfg", *flags]).splitlines()
                if line.startswith("target_feature=")}
    added = sorted(enabled("-C", f"target-cpu={cpu}") - enabled())
    unknown = [feature for feature in added if feature not in CPU_FLAGS]
    if unknown or not added:
        raise SystemExit(f"target-cpu={cpu} adds {added}; no /proc/cpuinfo flag is known for {unknown}")
    return {feature: CPU_FLAGS[feature] for feature in added}


GUARD = '''"""Refuse to load an engine compiled for {cpu} on a CPU that lacks it.

Written into the wheel by gnomon's examples/biobank/study/build_wheel.py. The engine uses these
instructions anywhere, so on such a CPU it would die of SIGILL mid-fit; there is no baseline fallback.
"""
from pathlib import Path

CPU = {cpu!r}
REQUIRED = {required!r}


class UnsupportedCPUError(ImportError):
    """This CPU lacks target features the gamfit engine was compiled to use."""

    def __init__(self, missing):
        self.missing = missing
        super().__init__(f"gamfit's engine was compiled for {{CPU}}; this CPU lacks {{', '.join(missing)}}")


def check(flags=None):
    if flags is None:
        try:
            text = Path("/proc/cpuinfo").read_text()
        except OSError as error:
            raise UnsupportedCPUError([f"/proc/cpuinfo ({{error}})"]) from error
        flags = next((line.split(":", 1)[1] for line in text.splitlines() if line.startswith("flags")), "")
    present = set(flags.split())
    missing = sorted(feature for feature, flag in REQUIRED.items() if flag not in present)
    if missing:
        raise UnsupportedCPUError(missing)


check()
'''


def add_cpu_guard(wheel, cpu, required):
    """Make ``import gamfit`` run the guard before anything loads the engine, and rewrite the RECORD."""
    import ast
    import base64
    with zipfile.ZipFile(wheel) as archive:
        entries = [(info, archive.read(info)) for info in archive.infolist()]
    init = next(data for info, data in entries if info.filename == "gamfit/__init__.py").decode()
    # After the docstring and any __future__ import, which must stay first.
    body = ast.parse(init).body
    first = next(node for index, node in enumerate(body)
                 if not (index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
                 and not (isinstance(node, ast.ImportFrom) and node.module == "__future__"))
    at = min([first.lineno, *(node.lineno for node in getattr(first, "decorator_list", []))]) - 1
    lines = init.splitlines(keepends=True)
    guarded = "".join(lines[:at] + [f"from . import _cpu_guard  # refuses a CPU without {cpu}\n"] + lines[at:]).encode()
    guard = GUARD.format(cpu=cpu, required=required).encode()
    files = []
    for info, data in entries:
        if info.filename.endswith(".dist-info/RECORD"):
            record = info
            continue
        files.append((info, guarded if info.filename == "gamfit/__init__.py" else data))
        if info.filename == "gamfit/__init__.py":
            added = zipfile.ZipInfo("gamfit/_cpu_guard.py", date_time=info.date_time)
            added.external_attr, added.compress_type = info.external_attr, zipfile.ZIP_DEFLATED
            files.append((added, guard))

    def digest(data):
        return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    rows = [f"{info.filename},sha256={digest(data)},{len(data)}" for info, data in files] + [f"{record.filename},,"]
    staged = wheel.with_name(wheel.name + ".partial")
    with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as archive:
        for info, data in [*files, (record, ("\n".join(rows) + "\n").encode())]:
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
    staged.replace(wheel)
    return {"cpu": cpu, "required": required, "guard_sha256": hashlib.sha256(guard).hexdigest()}


def fma_instructions(wheel, work):
    """Hardware FMA sites in the engine; a baseline build reaches fma through a software call instead."""
    _, extension = extract_extension(wheel, work)
    disassembly = subprocess.Popen(["objdump", "-d", "--no-show-raw-insn", extension], stdout=subprocess.PIPE)
    counted = subprocess.run(["grep", "-c", "-E", r"[[:space:]]vfn?m(add|sub)"], stdin=disassembly.stdout,
                             capture_output=True, text=True)
    disassembly.stdout.close()
    disassembly.wait()
    extension.unlink()
    return int(counted.stdout.strip() or 0)


def verified_build(args, sha, stamp, lines, out, pip_python, pip_info):
    required = cpu_features(args.cpu) if args.cpu else None
    wheel, compiled = build(args.gam, args.target_dir, out, args.maturin, args.jobs, args.build_timeout, args.profile,
                            args.cpu)
    print(f"BUILT {wheel.name} in {compiled['seconds']} s: {compiled['rustc_calls']} rustc calls, "
          f"{compiled['fresh_units']} fresh units", flush=True)
    guard = add_cpu_guard(wheel, args.cpu, required) if args.cpu else None
    extension = inspect_extension(wheel, out, args.profile)
    if args.cpu:
        extension["fma_instructions"] = fma_instructions(wheel, out)
        if not extension["fma_instructions"]:
            raise SystemExit(f"a target-cpu={args.cpu} engine with no hardware FMA instruction was not built for it")
    if compiled.setdefault("flags_engine_sha256", extension["engine_sha256"]) != extension["engine_sha256"]:
        raise SystemExit("gam_pyffi was fresh, but the saved flags belong to a different extension")
    write_json(out / "compile.json", {"final_rustc_flags": compiled["final_rustc_flags"],
                                      "engine_sha256": extension["engine_sha256"]})
    provenance = {"gam_commit": sha, "profile": args.profile, "policy": POLICY, "python": PYTHON,
                  "wheel": str(wheel), "wheel_sha256": sha256(wheel), "engine_sha256": extension["engine_sha256"],
                  "rustc": run(["rustc", "--version"]).strip(), "maturin": run([args.maturin, "--version"]).strip(),
                  "build": compiled, "extension": extension, "dependencies": stamp, "cpu_guard": guard,
                  "resolver": pip_info}
    tar = resolve(wheel, args.deps, lines, out, provenance, archive=args.profile == "release-pypi",
                  pip_python=pip_python)
    if tar:
        provenance["wheelhouse"] = {"path": str(tar), "sha256": sha256(tar)}
    return wheel, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gam", type=Path, required=True, help="existing gam clone to check out and build")
    parser.add_argument("--ref", required=True, help="gam commit to pin, or 'main' for GitHub main")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="release-pypi")
    parser.add_argument("--target-dir", type=Path, required=True, help="warm cargo target")
    parser.add_argument("--out", type=Path, required=True, help="artifact root; each build gets <out>/<sha>[-quick]")
    parser.add_argument("--deps", type=Path, required=True, help="dependency wheel cache, reused across pins")
    parser.add_argument("--requirements", type=Path, default=Path(__file__).with_name("requirements.txt"))
    parser.add_argument("--maturin", default="maturin")
    parser.add_argument("--jobs", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--build-timeout", type=int, default=2400)
    parser.add_argument("--venv-root", type=Path, help="publish venv-<sha>-<profile> here and point its venv symlink at it")
    parser.add_argument("--no-link", action="store_true", help="publish without moving the venv symlink")
    parser.add_argument("--note", help="caveat recorded in the build's provenance and at the top of its VENV.md")
    parser.add_argument("--tool", action="append", default=[],
                        help="build-host package for the base venv only (e.g. miniwdl), never in the wheelhouse")
    parser.add_argument("--unstripped", action="store_true",
                        help="release-pypi: also relink with symbols kept, as <out>/<sha>/unstripped/, for profiling")
    parser.add_argument("--reuse", action="store_true",
                        help="skip the build: reuse this pin's verified wheel recorded in its PROVENANCE.json")
    parser.add_argument("--cpu", help="compile for this target-cpu (e.g. x86-64-v3); the wheel then refuses, at import, "
                                      "a CPU without its features; build it in its own --target-dir")
    args = parser.parse_args()
    if f"{sys.version_info.major}.{sys.version_info.minor}" != PYTHON:
        raise SystemExit(f"run under CPython {PYTHON}, the AoU runtime's version")
    if args.unstripped and args.profile != "release-pypi":
        raise SystemExit("--unstripped relinks release-pypi; quick builds keep their symbols already")
    lines = requirements(args.requirements)
    # Before any checkout or cargo work, so a host whose resolver cannot run pip refuses in seconds.
    pip_python, pip_info = resolver(args.out)
    print(f"RESOLVER {json.dumps(pip_info)}", flush=True)
    sha = checkout(args.gam, args.ref)
    print(f"PIN gam {sha} profile {args.profile}", flush=True)
    stamp = download_dependencies(lines, args.deps, pip_python)
    variant = f"-{args.cpu.removeprefix('x86-64-')}" if args.cpu else ""
    out = args.out / (sha[:10] + ("" if args.profile == "release-pypi" else f"-{args.profile}") + variant)
    record = out / "PROVENANCE.json"
    if args.reuse:
        provenance = json.loads(record.read_text())
        wheel = Path(provenance["wheel"])
        if (provenance["gam_commit"], provenance["profile"], provenance["dependencies"],
                (provenance.get("cpu_guard") or {}).get("cpu")) != (sha, args.profile, stamp, args.cpu) \
                or sha256(wheel) != provenance["wheel_sha256"]:
            raise SystemExit(f"{record} does not describe this pin, profile and dependency set")
        print(f"REUSED {wheel} (engine {provenance['engine_sha256']})", flush=True)
    else:
        wheel, provenance = verified_build(args, sha, stamp, lines, out, pip_python, pip_info)
        write_json(record, provenance)
    if args.note:
        provenance["note"] = args.note
    if args.venv_root:
        base = ensure_base(args.venv_root, args.deps, stamp, lines, provenance["resolved"], args.tool)
        name = f"venv-{sha[:10]}{PROFILES[args.profile]['suffix']}{variant}"
        provenance["venv"] = publish(args.venv_root, base, wheel, name, provenance, args.jobs, not args.no_link)
        write_json(record, provenance)
    print("PUBLISHED " + json.dumps({key: provenance.get(key) for key in ("gam_commit", "profile", "engine_sha256",
                                                                       "wheelhouse", "venv")}), flush=True)
    if args.unstripped:
        provenance["unstripped"] = unstripped_twin(args.gam, args.target_dir, out, args.maturin, args.jobs,
                                                   args.build_timeout, wheel, args.cpu)
        write_json(record, provenance)
        print("UNSTRIPPED " + json.dumps(provenance["unstripped"]), flush=True)
    failed = [key for key, good in (("venv", provenance.get("venv", {"ok": True})["ok"]),
                                    ("unstripped", provenance.get("unstripped", {"same_code": True})["same_code"]))
              if not good]
    if failed:
        raise SystemExit(f"{failed} failed: {json.dumps({key: provenance[key] for key in failed}, default=str)[:3000]}")


if __name__ == "__main__":
    main()
