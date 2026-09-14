"""Reuse the rustc command captured by a verbose warm MSI Cargo build."""
from pathlib import Path
import os
import shlex
import subprocess
import json
import re

root = Path('/projects/standard/hsiehph/sauer354/gnomon/target/score-map')
commands = []
for line in (root/'build-score.log').read_text().splitlines():
    if 'Running `' in line and 'rustc --crate-name gnomon_score' in line:
        commands.append(shlex.split(line.split('Running `', 1)[1].rsplit('`', 1)[0]))
command = commands[-1]
index = next(i for i, arg in enumerate(command) if arg.endswith('/rustc'))
environment = os.environ.copy()
for assignment in command[:index]:
    key, value = assignment.split('=', 1)
    environment[key] = value
args = command[index:]
def build(source, output, extra, library_log=None):
    selected = [args[0]]
    i = 1
    while i < len(args):
        arg = args[i]
        if arg == 'cli/main.rs':
            i += 1
        elif arg in ['--crate-name','--crate-type','--emit','--out-dir','--error-format','--json','--check-cfg']:
            i += 2
        elif arg.startswith(('--emit=', '--error-format=', '--json=')):
            i += 1
        elif arg == '-C' and args[i+1].startswith(('metadata=', 'extra-filename=', 'panic=', 'debuginfo=', 'codegen-units=')):
            i += 2
        else:
            selected.append(arg)
            i += 1
    selected += ['--crate-name', output.replace('-', '_'), str(source), '-o', str(root/output), '-C', 'codegen-units=64'] + extra
    if library_log is not None:
        artifacts = {}
        finished = False
        for line in Path(library_log).read_text().splitlines():
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get('reason') == 'build-finished':
                finished = message['success']
            if message.get('reason') == 'build-script-executed' and 'gnomon-pgs' in message['package_id']:
                environment.update(message['env'])
            if message.get('reason') == 'compiler-artifact':
                metadata = [path for path in message['filenames'] if path.endswith('.rmeta')]
                if metadata:
                    assert len(metadata) == 1 and Path(metadata[0]).is_file(), metadata
                    artifacts[message['target']['name']] = metadata[0]
        if not finished or 'gnomon' not in artifacts:
            raise RuntimeError('A successful Cargo JSON library build is required')
        environment['CARGO_PKG_VERSION'] = re.search(
            r'^version = "([^"]+)"', (root / 'src/Cargo.toml').read_text(), re.MULTILINE
        ).group(1)
        assert 'GNOMON_BUILD_TIMESTAMP' in environment
        for i, arg in enumerate(selected[:-1]):
            if arg == '--extern':
                name, path = selected[i + 1].split('=', 1)
                if name == 'gnomon':
                    selected[i + 1] = name + '=' + str(Path(artifacts[name]).with_suffix(Path(path).suffix))
        for name in ['blake3', 'fs4']:
            for suffix in ['.rlib', '.rmeta']:
                selected += ['--extern', name + '=' + str(Path(artifacts[name]).with_suffix(suffix))]
        for directory in sorted({str(Path(path).parent) for path in artifacts.values()}):
            selected += ['-L', 'dependency=' + directory]
    result = subprocess.run(selected, env=environment, cwd=root/'src', timeout=45)
    if result.returncode:
        raise SystemExit(result.returncode)
