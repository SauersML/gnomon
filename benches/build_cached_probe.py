"""Reuse the rustc command captured by a verbose warm MSI Cargo build."""
from pathlib import Path
import os
import shlex
import subprocess

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
def build(source, output, extra):
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
    subprocess.run(selected, env=environment, cwd=root/'src', check=True, timeout=45)
