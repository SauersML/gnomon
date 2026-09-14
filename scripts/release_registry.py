"""Check synchronized release versions and skip already-published artifacts."""

import ast
import json
import os
from pathlib import Path
import sys
import tomllib
from urllib.error import HTTPError
from urllib.request import Request, urlopen

root = Path(__file__).resolve().parent.parent
core = tomllib.loads((root / "Cargo.toml").read_text())["package"]
python = tomllib.loads((root / "python/pyproject.toml").read_text())["project"]
native = tomllib.loads((root / "python/native/Cargo.toml").read_text())["package"]
bindings = ast.parse((root / "python/gnomon/__init__.py").read_text())
binding_versions = [
    ast.literal_eval(node.value)
    for node in bindings.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
]
version = core["version"]
assert binding_versions == [version], "Python import version differs from Cargo"
assert python["version"] == native["version"] == version, "Release versions differ"
assert core["name"] == python["name"] == "gnomon-pgs", "Unexpected registry package"
assert sys.argv[1:] in [["crates"], ["pypi"]], "Expected crates or pypi"
registry = sys.argv[1]
url = (
    "https://index.crates.io/gn/om/gnomon-pgs"
    if registry == "crates"
    else "https://pypi.org/pypi/gnomon-pgs/json"
)
try:
    with urlopen(Request(url, headers={"User-Agent": "SauersML/gnomon release workflow"}), timeout=30) as response:
        payload = response.read().decode()
except HTTPError as error:
    if error.code != 404:
        raise
    published = False
else:
    if registry == "crates":
        published = any(json.loads(line)["vers"] == version for line in payload.splitlines())
    else:
        published = bool(json.loads(payload)["releases"].get(version))

result = f"version={version}\npublish={str(not published).lower()}\n"
print(f"{registry}: {core['name']} {version}, already published: {published}")
if "GITHUB_OUTPUT" in os.environ:
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(result)
