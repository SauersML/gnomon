"""Tests for the gnomon Python wrapper.

A fake CLI script imitates gnomon's argv contract and output-file
conventions so we can exercise every code path without a Rust build.
"""

from __future__ import annotations

import json
import stat
import textwrap
from pathlib import Path

import pytest

from gnomon import (
    GnomonBinaryNotFound,
    GnomonError,
    GnomonFailed,
    InferredSex,
    InvalidConfig,
    ScoreResult,
    ScoreTable,
    SscoreParseError,
    TermsResult,
    locate_binary,
    read_sscore,
    run_all,
    score,
    terms,
)
import gnomon.map as gnomon_map
import gnomon.calibrate as gnomon_calibrate


# ---------------------------------------------------------------------------
# Fake binary plumbing
# ---------------------------------------------------------------------------


def _fake_binary(tmp_path: Path, body: str, name: str = "gnomon") -> Path:
    p = tmp_path / name
    p.write_text(body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return p


_SCORE_FAKE = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, pathlib, sys

    argv = sys.argv[1:]
    log = pathlib.Path(os.environ['GNOMON_ARGV_LOG'])
    log.write_text(json.dumps(argv))

    assert argv[0] == 'score', f'expected score subcmd, got {argv[0]!r}'
    score_arg = argv[1]
    input_path = pathlib.Path(argv[2])
    parent = input_path.parent or pathlib.Path('.')
    stem = input_path.stem if input_path.suffix in {'.bed', '.vcf', '.bcf', '.gz', '.txt'} else input_path.name

    parts = [p.strip() for p in score_arg.split(',')]
    if all(p.upper().startswith('PGS') for p in parts):
        suffix = '-'.join(p.upper() for p in parts)
        out = parent / f'{stem}_{suffix}.sscore'
    else:
        spath = pathlib.Path(score_arg)
        if spath.is_dir():
            out = parent / f'{stem}.sscore'
        else:
            out = parent / f'{stem}_{spath.stem}.sscore'

    # Write a small sscore: 2 samples, 2 scores when 2 IDs given.
    n_scores = max(1, len(parts))
    score_names = parts if all(p.upper().startswith('PGS') for p in parts) else ['SCORE']
    if len(score_names) < n_scores:
        score_names = ['SCORE']

    header = ['#FID', 'IID']
    for s in score_names:
        header += [f'{s}_AVG', f'{s}_SUM', f'{s}_DENOM']

    rows = [
        ['FAM', 'S1'] + [f'{0.1 + i:.3f}' for i in range(len(score_names) * 3)],
        ['FAM', 'S2'] + [f'{0.2 + i:.3f}' for i in range(len(score_names) * 3)],
    ]
    with open(out, 'w') as f:
        f.write('\\t'.join(header) + '\\n')
        for r in rows:
            f.write('\\t'.join(r) + '\\n')
    print(f'Wrote score output to {out}')
    """
)


_FAILING = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import sys
    sys.stderr.write('Error: simulated failure\\n')
    sys.exit(2)
    """
)


_TERMS_FAKE = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import os, pathlib, sys
    argv = sys.argv[1:]
    assert argv[0] == 'terms'
    gp = pathlib.Path(argv[1])
    parent = gp.parent or pathlib.Path('.')
    stem = gp.stem if gp.suffix in {'.bed', '.vcf', '.bcf', '.gz'} else gp.name
    out = parent / f'{stem}_sex.tsv'
    with open(out, 'w') as f:
        f.write('IID\\tSEX\\n')
        f.write('S1\\tfemale\\n')
        f.write('S2\\tmale\\n')
    print(f'Wrote sex table to {out}')
    """
)


def _make_genotype(tmp_path: Path, name: str = "geno") -> Path:
    bed = tmp_path / f"{name}.bed"
    bed.write_bytes(b"\x6c\x1b\x01\x00")
    (tmp_path / f"{name}.bim").write_text("1\trs1\t0\t1\tA\tG\n")
    (tmp_path / f"{name}.fam").write_text("FAM\tS1\t0\t0\t0\t-9\n")
    # Return a path that score()/terms() will recognise (typically the prefix).
    return tmp_path / name


# ---------------------------------------------------------------------------
# Sscore parsing
# ---------------------------------------------------------------------------


def _write_sscore(path: Path, header: list, rows: list) -> None:
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


def test_read_sscore_full_avg_sum_denom(tmp_path):
    p = tmp_path / "out.sscore"
    _write_sscore(
        p,
        ["#FID", "IID", "PGS001_AVG", "PGS001_SUM", "PGS001_DENOM"],
        [
            ["FAM", "S1", "0.10", "5.0", "50"],
            ["FAM", "S2", "0.20", "10.0", "50"],
        ],
    )
    t = read_sscore(p)
    assert t.iids == ("S1", "S2")
    assert t.fids == ("FAM", "FAM")
    assert t.score_names == ("PGS001",)
    assert t.avg == ((0.10,), (0.20,))
    assert t.sum == ((5.0,), (10.0,))
    assert t.denom == ((50.0,), (50.0,))
    assert len(t) == 2


def test_read_sscore_only_avg(tmp_path):
    p = tmp_path / "out.sscore"
    _write_sscore(p, ["#FID", "IID", "PGS001_AVG"], [["F", "S1", "1.5"]])
    t = read_sscore(p)
    assert t.avg == ((1.5,),)
    assert t.sum is None
    assert t.denom is None


def test_read_sscore_handles_missing_values(tmp_path):
    p = tmp_path / "out.sscore"
    _write_sscore(
        p,
        ["#FID", "IID", "PGS001_AVG"],
        [["F", "S1", "NA"], ["F", "S2", ""]],
    )
    t = read_sscore(p)
    import math
    assert math.isnan(t.avg[0][0])
    assert math.isnan(t.avg[1][0])


def test_read_sscore_empty_file_raises(tmp_path):
    p = tmp_path / "empty.sscore"
    p.write_text("")
    with pytest.raises(SscoreParseError):
        read_sscore(p)


def test_read_sscore_missing_iid_raises(tmp_path):
    p = tmp_path / "bad.sscore"
    p.write_text("#FID\tSAMPLE\tSCORE_AVG\nF\tS\t1.0\n")
    with pytest.raises(SscoreParseError):
        read_sscore(p)


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


def test_score_with_pgs_id_argv_and_output(tmp_path, monkeypatch):
    fake = _fake_binary(tmp_path, _SCORE_FAKE)
    geno = _make_genotype(tmp_path)
    monkeypatch.setenv("GNOMON_ARGV_LOG", str(tmp_path / "argv.json"))

    r = score("PGS001,PGS002", geno, binary=fake)
    assert isinstance(r, ScoreResult)
    assert r.output_path == tmp_path / f"{geno.name}_PGS001-PGS002.sscore"
    assert r.output_path.exists()
    assert r.score_names == ("PGS001", "PGS002")
    assert r.n_samples == 2

    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[0] == "score"
    assert argv[1] == "PGS001,PGS002"
    assert argv[2] == str(geno)


def test_score_with_list_of_ids(tmp_path, monkeypatch):
    fake = _fake_binary(tmp_path, _SCORE_FAKE)
    geno = _make_genotype(tmp_path)
    monkeypatch.setenv("GNOMON_ARGV_LOG", str(tmp_path / "argv.json"))

    r = score(["PGS001", "PGS002"], geno, binary=fake)
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[1] == "PGS001,PGS002"
    assert r.output_path.exists()


def test_score_with_file_path(tmp_path, monkeypatch):
    fake = _fake_binary(tmp_path, _SCORE_FAKE)
    geno = _make_genotype(tmp_path)
    score_file = tmp_path / "myscore.weights.tsv"
    score_file.write_text("# weights\n")
    monkeypatch.setenv("GNOMON_ARGV_LOG", str(tmp_path / "argv.json"))

    r = score(score_file, geno, binary=fake)
    # Expected output: geno_myscore.weights.sscore (using the score file stem).
    assert r.output_path.name.endswith(".sscore")
    assert r.output_path.exists()


def test_score_passes_optional_flags(tmp_path, monkeypatch):
    fake = _fake_binary(tmp_path, _SCORE_FAKE)
    geno = _make_genotype(tmp_path)
    keep = tmp_path / "keep.txt"
    keep.write_text("S1\nS2\n")
    monkeypatch.setenv("GNOMON_ARGV_LOG", str(tmp_path / "argv.json"))

    score(
        "PGS001",
        geno,
        binary=fake,
        keep=keep,
        build="38",
        inferred_sex="male",
        extra_args=["--foo", "bar"],
    )
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[argv.index("--keep") + 1] == str(keep)
    assert argv[argv.index("--build") + 1] == "38"
    assert argv[argv.index("--inferred-sex") + 1] == "male"
    assert "--foo" in argv and "bar" in argv


def test_score_propagates_nonzero_exit(tmp_path):
    fake = _fake_binary(tmp_path, _FAILING)
    geno = _make_genotype(tmp_path)
    with pytest.raises(GnomonFailed) as ei:
        score("PGS001", geno, binary=fake)
    assert ei.value.returncode == 2
    assert "simulated failure" in ei.value.stderr


def test_score_missing_output_raises(tmp_path, monkeypatch):
    # Fake that exits 0 but writes nothing.
    fake = _fake_binary(tmp_path, "#!/usr/bin/env python3\nprint('did nothing')\n")
    geno = _make_genotype(tmp_path)
    with pytest.raises(GnomonFailed):
        score("PGS001", geno, binary=fake)


def test_score_read_output_false_skips_parse(tmp_path, monkeypatch):
    fake = _fake_binary(tmp_path, _SCORE_FAKE)
    geno = _make_genotype(tmp_path)
    monkeypatch.setenv("GNOMON_ARGV_LOG", str(tmp_path / "argv.json"))
    r = score("PGS001", geno, binary=fake, read_output=False)
    assert r.output_path.exists()
    assert isinstance(r.scores, ScoreTable)
    assert r.scores.iids == ()


# ---------------------------------------------------------------------------
# terms()
# ---------------------------------------------------------------------------


def test_terms_parses_sex_tsv(tmp_path):
    fake = _fake_binary(tmp_path, _TERMS_FAKE)
    geno = _make_genotype(tmp_path)
    r = terms(geno, binary=fake)
    assert isinstance(r, TermsResult)
    assert r.sex_table == (("S1", "female"), ("S2", "male"))
    # Multiple rows → no single 'inferred_sex'.
    assert r.inferred_sex is None
    assert r.sex_output_path is not None


def test_terms_single_sample_yields_single_call(tmp_path):
    body = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import pathlib, sys
        gp = pathlib.Path(sys.argv[2])
        parent = gp.parent or pathlib.Path('.')
        stem = gp.stem if gp.suffix in {'.bed', '.vcf', '.bcf', '.gz'} else gp.name
        out = parent / f'{stem}_sex.tsv'
        out.write_text('IID\\tSEX\\nS1\\tfemale\\n')
        """
    )
    fake = _fake_binary(tmp_path, body)
    geno = _make_genotype(tmp_path)
    r = terms(geno, binary=fake)
    assert r.inferred_sex is InferredSex.FEMALE


def test_terms_requires_sex_for_now(tmp_path):
    fake = _fake_binary(tmp_path, _TERMS_FAKE)
    geno = _make_genotype(tmp_path)
    with pytest.raises(InvalidConfig):
        terms(geno, sex=False, binary=fake)


# ---------------------------------------------------------------------------
# map.fit / map.project
# ---------------------------------------------------------------------------


def test_map_fit_validates_components(tmp_path):
    fake = _fake_binary(tmp_path, "#!/usr/bin/env python3\nprint('ok')\n")
    geno = _make_genotype(tmp_path)
    with pytest.raises(InvalidConfig):
        gnomon_map.fit(geno, components=0, binary=fake)


def test_map_fit_rejects_window_without_ld(tmp_path):
    fake = _fake_binary(tmp_path, "#!/usr/bin/env python3\nprint('ok')\n")
    geno = _make_genotype(tmp_path)
    with pytest.raises(InvalidConfig):
        gnomon_map.fit(geno, components=5, sites_window=101, binary=fake)


def test_map_project_passes_model(tmp_path, monkeypatch):
    body = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import json, os, pathlib, sys
        log = pathlib.Path(os.environ['GNOMON_ARGV_LOG'])
        log.write_text(json.dumps(sys.argv[1:]))
        """
    )
    fake = _fake_binary(tmp_path, body)
    geno = _make_genotype(tmp_path)
    monkeypatch.setenv("GNOMON_ARGV_LOG", str(tmp_path / "argv.json"))
    gnomon_map.project(geno, model="hwe_1kg_hgdp_gsa_v3", binary=fake)
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[0] == "project"
    assert argv[argv.index("--model") + 1] == "hwe_1kg_hgdp_gsa_v3"


# ---------------------------------------------------------------------------
# calibrate.train / calibrate.infer
# ---------------------------------------------------------------------------


def test_calibrate_train_argv(tmp_path, monkeypatch):
    body = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import json, os, pathlib, sys
        pathlib.Path(os.environ['GNOMON_ARGV_LOG']).write_text(json.dumps(sys.argv[1:]))
        """
    )
    fake = _fake_binary(tmp_path, body)
    monkeypatch.setenv("GNOMON_ARGV_LOG", str(tmp_path / "argv.json"))
    data = tmp_path / "train.tsv"
    data.write_text("phenotype\tscore\tPC1\n0\t1.0\t0.5\n")
    gnomon_calibrate.train(data, num_pcs=4, binary=fake)
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[0] == "train"
    assert argv[1] == str(data)
    assert argv[argv.index("--num-pcs") + 1] == "4"
    assert argv[argv.index("--model-family") + 1] == "gam"


def test_calibrate_infer_argv(tmp_path, monkeypatch):
    body = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import json, os, pathlib, sys
        pathlib.Path(os.environ['GNOMON_ARGV_LOG']).write_text(json.dumps(sys.argv[1:]))
        """
    )
    fake = _fake_binary(tmp_path, body)
    monkeypatch.setenv("GNOMON_ARGV_LOG", str(tmp_path / "argv.json"))
    test = tmp_path / "test.tsv"
    test.write_text("score\tPC1\n1.0\t0.5\n")
    model = tmp_path / "model.toml"
    model.write_text("")
    gnomon_calibrate.infer(test, model=model, binary=fake)
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[0] == "infer"
    assert argv[1] == str(test)
    assert argv[argv.index("--model") + 1] == str(model)


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


def test_run_all_invokes_all_subcommand(tmp_path, monkeypatch):
    body = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import json, os, pathlib, sys
        log = pathlib.Path(os.environ['GNOMON_ARGV_LOG'])
        log.write_text(json.dumps(sys.argv[1:]))
        argv = sys.argv[1:]
        # Write a real sscore so AllResult.score gets populated.
        input_p = pathlib.Path(argv[2])
        stem = input_p.stem if input_p.suffix in {'.bed','.vcf','.bcf','.gz'} else input_p.name
        out = input_p.parent / f'{stem}_PGS001.sscore'
        out.write_text('#FID\\tIID\\tPGS001_AVG\\nFAM\\tS1\\t0.7\\n')
        """
    )
    fake = _fake_binary(tmp_path, body)
    monkeypatch.setenv("GNOMON_ARGV_LOG", str(tmp_path / "argv.json"))
    geno = _make_genotype(tmp_path)
    r = run_all("PGS001", geno, model="hwe_1kg_hgdp_gsa_v3", binary=fake)
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[0] == "all"
    assert argv[argv.index("--model") + 1] == "hwe_1kg_hgdp_gsa_v3"
    assert r.score is not None
    assert r.score.scores.iids == ("S1",)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_locate_binary_env(monkeypatch, tmp_path):
    fake = _fake_binary(tmp_path, "#!/usr/bin/env python3\nprint('hi')\n")
    monkeypatch.setenv("GNOMON_BIN", str(fake))
    assert locate_binary() == fake


def test_locate_binary_not_found(monkeypatch, tmp_path):
    monkeypatch.delenv("GNOMON_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(GnomonBinaryNotFound):
        locate_binary()


def test_error_hierarchy():
    for cls in (GnomonBinaryNotFound, GnomonFailed, InvalidConfig, SscoreParseError):
        assert issubclass(cls, GnomonError)
