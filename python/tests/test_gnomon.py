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


def _argv_log_path(fake_binary: Path) -> Path:
    """Where a fake binary records the argv it was called with."""
    return fake_binary.parent / "argv.json"


_SCORE_FAKE = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, pathlib, sys

    argv = sys.argv[1:]
    log = pathlib.Path(sys.argv[0]).parent / 'argv.json'
    log.write_text(json.dumps(argv))

    assert argv[0] == 'score', f'expected score subcmd, got {argv[0]!r}'
    score_arg = argv[1]
    input_path = pathlib.Path(argv[2])
    parent = input_path.parent or pathlib.Path('.')

    # Mirror score::main::score_output_stem.
    COMPOUND = ('.vcf.bgz', '.vcf.gz', '.bcf.bgz', '.bcf.gz')
    SIMPLE = ('.bed', '.vcf', '.bcf')
    name = input_path.name
    lower = name.lower()
    stem = name
    for s in COMPOUND + SIMPLE:
        if lower.endswith(s) and len(name) > len(s):
            stem = name[:-len(s)]
            break

    def fnv1a64_hex8(b):
        h = 0xCBF29CE484222325
        for ch in b:
            h ^= ch
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return f'{h:016x}'[-8:]

    spath = pathlib.Path(score_arg)
    if not spath.exists() and 'PGS' in score_arg:
        ids = [p.split('|', 1)[0].strip() for p in score_arg.split(',')]
        ids = [i for i in ids if i.startswith('PGS')]
        suffix = f'pgs{len(ids)}_{fnv1a64_hex8(score_arg.encode())}'
        out = parent / f'{stem}_{suffix}.sscore'
    elif spath.is_dir():
        out = parent / f'{stem}.sscore'
    else:
        out = parent / f'{stem}_{spath.stem}.sscore'

    parts = [p.strip() for p in score_arg.split(',')]

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


# Mirror of the *real* gnomon CLI sex-output contract:
#   * filename is `<stem>.sex.tsv`  (NOT `<stem>_sex.tsv`)
#     where `<stem>` strips compound suffixes like `.vcf.gz`
#     (see `map/io.rs::derive_output_stem_name` + `append_output_filename`).
#   * the TSV header has 12 columns: IID Build Sex <metrics...>
#     (see `terms/sex.rs::SEX_TSV_HEADER`).
# Keeping this fake faithful is the wrapper<->CLI contract test: if the Rust
# output naming/columns change, these tests must change in lockstep.
_SEX_TSV_HEADER = (
    "IID\\tBuild\\tSex\\tY_Density\\tX_AutoHet_Ratio\\tComposite_Index\\t"
    "Auto_Valid\\tAuto_Het\\tX_NonPAR_Valid\\tX_NonPAR_Het\\t"
    "Y_NonPAR_Valid\\tY_PAR_Valid"
)

# Stem-derivation helper matching map/io.rs::derive_output_stem_name, defined as
# a function inside the fake. Kept as real source lines (not an injected string)
# so indentation stays valid.
_STEM_FN = textwrap.dedent(
    """\
    def stem_of(p):
        name = p.name; low = name.lower()
        for s in ['.vcf.bgz','.vcf.gz','.bcf.bgz','.bcf.gz','.bed','.bim','.fam','.pgen','.pvar','.psam','.vcf','.bcf']:
            if low.endswith(s) and len(name) > len(s):
                return name[:-len(s)]
        return pathlib.Path(name).stem or 'dataset'
    """
)

_TERMS_FAKE = (
    "#!/usr/bin/env python3\n"
    "import os, pathlib, sys\n"
    + _STEM_FN
    + textwrap.dedent(
        """\
        argv = sys.argv[1:]
        assert argv[0] == 'terms'
        gp = pathlib.Path(argv[1])
        parent = gp.parent or pathlib.Path('.')
        out = parent / (stem_of(gp) + '.sex.tsv')
        with open(out, 'w') as f:
            f.write('{header}\\n')
            f.write('S1\\tGRCh38\\tfemale\\t0.012\\t0.350\\t-1.20\\t1900\\t620\\t40\\t12\\t2\\t1\\n')
            f.write('S2\\tGRCh38\\tmale\\t0.640\\t0.080\\t1.45\\t1950\\t90\\t38\\t1\\t30\\t1\\n')
        print('Sex inference results written to ' + str(out), file=sys.stderr)
        """
    ).format(header=_SEX_TSV_HEADER)
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


def test_score_table_indexable_by_iid(tmp_path):
    """ScoreTable supports `iid in t`, `t[iid]`, and `score_for`."""
    p = tmp_path / "out.sscore"
    _write_sscore(
        p,
        ["#FID", "IID", "PGS001_AVG", "PGS001_SUM", "PGS002_AVG", "PGS002_SUM"],
        [
            ["F", "S1", "0.10", "5.0", "0.20", "10.0"],
            ["F", "S2", "0.15", "7.5", "0.25", "12.5"],
        ],
    )
    t = read_sscore(p)
    assert "S1" in t and "S2" in t and "S3" not in t
    assert t.index_of("S2") == 1
    assert t["S1"]["PGS001"]["avg"] == 0.10
    assert t["S1"]["PGS002"]["sum"] == 10.0
    assert t.score_for("S2", "PGS001") == 0.15
    assert t.score_for("S2", "PGS002", kind="sum") == 12.5
    with pytest.raises(KeyError):
        t["nope"]
    with pytest.raises(KeyError):
        t.score_for("S1", "PGS999")
    with pytest.raises(ValueError):
        t.score_for("S1", "PGS001", kind="garbage")


def test_score_table_missing_column_kind_raises(tmp_path):
    p = tmp_path / "out.sscore"
    _write_sscore(p, ["#FID", "IID", "PGS001_AVG"], [["F", "S1", "1.5"]])
    t = read_sscore(p)
    with pytest.raises(KeyError, match="'sum'"):
        t.score_for("S1", "PGS001", kind="sum")


def test_read_sscore_missing_iid_raises(tmp_path):
    p = tmp_path / "bad.sscore"
    p.write_text("#FID\tSAMPLE\tSCORE_AVG\nF\tS\t1.0\n")
    with pytest.raises(SscoreParseError):
        read_sscore(p)


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


def test_public_expected_sscore_path():
    from gnomon import expected_sscore_path

    p = expected_sscore_path("/d/arrays.vcf.gz", "PGS001,PGS002", score_exists=False)
    assert p.parent == Path("/d")
    assert p.name.startswith("arrays_pgs2_")
    assert p.name.endswith(".sscore")

    # List/tuple of IDs also accepted.
    p2 = expected_sscore_path("/d/arrays.vcf.gz", ["PGS001", "PGS002"], score_exists=False)
    assert p2 == p


def test_expected_sscore_path_strips_compound_suffixes(tmp_path):
    """Regression: paths like sample.vcf.gz used to be passed through
    with only `.gz` stripped, producing `sample.vcf_<score>.sscore`.

    The naming convention here must stay byte-identical to
    score::main::score_output_stem + inline_pgs_output_suffix in the
    Rust source; otherwise the wrapper looks for the wrong file after
    a successful run.
    """
    from gnomon._api import _expected_sscore_path, _inline_pgs_output_suffix

    # Stem stripping: every input shape should converge to "arrays".
    pgs_suffix = _inline_pgs_output_suffix("PGS001")
    for raw in (
        "arrays",
        "arrays.bed",
        "arrays.vcf",
        "arrays.bcf",
        "arrays.vcf.gz",
        "arrays.bcf.gz",
        "arrays.vcf.bgz",
        "arrays.bcf.bgz",
    ):
        got = _expected_sscore_path(Path("/d") / raw, "PGS001", score_exists=False)
        assert str(got) == f"/d/arrays_{pgs_suffix}.sscore", (raw, got)

    # Multiple inline PGS IDs share a single hash-based suffix.
    multi_suffix = _inline_pgs_output_suffix("PGS001,PGS002")
    got = _expected_sscore_path(Path("/d/arrays"), "PGS001,PGS002", score_exists=False)
    assert str(got) == f"/d/arrays_{multi_suffix}.sscore"

    # Score file path on disk: use the score file's stem.
    score_file = tmp_path / "weights.tsv"
    score_file.write_text("")
    got = _expected_sscore_path(tmp_path / "data" / "arrays.bed", str(score_file))
    assert got.name == "arrays_weights.sscore"


def test_inline_pgs_suffix_matches_rust_format():
    """Sanity-check on the inline_pgs_output_suffix format."""
    from gnomon._api import _inline_pgs_output_suffix

    suffix = _inline_pgs_output_suffix("PGS001,PGS002,PGS003")
    assert suffix.startswith("pgs3_")
    assert len(suffix) == len("pgs3_") + 8  # 8 hex chars

    # Hash is sensitive to argument text — comma spacing matters.
    a = _inline_pgs_output_suffix("PGS001,PGS002")
    b = _inline_pgs_output_suffix("PGS001, PGS002")
    assert a != b


def test_score_with_pgs_id_argv_and_output(tmp_path):
    fake = _fake_binary(tmp_path, _SCORE_FAKE)
    geno = _make_genotype(tmp_path)

    r = score("PGS001,PGS002", geno, binary=fake)
    assert isinstance(r, ScoreResult)
    assert r.output_path.name.startswith(f"{geno.name}_pgs2_")
    assert r.output_path.name.endswith(".sscore")
    assert r.output_path.exists()
    assert r.score_names == ("PGS001", "PGS002")
    assert r.n_samples == 2

    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[0] == "score"
    assert argv[1] == "PGS001,PGS002"
    assert argv[2] == str(geno)


def test_score_with_list_of_ids(tmp_path):
    fake = _fake_binary(tmp_path, _SCORE_FAKE)
    geno = _make_genotype(tmp_path)

    r = score(["PGS001", "PGS002"], geno, binary=fake)
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[1] == "PGS001,PGS002"
    assert r.output_path.exists()


def test_score_with_file_path(tmp_path):
    fake = _fake_binary(tmp_path, _SCORE_FAKE)
    geno = _make_genotype(tmp_path)
    score_file = tmp_path / "myscore.weights.tsv"
    score_file.write_text("# weights\n")

    r = score(score_file, geno, binary=fake)
    # Expected output: geno_myscore.weights.sscore (using the score file stem).
    assert r.output_path.name.endswith(".sscore")
    assert r.output_path.exists()


def test_score_passes_optional_flags(tmp_path):
    fake = _fake_binary(tmp_path, _SCORE_FAKE)
    geno = _make_genotype(tmp_path)
    keep = tmp_path / "keep.txt"
    keep.write_text("S1\nS2\n")

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


def test_score_missing_output_raises(tmp_path):
    # Fake that exits 0 but writes nothing.
    fake = _fake_binary(tmp_path, "#!/usr/bin/env python3\nprint('did nothing')\n")
    geno = _make_genotype(tmp_path)
    with pytest.raises(GnomonFailed):
        score("PGS001", geno, binary=fake)


def test_score_read_output_false_skips_parse(tmp_path):
    fake = _fake_binary(tmp_path, _SCORE_FAKE)
    geno = _make_genotype(tmp_path)
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
    # Sex call comes from the *Sex* column (index 2), not Build (index 1).
    assert r.sex_table == (("S1", "female"), ("S2", "male"))
    # Multiple rows → no single 'inferred_sex'.
    assert r.inferred_sex is None
    assert r.sex_output_path is not None


def test_terms_finds_dotted_sex_tsv_filename(tmp_path):
    """The CLI writes `<stem>.sex.tsv`; the wrapper must resolve that name."""
    fake = _fake_binary(tmp_path, _TERMS_FAKE)
    geno = _make_genotype(tmp_path)
    r = terms(geno, binary=fake)
    assert r.sex_output_path is not None
    assert r.sex_output_path.name == "geno.sex.tsv"
    assert r.sex_output_path.exists()


def test_terms_resolves_sex_tsv_for_compound_vcf_gz(tmp_path):
    """`sample.vcf.gz` -> `sample.sex.tsv` (compound suffix stripped)."""
    fake = _fake_binary(tmp_path, _TERMS_FAKE)
    vcf = tmp_path / "sample.vcf.gz"
    vcf.write_bytes(b"\x1f\x8b\x08\x00")  # gzip magic; content unused by fake
    r = terms(vcf, binary=fake)
    assert r.sex_output_path is not None
    assert r.sex_output_path.name == "sample.sex.tsv"


def test_terms_surfaces_structured_metrics(tmp_path):
    fake = _fake_binary(tmp_path, _TERMS_FAKE)
    geno = _make_genotype(tmp_path)
    r = terms(geno, binary=fake)
    assert r.sex_metrics is not None
    assert len(r.sex_metrics) == 2
    s1, s2 = r.sex_metrics
    assert s1.iid == "S1"
    assert s1.sex is InferredSex.FEMALE
    assert s1.build == "GRCh38"
    assert s1.y_density == 0.012
    assert s1.x_autosome_het_ratio == 0.350
    assert s1.composite_index == -1.20
    assert s1.confidence == -1.20  # alias for composite_index
    assert s1.auto_valid == 1900
    assert s1.auto_het == 620
    assert s1.y_non_par_valid == 2
    assert s2.sex is InferredSex.MALE
    assert s2.composite_index == 1.45


def test_terms_metrics_na_becomes_none(tmp_path):
    body = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import pathlib, sys
        gp = pathlib.Path(sys.argv[2])
        parent = gp.parent or pathlib.Path('.')
        out = parent / (gp.name + '.sex.tsv')
        out.write_text(
            '{header}\\n'
            'S1\\tGRCh38\\tindeterminate\\tNA\\tNA\\tNA\\t0\\t0\\t0\\t0\\t0\\t0\\n'
        )
        """
    ).format(header=_SEX_TSV_HEADER)
    fake = _fake_binary(tmp_path, body)
    geno = tmp_path / "single"
    (tmp_path / "single.bed").write_bytes(b"\x6c\x1b\x01\x00")
    r = terms(geno, binary=fake)
    assert r.metrics is not None
    assert r.metrics.y_density is None
    assert r.metrics.composite_index is None
    assert r.metrics.confidence is None
    assert r.inferred_sex is InferredSex.UNKNOWN  # 'indeterminate' -> UNKNOWN


def test_terms_single_sample_yields_single_call(tmp_path):
    body = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import pathlib, sys
        gp = pathlib.Path(sys.argv[2])
        parent = gp.parent or pathlib.Path('.')
        out = parent / (gp.name + '.sex.tsv')
        out.write_text(
            '{header}\\n'
            'S1\\tGRCh38\\tfemale\\t0.01\\t0.4\\t-1.0\\t100\\t30\\t5\\t2\\t1\\t0\\n'
        )
        """
    ).format(header=_SEX_TSV_HEADER)
    fake = _fake_binary(tmp_path, body)
    geno = _make_genotype(tmp_path)
    r = terms(geno, binary=fake)
    assert r.inferred_sex is InferredSex.FEMALE
    # `metrics` convenience: the sole SexMetrics for a 1-sample run.
    assert r.metrics is not None
    assert r.metrics.sex is InferredSex.FEMALE


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


def test_map_project_passes_model(tmp_path):
    body = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import json, os, pathlib, sys
        log = pathlib.Path(sys.argv[0]).parent / 'argv.json'
        log.write_text(json.dumps(sys.argv[1:]))
        """
    )
    fake = _fake_binary(tmp_path, body)
    geno = _make_genotype(tmp_path)
    gnomon_map.project(geno, model="hwe_1kg_hgdp_gsa_v3", binary=fake)
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[0] == "project"
    assert argv[argv.index("--model") + 1] == "hwe_1kg_hgdp_gsa_v3"


# ---------------------------------------------------------------------------
# map.model_variant_keys()
# ---------------------------------------------------------------------------


# Mirrors the real `gnomon model-keys <name>` CLI contract: JSON on stdout,
# loader chatter on stderr. Shape matches `map::main::model_variant_keys_json`.
_MODEL_KEYS_FAKE = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, pathlib, sys
    argv = sys.argv[1:]
    log = pathlib.Path(sys.argv[0]).parent / 'argv.json'
    log.write_text(json.dumps(argv))
    assert argv[0] == 'model-keys'
    name = argv[1]
    # Loader noise must go to stderr so stdout is pure JSON.
    print('Using cached model: ' + name, file=sys.stderr)
    doc = {
        'model': name,
        'genome_build': 'GRCh38',
        'n_variants': 2,
        'variant_keys': [
            {'chromosome': '1', 'position': 12345, 'alleles': ['A', 'G']},
            {'chromosome': 'X', 'position': 67890},
        ],
    }
    print(json.dumps(doc))
    """
)


def test_model_variant_keys_argv_and_parse(tmp_path):
    fake = _fake_binary(tmp_path, _MODEL_KEYS_FAKE)
    result = gnomon_map.model_variant_keys("hwe_1kg_hgdp_gsa_v3", binary=fake)
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv == ["model-keys", "hwe_1kg_hgdp_gsa_v3"]
    assert result.model == "hwe_1kg_hgdp_gsa_v3"
    assert result.genome_build == "GRCh38"
    assert len(result) == 2
    keys = list(result)
    assert keys[0] == gnomon_map.VariantKey(chromosome="1", position=12345, alleles=("A", "G"))
    assert keys[1].chromosome == "X"
    assert keys[1].position == 67890
    assert keys[1].alleles is None


def test_model_variant_keys_unparseable_raises(tmp_path):
    body = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import sys
        print('not json at all')
        """
    )
    fake = _fake_binary(tmp_path, body)
    with pytest.raises(GnomonFailed):
        gnomon_map.model_variant_keys("whatever", binary=fake)


def test_model_variant_keys_propagates_cli_failure(tmp_path):
    fake = _fake_binary(tmp_path, _FAILING)
    with pytest.raises(GnomonFailed):
        gnomon_map.model_variant_keys("unknown_model", binary=fake)


# ---------------------------------------------------------------------------
# calibrate.train / calibrate.infer
# ---------------------------------------------------------------------------


def test_calibrate_train_argv(tmp_path):
    body = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import json, os, pathlib, sys
        (pathlib.Path(sys.argv[0]).parent / 'argv.json').write_text(json.dumps(sys.argv[1:]))
        """
    )
    fake = _fake_binary(tmp_path, body)
    data = tmp_path / "train.tsv"
    data.write_text("phenotype\tscore\tPC1\n0\t1.0\t0.5\n")
    gnomon_calibrate.train(data, num_pcs=4, binary=fake)
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert argv[0] == "train"
    assert argv[1] == str(data)
    assert argv[argv.index("--num-pcs") + 1] == "4"
    assert argv[argv.index("--model-family") + 1] == "gam"


def test_calibrate_infer_argv(tmp_path):
    body = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import json, os, pathlib, sys
        (pathlib.Path(sys.argv[0]).parent / 'argv.json').write_text(json.dumps(sys.argv[1:]))
        """
    )
    fake = _fake_binary(tmp_path, body)
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


def test_run_all_invokes_all_subcommand(tmp_path):
    body = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import json, os, pathlib, sys
        log = pathlib.Path(sys.argv[0]).parent / 'argv.json'
        log.write_text(json.dumps(sys.argv[1:]))
        argv = sys.argv[1:]
        # Write a real sscore so AllResult.score gets populated. Use the
        # same inline-PGS naming convention the wrapper expects.
        score_arg = argv[1]
        input_p = pathlib.Path(argv[2])
        stem = input_p.stem if input_p.suffix in {'.bed','.vcf','.bcf','.gz'} else input_p.name

        def fnv1a64_hex8(b):
            h = 0xCBF29CE484222325
            for ch in b:
                h ^= ch
                h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
            return f'{h:016x}'[-8:]

        ids = [p.split('|', 1)[0].strip() for p in score_arg.split(',')]
        ids = [i for i in ids if i.startswith('PGS')]
        suffix = f'pgs{len(ids)}_{fnv1a64_hex8(score_arg.encode())}'
        out = input_p.parent / f'{stem}_{suffix}.sscore'
        out.write_text('#FID\\tIID\\tPGS001_AVG\\nFAM\\tS1\\t0.7\\n')
        """
    )
    fake = _fake_binary(tmp_path, body)
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


def test_locate_binary_override(tmp_path):
    fake = _fake_binary(tmp_path, "#!/usr/bin/env python3\nprint('hi')\n")
    assert locate_binary(fake) == fake


def test_locate_binary_override_missing_raises(tmp_path):
    with pytest.raises(GnomonBinaryNotFound):
        locate_binary(tmp_path / "no-such-binary")


def test_locate_binary_not_on_path(monkeypatch, tmp_path):
    # PATH only — no override, nothing on PATH.
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(GnomonBinaryNotFound):
        locate_binary()


def test_error_hierarchy():
    for cls in (GnomonBinaryNotFound, GnomonFailed, InvalidConfig, SscoreParseError):
        assert issubclass(cls, GnomonError)
