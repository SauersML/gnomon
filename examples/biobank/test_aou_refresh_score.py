"""Exact component handoff from the native scorer, without participant data."""
import pytest

from aou_refresh_score import component_scores


def test_components_preserve_denominator_and_exclude_unobserved(tmp_path):
    path = tmp_path / "score.sscore"
    path.write_text("#SCORE_VARIANT_COUNT\tSCORE\tCOUNT\n"
                    "#SCORE_VARIANT_COUNT\tPGS004525\t4\n"
                    "#IID\tPGS004525_SUM\tPGS004525_MISSING_CT\n"
                    "observed\t6\t1\nmissing\t0\t4\nzero\t0\t0\n")
    scores, total, excluded, ids = component_scores(path, "PGS004525")
    assert total == 4 and excluded == 1
    assert ids == {"observed", "missing", "zero"}
    assert scores["#IID"].tolist() == ["observed", "zero"]
    assert scores["PGS004525_AVG"].tolist() == [2., 0.]
    assert scores["PGS004525_MISSING_PCT"].tolist() == [25., 0.]


@pytest.mark.parametrize("missing", ["-1", "5", "1.5", "nan"])
def test_invalid_native_missing_counts_fail(tmp_path, missing):
    path = tmp_path / "score.sscore"
    path.write_text("#SCORE_VARIANT_COUNT\tSCORE\tCOUNT\n"
                    "#SCORE_VARIANT_COUNT\tPGS004525\t4\n"
                    "#IID\tPGS004525_SUM\tPGS004525_MISSING_CT\n"
                    f"sample\t0\t{missing}\n")
    with pytest.raises(ValueError):
        component_scores(path, "PGS004525")
