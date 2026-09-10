"""Check the simulation oracle and independent BED fixture encoding."""

from decimal import Decimal, localcontext
from pathlib import Path
from subprocess import CompletedProcess

import numpy as np
import pandas as pd
import pytest

import sim_test


def test_hwe_contains_both_homozygotes_and_expected_frequencies():
    variants = pd.DataFrame({"af": [0.0, 1.0, 0.5]})
    genotypes = sim_test.generate_genotypes(variants, np.random.default_rng(42), 100_000)
    assert (genotypes[0] == 0).all()
    assert (genotypes[1] == 2).all()
    np.testing.assert_allclose(np.bincount(genotypes[2], minlength=3) / 100_000, [0.25, 0.5, 0.25], atol=0.01)


def test_hwe_fixed_alleles_handle_the_zero_random_draw():
    class ZeroDraws:
        def random(self, shape):
            return np.zeros(shape)

    actual = sim_test.generate_genotypes(pd.DataFrame({"af": [0.0, 1.0]}), ZeroDraws(), 1)
    np.testing.assert_array_equal(actual, [[0], [2]])


def fixture_rows():
    return pd.DataFrame({"chr": [1, 1], "id": ["1:1", "1:2"], "cm": [0, 0], "pos": [1, 2], "a1": ["A", "C"], "a2": ["G", "T"]})


def test_bed_encoding_matches_known_bytes_with_partial_final_byte(tmp_path):
    genotypes = pd.DataFrame([[0, 1, 2, -1, 2], [-1, 2, 1, 0, -1]])
    prefix = tmp_path / "known"
    sim_test._write_plink_files(prefix, fixture_rows(), list("abcde"), genotypes)
    assert prefix.with_suffix(".bed").read_bytes() == bytes([0x6C, 0x1B, 0x01, 0x78, 0x03, 0x2D, 0x01])


@pytest.mark.parametrize("invalid", [-2, 3, 0.5, np.nan])
def test_bed_rejects_invalid_dosages_without_publishing_files(tmp_path, invalid):
    with pytest.raises(ValueError, match="diploid dosages"):
        sim_test._write_plink_files(tmp_path / "invalid", fixture_rows(), ["a"], pd.DataFrame([[invalid], [0]]))
    assert not list(tmp_path.iterdir())


def test_ground_truth_preserves_cancellation_and_missingness():
    weights = [1e16, 1.0, -1e16]
    variants = pd.DataFrame({"effect_weight": weights, "effect_allele": ["G"] * 3, "alt": ["G"] * 3})
    actual = sim_test.calculate_ground_truth_prs(np.array([[1, -1], [1, 2], [1, -1]]), variants)
    with localcontext() as context:
        context.prec = 256
        expected = float(sum(Decimal.from_float(weight) for weight in weights) / 3)
    assert actual.PRS_AVG.tolist() == [expected, 2.0]


def test_ground_truth_matches_high_precision_oracle_for_both_effect_alleles():
    rng = np.random.default_rng(73)
    genotypes = rng.integers(-1, 3, size=(64, 9))
    weights = rng.normal(size=64)
    alt_effect = rng.random(64) > 0.5
    variants = pd.DataFrame({"effect_weight": weights, "effect_allele": np.where(alt_effect, "G", "A"), "alt": ["G"] * 64})
    actual = sim_test.calculate_ground_truth_prs(genotypes, variants).PRS_AVG.to_numpy()
    expected = []
    with localcontext() as context:
        context.prec = 256
        for sample in genotypes.T:
            terms = []
            for genotype, weight, is_alt in zip(sample, weights, alt_effect):
                if genotype != -1:
                    dosage = int(genotype if is_alt else 2 - genotype)
                    terms.append(Decimal.from_float(float(weight)) * dosage)
            expected.append(float(sum(terms) / len(terms)))
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-16)


def test_truth_calculation_does_not_add_metadata_to_tool_score_file(tmp_path):
    def failed_command(command, label, cwd):
        return CompletedProcess(command, 1, "", "intentional command stub")

    assert not sim_test.run_simple_dosage_test(tmp_path, Path("gnomon"), Path("plink2"), Path("pylink.py"), failed_command)
    assert pd.read_csv(tmp_path / "simple_test.score", sep="\t").columns.tolist() == ["variant_id", "effect_allele", "other_allele", "simple_score"]
