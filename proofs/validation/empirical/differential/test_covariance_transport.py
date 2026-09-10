"""Exercise the source-extracted covariance constructors in their genotype units."""

import math
import unittest
from functools import cache

import corpus
import emit
import lean_defs


@cache
def extracted(name):
    # The scalar API's random self-check rejects function-valued matrix outputs.
    # Compile their parsed bodies with the same resolver as the emitter, then
    # evaluate every matrix entry against the explicit fixtures below.
    definition = corpus.api.definition(name)
    source, _ = emit.translate_in_context(context(), definition, fname="tested")
    namespace = dict(vars(lean_defs))
    exec(compile(source, "<covariance-constructor>", "exec"), namespace)
    return namespace["tested"]


@cache
def context():
    return emit.build_context({
        "definitions": list(corpus.api.definition_table().values()),
        "structures": list(corpus.api.structures().values()),
    })


def popgen(mu):
    record = {"Ne": 1.0, "μ": mu, "mig": 0.0, "recomb": 0.0, "V_A": 1.0}
    for name in ("fstTransientAt", "mutationSharedRetentionAt", "migrationSharedBoostAt"):
        fn = extracted("Calibrator.GenerationalPopGenParameters." + name)
        record[name] = lambda t, fn=fn: fn(record, t)
    return record


def model(source_frequency, target_frequency, mu=0.0):
    source_variance = 2 * source_frequency * (1 - source_frequency)
    return {
        "popGen": popgen(mu),
        "tagAlleleFreqSource": lambda i: source_frequency,
        "causalAlleleFreqSource": lambda j: source_frequency,
        "tagAlleleFreqStandingTargetAt": lambda t, i: target_frequency,
        "causalAlleleFreqStandingTargetAt": lambda t, j: target_frequency,
        "tagAlleleFreqMutationShiftAt": lambda t, i: 0.0,
        "causalAlleleFreqMutationShiftAt": lambda t, j: 0.0,
        "tagDistance": lambda i, j: 0.0,
        "tagCausalDistance": lambda i, j: 0.0,
        "sigmaTagSource": lambda i, j: source_variance,
        "directCausalSource": lambda i, j: source_variance,
        "proxyTaggingSource": lambda i, j: 0.8 * source_variance,
        "novelDirectCausalCovarianceTemplateAt": lambda t, i, j: 0.1 + 0.05 * t,
        "novelProxyTaggingCovarianceTemplateAt": lambda t, i, j: -(0.1 + 0.05 * t),
    }


class CovarianceTransportTests(unittest.TestCase):
    def test_shared_constructors_preserve_variance_and_correlation(self):
        for source, target in ((0.2, 0.5), (0.5, 0.2), (0.184, 0.338),
                               (0.748, 0.458), (0.212, 0.668), (0.266, 1.0),
                               (0.478, 0.0), (0.5, 0.5)):
            with self.subTest(source=source, target=target):
                record = model(source, target)
                variance = extracted("sigmaTagTargetAt")(record, 1)(0, 0)
                direct = extracted("directCausalTargetAt")(record, 1)(0, 0)
                proxy = extracted("proxyTaggingTargetAt")(record, 1)(0, 0)
                self.assertAlmostEqual(variance, 2 * target * (1 - target))
                self.assertAlmostEqual(direct, variance)
                self.assertAlmostEqual(proxy, 0.8 * variance)
                self.assertGreaterEqual(variance - abs(proxy), -1e-14)

    def test_unequal_coordinate_scales_preserve_cross_covariance(self):
        for target in ((0.5, 0.3), (0.1, 0.5), (0.0, 0.5), (0.5, 1.0)):
            for rho in (-1.0, 0.8, 1.0):
                for source_standardized in (False, True):
                    with self.subTest(target=target, rho=rho,
                                      source_standardized=source_standardized):
                        source = (0.2, 0.5)
                        source_var = [2 * p * (1 - p) for p in source]
                        target_var = [2 * p * (1 - p) for p in target]
                        scales = source_var if source_standardized else (1.0, 1.0)
                        source_cov = rho * math.sqrt(
                            source_var[0] * source_var[1] / (scales[0] * scales[1]))
                        expected = rho * math.sqrt(
                            target_var[0] * target_var[1] / (scales[0] * scales[1]))
                        record = model(source[0], target[0])
                        record.update({
                            "tagAlleleFreqSource": lambda i: source[i],
                            "tagAlleleFreqStandingTargetAt": lambda t, i: target[i],
                            "causalAlleleFreqSource": lambda j: source[1],
                            "causalAlleleFreqStandingTargetAt": lambda t, j: target[1],
                            "sigmaTagSource": lambda i, j: (
                                source_var[i] / scales[i] if i == j else source_cov),
                            "directCausalSource": lambda i, j: source_cov,
                            "proxyTaggingSource": lambda i, j: source_cov,
                        })
                        sigma = extracted("sigmaTagTargetAt")(record, 1)
                        for i in (0, 1):
                            self.assertAlmostEqual(sigma(i, i), target_var[i] / scales[i])
                        self.assertAlmostEqual(sigma(0, 1), expected)
                        self.assertAlmostEqual(sigma(1, 0), expected)
                        for name in ("directCausalTargetAt", "proxyTaggingTargetAt"):
                            covariance = extracted(name)(record, 1)(0, 0)
                            self.assertAlmostEqual(covariance, expected)
                            self.assertGreaterEqual(
                                sigma(0, 0) * sigma(1, 1) - covariance ** 2, -1e-14)

    def test_novel_target_covariance_survives_source_fixation(self):
        for source_frequency in (0.0, 0.5, 1.0):
            record = model(source_frequency, 0.5, math.log(2) / 2)
            for t in (0, 1, 2):
                expected = (0.1 + 0.05 * t) * (1 - 2 ** -t)
                direct = extracted("novelDirectCausalTargetAt")(record, t)(0, 0)
                proxy = extracted("novelProxyTaggingTargetAt")(record, t)(0, 0)
                self.assertAlmostEqual(direct, expected)
                self.assertAlmostEqual(proxy, -expected)
                self.assertTrue(math.isfinite(direct))
                self.assertLessEqual(abs(direct), 0.1 + 0.05 * t)
                if t > 0:
                    self.assertGreater(direct, 0.0)


if __name__ == "__main__":
    unittest.main()
