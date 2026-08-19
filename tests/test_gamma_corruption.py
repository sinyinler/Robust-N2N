import unittest

import torch

from losses.masked_prediction import apply_local_gamma_noise


class GammaCorruptionTest(unittest.TestCase):
    def test_only_changes_selected_region_and_is_deterministic(self):
        raw = torch.linspace(1.0, 20.0, steps=2 * 16 * 16).reshape(2, 1, 16, 16)
        log_image = torch.log1p(raw)
        visible = torch.ones((2, 1, 16, 16))
        visible[:, :, 4:12, 4:12] = 0.0

        first_generator = torch.Generator().manual_seed(40_188)
        second_generator = torch.Generator().manual_seed(40_188)
        first, first_cvs = apply_local_gamma_noise(
            log_image, visible, 0.025, 0.075, generator=first_generator
        )
        second, second_cvs = apply_local_gamma_noise(
            log_image, visible, 0.025, 0.075, generator=second_generator
        )

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first_cvs, second_cvs))
        self.assertTrue(torch.equal(first * visible, log_image * visible))
        self.assertGreater(float(((first - log_image).abs() * (1.0 - visible)).sum()), 0.0)
        self.assertTrue(bool(((first_cvs >= 0.025) & (first_cvs <= 0.075)).all()))

    def test_mean_one_gamma_is_raw_domain_unbiased_in_large_sample(self):
        raw = torch.full((64, 1, 64, 64), 10.0)
        log_image = torch.log1p(raw)
        visible = torch.zeros((64, 1, 64, 64))
        generator = torch.Generator().manual_seed(187)

        corrupted, cvs = apply_local_gamma_noise(
            log_image, visible, 0.05, 0.05, generator=generator
        )
        raw_ratio = torch.expm1(corrupted).mean() / raw.mean()

        self.assertTrue(torch.equal(cvs, torch.full_like(cvs, 0.05)))
        self.assertAlmostEqual(float(raw_ratio), 1.0, delta=1e-3)
        self.assertGreaterEqual(float(corrupted.min()), 0.0)

if __name__ == "__main__":
    unittest.main()
