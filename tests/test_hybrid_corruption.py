import unittest

import torch

from losses.masked_prediction import (
    apply_local_hybrid_noise,
    make_block_visible_mask,
    split_hidden_regions_by_patch,
)


class HybridCorruptionTest(unittest.TestCase):
    def setUp(self):
        raw = torch.linspace(1.0, 40.0, steps=2 * 32 * 32).reshape(2, 1, 32, 32)
        self.log_image = torch.log1p(raw)
        self.visible = make_block_visible_mask(
            2,
            32,
            32,
            ratio=0.5,
            patch=4,
            device=raw.device,
            dtype=raw.dtype,
            generator=torch.Generator().manual_seed(20_043),
        )

    def test_patch_split_is_disjoint_complete_and_deterministic(self):
        first = split_hidden_regions_by_patch(
            self.visible,
            patch=4,
            gamma_probability=0.5,
            generator=torch.Generator().manual_seed(60_043),
        )
        second = split_hidden_regions_by_patch(
            self.visible,
            patch=4,
            gamma_probability=0.5,
            generator=torch.Generator().manual_seed(60_043),
        )
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))

        gamma_visible, gaussian_visible, fractions = first
        hidden = 1.0 - self.visible
        gamma_hidden = 1.0 - gamma_visible
        gaussian_hidden = 1.0 - gaussian_visible
        self.assertTrue(torch.equal(gamma_hidden * gaussian_hidden, torch.zeros_like(hidden)))
        self.assertTrue(torch.equal(gamma_hidden + gaussian_hidden, hidden))
        self.assertTrue(bool(((fractions > 0.0) & (fractions < 1.0)).all()))

    def test_mixture_changes_only_selected_region(self):
        corrupted, cvs, sigmas, fractions = apply_local_hybrid_noise(
            self.log_image,
            self.visible,
            patch=4,
            gamma_probability=0.5,
            gamma_cv_min=0.025,
            gamma_cv_max=0.075,
            gaussian_sigma_min=0.02,
            gaussian_sigma_max=0.06,
            strategy="mixture",
            gamma_generator=torch.Generator().manual_seed(40_043),
            gaussian_generator=torch.Generator().manual_seed(50_043),
            choice_generator=torch.Generator().manual_seed(60_043),
        )
        hidden = 1.0 - self.visible
        self.assertTrue(torch.equal(corrupted * self.visible, self.log_image * self.visible))
        self.assertGreater(float(((corrupted - self.log_image).abs() * hidden).sum()), 0.0)
        self.assertTrue(bool(((cvs >= 0.025) & (cvs <= 0.075)).all()))
        self.assertTrue(bool(((sigmas >= 0.02) & (sigmas <= 0.06)).all()))
        self.assertTrue(bool(((fractions > 0.0) & (fractions < 1.0)).all()))

    def test_sequential_is_deterministic_and_uses_all_selected_patches(self):
        def run():
            return apply_local_hybrid_noise(
                self.log_image,
                self.visible,
                patch=4,
                gamma_probability=0.5,
                gamma_cv_min=0.0176776695,
                gamma_cv_max=0.0530330086,
                gaussian_sigma_min=0.0141421356,
                gaussian_sigma_max=0.0424264069,
                strategy="sequential",
                gamma_generator=torch.Generator().manual_seed(40_043),
                gaussian_generator=torch.Generator().manual_seed(50_043),
            )

        first = run()
        second = run()
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(first[3], torch.ones_like(first[3])))
        self.assertTrue(torch.equal(first[0] * self.visible, self.log_image * self.visible))

    def test_rejects_invalid_strategy(self):
        with self.assertRaisesRegex(ValueError, "unsupported hybrid strategy"):
            apply_local_hybrid_noise(
                self.log_image,
                self.visible,
                patch=4,
                gamma_probability=0.5,
                gamma_cv_min=0.025,
                gamma_cv_max=0.075,
                gaussian_sigma_min=0.02,
                gaussian_sigma_max=0.06,
                strategy="unknown",
            )


if __name__ == "__main__":
    unittest.main()
