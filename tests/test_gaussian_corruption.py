import unittest

import torch

from losses.masked_prediction import apply_local_gaussian_noise, make_block_visible_mask


class GaussianCorruptionTest(unittest.TestCase):
    def test_zero_sigma_keeps_image_exactly_unchanged(self):
        image = torch.linspace(0.0, 4.0, steps=2 * 32 * 32).reshape(2, 1, 32, 32)
        visible = make_block_visible_mask(
            2,
            32,
            32,
            ratio=0.25,
            patch=8,
            device=image.device,
            dtype=image.dtype,
            generator=torch.Generator().manual_seed(20_043),
        )
        corrupted, sigmas = apply_local_gaussian_noise(
            image,
            visible,
            0.0,
            0.0,
            generator=torch.Generator().manual_seed(40_043),
        )

        self.assertTrue(bool(((1.0 - visible).sum() > 0).item()))
        self.assertTrue(torch.equal(sigmas, torch.zeros_like(sigmas)))
        self.assertTrue(torch.equal(corrupted, image))


if __name__ == "__main__":
    unittest.main()
