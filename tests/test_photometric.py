import unittest

import numpy as np

from utils.photometric import common_center_crop, fit_reference_affine


class PhotometricDiagnosticTest(unittest.TestCase):
    def test_recovers_known_scale_and_offset(self):
        rng = np.random.default_rng(42)
        output = rng.normal(loc=70.0, scale=15.0, size=(32, 40))
        reference = 1.75 * output - 12.5

        fit = fit_reference_affine(output, reference)

        self.assertAlmostEqual(fit.scale, 1.75, places=10)
        self.assertAlmostEqual(fit.offset, -12.5, places=10)
        np.testing.assert_allclose(fit.corrected, reference, atol=1e-10)

    def test_uses_same_common_center_crop_as_metrics(self):
        output = np.arange(8 * 10, dtype=np.float64).reshape(8, 10)
        reference = 2.0 * output[1:7, 2:8] + 3.0

        cropped_output, cropped_reference = common_center_crop(output, reference)
        fit = fit_reference_affine(output, reference)

        self.assertEqual(cropped_output.shape, (6, 6))
        self.assertEqual(cropped_reference.shape, (6, 6))
        self.assertAlmostEqual(fit.scale, 2.0, places=10)
        self.assertAlmostEqual(fit.offset, 3.0, places=10)

    def test_constant_output_falls_back_to_reference_mean(self):
        output = np.full((8, 8), 7.0)
        reference = np.arange(64, dtype=np.float64).reshape(8, 8)

        fit = fit_reference_affine(output, reference)

        self.assertEqual(fit.scale, 0.0)
        self.assertAlmostEqual(fit.offset, float(reference.mean()))
        np.testing.assert_allclose(fit.corrected, reference.mean())


if __name__ == "__main__":
    unittest.main()
