import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import summarize_level4_k0_k1 as summary


class SummarizeLevel4K0K1Test(unittest.TestCase):
    def write_eval(self, root: Path, tag: str, candidate_psnr: list[float]) -> None:
        out_dir = root / f"level4_seen_{tag}_vs_W1_s42"
        out_dir.mkdir(parents=True)
        with (out_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "frame",
                    "robust_psnr",
                    "robust_mssim",
                    "robust_r",
                    "base_psnr",
                    "base_mssim",
                    "base_r",
                    "dPSNR",
                ],
            )
            writer.writeheader()
            for index, psnr in enumerate(candidate_psnr):
                base_psnr = 30.0 + index
                writer.writerow(
                    {
                        "frame": str(index),
                        "robust_psnr": psnr,
                        "robust_mssim": 0.81 + index * 0.01,
                        "robust_r": 0.91 + index * 0.01,
                        "base_psnr": base_psnr,
                        "base_mssim": 0.80 + index * 0.01,
                        "base_r": 0.90 + index * 0.01,
                        "dPSNR": psnr - base_psnr,
                    }
                )

    def test_writes_paired_summary_and_curve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_eval(root, "K0", [30.5, 30.5])
            self.write_eval(root, "K1", [31.0, 32.0])

            with patch(
                "sys.argv",
                ["summarize_level4_k0_k1.py", "--eval_root", str(root), "--seed", "42"],
            ), redirect_stdout(io.StringIO()):
                summary.main()

            csv_path = root / "K0_K1_vs_W1_s42_summary.csv"
            markdown_path = root / "K0_K1_vs_W1_s42_summary.md"
            curve_path = root / "K0_K1_vs_W1_s42_psnr_curve.png"
            self.assertTrue(csv_path.is_file())
            self.assertTrue(markdown_path.is_file())
            self.assertTrue(curve_path.is_file())

            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                records = {row["method"]: row for row in csv.DictReader(handle)}
            self.assertAlmostEqual(float(records["K1"]["delta_psnr_vs_W1"]), 1.0)
            self.assertEqual(int(records["K1"]["wins_vs_W1"]), 2)
            self.assertAlmostEqual(float(records["K1"]["delta_psnr_vs_K0"]), 1.0)


if __name__ == "__main__":
    unittest.main()
