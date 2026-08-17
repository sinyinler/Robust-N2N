import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from train_masked import (
    capture_rng_state,
    position_onecycle_after_steps,
    restore_rng_state,
    validate_resume_args,
)
from train_n2n import build_onecycle


class MaskedResumeTest(unittest.TestCase):
    def test_positions_onecycle_at_completed_global_step(self):
        args = SimpleNamespace(
            epochs=10,
            lr_max=0.01,
            lr_final=0.0005,
            warmup_pct=0.1,
        )
        reference_parameter = torch.nn.Parameter(torch.tensor(1.0))
        reference_optimizer = torch.optim.AdamW([reference_parameter], lr=args.lr_max)
        reference_scheduler = build_onecycle(reference_optimizer, 7, args)
        for _ in range(23):
            reference_optimizer.step()
            reference_scheduler.step()

        resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
        resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=args.lr_max)
        resumed_scheduler = build_onecycle(resumed_optimizer, 7, args)
        position_onecycle_after_steps(resumed_scheduler, 23)

        self.assertEqual(resumed_scheduler.last_epoch, reference_scheduler.last_epoch)
        self.assertAlmostEqual(
            resumed_scheduler.get_last_lr()[0],
            reference_scheduler.get_last_lr()[0],
            places=14,
        )

    def test_rng_state_round_trip(self):
        random.seed(187)
        np.random.seed(187)
        torch.manual_seed(187)
        train_loader = SimpleNamespace(generator=torch.Generator().manual_seed(10_188))
        val_loader = SimpleNamespace(generator=torch.Generator().manual_seed(10_189))
        mask_generator = torch.Generator().manual_seed(20_188)
        noise_generator = torch.Generator().manual_seed(40_188)
        extra_generators = {
            "hybrid_gaussian": torch.Generator().manual_seed(50_188),
            "hybrid_choice": torch.Generator().manual_seed(60_188),
        }

        state = capture_rng_state(
            train_loader,
            val_loader,
            mask_generator,
            noise_generator,
            extra_generators,
        )
        expected = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
            float(torch.rand((), generator=train_loader.generator)),
            float(torch.rand((), generator=val_loader.generator)),
            float(torch.rand((), generator=mask_generator)),
            float(torch.rand((), generator=noise_generator)),
            float(torch.rand((), generator=extra_generators["hybrid_gaussian"])),
            float(torch.rand((), generator=extra_generators["hybrid_choice"])),
        )

        restore_rng_state(
            state,
            train_loader,
            val_loader,
            mask_generator,
            noise_generator,
            extra_generators,
        )
        actual = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
            float(torch.rand((), generator=train_loader.generator)),
            float(torch.rand((), generator=val_loader.generator)),
            float(torch.rand((), generator=mask_generator)),
            float(torch.rand((), generator=noise_generator)),
            float(torch.rand((), generator=extra_generators["hybrid_gaussian"])),
            float(torch.rand((), generator=extra_generators["hybrid_choice"])),
        )
        self.assertEqual(actual, expected)

    def test_rejects_critical_configuration_mismatch(self):
        args = SimpleNamespace(epochs=100, batch_size=6)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            validate_resume_args(
                {"epochs": 100, "batch_size": 8},
                args,
            )


if __name__ == "__main__":
    unittest.main()
