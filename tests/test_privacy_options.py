import unittest

from model.cli import parse_args


class PrivacyOptionsTests(unittest.TestCase):
    def test_external_logging_is_off_by_default(self):
        args = parse_args([])
        self.assertFalse(args.wandb)
        self.assertFalse(args.wandb_artifacts)

    def test_metrics_opt_in_does_not_enable_media(self):
        args = parse_args(["--wandb"])
        self.assertTrue(args.wandb)
        self.assertFalse(args.wandb_artifacts)

    def test_artifacts_require_an_explicit_logging_opt_in(self):
        with self.assertRaises(SystemExit):
            parse_args(["--wandb-artifacts"])
        args = parse_args(["--wandb", "--wandb-artifacts"])
        self.assertTrue(args.wandb_artifacts)

    def test_disable_flag_is_incompatible_with_media_upload(self):
        with self.assertRaises(SystemExit):
            parse_args(["--wandb", "--wandb-artifacts", "--no_wandb"])
