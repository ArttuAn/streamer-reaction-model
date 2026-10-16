"""Training options without importing model runtimes or contacting services."""
import argparse
from pathlib import Path

BASE = Path(__file__).parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train SigLIP emotion predictor")
    parser.add_argument("--dataset_dir", type=Path,  default=BASE.parent / "dataset-assembler" / "dataset")
    parser.add_argument("--out",         type=Path,  default=BASE / "checkpoints")
    parser.add_argument("--model_name",  type=str,   default="google/siglip-base-patch16-224")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--n_frames",    type=int,   default=4)
    parser.add_argument("--workers",     type=int,   default=0)
    parser.add_argument("--val_split",   type=float, default=0.15)
    # wandb
    parser.add_argument("--wandb_project", type=str, default="streamer-reaction-model")
    parser.add_argument("--no_wandb",    action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--wandb", action="store_true", help="Opt in to external metrics and run metadata logging")
    parser.add_argument("--wandb_artifacts", "--wandb-artifacts", action="store_true",
                        help="Also upload dataset media, identifiers, and model checkpoints")
    args = parser.parse_args(argv)
    if args.wandb_artifacts and (not args.wandb or args.no_wandb):
        parser.error("--wandb-artifacts requires --wandb and cannot be combined with --no_wandb")
    return args
