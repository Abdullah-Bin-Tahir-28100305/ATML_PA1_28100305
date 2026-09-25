# =============================================================================
# task4/run_task4.py
# -----------------------------------------------------------------------------
# TOP-LEVEL ORCHESTRATOR for Task 4, in the order the assignment prescribes:
#   1. Train Vanilla                     (closed-set baseline)
#   2. Train GCSC                        (Vanilla recipe + RandAugment)
#   3. Train PROSER                      (init from Vanilla; placeholders)
#   (optional) Train RPL                 (--include_rpl)
#   4. Extract cached outputs for each fixed checkpoint (score-extraction stage)
#   5. Evaluate: Table A (post-hoc scores), Table B (model comparison),
#      score-distribution figure, failure analysis
#
# HOW TO RUN (from inside task4/):
#   python run_task4.py                    # vanilla + gcsc + proser, then eval
#   python run_task4.py --include_rpl      # also train + evaluate the RPL row
#   python run_task4.py --skip_train       # only extract + evaluate saved ckpts
#
# WHY THIS ORDER: unknowns (CIFAR-100) are touched ONLY in the extract/evaluate
# stages, after every model is trained and selected — enforcing the spec rule
# that unknowns never influence training, selection, score design, or thresholds.
# =============================================================================

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from utils import load_config, ensure_dir
from train import train_one
from extract_outputs import extract_for_checkpoint
from evaluate_osr import evaluate_all

CONFIG_DIR = os.path.join(THIS_DIR, "configs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--include_rpl", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(os.path.join(CONFIG_DIR, "vanilla.yaml"))
    ensure_dir(base_cfg["paths"]["results_dir"])
    ckpt_dir = base_cfg["paths"]["checkpoints_dir"]

    run_names = ["vanilla", "gcsc", "proser"]
    if args.include_rpl:
        run_names.append("rpl")

    # ---- Steps 1-4: train each model ----
    if not args.skip_train:
        # Vanilla first (PROSER initializes from it).
        train_one(load_config(os.path.join(CONFIG_DIR, "vanilla.yaml")), run_name="vanilla")
        # GCSC
        train_one(load_config(os.path.join(CONFIG_DIR, "gcsc.yaml")), run_name="gcsc")
        # PROSER, initialized from the selected Vanilla checkpoint.
        vanilla_ckpt = os.path.join(ckpt_dir, "vanilla.pt")
        train_one(load_config(os.path.join(CONFIG_DIR, "proser.yaml")), run_name="proser",
                  init_checkpoint=vanilla_ckpt)
        if args.include_rpl:
            train_one(load_config(os.path.join(CONFIG_DIR, "rpl.yaml")), run_name="rpl")

    # ---- Step 4: extract cached outputs for each fixed checkpoint ----
    print("\n########## EXTRACT OUTPUTS ##########")
    for name in run_names:
        cfg_map = {"vanilla": "vanilla", "gcsc": "gcsc", "proser": "proser", "rpl": "rpl"}
        cfg = load_config(os.path.join(CONFIG_DIR, f"{cfg_map[name]}.yaml"))
        print(f"-- extracting {name} --")
        extract_for_checkpoint(cfg, name)

    # ---- Step 5: evaluate everything ----
    print("\n########## EVALUATE OSR ##########")
    evaluate_all(base_cfg, include_rpl=args.include_rpl)
    print("\n[all done] see results/ for tables, the figure, and the failure analysis.")


if __name__ == "__main__":
    main()
