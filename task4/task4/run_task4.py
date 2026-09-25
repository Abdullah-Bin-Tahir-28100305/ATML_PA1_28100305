

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

    if not args.skip_train:
        train_one(load_config(os.path.join(CONFIG_DIR, "vanilla.yaml")), run_name="vanilla")
        # GCSC
        train_one(load_config(os.path.join(CONFIG_DIR, "gcsc.yaml")), run_name="gcsc")
        vanilla_ckpt = os.path.join(ckpt_dir, "vanilla.pt")
        train_one(load_config(os.path.join(CONFIG_DIR, "proser.yaml")), run_name="proser",
                  init_checkpoint=vanilla_ckpt)
        if args.include_rpl:
            train_one(load_config(os.path.join(CONFIG_DIR, "rpl.yaml")), run_name="rpl")

    print("\n########## EXTRACT OUTPUTS ##########")
    for name in run_names:
        cfg_map = {"vanilla": "vanilla", "gcsc": "gcsc", "proser": "proser", "rpl": "rpl"}
        cfg = load_config(os.path.join(CONFIG_DIR, f"{cfg_map[name]}.yaml"))
        print(f"-- extracting {name} --")
        extract_for_checkpoint(cfg, name)

    print("\n########## EVALUATE OSR ##########")
    evaluate_all(base_cfg, include_rpl=args.include_rpl)
    print("\n[all done] see results/ for tables, the figure, and the failure analysis.")


if __name__ == "__main__":
    main()
