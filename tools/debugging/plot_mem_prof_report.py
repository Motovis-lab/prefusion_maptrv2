import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from copious.io.fs import parent_ensured_path


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--report-path', type=Path, required=True, help='Input file path')
    parser.add_argument('--plot-save-path', type=parent_ensured_path, required=True, help='Input file path')
    parser.add_argument('--ylim', nargs="*", type=float)
    parser.add_argument('--interval', type=int, default=10)
    return parser.parse_args()

def main(args):
    df = pd.read_csv(args.report_path, sep=" ")
    df.columns = ["item", "mem_use", "ts"]
    fig = plt.figure(figsize=(18, 4))
    if args.interval:
        df = df[df.index % args.interval == 0]
    plt.plot(df.ts, df.mem_use)
    if args.ylim and len(args.ylim) == 2:
        plt.gca().set_ylim(args.ylim)
        logger.info(f"Set ylim to {args.ylim}")
    plt.savefig(args.plot_save_path)


if __name__ == '__main__':
    main(parse_args())
