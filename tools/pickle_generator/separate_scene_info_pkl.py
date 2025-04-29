import argparse
import pickle
from pathlib import Path

from loguru import logger
from tqdm import tqdm
from copious.io.fs import parent_ensured_path

def parse_args():
    parser = argparse.ArgumentParser(description='Separate specified items from input pkl file.')
    parser.add_argument('--input-pkl-path', type=Path, required=True, help='Input file path')
    parser.add_argument('--output-pkl-path', type=Path, required=True, help='Output file path')
    parser.add_argument("--mv4d-data-save-root", type=Path, required=True,
                        help='Root directory to save the separated frame pickles.')
    parser.add_argument("--scene-info-pkl-filename", type=str, default="scene_info.pkl")
    return parser.parse_args()

def main(args):
    # Step 1: Load the input pickle file
    logger.info(f"Loading input data from {args.input_pkl_path}")
    with args.input_pkl_path.open('rb') as f:
        data = pickle.load(f)
    logger.info("Input data loaded successfully.")

    # Step 2: Separate the frame info pkl for each scene
    for scene_id in tqdm(data.keys()):
        save_path = parent_ensured_path(args.mv4d_data_save_root / scene_id / args.scene_info_pkl_filename)
        with save_path.open('wb') as f_out:
            pickle.dump(data[scene_id]["scene_info"], f_out)
        data[scene_id]["scene_info"] = str(save_path.relative_to(args.mv4d_data_save_root))

    # Step 3: Save the modified data to the output pickle file
    logger.info(f"Saving modified data to {args.output_pkl_path}")
    with args.output_pkl_path.open('wb') as f_out:
        pickle.dump(data, f_out)

    logger.success("Processing completed successfully.")

if __name__ == '__main__':
    main(parse_args())
