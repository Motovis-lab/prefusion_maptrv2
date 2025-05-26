import argparse
import pickle
from pathlib import Path

from loguru import logger
from copious.io.fs import parent_ensured_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Separate specified items from input pkl file."
    )
    parser.add_argument(
        "--input-pkl-path", type=Path, required=True, help="Input file path"
    )
    parser.add_argument(
        "--output-pkl-path", type=Path, required=True, help="Output file path"
    )
    parser.add_argument(
        "--mv4d-data-save-root",
        type=Path,
        required=True,
        help="Root directory to save the separated frame pickles.",
    )
    parser.add_argument(
        "--store-scene-info-to-frame-pkl", action="store_true", default=False
    )
    parser.add_argument("--frame-info-pkl-dir-name", type=str, default="frame_info_pkl")
    return parser.parse_args()


def main(args):
    # Step 1: Load the input pickle file
    logger.info(f"Loading input data from {args.input_pkl_path}")
    with args.input_pkl_path.open("rb") as f:
        data = pickle.load(f)
    logger.info("Input data loaded successfully.")

    # Step 2: Separate the frame info pkl for each scene
    for scene_id, scene_data in data.items():
        total_separated = 0
        frame_info = scene_data.get("frame_info", {})
        for frame_id in frame_info.keys():
            if args.store_scene_info_to_frame_pkl:
                frame_info[frame_id]["scene_info"] = scene_data["scene_info"]

            save_path = parent_ensured_path(
                args.mv4d_data_save_root
                / scene_id
                / args.frame_info_pkl_dir_name
                / f"{frame_id}.pkl"
            )

            # Save the frame pickle data to the constructed path
            with save_path.open("wb") as f_out:
                pickle.dump(frame_info[frame_id], f_out)

            # Update the frame_data to reference the saved file path
            frame_info[frame_id] = str(save_path.relative_to(args.mv4d_data_save_root))
            total_separated += 1

        logger.info(f"Processed {total_separated} pickles in scene {scene_id}.")

    # Step 3: Save the modified data to the output pickle file
    logger.info(f"Saving modified data to {args.output_pkl_path}")
    with args.output_pkl_path.open("wb") as f_out:
        pickle.dump(data, f_out)

    logger.success("Processing completed successfully.")


if __name__ == "__main__":
    main(parse_args())
