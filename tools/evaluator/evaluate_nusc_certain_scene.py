import argparse
from pathlib import Path

from loguru import logger
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
from copious.io.args import declare_vars_as_global, g
from copious.io.fs import read_json, write_json, mktmpdir, ensured_path


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--nusc-data-root", type=Path, required=True, help="Input file path"
    )
    parser.add_argument(
        "--scene-names", nargs="+", required=True, help="scene names, e.g.: scene-0001"
    )
    parser.add_argument(
        "--model-infer-results",
        type=Path,
        required=True,
        help="Path to the result.json that inferred by model",
    )
    parser.add_argument(
        "--output-dir", type=ensured_path, required=True, help="Output directory"
    )
    parser.add_argument(
        "--nusc-eval-set", default="val", choices=["train", "val", "test"]
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def main(args):
    declare_vars_as_global(verbose=args.verbose)
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nusc_data_root)
    sample_tokens = []
    for scn_name in args.scene_names:
        _sample_tokens_in_scene = get_sample_tokens(nusc, scn_name)
        sample_tokens.extend(_sample_tokens_in_scene)
        if g("verbose"):
            logger.debug(f"{scn_name}: {_sample_tokens_in_scene}")
        
    # FIXME: e93e98b63d3b40209056d129dc53ceee
    # sample_tokens = ["e93e98b63d3b40209056d129dc53ceee"]

    infer_results = read_json(args.model_infer_results)

    # 过滤出目标样本的预测
    filtered_preds = {
        "meta": infer_results["meta"],
        "results": {
            token: infer_results["results"][token]
            for token in sample_tokens
            if token in infer_results["results"]
        },
    }

    # 保存到临时文件
    tmpdir = mktmpdir()
    write_json(filtered_preds, tmpdir / "filtered_preds.json")
    logger.info(f"Saved filtered_preds.json to {tmpdir}/filtered_preds.json")

    # 初始化评估器 (To make it work, we have to comment the sample_token assertion in /opt/conda/lib/python3.10/site-packages/nuscenes/eval/detection/evaluate.py)
    logger.info(f"Loading nuscenes data ({args.nusc_eval_set})")
    nusc_eval = NuScenesEval(
        nusc,
        config=config_factory("detection_cvpr_2019"),  # 加载配置文件
        result_path=str(tmpdir / "filtered_preds.json"),
        eval_set=args.nusc_eval_set,
        output_dir=str(args.output_dir),
        verbose=True,
    )

    # 覆盖样本Tokens
    nusc_eval.sample_tokens = sample_tokens
    nusc_eval.gt_boxes.boxes = {t: v for t, v in nusc_eval.gt_boxes.boxes.items() if t in sample_tokens}

    # 执行评估
    nusc_eval.main(render_curves=False)


def get_sample_tokens(nusc, scene_name: str):
    # 指定场景名称或直接使用scene_token
    target_scene = [s for s in nusc.scene if s["name"] == scene_name][0]

    # 收集所有样本Tokens
    sample_tokens = []
    current_token = target_scene["first_sample_token"]
    while current_token:
        sample = nusc.get("sample", current_token)
        sample_tokens.append(sample["token"])
        current_token = sample["next"]

    return sample_tokens


if __name__ == "__main__":
    main(parse_args())
