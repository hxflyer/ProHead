from __future__ import annotations

from multiprocessing import freeze_support

from dense_image_train_core import create_arg_parser, launch_linux, launch_windows, _resolve_prediction_targets, _target_summary
from train_launcher_common import data_roots_from_folders, launch_for_platform, resolve_platform_key, resolve_platform_value


DATA_FOLDERS_BY_PLATFORM = {
    "linux": [
        ("/hy-tmp/CapturedFrames_final1_processed", "synthetic"),
        ("/hy-tmp/CapturedFrames_final2_processed", "synthetic"),
        ("/hy-tmp/CapturedFrames_final3_processed", "synthetic"),
        ("/hy-tmp/CapturedFrames_final4_processed", "synthetic"),
        ("/hy-tmp/CapturedFrames_final5_processed", "synthetic"),
        ("/hy-tmp/CapturedFrames_final6_processed", "synthetic"),
        ("/hy-tmp/CapturedFrames_final7_processed", "synthetic"),
        ("/hy-tmp/CapturedFrames_final8_processed", "synthetic"),
        ("/hy-tmp/CapturedFrames_final9_processed", "synthetic"),
    ],
    "windows": [
        ("G:/CapturedFrames_final1_processed", "synthetic"),
        ("G:/CapturedFrames_final8_processed", "synthetic"),
    ],
}


def main(platform_key: str | None = None) -> None:
    freeze_support()
    resolved_platform = resolve_platform_key(platform_key)

    parser = create_arg_parser(f"Train Dense Image Transformer ({resolved_platform.capitalize()})")
    args = parser.parse_args()
    data_folders = resolve_platform_value(DATA_FOLDERS_BY_PLATFORM, resolved_platform)
    args.data_roots = data_roots_from_folders(data_folders)

    predict_basecolor, predict_geo, predict_normal = _resolve_prediction_targets(args)
    print(f"Dense image training roots: {args.data_roots}")
    print(f"Targets: {_target_summary(predict_basecolor, predict_geo, predict_normal)}")
    print(f"Epochs: {args.epochs} (default 50)")

    launch_for_platform(args, launch_linux, launch_windows, resolved_platform)


if __name__ == "__main__":
    main()
