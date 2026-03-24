from __future__ import annotations

from multiprocessing import freeze_support

from dense2geometry_train_core import create_arg_parser, launch_linux, launch_windows
from train_launcher_common import data_roots_from_folders, launch_for_platform, resolve_platform_key, resolve_platform_value


DATA_FOLDERS_BY_PLATFORM = {
    "linux": [
        "/hy-tmp/CapturedFrames_final1_processed",
        "/hy-tmp/CapturedFrames_final2_processed",
        "/hy-tmp/CapturedFrames_final3_processed",
        "/hy-tmp/CapturedFrames_final4_processed",
        "/hy-tmp/CapturedFrames_final5_processed",
        "/hy-tmp/CapturedFrames_final6_processed",
        "/hy-tmp/CapturedFrames_final7_processed",
        "/hy-tmp/CapturedFrames_final8_processed",
        "/hy-tmp/CapturedFrames_final9_processed",
    ],
    "windows": [
        "G:/CapturedFrames_final1_processed",
        "G:/CapturedFrames_final2_processed",
        "G:/CapturedFrames_final3_processed",
        "G:/CapturedFrames_final4_processed",
        "G:/CapturedFrames_final5_processed",
        "G:/CapturedFrames_final6_processed",
        "G:/CapturedFrames_final7_processed",
        "G:/CapturedFrames_final8_processed",
        "G:/CapturedFrames_final9_processed",
    ],
}

TEXTURE_ROOT_BY_PLATFORM = {
    "linux": "/hy-tmp/textures",
    "windows": "G:/textures",
}

DEFAULT_DENSE_CKPT = "artifacts/checkpoints/best_dense_image_transformer_ch10.pth"


def main(platform_key: str | None = None) -> None:
    freeze_support()
    resolved_platform = resolve_platform_key(platform_key)

    parser = create_arg_parser(f"Train Dense2Geometry ({resolved_platform.capitalize()})")
    args = parser.parse_args()

    args.data_roots = data_roots_from_folders(resolve_platform_value(DATA_FOLDERS_BY_PLATFORM, resolved_platform))
    args.texture_root = str(resolve_platform_value(TEXTURE_ROOT_BY_PLATFORM, resolved_platform))
    args.image_size = 1024
    args.load_dense_model = str(args.load_dense_model or DEFAULT_DENSE_CKPT)

    print(f"Dense2Geometry training roots: {args.data_roots}")
    print(f"Texture root: {args.texture_root}")
    print(f"Image size: {args.image_size}")
    print(f"Dense checkpoint: {args.load_dense_model}")

    launch_for_platform(args, launch_linux, launch_windows, resolved_platform)


if __name__ == "__main__":
    main()
