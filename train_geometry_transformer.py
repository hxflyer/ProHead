from __future__ import annotations

from multiprocessing import freeze_support

from geometry_train_core import create_arg_parser, launch_linux, launch_windows
from train_launcher_common import (
    data_roots_from_folders,
    launch_for_platform,
    resolve_platform_key,
    resolve_platform_value,
    synthetic_data_roots_from_folders,
)


DATA_FOLDERS_BY_PLATFORM = {
    "linux": [
        ("/hy-tmp/CapturedFrames_final1_processed", "synthetic"),
        ("/hy-tmp/CapturedFrames_final7_processed", "synthetic"),
        ("/hy-tmp/CapturedFrames_final8_processed", "real"),
        ("/hy-tmp/CapturedFrames_final9_processed", "real"),
    ],
    "windows": [
        ("G:/CapturedFrames_final8_processed", "synthetic"),
        ("G:/CapturedFrames_final9_processed", "real"),
    ],
}

TEXTURE_ROOT_BY_PLATFORM = {
    "linux": "/hy-tmp/textures",
    "windows": "G:/textures",
}


def main(platform_key: str | None = None) -> None:
    freeze_support()
    resolved_platform = resolve_platform_key(platform_key)

    parser = create_arg_parser(f"Train Geometry Transformer ({resolved_platform.capitalize()})")
    args = parser.parse_args()

    data_folders = resolve_platform_value(DATA_FOLDERS_BY_PLATFORM, resolved_platform)
    args.data_roots = data_roots_from_folders(data_folders)
    args.texture_root = str(resolve_platform_value(TEXTURE_ROOT_BY_PLATFORM, resolved_platform))
    args.synthetic_data_roots = synthetic_data_roots_from_folders(data_folders)

    if args.synthetic_data_roots:
        print(f"Synthetic folders (full GT): {args.synthetic_data_roots}")
        print(f"Real folders (render only): {[path for path, dtype in data_folders if dtype == 'real']}")
    else:
        print("All folders will use full supervision (no synthetic/real separation)")

    launch_for_platform(args, launch_linux, launch_windows, resolved_platform)


if __name__ == "__main__":
    main()
