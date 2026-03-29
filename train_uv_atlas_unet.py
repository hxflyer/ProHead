from __future__ import annotations

from multiprocessing import freeze_support

from train_launcher_common import launch_for_platform, resolve_platform_key
from uv_atlas_train_core import create_arg_parser, launch_linux, launch_windows


def main(platform_key: str | None = None) -> None:
    freeze_support()
    resolved_platform = resolve_platform_key(platform_key)
    parser = create_arg_parser(f"Train UV Atlas UNet ({resolved_platform.capitalize()})")
    args = parser.parse_args()

    print(f"UV atlas cache root: {args.cache_root}")
    print(f"Epochs: {args.epochs}")
    print(f"Checkpoint path: {args.save_path}")
    launch_for_platform(args, launch_linux, launch_windows, resolved_platform)


if __name__ == "__main__":
    main()
