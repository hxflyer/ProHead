from multiprocessing import freeze_support

from geometry_train_core import create_arg_parser, launch_linux

# Dataset configuration: list of (path, type) tuples
# type: "synthetic" = full supervision (GT 2D, 3D, texture)
#       "real" = render-only supervision (no GT geometry)
DATA_FOLDERS = [
    ("/hy-tmp/CapturedFrames_final1_processed", "synthetic"),
    ("/hy-tmp/CapturedFrames_final7_processed", "synthetic"),
    ("/hy-tmp/CapturedFrames_final8_processed", "real"),
    ("/hy-tmp/CapturedFrames_final9_processed", "real"),
]

TEXTURE_ROOT = "/hy-tmp/textures"


def main() -> None:
    parser = create_arg_parser("Train Geometry Transformer (Linux)")
    args = parser.parse_args()

    # Override data_roots and texture_root from config
    args.data_roots = [path for path, _ in DATA_FOLDERS]
    args.texture_root = TEXTURE_ROOT
    args.synthetic_data_roots = [path for path, dtype in DATA_FOLDERS if dtype == "synthetic"]

    if args.synthetic_data_roots:
        print(f"Synthetic folders (full GT): {args.synthetic_data_roots}")
        print(f"Real folders (render only): {[p for p, t in DATA_FOLDERS if t == 'real']}")
    else:
        print("All folders will use full supervision (no synthetic/real separation)")

    launch_linux(args)


if __name__ == "__main__":
    freeze_support()
    main()
