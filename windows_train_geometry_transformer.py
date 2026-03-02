import sys
from geometry_train_core import create_arg_parser, launch_windows

# Dataset configuration: list of (path, type) tuples
# type: "synthetic" = full supervision (GT 2D, 3D, texture)
#       "real" = render-only supervision (no GT geometry)
DATA_FOLDERS = [
    #("G:/CapturedFrames_final1_processed", "synthetic"),
    #("G:/CapturedFrames_final7_processed", "synthetic"),
    ("G:/CapturedFrames_final8_processed", "synthetic"),
    ("G:/CapturedFrames_final9_processed", "real"),
]

TEXTURE_ROOT = "G:/textures"

if __name__ == '__main__':
    parser = create_arg_parser("Train Geometry Transformer (Windows)")
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

    launch_windows(args)
