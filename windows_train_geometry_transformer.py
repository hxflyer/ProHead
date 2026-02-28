from geometry_train_core import create_arg_parser, launch_windows


if __name__ == "__main__":
    parser = create_arg_parser("RGB Geometry Transformer Training (Windows Launcher)")
    args = parser.parse_args()
    launch_windows(args)
