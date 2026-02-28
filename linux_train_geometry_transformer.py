from geometry_train_core import create_arg_parser, launch_linux


if __name__ == "__main__":
    parser = create_arg_parser("Multi-GPU RGB Geometry Transformer Training (Linux Launcher)")
    args = parser.parse_args()
    launch_linux(args)
