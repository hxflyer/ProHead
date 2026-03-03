from multiprocessing import freeze_support

from dense_image_train_core import create_arg_parser, launch_linux

DATA_FOLDERS = [
    ("/hy-tmp/CapturedFrames_final1_processed", "synthetic"),
    ("/hy-tmp/CapturedFrames_final7_processed", "synthetic"),
    ("/hy-tmp/CapturedFrames_final8_processed", "real"),
    ("/hy-tmp/CapturedFrames_final9_processed", "real"),
]


def main() -> None:
    parser = create_arg_parser("Train Dense Image Transformer (Linux)")
    args = parser.parse_args()
    args.data_roots = [path for path, _ in DATA_FOLDERS]
    print(f"Dense image training roots: {args.data_roots}")
    print(f"Target output channels: {args.output_channels}")
    print(f"Epochs: {args.epochs} (default 50)")
    launch_linux(args)


if __name__ == "__main__":
    freeze_support()
    main()
