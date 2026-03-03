from dense_image_train_core import create_arg_parser, launch_windows

# Dataset configuration: list of (path, type) tuples
# type field is informational for this task; training uses basecolor supervision.
DATA_FOLDERS = [
    # ("G:/CapturedFrames_final1_processed", "synthetic"),
    # ("G:/CapturedFrames_final7_processed", "synthetic"),
    ("G:/CapturedFrames_final8_processed", "synthetic"),
    ("G:/CapturedFrames_final9_processed", "real"),
]


if __name__ == "__main__":
    parser = create_arg_parser("Train Dense Image Transformer (Windows)")
    args = parser.parse_args()
    args.data_roots = [path for path, _ in DATA_FOLDERS]

    print(f"Dense image training roots: {args.data_roots}")
    print(f"Target output channels: {args.output_channels}")
    print(f"Epochs: {args.epochs} (default 50)")
    launch_windows(args)
