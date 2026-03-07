from dense_image_train_core import create_arg_parser, launch_windows, _resolve_prediction_targets, _target_summary

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
    predict_basecolor, predict_geo, predict_normal = _resolve_prediction_targets(args)

    print(f"Dense image training roots: {args.data_roots}")
    print(f"Targets: {_target_summary(predict_basecolor, predict_geo, predict_normal)}")
    print(f"Epochs: {args.epochs} (default 50)")
    launch_windows(args)
