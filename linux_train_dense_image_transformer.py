from multiprocessing import freeze_support

from dense_image_train_core import create_arg_parser, launch_linux, _resolve_prediction_targets, _target_summary

DATA_FOLDERS = [
    ("/hy-tmp/CapturedFrames_final1_processed", "synthetic"),
    ("/hy-tmp/CapturedFrames_final2_processed", "synthetic"),
    ("/hy-tmp/CapturedFrames_final3_processed", "synthetic"),
    ("/hy-tmp/CapturedFrames_final4_processed", "synthetic"),
    ("/hy-tmp/CapturedFrames_final5_processed", "synthetic"),
    ("/hy-tmp/CapturedFrames_final6_processed", "synthetic"),
    ("/hy-tmp/CapturedFrames_final7_processed", "synthetic"),
    ("/hy-tmp/CapturedFrames_final8_processed", "synthetic"),
    ("/hy-tmp/CapturedFrames_final9_processed", "synthetic"),
]


def main() -> None:
    parser = create_arg_parser("Train Dense Image Transformer (Linux)")
    args = parser.parse_args()
    args.data_roots = [path for path, _ in DATA_FOLDERS]
    predict_basecolor, predict_geo, predict_normal = _resolve_prediction_targets(args)
    print(f"Dense image training roots: {args.data_roots}")
    print(f"Targets: {_target_summary(predict_basecolor, predict_geo, predict_normal)}")
    print(f"Epochs: {args.epochs} (default 50)")
    launch_linux(args)


if __name__ == "__main__":
    freeze_support()
    main()
