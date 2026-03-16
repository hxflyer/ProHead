from multiprocessing import freeze_support

from dense2geometry_train_core import create_arg_parser, launch_linux


DATA_FOLDERS = [
    "/hy-tmp/CapturedFrames_final1_processed",
    "/hy-tmp/CapturedFrames_final2_processed",
    "/hy-tmp/CapturedFrames_final3_processed",
    "/hy-tmp/CapturedFrames_final4_processed",
    "/hy-tmp/CapturedFrames_final5_processed",
    "/hy-tmp/CapturedFrames_final6_processed",
    "/hy-tmp/CapturedFrames_final7_processed",
    "/hy-tmp/CapturedFrames_final8_processed",
    "/hy-tmp/CapturedFrames_final9_processed",
]

TEXTURE_ROOT = "/hy-tmp/textures"
DEFAULT_DENSE_CKPT = "best_dense_image_transformer_ch10.pth"


def main() -> None:
    parser = create_arg_parser("Train Dense2Geometry (Linux)")
    args = parser.parse_args()

    args.data_roots = list(DATA_FOLDERS)
    args.texture_root = TEXTURE_ROOT
    args.image_size = 1024
    args.load_dense_model = str(args.load_dense_model or DEFAULT_DENSE_CKPT)

    print(f"Dense2Geometry training roots: {args.data_roots}")
    print(f"Texture root: {args.texture_root}")
    print(f"Image size: {args.image_size}")
    print(f"Dense checkpoint: {args.load_dense_model}")

    launch_linux(args)


if __name__ == "__main__":
    freeze_support()
    main()
