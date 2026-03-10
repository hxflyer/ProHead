"""
Dataset examination tool for DenseImageDataset
Analyzes dataset health, checks for missing data, and reports statistics
"""
import argparse
from torch.utils.data import Subset

from dense_image_dataset import DenseImageDataset


def _dataset_basecolor_valid_stats(dataset) -> tuple[int, int]:
    """Calculate valid basecolor statistics for dataset."""
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = list(dataset.indices)
        if hasattr(base_dataset, "samples"):
            valid_count = sum(
                1
                for idx in indices
                if idx < len(base_dataset.samples)
                and len(base_dataset.samples[idx]) >= 3
                and base_dataset.samples[idx][2] is not None
            )
            return len(indices), int(valid_count)
        return len(indices), -1

    if hasattr(dataset, "samples"):
        total = len(dataset.samples)
        valid = sum(
            1
            for sample in dataset.samples
            if len(sample) >= 3 and sample[2] is not None
        )
        return int(total), int(valid)

    try:
        return int(len(dataset)), -1
    except Exception:
        return -1, -1


def print_dataset_summary(prefix: str, dataset) -> None:
    """Print comprehensive dataset summary including health checks."""
    print(f"\n{'=' * 60}")
    print(f"{prefix} Dataset Summary")
    print(f"{'=' * 60}")
    
    total, valid = _dataset_basecolor_valid_stats(dataset)
    if total >= 0 and valid >= 0:
        ratio = float(valid / max(total, 1))
        print(f"Basecolor valid: {valid}/{total} ({ratio:.2%})")
    elif total >= 0:
        print(f"Total samples: {total} (valid basecolor count unavailable)")

    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(base_dataset, "get_debug_summary"):
        try:
            summary = base_dataset.get_debug_summary()
            
            # Basecolor missing examples
            missing_examples = list(summary.get("missing_examples", []))
            if missing_examples:
                print(f"\nMissing basecolor mappings ({len(missing_examples)} total):")
                for ex in missing_examples[:10]:
                    print(f"  {ex}")
                if len(missing_examples) > 10:
                    print(f"  ... and {len(missing_examples) - 10} more")
            
            # Geo, Normal, Face mask statistics
            for key in ("geo", "normal", "face_mask"):
                valid_key = f"{key}_valid"
                ratio_key = f"{key}_valid_ratio"
                if valid_key in summary:
                    total_count = int(summary.get("total", 0))
                    valid_count = int(summary.get(valid_key, 0))
                    ratio = float(summary.get(ratio_key, 0.0))
                    print(f"{key.capitalize()} valid: {valid_count}/{total_count} ({ratio:.2%})")
            
            # Geo missing examples
            missing_geo_examples = list(summary.get("missing_geo_examples", []))
            if missing_geo_examples:
                print(f"\nMissing geo mappings ({len(missing_geo_examples)} total):")
                for ex in missing_geo_examples[:10]:
                    print(f"  {ex}")
                if len(missing_geo_examples) > 10:
                    print(f"  ... and {len(missing_geo_examples) - 10} more")
            
            # Normal missing examples
            missing_normal_examples = list(summary.get("missing_normal_examples", []))
            if missing_normal_examples:
                print(f"\nMissing normal mappings ({len(missing_normal_examples)} total):")
                for ex in missing_normal_examples[:10]:
                    print(f"  {ex}")
                if len(missing_normal_examples) > 10:
                    print(f"  ... and {len(missing_normal_examples) - 10} more")
            
            # Face mask missing examples
            missing_face_mask_examples = list(summary.get("missing_face_mask_examples", []))
            if missing_face_mask_examples:
                print(f"\nMissing face-mask mappings ({len(missing_face_mask_examples)} total):")
                for ex in missing_face_mask_examples[:10]:
                    print(f"  {ex}")
                if len(missing_face_mask_examples) > 10:
                    print(f"  ... and {len(missing_face_mask_examples) - 10} more")
                    
        except Exception as e:
            print(f"Error getting debug summary: {e}")
    
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Examine DenseImageDataset health and statistics")
    parser.add_argument("--data_roots", nargs="+", required=True, help="Dataset root directories")
    parser.add_argument("--image_size", type=int, default=512, help="Image size for dataset")
    parser.add_argument("--train_ratio", type=float, default=0.95, help="Train/val split ratio")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit samples for testing (0=all)")
    args = parser.parse_args()

    print(f"Examining dataset from roots: {args.data_roots}")
    print(f"Image size: {args.image_size}")
    print(f"Train ratio: {args.train_ratio}")
    
    # Create train dataset
    print("\nLoading train dataset...")
    train_dataset = DenseImageDataset(
        data_roots=args.data_roots,
        split="train",
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        augment=False,  # Disable augmentation for examination
    )
    
    # Create val dataset
    print("Loading validation dataset...")
    val_dataset = DenseImageDataset(
        data_roots=args.data_roots,
        split="val",
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        augment=False,
    )
    
    # Print summaries
    print_dataset_summary("Train", train_dataset)
    print_dataset_summary("Validation", val_dataset)
    
    # If max_samples specified, also check subset
    if args.max_samples > 0 and len(train_dataset) > args.max_samples:
        print(f"Testing with limited train dataset ({args.max_samples} samples)...")
        train_subset = Subset(train_dataset, list(range(args.max_samples)))
        print_dataset_summary("Train (limited)", train_subset)
    
    # Summary
    print("\n" + "=" * 60)
    print("Examination Complete")
    print("=" * 60)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Total samples: {len(train_dataset) + len(val_dataset)}")


if __name__ == "__main__":
    main()
