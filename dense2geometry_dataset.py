from torch.utils.data import Dataset

from metahuman_geometry_dataset import FastGeometryDataset, fast_collate_fn


class Dense2GeometryDataset(Dataset):
    """Synthetic-only wrapper around FastGeometryDataset for dense2geometry training."""

    def __init__(
        self,
        data_roots,
        split: str = "train",
        image_size: int = 512,
        train_ratio: float = 0.95,
        augment: bool = True,
        texture_root: str | None = None,
    ):
        self.base_dataset = FastGeometryDataset(
            data_roots=data_roots,
            split=split,
            image_size=image_size,
            train_ratio=train_ratio,
            augment=augment,
            texture_root=texture_root,
            texture_png_cache_max_items=0,
            combined_texture_cache_max_items=0,
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]
        mesh_found = sample.get("mesh_found_mask", None)
        mesh_weights = sample.get("mesh_weights", None)
        if mesh_found is not None and mesh_weights is not None:
            sample["mesh_loss_weights"] = mesh_found * mesh_weights
        elif mesh_weights is not None:
            sample["mesh_loss_weights"] = mesh_weights
        return sample


def dense2geometry_collate_fn(batch):
    collated = fast_collate_fn(batch)
    if all("mesh_loss_weights" in item for item in batch):
        collated["mesh_loss_weights"] = collated["mesh_weights"] * collated["mesh_found_mask"]
    return collated
