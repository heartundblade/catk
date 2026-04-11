# src/vbd/datamodule/datamodule.py
from typing import Optional

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from src.vbd.data.dataset import SmartToVBDDataset


class VBDDataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        train_raw_dir: str,
        val_raw_dir: str,
        test_raw_dir: str,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        max_num_objects: int = 64,
        max_polylines: int = 256,
        num_points_polyline: int = 30,
        current_index: int = 10,
    ) -> None:
        super(VBDDataModule, self).__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        
        # VBD settings
        self.max_num_objects = max_num_objects
        self.max_polylines = max_polylines
        self.num_points_polyline = num_points_polyline
        self.current_index = current_index

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = SmartToVBDDataset(
                smart_data_dir=self.train_raw_dir,
                max_num_objects=self.max_num_objects,
                max_polylines=self.max_polylines,
                num_points_polyline=self.num_points_polyline,
                current_index=self.current_index,
            )
            self.val_dataset = SmartToVBDDataset(
                smart_data_dir=self.val_raw_dir,
                max_num_objects=self.max_num_objects,
                max_polylines=self.max_polylines,
                num_points_polyline=self.num_points_polyline,
                current_index=self.current_index,
            )
        elif stage == "validate":
            self.val_dataset = SmartToVBDDataset(
                smart_data_dir=self.val_raw_dir,
                max_num_objects=self.max_num_objects,
                max_polylines=self.max_polylines,
                num_points_polyline=self.num_points_polyline,
                current_index=self.current_index,
            )
        elif stage == "test":
            self.test_dataset = SmartToVBDDataset(
                smart_data_dir=self.test_raw_dir,
                max_num_objects=self.max_num_objects,
                max_polylines=self.max_polylines,
                num_points_polyline=self.num_points_polyline,
                current_index=self.current_index,
            )
        else:
            raise ValueError(f"{stage} should be one of [fit, validate, test]")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
