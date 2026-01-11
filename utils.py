from dataset import CarvanaDataset
from torch.utils.data import DataLoader


def get_loaders(train_img_dir, train_labels_dir, val_img_dir, val_labels_dir,
                batch_size, num_workers, pin_memory=True):
    train_ds = None
    val_ds = None

    if train_img_dir and train_labels_dir:
        train_ds = CarvanaDataset(images_dir=train_img_dir, labels_dir=train_labels_dir, transform=None)

    if val_img_dir and val_labels_dir:
        val_ds = CarvanaDataset(images_dir=val_img_dir, labels_dir=val_labels_dir, transform=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory, shuffle=True) if train_ds else None
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, shuffle=False) if val_ds else None

    return train_loader, val_loader