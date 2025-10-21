import torch
import torch.utils.data
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Make sure this is imported


DATASET_PATH = Path("Jute_Pest_Dataset/train") 
RANDOM_SEED = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
IMG_SIZE = 224

class JutePestDataset(Dataset):
    """
    Custom Dataset for loading jute pest images.
    It takes a DataFrame (with 'filepath' and 'label' columns)
    and a transform to apply.
    """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

        self.classes = sorted(df['label'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['filepath']
        label_str = self.df.iloc[idx]['label']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Could not find image: {img_path}. Skipping.")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        except Exception as e:
            print(f"Error loading {img_path}: {e}. Skipping.")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        label = self.class_to_idx[label_str]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_data_from_folders(dataset_dir):
    """Scans a directory and loads image paths/labels."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    print(f"Scanning for images in: {dataset_dir}")
    filepaths = []
    labels = []
    
    if not dataset_dir.exists():
        print(f"Error: Dataset path not found: {dataset_dir}", file=sys.stderr)
        return pd.DataFrame()
        
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            print(f"  Loading class: {class_name}")
            for img_path in class_dir.rglob('*'):
                if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                    filepaths.append(str(img_path))
                    labels.append(class_name)

    if not filepaths:
        print(f"Error: No images found in {dataset_dir}.", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame({'filepath': filepaths, 'label': labels})
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"\nSuccessfully loaded {len(df)} images from {len(df['label'].unique())} classes.")
    return df

def get_training_set_stats(train_df, num_workers=4, batch_size=64):
    """Calculates the mean and std of the training set."""
    print("Calculating training set statistics (mean, std)...")
    
    stats_transform = T.Compose([
        T.Resize((256, 256)), 
        T.ToTensor()
    ])
    
    stats_dataset = JutePestDataset(df=train_df, transform=stats_transform)

    stats_loader = DataLoader(
        stats_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False
    )
    
    psum = torch.zeros(3)
    psum_sq = torch.zeros(3)
    total_samples = 0
    
    for inputs, _ in stats_loader:
        psum += inputs.sum(dim=[0, 2, 3])
        psum_sq += (inputs ** 2).sum(dim=[0, 2, 3])
        total_samples += inputs.size(0)
        print(f"\rProcessed {total_samples}/{len(stats_dataset)} images", end='')
    
    print("\nCalculation complete.")
    
    count = total_samples * 256 * 256 
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)
    
    return total_mean, total_std

def visualize_preprocessing(df, transform, train_mean, train_std):
    """Shows a before-and-after of preprocessing."""
    if len(df) < 1:
        print("\nCannot visualize: No image data found.")
        return
        
    img_path = df.iloc[0]['filepath']
    img_original = Image.open(img_path).convert('RGB')
    img_transformed = transform(img_original)
    
    mean = train_mean.numpy()
    std = train_std.numpy()
    
    img_display = img_transformed.clone()
    mean = mean.reshape(3, 1, 1)
    std = std.reshape(3, 1, 1)
    img_display = img_display * std + mean
    img_display = np.clip(img_display.numpy(), 0, 1)
    img_display = img_display.transpose(1, 2, 0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_original)
    ax[0].set_title(f"Before\nOriginal Size: {img_original.size}")
    ax[0].axis('off')
    
    ax[1].imshow(img_display)
    ax[1].set_title(f"After (Preprocessed)\nSize: {img_display.shape[:2]}")
    ax[1].axis('off')
    
    plt.suptitle("Preprocessing Visualization", fontsize=16)
    plt.show()


if __name__ == '__main__':

    full_df = load_data_from_folders(DATASET_PATH)

    if full_df.empty:
        print("Halting execution due to data loading error.")
        sys.exit(1) 

    X_train, X_temp, y_train, y_temp = train_test_split(
        full_df['filepath'], full_df['label'], train_size=TRAIN_SIZE, 
        stratify=full_df['label'], random_state=RANDOM_SEED
    )
    train_df_final = pd.DataFrame({'filepath': X_train, 'label': y_train})
    print(f"Loaded and split data. Using {len(train_df_final)} images for stats calculation.")
    
    try:
        TRAIN_MEAN = torch.load("train_mean.pt")
        TRAIN_STD = torch.load("train_std.pt")
        print("Loaded saved statistics.")
    except FileNotFoundError:
        print("Statistics files not found. Calculating from scratch...")
        TRAIN_MEAN, TRAIN_STD = get_training_set_stats(train_df_final, num_workers=4)
        
        torch.save(TRAIN_MEAN, "train_mean.pt")
        torch.save(TRAIN_STD, "train_std.pt")
        print("Saved new statistics to train_mean.pt and train_std.pt")

    print(f"\n--- Computed Statistics ---")
    print(f"TRAINING MEAN: {TRAIN_MEAN.numpy()}")
    print(f"TRAINING STD:  {TRAIN_STD.numpy()}")
    print(f"---------------------------")

    data_transforms = {
        'train': T.Compose([
            T.RandomResizedCrop(IMG_SIZE),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(TRAIN_MEAN, TRAIN_STD)
        ]),
        'val': T.Compose([
            T.Resize(256),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(TRAIN_MEAN, TRAIN_STD)
        ]),
        'test': T.Compose([
            T.Resize(256),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(TRAIN_MEAN, TRAIN_STD)
        ])
    }
    print("\nSuccessfully created 'data_transforms' dictionary.")

    visualize_preprocessing(train_df_final, data_transforms['val'], TRAIN_MEAN, TRAIN_STD)