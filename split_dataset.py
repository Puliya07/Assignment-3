import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

DATASET_PATH = Path("Jute_Pest_Dataset/train")

RANDOM_SEED = 42

TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

def load_data_from_folders(dataset_dir):
    """
    Scans a directory structured as 'class_label/image_files'
    and loads all image paths and labels into a pandas DataFrame.
    """

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', 'gif'}
    print(f"Scanning for images in: {dataset_dir}")

    filepaths = []
    labels = []

    if not dataset_dir.exists():
        print(f"Error: Dataset path not found: {dataset_dir}", file=sys.stderr)
        return pd.DataFrame() # Return empty DataFrame
    
    for class_dir in dataset_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            print(f" Loading class: {class_name}")

            for img_path in class_dir.rglob('*'):
                if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                    filepaths.append(str(img_path)) # Store path as string
                    labels.append(class_name)

    if not filepaths:
        print(f"Error: No images found in {dataset_dir}. Check the path and folder structure.", file=sys.stderr)
        return pd.DataFrame()
    
    df = pd.DataFrame({
        'filepath': filepaths,
        'label': labels
    })

    df = df.sample(frac=1, random_state = RANDOM_SEED).reset_index(drop=True)  # Shuffle the DataFrame

    print(f"\nSuccessfully loaded {len(df)} images from {len(df['label'].unique())} classes.")
    return df

full_df = load_data_from_folders(DATASET_PATH)

if full_df.empty:
    print("Haltingexecution due to data loading error.")
    sys.exit(1)

X = full_df['filepath']
y = full_df['label']

print(f"\nSplitting {len(full_df)} images with seed {RANDOM_SEED}...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    train_size=TRAIN_SIZE,
    shuffle=True,
    stratify=y, 
    random_state=RANDOM_SEED
)

test_split_ratio = TEST_SIZE / (1.0 - TRAIN_SIZE)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=test_split_ratio,
    shuffle=True,
    stratify=y_temp, 
    random_state=RANDOM_SEED
)

print("Splitting complete.")

print("\n" + "="*50)
print("     STRATIFICATION VERIFICATION REPORT")
print("="*50)

# Re-create DataFrames for easy analysis
train_df = pd.DataFrame({'label': y_train})
val_df = pd.DataFrame({'label': y_val})
test_df = pd.DataFrame({'label': y_test})

print("\n--- Class Distribution (Percentages) ---")

orig_pct = full_df['label'].value_counts(normalize=True).sort_index()
train_pct = train_df['label'].value_counts(normalize=True).sort_index()
val_pct = val_df['label'].value_counts(normalize=True).sort_index()
test_pct = test_df['label'].value_counts(normalize=True).sort_index()

pct_report = pd.DataFrame({
    'Original': orig_pct,
    'Training (70%)': train_pct,
    'Validation (15%)': val_pct,
    'Test (15%)': test_pct
})
print(pct_report.to_string(float_format="%.4f"))

print("\n--- Class Distribution (Absolute Counts) ---")

orig_counts = full_df['label'].value_counts().sort_index()
train_counts = train_df['label'].value_counts().sort_index()
val_counts = val_df['label'].value_counts().sort_index()
test_counts = test_df['label'].value_counts().sort_index()

count_report = pd.DataFrame({
    'Original': orig_counts,
    'Training (70%)': train_counts,
    'Validation (15%)': val_counts,
    'Test (15%)': test_counts
})
print(count_report)


print("\n--- Edge Case Analysis (Low Sample Count) ---")
LOW_SAMPLE_THRESHOLD = 5

val_low = count_report[count_report['Validation (15%)'] < LOW_SAMPLE_THRESHOLD]
test_low = count_report[count_report['Test (15%)'] < LOW_SAMPLE_THRESHOLD]

if val_low.empty and test_low.empty:
    print(f"SUCCESS: All classes in Validation and Test sets have >= {LOW_SAMPLE_THRESHOLD} samples.")
else:
    if not val_low.empty:
        print(f" WARNING: The following classes have < {LOW_SAMPLE_THRESHOLD} samples in the VALIDATION set:")
        print(val_low.index.tolist())
    if not test_low.empty:
        print(f" WARNING: The following classes have < {LOW_SAMPLE_THRESHOLD} samples in the TEST set:")
        print(test_low.index.tolist())
    
print("\nVerification finished.")