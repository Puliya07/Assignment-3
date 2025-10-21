import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import torch

def explore_jute_pest_dataset(dataset_path):
    """Exploration of Jute Pest Dataset"""

    print("EXPLORING JUTE PEST DATASET")
    print("=" * 50)

    # Find all classes (subdirectories)
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path,d))]
    classes.sort()

    print(f"Found {len(classes)} classes: {classes}")

    # Gather Statistics
    class_counts = defaultdict(int)
    image_dimensions = []
    color_channels = []
    all_image_paths = []
    class_image_paths = defaultdict(list)

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        class_counts[class_name] = len(image_files)

        # Analyze multiple images per class
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                with Image.open(img_path) as img:
                    image_dimensions.append(img.size) # (width, height)
                    color_channels.append(len(img.getbands()))
                    all_image_paths.append(img_path)
                    class_image_paths[class_name].append(img_path)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return classes, class_counts, image_dimensions, color_channels, all_image_paths, class_image_paths

def print_dataset_statistics(classes, class_counts, image_dimensions, color_channels):
    """Print comprehensive dataset statistics"""
    
    print("\n DATASET STATISTICS")
    print("-" * 30)
    
    # Basic counts
    total_images = sum(class_counts.values())
    print(f"Total images: {total_images}")
    print(f"Number of classes: {len(classes)}")
    
    # Class distribution
    print(f"\nClass distribution:")
    for class_name in classes:
        count = class_counts[class_name]
        percentage = (count / total_images) * 100
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    # Image dimension analysis
    if image_dimensions:
        widths, heights = zip(*image_dimensions)
        unique_sizes = set(image_dimensions)
        
        print(f"\nImage dimensions:")
        print(f"  Unique sizes: {len(unique_sizes)}")
        print(f"  Width range: {min(widths)} - {max(widths)} pixels")
        print(f"  Height range: {min(heights)} - {max(heights)} pixels")
        print(f"  Most common size: {Counter(image_dimensions).most_common(1)[0][0]}")
    
    # Color channel analysis
    if color_channels:
        channel_counts = Counter(color_channels)
        print(f"\nColor channels:")
        for channels, count in channel_counts.items():
            if channels == 1:
                print(f"  Grayscale: {count} images")
            elif channels == 3:
                print(f"  RGB: {count} images")
            elif channels == 4:
                print(f"  RGBA: {count} images")
    
    # Class balance analysis
    counts = list(class_counts.values())
    print(f"\nClass balance:")
    print(f"  Min images per class: {min(counts)}")
    print(f"  Max images per class: {max(counts)}")
    print(f"  Average images per class: {np.mean(counts):.1f}")
    print(f"  Standard deviation: {np.std(counts):.1f}")
    
    # Imbalance ratio
    imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}")

def visualize_class_examples(dataset_path, classes, class_counts, class_image_paths):
    """Create a grid of example images from each class"""
    
    print(f"\n VISUALIZING CLASS EXAMPLES")
    print("-" * 30)
    
    # Set up the plot
    n_classes = len(classes)
    fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=(20, 10))
    if n_classes > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, class_name in enumerate(classes):
        if i >= len(axes):
            break
            
        if class_image_paths[class_name]:
            # Load first image from class
            img_path = class_image_paths[class_name][0]
            try:
                img = Image.open(img_path)
                
                axes[i].imshow(np.array(img))
                axes[i].set_title(f'{class_name}\n({class_counts[class_name]} images)', 
                                fontsize=12, pad=10, weight='bold')
                axes[i].axis('off')
                
                # Print what we observe about this image
                print(f"{class_name}: Size {img.size}, Mode {img.mode}")
                
            except Exception as e:
                print(f"Error loading example for {class_name}: {e}")
                axes[i].set_title(f'{class_name}\n(Load error)', fontsize=12)
                axes[i].axis('off')
        else:
            axes[i].set_title(f'{class_name}\n(No images)', fontsize=12)
            axes[i].axis('off')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_image_quality(dataset_path, classes, class_image_paths, sample_per_class=2):
    """Analyze potential challenges in the dataset"""
    
    print(f"\n IMAGE QUALITY AND CHALLENGE ANALYSIS")
    print("-" * 40)
    
    challenges = {
        'varying_sizes': False,
        'varying_quality': False,
        'background_clutter': False,
        'lighting_variation': False,
        'occlusion': False,
        'scale_variation': False,
        'similar_classes': False
    }
    
    formats = set()
    sizes = set()
    quality_observations = []
    
    print("Sample images from each class:")
    for class_name in classes:
        print(f"\n{class_name}:")
        sample_paths = class_image_paths[class_name][:sample_per_class]
        
        for i, img_path in enumerate(sample_paths):
            try:
                with Image.open(img_path) as img:
                    img_array = np.array(img)
                    formats.add(img.format)
                    sizes.add(img.size)
                    
                    print(f"  Image {i+1}: {img.size}, {img.mode}")
                    
                    # Quality observations
                    if img_array.shape[0] < 100 or img_array.shape[1] < 100:
                        quality_observations.append(f"Small image in {class_name}")
                    
                    # Check for complex backgrounds
                    if len(img_array.shape) == 3:  # Color image
                        brightness_variation = np.std(img_array)
                        if brightness_variation > 80:
                            challenges['lighting_variation'] = True
                            
            except Exception as e:
                print(f"  Error analyzing {img_path}: {e}")
    
    # Report findings
    print(f"\n IDENTIFIED CHALLENGES:")
    if len(sizes) > 3:
        challenges['varying_sizes'] = True
        print(f"  Varying image sizes: {len(sizes)} different dimensions")
    
    if len(formats) > 1:
        challenges['varying_quality'] = True
        print(f"  Multiple image formats: {formats}")
    
    # Assess class similarity (this would require domain knowledge)
    print(f" Class similarity assessment:")
    print(f"   - Requires domain knowledge of jute pests")
    print(f"   - Some pests may have similar visual characteristics")
    
    return challenges

def personal_class_assessment(classes, class_image_paths):
    """Personal assessment of whether I can distinguish between classes"""
    
    print(f"\n PERSONAL CLASS DISTINCTION ASSESSMENT")
    print("-" * 40)
    
    print("Looking at the example images, here's my assessment:")
    
    # This is a subjective assessment - I'll look for visual cues
    for i, class_name in enumerate(classes):
        if class_image_paths[class_name]:
            img_path = class_image_paths[class_name][0]
            try:
                img = Image.open(img_path)
                print(f"  {class_name}: Can identify visual patterns? [Need to see images]")
            except:
                print(f"  {class_name}: Cannot load image for assessment")
    
    print("\nKey questions for manual inspection:")
    print("  1. Are there clear visual differences between classes?")
    print("  2. Do some classes look very similar?")
    print("  3. Is the pest clearly visible in each image?")
    print("  4. Are there consistent features within each class?")

# Main execution
if __name__ == "__main__":
    # Assuming the dataset is in a folder called "jute_pest_dataset"
    dataset_path = "Jute_Pest_Dataset/train"
    
    if os.path.exists(dataset_path):
        print(" Dataset found! Starting analysis...")
        
        # Explore dataset structure
        classes, class_counts, image_dimensions, color_channels, all_image_paths, class_image_paths = explore_jute_pest_dataset(dataset_path)
        
        # Print statistics
        print_dataset_statistics(classes, class_counts, image_dimensions, color_channels)
        
        # Visualize examples
        visualize_class_examples(dataset_path, classes, class_counts, class_image_paths)
        
        # Analyze challenges
        challenges = analyze_image_quality(dataset_path, classes, class_image_paths)
        
        # Personal assessment
        personal_class_assessment(classes, class_image_paths)
        
    else:
        print(f" Dataset path '{dataset_path}' not found!")
        print("Please make sure the dataset is unzipped in the current directory")
        print("Current directory contents:", os.listdir('.'))