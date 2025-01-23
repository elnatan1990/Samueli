# pip install openslide-python opencv-python torch torchvision pandas scikit-learn

import os
import pandas as pd
import numpy as np
import openslide
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class DigitalPathologyProcessor:
    def __init__(self, config):
        self.svs_files_dir = config['svs_files_dir']
        self.output_dir = config['output_dir']
        self.labels_csv = config['labels_csv']

        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'tiles'), exist_ok=True)

    def load_labels(self):
        """Load labels from CSV file"""
        try:
            self.labels_df = pd.read_csv(
                self.labels_csv,
                sep=',',  # Use a comma as the default separator
                header=None,
                names=['filename', 'label'],
                dtype={'filename': str, 'label': int}  # Ensure correct data types
            )
        except ValueError:
            # Fallback to whitespace separator if necessary
            self.labels_df = pd.read_csv(
                self.labels_csv,
                sep='\s+',  # Use whitespace as separator
                header=None,
                names=['filename', 'label'],
                dtype={'filename': str, 'label': int}  # Ensure correct data types
            )

        return self.labels_df

    def tile_svs_images(self, tile_size=256, overlap=0):
        """
        Tile SVS images and save preprocessed tiles, ignoring tiles with >50% white/black area.

        Args:
            tile_size (int): Size of each square tile
            overlap (int): Pixel overlap between tiles
        """
        tiles_info_path = os.path.join(self.output_dir, 'tiles_info.csv')

        # Check if tiles_info.csv already exists
        if os.path.exists(tiles_info_path):
            print(f"Tiles info file already exists at {tiles_info_path}. Loading existing data...")
            tiles_df = pd.read_csv(tiles_info_path)
            return tiles_df

        print("Tiles info file does not exist. Generating tiles...")
        tiles_info = []

        for index, row in self.labels_df.iterrows():
            svs_path = os.path.join(self.svs_files_dir, row['filename'])

            # Open slide
            slide = openslide.OpenSlide(svs_path)

            # Get dimensions
            width, height = slide.dimensions

            # Generate tiles
            for x in range(0, width, tile_size - overlap):
                for y in range(0, height, tile_size - overlap):
                    try:
                        # Extract tile
                        tile = slide.read_region((x, y), 0, (tile_size, tile_size))
                        tile = tile.convert('RGB')  # Convert to RGB

                        # Convert to NumPy array
                        tile_np = np.array(tile)

                        # Check white pixel ratio
                        white_threshold = 220  # Threshold for considering a pixel "white"
                        white_pixels = np.sum(np.all(tile_np >= white_threshold, axis=2))
                        total_pixels = tile_np.shape[0] * tile_np.shape[1]
                        white_ratio = white_pixels / total_pixels

                        # Check black pixel ratio
                        black_threshold = 30  # Threshold for considering a pixel "black"
                        black_pixels = np.sum(np.all(tile_np <= black_threshold, axis=2))
                        black_ratio = black_pixels / total_pixels

                        if white_ratio > 0.5:
                            print(f"Skipping tile at ({x}, {y}) due to high WHITE area ({white_ratio:.2%})")
                            continue
                        if black_ratio > 0.5:
                            print(f"Skipping tile at ({x}, {y}) due to high BLACK area ({black_ratio:.2%})")
                            continue

                        # Preprocess and save tile
                        tile_preprocessed = self.preprocess_tile(tile_np)
                        tile_filename = f"{row['filename']}_{x}_{y}.png"
                        tile_path = os.path.join(self.output_dir, 'tiles', tile_filename)
                        cv2.imwrite(tile_path, tile_preprocessed)

                        # Store tile information
                        tiles_info.append({
                            'tile_path': tile_path,
                            'label': row['label']
                        })

                    except Exception as e:
                        print(f"Error processing tile at ({x},{y}) in {row['filename']}: {e}")

        # Create tiles dataframe
        tiles_df = pd.DataFrame(tiles_info)
        tiles_df.to_csv(tiles_info_path, index=False)

        return tiles_df

    def preprocess_tile(self, tile):
        """
        Preprocess image tile

        Args:
            tile (np.array): Input tile image

        Returns:
            np.array: Preprocessed tile
        """
        # Resize
        tile_resized = cv2.resize(tile, (224, 224))

        # Normalize
        tile_normalized = tile_resized / 255.0

        # Scale back to 0-255 and convert to uint8 for saving
        tile_preprocessed = (tile_normalized * 255).astype(np.uint8)

        return tile_preprocessed

    def create_dataset(self, tiles_df):
        """
        Create train/validation split

        Args:
            tiles_df (pd.DataFrame): DataFrame with tile paths and labels

        Returns:
            tuple: Train and validation DataLoaders
        """
        # Split data maintaining class proportions
        train_df, val_df = train_test_split(
            tiles_df,
            test_size=0.3,
            stratify=tiles_df['label'],
            random_state=42
        )

        # Create custom datasets
        train_dataset = PathologyTileDataset(train_df, transform=self.get_transforms())
        val_dataset = PathologyTileDataset(val_df, transform=self.get_transforms())

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        return train_loader, val_loader

    def get_transforms(self):
        """
        Define image transformations

        Returns:
            torchvision.transforms: Transformation pipeline
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def train_model(self, train_loader, val_loader):
        """
        Train classification model

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader

        Returns:
            torch.nn.Module: Trained model
        """
        # Load pre-trained ResNet
        model = models.resnet50(pretrained=True)

        # Modify last layer for binary classification
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        best_val_accuracy = 0

        for epoch in range(10):  # 10 epochs
            model.train()
            train_loss = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_accuracy = val_correct / val_total

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))

            print(f'Epoch {epoch + 1}: Val Accuracy = {val_accuracy:.4f}')

        return model


class PathologyTileDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Custom Dataset for Pathology Tiles

        Args:
            dataframe (pd.DataFrame): DataFrame with tile paths and labels
            transform (callable, optional): Optional transform to be applied
        """
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['tile_path']
        label = self.data.iloc[idx]['label']

        # Read image
        image = cv2.imread(img_path)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    # Configuration
    config = {
        'svs_files_dir': r"C:\Users\User\Documents\SVS",
        'output_dir': r"C:\Users\User\Documents\SVS\save",
        'labels_csv': r"C:\Users\User\Documents\SVS\labels.csv"
    }

    # Initialize processor
    processor = DigitalPathologyProcessor(config)

    # Step 1: Load labels
    labels_df = processor.load_labels()
    print("Labels loaded successfully.")

    # Step 2: Tile and preprocess images
    tiles_df = processor.tile_svs_images()
    print(f"Total tiles created: {len(tiles_df)}")

    # Step 3: Create datasets
    train_loader, val_loader = processor.create_dataset(tiles_df)

    # Step 4: Train model
    model = processor.train_model(train_loader, val_loader)

    print("Digital Pathology Classification Pipeline Complete!")


if __name__ == "__main__":
    main()