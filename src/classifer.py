import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pickle
import json
from datetime import datetime
import random
import copy

def calculate_binary_correlation_strength(df, y_col, c_col):
    """
    df: pd.DataFrame
    y_col: str, target column (e.g. "Pneumothorax")
    c_col: str, correlated variable (e.g. "Support Devices")
    """
    # Ensure binary
    y = df[y_col].astype(int)
    c = df[c_col].astype(int)

    # Compute Pearson correlation
    corr = np.corrcoef(y, c)[0, 1]
    return corr

class ImageDataset(Dataset):
    def __init__(self, df, root_folder, label, transform=None, split_name="unknown", all_dfs=None):
        self.original_df = df.reset_index(drop=True)
        self.root_folder = root_folder
        self.transform = transform
        self.split_name = split_name
        self.all_dfs = all_dfs
        self.label = label # Ensure label is a string
        
        # Build file map and filter valid samples
        self.file_map = self._build_file_map()
        self.df, self.missing_info = self._filter_valid_samples()
        
        
    def _build_file_map(self):
        # Try to load cache first
        cache_file = "file_map_cache_complete.pkl"
        
        if os.path.exists(cache_file):
            print("Loading cached complete file map...")
            try:
                with open(cache_file, 'rb') as f:
                    file_map = pickle.load(f)
                print(f"Cache loaded successfully! Found {len(file_map)} pairs.")
                return file_map
            except (EOFError, pickle.UnpicklingError, Exception) as e:
                print(f"Cache file corrupted ({e}), rebuilding...")
        
        # Build COMPLETE file map - scan ALL files regardless of which dataset
        print("Building COMPLETE file map (scanning all files)...")
        file_map = {}

        for root, dirs, files in tqdm(os.walk(self.root_folder), desc="Scanning directories"):
            parts = root.split(os.sep)
            if len(parts) < 3:  # Need at least 3 levels for MIMIC-CXR
                continue

            # MIMIC-CXR structure: .../p10/p10000032/s50414267/
            subject_folder = parts[-2]  # p10000032
            study_folder = parts[-1]    # s50414267
            
            # Extract numeric IDs (remove 'p' and 's' prefixes)
            if subject_folder.startswith('p') and study_folder.startswith('s'):
                subject_id = subject_folder[1:]  # Remove 'p' -> 10000032
                study_id = study_folder[1:]      # Remove 's' -> 50414267
                
                # Add ALL files found, not just those in current dataset
                for file in files:
                    if file.lower().endswith('.jpg'):
                        file_path = os.path.join(root, file)
                        key = (subject_id, study_id)
                        if key not in file_map:
                            file_map[key] = []
                        file_map[key].append(file_path)

        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(file_map, f)
        print(f"Complete file map saved to cache. Found {len(file_map)} pairs.")
        
        return file_map

    def _filter_valid_samples(self):
        """Filter out samples without corresponding image files"""
        print(f"Filtering valid samples for {self.split_name} set...")
        
        valid_indices = []
        missing_samples = []
        
        for idx, row in tqdm(self.original_df.iterrows(), total=len(self.original_df)):
            subject_id = str(int(row['subject_id']))
            study_id = str(int(row['study_id']))
            
            file_list = self.file_map.get((subject_id, study_id))
            
            if file_list and len(file_list) > 0:
                # Check if files actually exist
                existing_files = [f for f in file_list if os.path.exists(f)]
                if existing_files:
                    valid_indices.append(idx)
                else:
                    missing_samples.append({
                        'original_index': idx,
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'reason': 'Files in map but not on disk',
                        'mapped_files': file_list
                    })
            else:
                missing_samples.append({
                    'original_index': idx,
                    'subject_id': subject_id,
                    'study_id': study_id,
                    'reason': 'No files found in map',
                    'mapped_files': []
                })
        
        # Create filtered dataframe
        filtered_df = self.original_df.iloc[valid_indices].reset_index(drop=True)
        
        print(f"Original {self.split_name} samples: {len(self.original_df)}")
        print(f"Valid {self.split_name} samples: {len(filtered_df)}")
        print(f"Missing {self.split_name} samples: {len(missing_samples)}")
        
        missing_info = {
            'split_name': self.split_name,
            'total_original': len(self.original_df),
            'total_valid': len(filtered_df),
            'total_missing': len(missing_samples),
            'missing_samples': missing_samples,
            'timestamp': datetime.now().isoformat()
        }
        
        return filtered_df, missing_info

   

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject_id = str(int(row['subject_id']))
        study_id = str(int(row['study_id']))

        file_list = self.file_map.get((subject_id, study_id))
        if not file_list:
            raise FileNotFoundError(f"No images found for subject {subject_id}, study {study_id}")

        # Take first existing file
        file_path = None
        for f in file_list:
            if os.path.exists(f):
                file_path = f
                break
                
        if file_path is None:
            raise FileNotFoundError(f"No existing images found for subject {subject_id}, study {study_id}")

        try:
            img = Image.open(file_path).convert('L')  # Convert to grayscale
            img = img.resize((224, 224))  # Resize
            img = np.stack([np.array(img)] * 3, axis=-1)  # Convert to 3-channel
            img = Image.fromarray(img.astype(np.uint8))

            if self.transform:
                image = self.transform(img)
            else:
                image = transforms.ToTensor()(img)

            label_value = row[self.label]
            label = torch.tensor(label_value, dtype=torch.long)
            return image, label

        except Exception as e:
            raise RuntimeError(f"Error loading image {file_path}: {e}")


class ResNet18Classifier(nn.Module):
    def __init__(self, pretrained=True, freeze_base=False):
        super().__init__()
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        if freeze_base:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_model(train_df, val_df, root_folder, label_1, label_2, checkpointpath=None, epochs=10):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets with missing file handling
    # Pass all dataframes so complete file map can be built
    all_dfs = [train_df, val_df]
    train_dataset = ImageDataset(train_df, root_folder, label_1, transform=transform_train, split_name="train", all_dfs=all_dfs)
    val_dataset = ImageDataset(val_df, root_folder, label_1, transform=transform, split_name="validation", all_dfs=all_dfs)

    clean_train_df = train_dataset.df.copy()

    strength = calculate_binary_correlation_strength(clean_train_df,label_1, label_2)
    print("========== Correlation Analysis =========")
    print(f"Correlation strength between {label_1} and {label_2} in clean train set): {strength:.3f}")

    
    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Classifier().to(device)

    if checkpointpath is not None:
        checkpoint = torch.load(checkpointpath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model with accuracy: {checkpoint['accuracy']:.4f}")

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_model = None
    patience = 3
    counter = 0

    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            try:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue

        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1} done. Avg Loss = {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} - No valid batches!")
            continue

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        val_batch_count = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating"):
                try:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    preds = output.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(target.cpu().numpy())
                    val_batch_count += 1
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue

        if len(all_preds) > 0:
            acc = accuracy_score(all_labels, all_preds)
            print(f"Val Accuracy: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break
        else:
            print("No valid validation batches!")
            
    return best_acc, best_model, strength 
        


def evaluate_best_model(best_model_state_dict, test_df, root_folder, label):
    print("Evaluating best model...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageDataset(test_df, root_folder, label, transform=transform, split_name="test", all_dfs=[test_df])
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Classifier().to(device)
    model.load_state_dict(best_model_state_dict)
    model.eval()

    all_preds, all_labels = [], []
    test_indices = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
            try:
                data, target = data.to(device), target.to(device)
                output = model(data)
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                start_idx = batch_idx * test_loader.batch_size
                end_idx = min(start_idx + len(target), len(test_dataset))
                test_indices.extend(range(start_idx, end_idx))
            except Exception as e:
                print(f"Error in test batch {batch_idx}: {e}")
                continue

    if len(all_preds) > 0:
        acc = accuracy_score(all_labels, all_preds)
        print(f"Test Accuracy: {acc:.4f}")
        test_results = test_dataset.df.iloc[test_indices].copy()
        test_results["predicted"] = all_preds
        test_results["true"] = all_labels
        return test_results
    else:
        print("No valid test samples!")
        return pd.DataFrame()


  

    
    