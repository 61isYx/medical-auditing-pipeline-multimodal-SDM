import zipfile
import os
from tqdm import tqdm

# ZIP file path
zip_path = '/vol/bitbucket/yl28218/physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip'
# Custom extraction directory
extract_path = '/vol/bitbucket/yl28218/thesis/physionet.org/files/mimic-cxr-jpg/2.1.0/files/mimic_reports'

# Create directory
os.makedirs(extract_path, exist_ok=True)

print(f"Extracting report files to {extract_path}...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
 
    file_list = zip_ref.namelist()
    
  
    for file in tqdm(file_list, desc="Extracting", unit="file"):
        zip_ref.extract(file, extract_path)

print("Extraction completed!")