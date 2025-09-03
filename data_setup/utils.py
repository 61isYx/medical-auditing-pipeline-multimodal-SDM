import pandas as pd
import numpy as np

def convert_medical_metadata_to_text(df):
    """
    Convert medical imaging metadata to unified text descriptions
    """
    
    def clean_procedure_type(row):
        """Determine examination type from multiple fields"""
        # Try ProcedureCodeSequence_CodeMeaning first (more standardized)
        procedure = row.get('ProcedureCodeSequence_CodeMeaning', '')
        if pd.isna(procedure) or procedure == '':
            # Fall back to PerformedProcedureStepDescription
            procedure = row.get('PerformedProcedureStepDescription', '')
        
        if pd.isna(procedure) or procedure == '' or procedure == 'Performed Desc':
            return "chest radiography"
        
        procedure = str(procedure).upper()
        
        # Categorize exam types
        if any(word in procedure for word in ['PORTABLE', 'PORT']):
            if any(word in procedure for word in ['LINE', 'TUBE', 'PICC']):
                return "bedside chest imaging for line placement"
            else:
                return "bedside portable chest radiography"
        elif 'TRAUMA' in procedure:
            return "emergency trauma imaging"
        elif any(word in procedure for word in ['LINE', 'TUBE', 'PICC']):
            return "chest imaging for catheter placement"
        elif 'PRE-OP' in procedure:
            return "preoperative chest examination"
        elif 'ABDOMEN' in procedure:
            return "abdominal radiography"
        elif 'RIB' in procedure:
            return "rib examination"
        else:
            return "chest radiography"
    
    def clean_view_position(row):
        """Determine view position from multiple fields"""
        # Try ViewPosition first
        view = row.get('ViewPosition', '')
        if pd.isna(view) or view == '':
            # Fall back to ViewCodeSequence_CodeMeaning
            view = row.get('ViewCodeSequence_CodeMeaning', '')
        
        if pd.isna(view) or view == '':
            return "standard view"
        
        view = str(view).upper()
        
        # Standardize view descriptions
        if view in ['AP', 'ANTERO-POSTERIOR']:
            return "anterior-posterior view"
        elif view in ['PA', 'POSTERO-ANTERIOR']:
            return "posterior-anterior view"
        elif view in ['LATERAL', 'LAT', 'LL', 'LEFT LATERAL']:
            return "lateral view"
        elif view in ['LAO', 'LEFT ANTERIOR OBLIQUE']:
            return "left anterior oblique view"
        elif view in ['RAO']:
            return "right anterior oblique view"
        else:
            return "standard view"
    
    def clean_patient_orientation(row):
        """Determine patient orientation"""
        orientation = row.get('PatientOrientationCodeSequence_CodeMeaning', '')
        
        if pd.isna(orientation) or orientation == '':
            # Most chest X-rays are performed with patient erect
            return "upright position"
        
        orientation = str(orientation).lower()
        if 'erect' in orientation:
            return "upright position"
        elif 'recumbent' in orientation:
            return "lying position"
        else:
            return "standard position"
    
    def convert_image_quality(row):
        """Convert image dimensions to quality description"""
        rows = row.get('Rows', np.nan)
        cols = row.get('Columns', np.nan)
        
        if pd.isna(rows) or pd.isna(cols):
            return "standard resolution"
        
        try:
            total_pixels = int(rows) * int(cols)
            if total_pixels > 7000000:  # >7MP
                return "high-resolution"
            elif total_pixels > 3000000:  # 3-7MP
                return "standard resolution"
            else:
                return "low-resolution"
        except:
            return "standard resolution"
    
    def convert_study_time(row):
        """Convert study date and time to readable format"""
        study_date = row.get('StudyDate', '')
        study_time = row.get('StudyTime', '')
        
        date_desc = "on unspecified date"
        time_desc = ""
        
        # Handle date
        if not pd.isna(study_date) and study_date != '':
            try:
                date_str = str(study_date)
                if len(date_str) == 8:
                    year = date_str[:4]
                    month = date_str[4:6]
                    day = date_str[6:8]
                    
                    # Convert month to name
                    months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                             'July', 'August', 'September', 'October', 'November', 'December']
                    try:
                        month_name = months[int(month)]
                        date_desc = f"in {month_name} {year}"
                    except:
                        date_desc = f"in {year}"
            except:
                date_desc = "on unspecified date"
        
        # Handle time (simplified)
        if not pd.isna(study_time) and study_time != '':
            try:
                time_str = str(study_time).split('.')[0]
                if len(time_str) >= 4:
                    hour = int(time_str[:2])
                    if 6 <= hour < 12:
                        time_desc = " during morning hours"
                    elif 12 <= hour < 18:
                        time_desc = " during afternoon hours"
                    elif 18 <= hour < 24:
                        time_desc = " during evening hours"
                    else:
                        time_desc = " during night hours"
            except:
                time_desc = ""
        
        return date_desc + time_desc
    
    # Apply all conversions
    df_copy = df.copy()
    
    # Convert categorical fields to text
    df_copy['exam_type_text'] = df_copy.apply(clean_procedure_type, axis=1)
    df_copy['view_position_text'] = df_copy.apply(clean_view_position, axis=1)
    df_copy['patient_orientation_text'] = df_copy.apply(clean_patient_orientation, axis=1)
    df_copy['image_quality_text'] = df_copy.apply(convert_image_quality, axis=1)
    df_copy['study_time_text'] = df_copy.apply(convert_study_time, axis=1)
    
    # Create unified text description
    def create_unified_description(row):
        exam_type = row['exam_type_text']
        view = row['view_position_text']
        position = row['patient_orientation_text']
        quality = row['image_quality_text']
        time = row['study_time_text']
        
        description = f"Patient underwent {exam_type} {time}. " \
                     f"The examination was performed in {view} with patient in {position}. " \
                     f"A {quality} digital image was acquired."
        
        return description
    
    df_copy['metadata_description'] = df_copy.apply(create_unified_description, axis=1)
    
    return df_copy

def visualize_umap_with_domino(df, embedding_col, domino_col, threshold=0.5, 
                               n_neighbors=15, min_dist=0.5):
    """
    Apply UMAP to embeddings and visualize, coloring points based on domino_col threshold.
    
    Parameters:
    - df: pandas DataFrame
    - embedding_col: str, column containing embedding arrays/lists
    - domino_col: str, column containing domino scores
    - threshold: float, threshold for labeling
    - n_neighbors: int, UMAP parameter
    - min_dist: float, UMAP parameter
    """
    # Extract embeddings into array
    embeddings = np.stack(df[embedding_col].values)
    
    # Compute labels on the fly: 1 if > threshold else 0
    labels = np.where(df[domino_col] > threshold, 1, 0)
    
    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                          c=labels, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, label=f"{domino_col} > {threshold}")
    plt.title('UMAP Visualization of Embeddings with Domino Threshold')
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.show()
    

def visualize_pca_with_domino(df, embedding_col, domino_col, threshold=0.5):
    """
    Apply PCA to embeddings and visualize, coloring points based on domino_col threshold.
    
    Parameters:
    - df: pandas DataFrame
    - embedding_col: str, column containing embedding arrays/lists
    - domino_col: str, column containing domino scores
    - threshold: float, threshold for labeling
    """
    # Extract embeddings into array
    embeddings = np.stack(df[embedding_col].values)
    
    # Compute labels on the fly: 1 if > threshold else 0
    labels = np.where(df[domino_col] > threshold, 1, 0)
    
    # Apply PCA
    reducer = PCA(n_components=2)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                          c=labels, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, label=f"{domino_col} > {threshold}")
    plt.title('PCA Visualization of Embeddings with Domino Threshold')
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.show()

def read_mimic_report(subject_id, study_id):
    subject_dir = f"p{str(subject_id)[:2]}"
    subject_full = f"p{subject_id}"
    report_file = f"s{study_id}.txt"
    file_path = os.path.join(REPORTS_DIR, 'files', subject_dir, subject_full, report_file)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

    

