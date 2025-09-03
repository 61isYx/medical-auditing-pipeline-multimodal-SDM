import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import meerkat as mk
from domino import DominoSlicer
#from classifer_v5 import *
from classifer import *

# ========== Utility Functions ==========
def create_subset(df, max_percent=0.4, min_percent=0.8, seed=42):
    rng = np.random.default_rng(seed)
    df_new = df.copy()
    min_rows = int(len(df_new) * min_percent)
    max_rows = int(len(df_new) * max_percent)
    n_rows = rng.integers(min_rows, max_rows + 1)
    df_subset = df_new.sample(n=n_rows, random_state=seed)
    return df_subset, n_rows

def get_df_with_rare_attr(df, label_1, label_2, size=1000, strength=0.05, 
                          label1_value=1, label2_value=1, seed=42):
   

    np.random.seed(seed)

 
    per_class_size = size // 2


    rare_size = int(size * strength)


    df_rare = df[(df[label_1] == label1_value) & (df[label_2] == label2_value)] \
                .sample(n=rare_size, random_state=seed)


    df_not_rare_target = df[(df[label_1] == label1_value) & (df[label_2] != label2_value)] \
                .sample(n=per_class_size - rare_size, random_state=seed)


    df_non_target = df[df[label_1] != label1_value] \
                .sample(n=per_class_size, random_state=seed)


    df_combined = pd.concat([df_rare, df_not_rare_target, df_non_target]).reset_index(drop=True)
    df_combined = df_combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df_combined

def add_noisy_labels(df, label_1, label_2, noise_rate, label1_value=1, label2_value=1, seed=42):
    df = df.copy()
    np.random.seed(seed)
    df['noisy_label'] = df[label_1]

    mask = (df[label_1] == label1_value) & (df[label_2] == label2_value)

    flip_mask = mask & (np.random.rand(len(df)) < noise_rate)
    df.loc[flip_mask, 'noisy_label'] = 1 - df.loc[flip_mask, 'noisy_label']
    df[label_1] = df['noisy_label']
    return df

def subsample_with_correlation(df, y_col, c_col, target_corr=0.5, sample_size=1000, seed=42):
    np.random.seed(seed)
    

    df = df[(df[y_col].isin([0,1])) & (df[c_col].isin([0,1]))].copy()


    groups = {
        (0, 0): df[(df[y_col] == 0) & (df[c_col] != 1)],
        (0, 1): df[(df[y_col] == 0) & (df[c_col] == 1)],
        (1, 0): df[(df[y_col] == 1) & (df[c_col] != 1)],
        (1, 1): df[(df[y_col] == 1) & (df[c_col] == 1)],
    }

   
    best_subset = None
    best_corr = None
    min_diff = float("inf")

    for _ in range(500): 
        samples = []
        for k, g in groups.items():
            if len(g) == 0:
                continue
            take = np.random.randint(0, min(len(g), sample_size))
            samples.append(g.sample(take))

        sub_df = pd.concat(samples)
        if len(sub_df) < sample_size:
            continue
        sub_df = sub_df.sample(sample_size)

        r = np.corrcoef(sub_df[y_col], sub_df[c_col])[0, 1]
        if np.isnan(r):
            continue
        diff = abs(r - target_corr)
        if diff < min_diff:
            min_diff = diff
            best_subset = sub_df
            best_corr = r
            if diff < 0.01:
                break  

    if best_subset is not None:
        print(f"Sampled subset with correlation ~ {best_corr:.3f}")
        return best_subset
    else:
        
        raise ValueError("Could not generate a subset with the desired correlation.")


def get_combined_embedding(df, img_col='image_embedding', text_col='report_embedding', meta_col=None, n_components=512):
    img_emb = df[img_col].tolist()
    text_emb = df[text_col].tolist() if text_col is not None else [np.array([])] * len(img_emb)
    if meta_col is not None:
        meta_emb = df[meta_col].tolist()
        combined = [np.concatenate([i, t, m]) for i, t, m in zip(img_emb, text_emb, meta_emb)]
    else:
        combined = [np.concatenate([i, t]) for i, t in zip(img_emb, text_emb)]
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(combined)
    return [np.array(row) for row in reduced]

def create_meerkat_datapanel(df, selected_columns):
    df_selected = df[selected_columns].copy()
    panel_data = {}
    for col in selected_columns:
        if col == 'path':
            panel_data['image'] = mk.ImageColumn(df_selected[col])
        elif col in ['image_embedding', 'report_embedding', 'metadata_embedding', 'combined_embedding', 'combined_emebdding_v2']:
            panel_data[col] = mk.TensorColumn(df_selected[col].tolist())
        else:
            panel_data[col] = df_selected[col].tolist()
    return mk.DataPanel(panel_data)

def domino_wrapper(mk_df, embeddings_col, targets_col, slice_name, pred_probs_col, seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    domino = DominoSlicer(
        y_log_likelihood_weight=10,
        y_hat_log_likelihood_weight=10,
        n_mixture_components=20,
        n_slices=5,
        random_state=seed,
        init_params="kmeans"
    )
    domino.fit(data=mk_df, embeddings=embeddings_col, targets=targets_col, pred_probs=pred_probs_col)
    mk_df[slice_name] = domino.predict_proba(data=mk_df, embeddings=embeddings_col, targets=targets_col, pred_probs=pred_probs_col)
    return mk_df

def error_slice_producer(slice_id, slices_cloumn, mk_df, label, slice_threshold=0.5):
    slices = np.stack(mk_df[slices_cloumn].to_numpy())
    slice_scores = slices[:, slice_id]
    high_prob_mask = slice_scores > slice_threshold
    filtered_scores = slice_scores[high_prob_mask]
    filtered_indices = np.where(high_prob_mask)[0]
    columns_to_keep = ['true', 'predicted', 'report_text', 'metadata_description', label]
    mk_pd_df = mk_df[columns_to_keep].to_pandas()
    slice_df = mk_pd_df.iloc[filtered_indices].copy()
    slice_df["slice_score"] = filtered_scores
    return slice_df.sort_values(by="slice_score", ascending=False)

def analyze_error_slice_tokens(slice_df, test_df, k=20, max_features=1000, seed=42):
    most_common_class = slice_df['true'].value_counts().idxmax()
    correct_df = test_df[(test_df['true'] == most_common_class) & (test_df['predicted'] == most_common_class)]

    error_texts = (slice_df['report_text'].fillna('') + ' ' + slice_df['metadata_description'].fillna('')).tolist()
    normal_texts = (correct_df['report_text'].fillna('') + ' ' + correct_df['metadata_description'].fillna('')).tolist()
    all_texts = normal_texts + error_texts
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 4),
        stop_words="english",
        token_pattern=r'\b[a-zA-Z]{2,}\b'
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    feature_names = vectorizer.get_feature_names_out()
    split_index = len(normal_texts)
    tfidf_array = tfidf_matrix.toarray()
    normal_avg = np.mean(tfidf_array[:split_index], axis=0)
    error_avg = np.mean(tfidf_array[split_index:], axis=0)
    #diff_scores = error_avg - normal_avg
    diff_scores = np.abs(error_avg - normal_avg)
    analyzer = vectorizer.build_analyzer()
    error_tokens = sum((analyzer(t) for t in error_texts), [])
    normal_tokens = sum((analyzer(t) for t in normal_texts), [])
    error_counts = Counter(error_tokens)
    normal_counts = Counter(normal_tokens)
    result_df = pd.DataFrame({
        'token': feature_names,
        'diff_score': diff_scores,
        'error_count': [error_counts[token] for token in feature_names],
        'normal_count': [normal_counts[token] for token in feature_names]
    })
    return result_df.sort_values(by='diff_score', ascending=False).head()

def summarize_token_results(token_result_list):
    all_tokens = pd.concat(token_result_list, axis=0)
    summary = (
        all_tokens.groupby("token")
        .agg(
            freq_in_top20=("token", "count"),
            mean_diff_score=("diff_score", "mean"),
            std_diff_score=("diff_score", "std"),
            total_error_count=("error_count", "sum"),
            total_normal_count=("normal_count", "sum")
        )
        .reset_index()
        .sort_values(by="freq_in_top20", ascending=False)
    )
    return summary

def compute_precision_k(df, label, k=10, attribute=True):
    df = df.copy()
    

    df[label] = df[label].fillna(0)
    df[label] = df[label].apply(lambda x: 1 if x == 1 else 0)

    top_k = df.head(k)

    if attribute:
     
        precision_k = (top_k[label] * top_k["slice_score"]).sum() / k
    else:
        # Flip label：1 → 0, 0 → 1
        flipped_label = 1 - top_k[label]
        precision_k = (flipped_label * top_k["slice_score"]).sum() / k

    return precision_k

def best_precision_slice(embedding_name, mk_df, label_2, label_1_values, k, attribute, num_slices=5):
    best_slice = None
    best_precision = -1
    best_id = -1

    for slice_id in range(num_slices):
   
        current_slice = error_slice_producer(slice_id, embedding_name, mk_df, label_2)
        most_common_class = current_slice['true'].value_counts().idxmax() 
        if most_common_class == label_1_values:
            current_precision = compute_precision_k(current_slice, label_2, k, attribute)
        else:
            current_precision = 0

        if current_precision > best_precision:
            best_precision = current_precision
            best_slice = current_slice
            best_id = slice_id

    return best_slice, best_precision

def compute_baseline_precision_k(test_results, label_1, label_2, attribute):
    """
    Baseline precision@K: directly on incorrect samples, no slice discovery.
    """

    incorrect_df = test_results[test_results['true'] != test_results['predicted']].copy()
    if len(incorrect_df) == 0:
        return None

    if attribute:
        precision_k = (incorrect_df[label_2] == 1).sum() / len(incorrect_df)
    else:
        precision_k = (top_k[label_2] != 1).sum() / len(incorrect_df)

    return precision_k

def run_slice_analysis(
    test_results,   
    test_df,       
    selected_columns,
    label_1,
    label_2,
    label_1_values,
    k,
    attribute,
    max_features,
    iteration,
    test_ratio
):
    """
    Run slice discovery, precision@K, token analysis for one test set.
    Returns:
        precision_log_entry (dict)
        token_results (dict of lists of DataFrames)
    """
   
    token_results = {
        "image_only": [],
        "image_text": [],
        "image_text_meta": [],
        "text": [],
        "meta": [],
        "report_metadata": [],
        "image_metadata": [],
        "baseline": []
    }

    # === Domino slice discovery ===
    mk_df = create_meerkat_datapanel(test_results, selected_columns)
    mk_df = domino_wrapper(mk_df, 'image_embedding', 'true', 'slice_image_only', 'predicted')
    mk_df = domino_wrapper(mk_df, 'combined_embedding', 'true', 'slice_image_text', 'predicted')
    mk_df = domino_wrapper(mk_df, 'combined_emebdding_v2', 'true', 'slice_image_text_meta', 'predicted')
    mk_df = domino_wrapper(mk_df, "report_embedding", 'true', 'slice_report_text', 'predicted')
    mk_df = domino_wrapper(mk_df, "metadata_embedding", 'true', 'slice_metadata', 'predicted')
    mk_df = domino_wrapper(mk_df, "combined_embedding_v3", 'true', 'slice_report_metadata', 'predicted')
    mk_df = domino_wrapper(mk_df, "combined_embedding_v4", 'true', 'slice_image_metadata', 'predicted')

    # === Best precision slice ===
    slice_1, precision_k_1 = best_precision_slice('slice_image_only', mk_df, label_2, label_1_values, k, attribute)
    slice_2, precision_k_2 = best_precision_slice('slice_image_text', mk_df, label_2, label_1_values, k, attribute)
    slice_3, precision_k_3 = best_precision_slice('slice_image_text_meta', mk_df, label_2, label_1_values, k, attribute)
    slice_4, precision_k_4 = best_precision_slice('slice_report_text', mk_df, label_2, label_1_values, k, attribute)
    slice_5, precision_k_5 = best_precision_slice('slice_metadata', mk_df, label_2, label_1_values, k, attribute)
    slice_6, precision_k_6 = best_precision_slice('slice_report_metadata', mk_df, label_2, label_1_values, k, attribute)
    slice_7, precision_k_7 = best_precision_slice('slice_image_metadata', mk_df, label_2, label_1_values, k, attribute)

    # === Token analysis ===
    tokens_slice_1 = analyze_error_slice_tokens(slice_1, test_results, k, max_features)
    tokens_slice_2 = analyze_error_slice_tokens(slice_2, test_results, k, max_features)
    tokens_slice_3 = analyze_error_slice_tokens(slice_3, test_results, k, max_features)
    tokens_slice_4 = analyze_error_slice_tokens(slice_4, test_results, k, max_features)
    tokens_slice_5 = analyze_error_slice_tokens(slice_5, test_results, k, max_features)
    tokens_slice_6 = analyze_error_slice_tokens(slice_6, test_results, k, max_features)
    tokens_slice_7 = analyze_error_slice_tokens(slice_7, test_results, k, max_features)
    tokens_baseline = analyze_error_slice_tokens(test_results[test_results['true'] != test_results['predicted']], test_results, k, max_features)

    
    if tokens_slice_1 is not None:
        tokens_slice_1['bootstrap_iteration'] = iteration
        tokens_slice_1['test_ratio'] = test_ratio
        token_results["image_only"].append(tokens_slice_1)
    if tokens_slice_2 is not None:
        tokens_slice_2['bootstrap_iteration'] = iteration
        tokens_slice_2['test_ratio'] = test_ratio
        token_results["image_text"].append(tokens_slice_2)
    if tokens_slice_3 is not None:
        tokens_slice_3['bootstrap_iteration'] = iteration
        tokens_slice_3['test_ratio'] = test_ratio
        token_results["image_text_meta"].append(tokens_slice_3)
    if tokens_slice_4 is not None:
        tokens_slice_4['bootstrap_iteration'] = iteration
        tokens_slice_4['test_ratio'] = test_ratio
        token_results["text"].append(tokens_slice_4)
    if tokens_slice_5 is not None:
        tokens_slice_5['bootstrap_iteration'] = iteration
        tokens_slice_5['test_ratio'] = test_ratio
        token_results["meta"].append(tokens_slice_5)
    if tokens_slice_6 is not None:
        tokens_slice_6['bootstrap_iteration'] = iteration
        tokens_slice_6['test_ratio'] = test_ratio
        token_results["report_metadata"].append(tokens_slice_6)
    if tokens_slice_7 is not None:
        tokens_slice_7['bootstrap_iteration'] = iteration
        tokens_slice_7['test_ratio'] = test_ratio
        token_results["image_metadata"].append(tokens_slice_7)
    if tokens_baseline is not None:
        tokens_baseline['bootstrap_iteration'] = iteration
        tokens_baseline['test_ratio'] = test_ratio
        token_results["baseline"].append(tokens_baseline)
    
    baseline_precision_k = compute_baseline_precision_k(test_results, label_1, label_2, attribute)

   
    precision_log_entry = {
        'bootstrap_iteration': iteration,
        'test_ratio': test_ratio,
        'precision_k_image_only': precision_k_1,
        'precision_k_image_text': precision_k_2,
        'precision_k_image_text_meta': precision_k_3,
        'precision_k_report_text': precision_k_4,
        'precision_k_metadata': precision_k_5,
        'precision_k_report_metadata': precision_k_6,
        'precision_k_image_metadata': precision_k_7,
        'precision_k_baseline': baseline_precision_k
    }

    return precision_log_entry, token_results
