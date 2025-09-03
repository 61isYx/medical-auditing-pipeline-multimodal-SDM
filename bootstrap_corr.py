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
from bootstrap_utils import *


# ========== Main Bootstrap Loop ==========
if __name__ == "__main__":
    label_1 = "Pneumothorax"
    label_2 = 'Support Devices'
    attribute = True
    label_1_values = 0   

    print("=================== Bootstrap Experiment (exp1) ==========")
    print(f"Experiment for {label_1} and {label_2} labels")

    # === Load Data ===
    test_df = pd.read_parquet('test_df_with_embeddings.parquet')
    #test_df = test_df[test_df["Cardiomegaly"].isin([1, 0])].reset_index(drop=True)
    print("Loading test and train DataFrames...")
    train_df = pd.read_csv("/vol/bitbucket/yl28218/thesis/mimic_cxr_exp/data/train_df.csv")
\

    # === Build Embeddings ===
    print("Create embeddings for the test DataFrame...")
    test_df["combined_embedding"] = get_combined_embedding(test_df, 'image_embedding', 'report_embedding', n_components=512)
    test_df["combined_emebdding_v2"] = get_combined_embedding(test_df, 'image_embedding', 'report_embedding', 'metadata_embedding', 512)
    test_df["combined_embedding_v3"] = get_combined_embedding(test_df, 'report_embedding', 'metadata_embedding', n_components=512)
    test_df["combined_embedding_v4"] = get_combined_embedding(test_df, 'image_embedding', 'metadata_embedding')

    root_folder = "/vol/bitbucket/yl28218/thesis/physionet.org/files/mimic-cxr-jpg/2.1.0/files"
    selected_columns = [
        label_2, 'predicted', 'true', 'id', 'path', 'report_text',
        'image_embedding', 'report_embedding', 'metadata_embedding',
        "Rows", "Columns", "metadata_description",
        "combined_embedding", "combined_emebdding_v2", "combined_embedding_v3", "combined_embedding_v4"
    ]

    checkpointpath = None
    token_results = {"image_only": [], "image_text": [], "image_text_meta": [], "text": [], "meta": [], "report_metadata": [], "image_metadata": [], "baseline": []}
    bootstrap_times = 150
    precision_log = []

    seed = 42
    k = 10
    max_features = 1000
    threshold = 0.1

    if not os.path.exists("bootstrap_results"):
        os.makedirs("bootstrap_results")

    corr =0.7

    for i in tqdm(range(bootstrap_times)):
        seed += 1
        np.random.seed(seed)
        print(f"\nBootstrap iteration {i+1}/{bootstrap_times}, seed={seed}")

        # ===== Train set: random + noisy labels =====
        train = subsample_with_correlation(train_df, label_1, label_2, target_corr=corr, sample_size=1000, seed=seed)
        # ===== Build test sets (20% / 10% / 5%) =====
        test_underperforming = test_df[(test_df[label_2] == 1) & (test_df[label_1] == 0)].copy()
        test_rest = test_df[(test_df[label_2] != 1) | (test_df[label_1] != 0)].copy()

        test = pd.concat([
            test_underperforming.sample(n=60, random_state=seed).reset_index(drop=True),
            test_rest.sample(n=240, random_state=seed).reset_index(drop=True)
        ], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

        test_small = pd.concat([
            test_underperforming.sample(n=100, random_state=seed).reset_index(drop=True),
            test_rest.sample(n=200, random_state=seed).reset_index(drop=True)
        ], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

        test_tiny = pd.concat([
            test_underperforming.sample(n=120, random_state=seed).reset_index(drop=True),
            test_rest.sample(n=180, random_state=seed).reset_index(drop=True)
        ], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

        # ===== Validation set =====
        if attribute:
            val = test[test[label_2] != 1].copy()
        else:
            val = test[test[label_2] == 1].copy()

        # ===== Train classifier =====
        best_acc, best_model, strength = train_model(train, val, root_folder, label_1, label_2, checkpointpath, epochs=10)
        if best_model is None:
            print("No valid model trained, skipping.")
            continue

        test_results = evaluate_best_model(best_model, test, root_folder, label_1)
        test_results_small = evaluate_best_model(best_model, test_small, root_folder, label_1)
        test_results_tiny = evaluate_best_model(best_model, test_tiny, root_folder, label_1)

        if len(test_results) == 0:
            print("No test results to analyze, skipping.")
            continue

        # ===== Attribute-specific check =====
        test_results_1 = test_results[test_results[label_1] == 0].copy()
        support_acc, no_support_acc = None, None
        if label_2 in test_results.columns:
            mask_support = test_results_1[label_2] == 1
            if mask_support.sum() > 0:
                support_acc = accuracy_score(test_results_1[mask_support]["true"], test_results_1[mask_support]["predicted"])
                print(f" {label_2} X-ray Accuracy: {support_acc:.4f}")
            mask_no_support = test_results_1[label_2] != 1
            if mask_no_support.sum() > 0:
                no_support_acc = accuracy_score(test_results_1[mask_no_support]["true"], test_results_1[mask_no_support]["predicted"])
                print(f" Non-{label_2} X-ray Accuracy: {no_support_acc:.4f}")

        if support_acc is None or no_support_acc is None:
            print("Could not compute support/no-support accuracies, skipping this iteration.")
            continue


        difference = (no_support_acc - support_acc) if attribute else (support_acc - no_support_acc)
        if difference < threshold:
            print("The training was unsuccessful, skipping this iteration.")
            continue
        else:
            print(f"{label_2} X-ray Accuracy  is significantly lower than no {label_2} X-ray Accuracy by {no_support_acc - support_acc:.4f}, proceeding with analysis.")

        # ===== Slice discovery + token analysis =====
        log20, tokens20 = run_slice_analysis(test_results, test, selected_columns, label_1, label_2, label_1_values, k, attribute, max_features, i, "20%")
        log10, tokens10 = run_slice_analysis(test_results_small, test_small, selected_columns, label_1, label_2, label_1_values, k, attribute, max_features, i, "10%")
        log5, tokens5 = run_slice_analysis(test_results_tiny, test_tiny, selected_columns, label_1, label_2, label_1_values, k, attribute, max_features, i, "5%")

        precision_log.extend([log20, log10, log5])
        for key in token_results.keys():
            token_results[key].extend(tokens20[key])
            token_results[key].extend(tokens10[key])
            token_results[key].extend(tokens5[key])

      
        for key in token_results.keys():
            if len(token_results[key]) > 0:
                df_tokens = pd.concat(token_results[key], ignore_index=True)
                df_tokens.to_parquet(f"bootstrap_results/all_tokens_{key}_exp1_2.parquet", index=False)
                df_summary = (
                    df_tokens.groupby(["test_ratio", "token"])
                    .agg(
                        freq_in_top20=("token", "count"),
                        mean_diff_score=("diff_score", "mean"),
                        std_diff_score=("diff_score", "std"),
                        total_error_count=("error_count", "sum"),
                        total_normal_count=("normal_count", "sum"),
                    )
                    .reset_index()
                    .sort_values(by=["test_ratio", "freq_in_top20"], ascending=[True, False])
                )
                df_summary.to_csv(f"bootstrap_results/summary_{key}_exp1_2.csv", index=False)

        if len(precision_log) > 0:
            pd.DataFrame(precision_log).to_csv("bootstrap_results/precision_log_exp1_2.csv", index=False)

        print(f"Iteration {i+1} saved ")

    print("All bootstrap iterations completed and saved.")




