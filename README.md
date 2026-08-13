"""
Analysis/interpretability_ig.py
==================================
Comment 1 core deliverable: quantitative cross-lingual interpretability
using Integrated Gradients (via captum), run on PARALLEL English-Bengali
test pairs (matched by row_id, which every Dataset/*/mental_*_test_data.xlsx
file carries).

Why "position-bin profiles" instead of raw token overlap:
English and Bengali tokenizers produce completely different token
vocabularies/subwords, so there's no shared token ID to compare directly
across languages. Instead, for each sample we:
  1. Get per-token IG attribution scores (magnitude of contribution to
     the predicted class).
  2. Normalize each sequence's attribution into K fixed position bins
     (relative position 0%-100% through the sentence), so a 40-token
     English sentence and a 55-token Bengali translation become
     directly comparable K-length vectors.
  3. Compare the English bin-profile to the Bengali bin-profile for the
     SAME underlying post via cosine similarity and Spearman rank
     correlation.
A high similarity means the model is "looking at" the same relative
part of the post in both languages; a low similarity is quantitative
evidence of reduced cross-lingual interpretability -- directly what the
reviewer asked for instead of the previous 4 hand-picked examples.

Requires two ALREADY-TRAINED models on the SAME architecture, one per
language (e.g. Results/English/xlm_roberta/best_model and
Results/Bengali/xlm_roberta/best_model) -- use a multilingual model
(mbert/xlm_roberta/modern_bert) since BanglaBERT/BERT-base only cover
one language and can't be compared this way.

Install: pip install captum

Usage:
    python Analysis/interpretability_ig.py \
        --en_model_dir Results/English/xlm_roberta/best_model \
        --bn_model_dir Results/Bengali/xlm_roberta/best_model \
        --en_test_file Dataset/english/mental_english_test_data.xlsx \
        --bn_test_file Dataset/bengali/mental_bengali_test_data.xlsx \
        --output_dir Results/Interpretability/xlm_roberta_IG \
        --n_bins 10 \
        --n_steps 50 \
        --internal_batch_size 4
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from scipy.stats import spearmanr
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_ig_attributions(model, tokenizer, text, device, max_length=512, n_steps=50, internal_batch_size=4):
    """Returns (tokens, per-token attribution magnitude) for the model's
    predicted class, using the [PAD] token embedding as baseline.

    internal_batch_size caps how many of the n_steps interpolated copies
    of the input Captum processes at once. Without this, Captum builds
    ALL n_steps copies as one giant batch before running them through
    the model -- for a long sequence (e.g. ModernBERT's longer inputs)
    that single batch can exceed GPU memory even though the model
    itself is small. Lowering this trades a bit of speed for a much
    lower peak memory footprint, independent of sequence length.
    """
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    pad_id = tokenizer.pad_token_id or 0
    baseline_ids = torch.full_like(input_ids, pad_id)

    embeddings_layer = model.get_input_embeddings()
    lig = LayerIntegratedGradients(
        lambda ids: model(input_ids=ids, attention_mask=attention_mask).logits,
        embeddings_layer,
    )

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred_class = int(torch.argmax(logits, dim=-1).item())

    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        target=pred_class,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )
    # sum over embedding dim -> one score per token, then take magnitude
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    token_scores = np.abs(token_scores)

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().tolist())

    # free the large intermediate tensors/graph explicitly rather than
    # waiting on Python's garbage collector, and release cached CUDA
    # memory back to the allocator so it doesn't fragment/accumulate
    # across thousands of sequential single-sample calls.
    del attributions, logits, enc, input_ids, attention_mask, baseline_ids
    if device == "cuda":
        torch.cuda.empty_cache()

    return tokens, token_scores, pred_class


def bin_profile(scores, n_bins):
    """Normalize a variable-length attribution vector into n_bins equal
    relative-position bins (sum-normalized to 1 so different sequence
    lengths / total attribution magnitudes are comparable)."""
    if len(scores) == 0 or scores.sum() == 0:
        return np.zeros(n_bins)
    bin_edges = np.linspace(0, len(scores), n_bins + 1).astype(int)
    profile = np.array([
        scores[bin_edges[i]:bin_edges[i + 1]].sum() for i in range(n_bins)
    ])
    total = profile.sum()
    return profile / total if total > 0 else profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--en_model_dir", required=True)
    p.add_argument("--bn_model_dir", required=True)
    p.add_argument("--en_test_file", required=True)
    p.add_argument("--bn_test_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--n_steps", type=int, default=50)
    p.add_argument("--internal_batch_size", type=int, default=4,
                    help="lower this if you hit CUDA OOM on long sequences (e.g. ModernBERT); "
                         "raise it for speed if you have memory to spare")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="optional cap on rows for a quick test run")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    en_test = pd.read_excel(args.en_test_file)
    bn_test = pd.read_excel(args.bn_test_file)
    paired = en_test.merge(bn_test, on="row_id", suffixes=("_en", "_bn"))
    if args.limit:
        paired = paired.head(args.limit)

    print(f"Loading models on {device} ...")
    en_tokenizer = AutoTokenizer.from_pretrained(args.en_model_dir)
    en_model = AutoModelForSequenceClassification.from_pretrained(args.en_model_dir).to(device).eval()
    bn_tokenizer = AutoTokenizer.from_pretrained(args.bn_model_dir)
    bn_model = AutoModelForSequenceClassification.from_pretrained(args.bn_model_dir).to(device).eval()

    rows = []
    for i, row in paired.iterrows():
        en_text = str(row["English_Description"])
        bn_text = str(row["Bengali_Description"])
        try:
            _, en_scores, en_pred = get_ig_attributions(
                en_model, en_tokenizer, en_text, device, args.max_length, args.n_steps,
                args.internal_batch_size,
            )
            _, bn_scores, bn_pred = get_ig_attributions(
                bn_model, bn_tokenizer, bn_text, device, args.max_length, args.n_steps,
                args.internal_batch_size,
            )
        except Exception as e:
            print(f"[row {row['row_id']}] skipped due to error: {e}")
            continue

        en_profile = bin_profile(en_scores, args.n_bins)
        bn_profile = bin_profile(bn_scores, args.n_bins)

        # cosine similarity
        denom = (np.linalg.norm(en_profile) * np.linalg.norm(bn_profile))
        cosine_sim = float(np.dot(en_profile, bn_profile) / denom) if denom > 0 else 0.0

        # spearman rank correlation across bins
        if np.std(en_profile) > 0 and np.std(bn_profile) > 0:
            rho, _ = spearmanr(en_profile, bn_profile)
        else:
            rho = 0.0

        rows.append({
            "row_id": row["row_id"],
            "Mental_State": row.get("Mental_State_en", row.get("Mental_State")),
            "en_pred_class_idx": en_pred,
            "bn_pred_class_idx": bn_pred,
            "en_n_tokens": len(en_scores),
            "bn_n_tokens": len(bn_scores),
            "cosine_similarity": cosine_sim,
            "spearman_rho": float(rho) if rho == rho else 0.0,  # guard NaN
        })

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(paired)}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(f"{args.output_dir}/ig_cross_lingual_overlap.csv", index=False)

    def agg(series):
        return {"mean": float(series.mean()), "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0}

    summary = {
        "n_pairs_evaluated": int(len(result_df)),
        "n_bins": args.n_bins,
        "n_steps": args.n_steps,
        "internal_batch_size": args.internal_batch_size,
        "overall_cosine_similarity": agg(result_df["cosine_similarity"]),
        "overall_spearman_rho": agg(result_df["spearman_rho"]),
        "by_class": {},
    }
    if "Mental_State" in result_df.columns:
        for cls, g in result_df.groupby("Mental_State"):
            summary["by_class"][cls] = {
                "n": int(len(g)),
                "cosine_similarity": agg(g["cosine_similarity"]),
                "spearman_rho": agg(g["spearman_rho"]),
            }

    with open(f"{args.output_dir}/ig_cross_lingual_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nOverall cosine similarity: {summary['overall_cosine_similarity']}")
    print(f"Overall spearman rho:      {summary['overall_spearman_rho']}")
    print(f"Wrote {args.output_dir}/ig_cross_lingual_overlap.csv and ig_cross_lingual_summary.json")


if __name__ == "__main__":
    main()
