"""
evaluate.py - Evaluates the speech-to-text predictions

This script reads the predictions CSV and calculates:
- Word Error Rate (WER) - how many words were wrong
- Character Error Rate (CER) - how many characters were wrong
- Average inference latency - how fast the model is
- Summary statistics

Then it saves everything to metrics.json and report.md

Author: Mohit Mundria, AI Enginner(Fresher)
"""

import os
import json
import pandas as pd
from jiwer import wer, cer


def load_predictions(csv_path="results/predictions.csv"):
    print("=" * 50)
    print("Loading Predictions...")
    print("=" * 50)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} predictions from {csv_path}\n")
    return df


def calculate_metrics(df):
    """
    Calculates all the evaluation metrics.
    
    WER (Word Error Rate):
        - Measures how many words the model got wrong
        - WER = (Substitutions + Insertions + Deletions) / Total Words
        - Lower is better. 0 = perfect, 1 = 100% wrong
    
    CER (Character Error Rate):
        - Same as WER but at character level
        - More detailed measurement of errors
    
    Latency:
        - How long each prediction took in seconds
    """
    print("=" * 50)
    print("Calculating Evaluation Metrics...")
    print("=" * 50)

    # Get ground truth and predictions as lists of strings
    ground_truths = df["ground_truth"].astype(str).tolist()
    predictions = df["prediction"].astype(str).tolist()
    latencies = df["latency"].tolist()

    # ---- Calculate WER (Word Error Rate) ----
    # We calculate both overall WER and per-sample WER
    overall_wer = wer(ground_truths, predictions)

    # Per-sample WER for detailed analysis
    sample_wers = []
    for gt, pred in zip(ground_truths, predictions):
        try:
            sample_wer = wer(gt, pred)
        except:
            sample_wer = 1.0  # if something goes wrong, assume worst case
        sample_wers.append(round(sample_wer, 4))

    # ---- Calculate CER (Character Error Rate) ----
    overall_cer = cer(ground_truths, predictions)

    # ---- Calculate Latency Statistics ----
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    # ---- Count perfect predictions ----
    perfect_count = sum(1 for gt, pred in zip(ground_truths, predictions)
                        if gt.strip() == pred.strip())
    accuracy = perfect_count / len(ground_truths)

    # Put everything in a dictionary
    metrics = {
        "model": "openai/whisper-small",
        "dataset": "librispeech_asr (test.clean)",
        "num_samples": len(df),
        "word_error_rate": round(overall_wer, 4),
        "character_error_rate": round(overall_cer, 4),
        "exact_match_accuracy": round(accuracy, 4),
        "perfect_predictions": perfect_count,
        "latency": {
            "average_seconds": round(avg_latency, 4),
            "min_seconds": round(min_latency, 4),
            "max_seconds": round(max_latency, 4),
            "total_seconds": round(sum(latencies), 4)
        }
    }

    # Print the metrics nicely
    print(f"\n  Word Error Rate (WER):      {overall_wer:.2%}")
    print(f"  Character Error Rate (CER): {overall_cer:.2%}")
    print(f"  Exact Match Accuracy:       {accuracy:.2%}")
    print(f"  Perfect Predictions:        {perfect_count}/{len(df)}")
    print(f"  Average Latency:            {avg_latency:.2f} seconds")
    print(f"  Total Processing Time:      {sum(latencies):.2f} seconds\n")

    return metrics


def save_metrics(metrics, output_path="results/metrics.json"):
    """
    Saves the metrics dictionary to a JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {output_path}")


def generate_report(metrics, df, output_path="results/report.md"):
    """
    Creates a nice markdown report summarizing everything.
    This makes it easy to read and present the results.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report = f"""# Speech-to-Text Evaluation Report

## Model Information
- **Model**: {metrics['model']}
- **Dataset**: {metrics['dataset']}
- **Number of Samples**: {metrics['num_samples']}

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| Word Error Rate (WER) | {metrics['word_error_rate']:.2%} |
| Character Error Rate (CER) | {metrics['character_error_rate']:.2%} |
| Exact Match Accuracy | {metrics['exact_match_accuracy']:.2%} |
| Perfect Predictions | {metrics['perfect_predictions']}/{metrics['num_samples']} |

## Latency Statistics

| Metric | Value |
|--------|-------|
| Average Latency | {metrics['latency']['average_seconds']:.4f} seconds |
| Min Latency | {metrics['latency']['min_seconds']:.4f} seconds |
| Max Latency | {metrics['latency']['max_seconds']:.4f} seconds |
| Total Processing Time | {metrics['latency']['total_seconds']:.2f} seconds |

## What These Metrics Mean

### Word Error Rate (WER)
WER tells us what percentage of words the model got wrong. It counts three types of errors:
- **Substitutions**: Wrong word (e.g., "cat" instead of "cut")
- **Insertions**: Extra words added by the model
- **Deletions**: Words that were missed

A WER of 0% means perfect transcription. Lower is better.

### Character Error Rate (CER)
CER is similar to WER but works at the character level. It gives a more fine-grained view of errors. For example, if the model predicts "helo" instead of "hello", the WER would be 100% (whole word wrong), but CER would only be about 20% (1 character missing out of 5).

## Sample Predictions

Here are the first 10 predictions for reference:

| Audio ID | Ground Truth | Prediction | Match? |
|----------|-------------|------------|--------|
"""

    # Add first 10 sample predictions to the report
    for i, row in df.head(10).iterrows():
        gt = str(row['ground_truth'])
        pred = str(row['prediction'])
        match = "✅" if gt.strip() == pred.strip() else "❌"

        # Truncate long texts for readability
        gt_display = gt[:60] + "..." if len(gt) > 60 else gt
        pred_display = pred[:60] + "..." if len(pred) > 60 else pred

        report += f"| {row['audio_id']} | {gt_display} | {pred_display} | {match} |\n"

    report += f"""
## Conclusion

The Whisper-small model achieved a Word Error Rate of **{metrics['word_error_rate']:.2%}** on {metrics['num_samples']} samples from the LibriSpeech test-clean dataset. The model processed each sample in an average of **{metrics['latency']['average_seconds']:.2f} seconds**.

{'The model performed well with a low error rate, showing that Whisper is effective for clean speech recognition.' if metrics['word_error_rate'] < 0.15 else 'The error rate suggests there is room for improvement, which could be achieved by using a larger model variant (whisper-medium or whisper-large).'}

---
*Report generated by the evaluation pipeline*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to: {output_path}\n")


def main():
    """
    Main function that runs the entire evaluation pipeline.
    """
    print("\n" + "=" * 60)
    print("  EVALUATION PIPELINE")
    print("=" * 60 + "\n")

    # Step 1: Load the predictions
    df = load_predictions()

    # Step 2: Calculate metrics
    metrics = calculate_metrics(df)

    # Step 3: Save metrics to JSON
    save_metrics(metrics)

    # Step 4: Generate markdown report
    generate_report(metrics, df)

    return metrics


if __name__ == "__main__":
    main()
