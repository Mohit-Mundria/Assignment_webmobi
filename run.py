"""
run.py - Main entry point for the entire pipeline

Just run: python run.py
And it will do everything:
1. Load the Whisper model
2. Download LibriSpeech audio samples
3. Run speech-to-text on 30 samples
4. Calculate WER, CER, and latency
5. Save all results to the results/ folder

Author: BCA Graduate Assignment
"""

import sys
import os

# Add the project root to Python path so imports work properly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference import main as run_inference
from src.evaluate import main as run_evaluation


def main():
    """
    Runs the complete pipeline: inference + evaluation
    """
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#   SPEECH-TO-TEXT: INFERENCE & EVALUATION PIPELINE" + " " * 7 + "#")
    print("#   Model: OpenAI Whisper-Small" + " " * 27 + "#")
    print("#   Dataset: LibriSpeech (test-clean)" + " " * 21 + "#")
    print("#" + " " * 58 + "#")
    print("#" * 60 + "\n")

    # ---- PART 1: Run Inference ----
    print(">>> STARTING INFERENCE PIPELINE...\n")
    results = run_inference()

    # ---- PART 2: Run Evaluation ----
    print("\n>>> STARTING EVALUATION PIPELINE...\n")
    metrics = run_evaluation()

    # ---- DONE ----
    print("\n" + "=" * 60)
    print("  ALL DONE! Check the results/ folder:")
    print("  - results/predictions.csv  (model predictions)")
    print("  - results/metrics.json     (evaluation metrics)")
    print("  - results/report.md        (detailed report)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
