"""
inference.py - Runs speech-to-text inference using OpenAI Whisper model

This script does 3 things:
1. Downloads the Whisper model and processor from Hugging Face
2. Loads the LibriSpeech test dataset
3. Runs inference on audio samples and saves predictions to CSV

Author: BCA Graduate Assignment
"""

import os
import time
import pandas as pd
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset


def load_model():
    """
    Downloads and loads the Whisper-small model and its processor.
    The processor converts audio to features that the model can understand.
    The model then generates text from those features.
    """
    print("=" * 50)
    print("Step 1: Loading Whisper Model from Hugging Face...")
    print("=" * 50)

    model_name = "openai/whisper-small"

    # Processor = converts raw audio into mel spectrogram features
    processor = WhisperProcessor.from_pretrained(model_name)

    # Model = the actual neural network that does speech-to-text
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on: {device}")
    print(f"Model size: whisper-small (~244M parameters)\n")

    return processor, model, device


def load_speech_dataset(num_samples=30):
    """
    Loads the LibriSpeech ASR dataset from Hugging Face.
    We use the 'test.clean' split because it has clean audio recordings.
    
    Args:
        num_samples: How many audio samples to process (default 30)
    """
    print("=" * 50)
    print("Step 2: Loading LibriSpeech Dataset...")
    print("=" * 50)

    # Load the clean test split of LibriSpeech
    # streaming=True means we don't download the entire dataset at once
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="test",
        streaming=True,
        trust_remote_code=True
    )

    # Take only the number of samples we need
    samples = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(sample)

    print(f"Loaded {len(samples)} audio samples")
    print(f"Sample rate: 16000 Hz (required by Whisper)\n")

    return samples


def run_inference(processor, model, device, samples):
    """
    Runs the Whisper model on each audio sample to get text predictions.
    Also records how long each prediction takes (latency).
    
    Returns:
        results: list of dicts with audio_id, ground_truth, prediction, latency
    """
    print("=" * 50)
    print("Step 3: Running Inference on Audio Samples...")
    print("=" * 50)

    results = []

    for i, sample in enumerate(samples):
        # Get the audio data and the correct transcription
        audio = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        ground_truth = sample["text"].upper()  # Whisper outputs uppercase

        # Convert audio to model input features (mel spectrogram)
        input_features = processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_features.to(device)

        # Measure how long inference takes
        start_time = time.time()

        # Generate prediction - the model converts audio features to text
        with torch.no_grad():  # no_grad = we're not training, just predicting
            predicted_ids = model.generate(input_features)

        # Decode the predicted token IDs back to readable text
        prediction = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0].upper().strip()

        end_time = time.time()
        latency = end_time - start_time

        # Store the result
        results.append({
            "audio_id": f"{i + 1:04d}",  # like 0001, 0002, etc.
            "ground_truth": ground_truth,
            "prediction": prediction,
            "latency": round(latency, 4)
        })

        # Print progress every 5 samples
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples | "
                  f"Latency: {latency:.2f}s")

    print(f"\nDone! Processed all {len(samples)} samples.\n")
    return results


def save_predictions(results, output_path="results/predictions.csv"):
    """
    Saves the inference results to a CSV file.
    """
    # Create the results folder if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(f"Total samples: {len(results)}\n")

    return df


def main():
    """
    Main function that runs the entire inference pipeline.
    """
    print("\n" + "=" * 60)
    print("  WHISPER SPEECH-TO-TEXT INFERENCE PIPELINE")
    print("=" * 60 + "\n")

    # Step 1: Load the model
    processor, model, device = load_model()

    # Step 2: Load the dataset (30 samples as per assignment requirement)
    samples = load_speech_dataset(num_samples=30)

    # Step 3: Run inference
    results = run_inference(processor, model, device, samples)

    # Step 4: Save predictions to CSV
    save_predictions(results)

    return results


if __name__ == "__main__":
    main()
