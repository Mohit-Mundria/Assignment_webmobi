# Speech-to-Text Inference & Evaluation Pipeline

A simple and reproducible pipeline that uses **OpenAI's Whisper-Small** model to convert speech to text, and then evaluates how accurate the transcriptions are.

Built as a BCA assignment to demonstrate understanding of ML models, Hugging Face integration, and evaluation metrics.

---

## What This Project Does

1. **Downloads** the Whisper-small speech recognition model from Hugging Face
2. **Loads** audio samples from the LibriSpeech dataset (a popular speech benchmark)
3. **Runs inference** — feeds audio to the model and gets text predictions
4. **Evaluates** the predictions using WER (Word Error Rate) and CER (Character Error Rate)
5. **Saves** everything (predictions CSV, metrics JSON, and a readable report)

---

## Project Structure

```
Assignment_webmobi/
├── run.py                  # Main entry point — just run this!
├── requirements.txt        # Python packages needed
├── README.md               # You're reading this
│
├── src/
│   ├── __init__.py         # Makes src a Python package
│   ├── inference.py        # Loads model + runs speech-to-text
│   └── evaluate.py         # Calculates WER, CER, latency
│
├── research/
│   └── research_summary.md # 1-2 page summary of the Whisper paper
│
└── results/                # Created after running the pipeline
    ├── predictions.csv     # Model predictions vs ground truth
    ├── metrics.json        # Evaluation metrics in JSON format
    └── report.md           # Human-readable evaluation report
```

---

## How to Run

### Step 1: Install Dependencies

Make sure you have Python 3.8+ installed. Then run:

```bash
pip install -r requirements.txt
```

> **Note**: If you have a GPU with CUDA, PyTorch will use it automatically for faster inference. If not, it'll run on CPU (slower but works fine).

### Step 2: Run the Pipeline

```bash
python run.py
```

That's it! The script will:
- Download the model (first run only, ~1GB)
- Download audio samples from LibriSpeech
- Run speech-to-text on 30 samples
- Calculate evaluation metrics
- Save everything to the `results/` folder

### Step 3: Check Results

After running, open the `results/` folder:
- **predictions.csv** — See what the model predicted vs the correct text
- **metrics.json** — Machine-readable metrics (WER, CER, latency)
- **report.md** — A nice report you can read or submit

---

## Model Details

| Detail | Value |
|--------|-------|
| Model | openai/whisper-small |
| Parameters | ~244 Million |
| Task | Automatic Speech Recognition (ASR) |
| Source | [Hugging Face](https://huggingface.co/openai/whisper-small) |
| Dataset | LibriSpeech test-clean |
| Samples | 30 audio clips |

---

## Evaluation Metrics Explained

- **WER (Word Error Rate)**: What percentage of words were transcribed incorrectly. Lower = better.
- **CER (Character Error Rate)**: Same as WER but at character level. More fine-grained.
- **Exact Match Accuracy**: How many transcriptions were 100% correct.
- **Latency**: How long each audio sample took to process.

---

## Technologies Used

- **Python 3.8+**
- **Hugging Face Transformers** — For loading the Whisper model
- **Hugging Face Datasets** — For loading LibriSpeech audio data
- **PyTorch** — The deep learning framework Whisper runs on
- **jiwer** — Library to calculate WER and CER
- **pandas** — For handling CSV data

---

## Research Summary

The research summary of the Whisper paper is available at: [research/research_summary.md](research/research_summary.md)

It covers:
- What problem Whisper solves
- How the architecture works
- Why it's better than previous models
- Datasets used for training
- Limitations of the model

---

## Troubleshooting

**"Out of memory" error?**
- This usually happens on systems with less than 4GB RAM. Try closing other applications.

**"No module named transformers"?**
- Make sure you installed the requirements: `pip install -r requirements.txt`

**Takes too long?**
- On CPU, each sample takes about 5-15 seconds. The full pipeline (30 samples) should finish in about 5-10 minutes.
- If you have an NVIDIA GPU, it'll be much faster (under 2 minutes).

**Internet error while downloading?**
- The first run needs internet to download the model and dataset. After that, they're cached locally.

---

## Author

BCA Graduate — Assignment for ML/AI practical demonstration.

---

## License

This project is for educational purposes only. The Whisper model is licensed by OpenAI under MIT License.
