# Prompt Injection Detector

A binary text classifier (`safe` vs `injection`) built by fine-tuning DeBERTa-v3-small.

- Model: https://huggingface.co/Anant9923/prompt-injection-detector


## Task
Label 1 = prompt injection (text that tries to override or leak the instructions of an AI system).
Label 0 = normal text.

## Data
- `deepset/prompt-injections` and `xTRam1/safe-guard-prompt-injection` (Hugging Face)
- About 10.8k samples after removing duplicates, split 80/10/10 (train/val/test)
- The final model (v3) adds about 230 template-based examples that use words like
  "ignore", "forget", "act as" in both harmless and malicious ways (training set only)

## Results (held-out test set, 1,080 samples)

| Model | Accuracy | Injection F1 | False positives | Missed injections | Hand-made hard set (14) |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.970 | 0.950 | 0 | 32 | 7/14 |
| DeBERTa-v3-small v1 | 0.990 | 0.984 | 5 | 6 | 11/14 |
| DeBERTa-v3-small v3 (final) | 0.994 | 0.989 | 0 | 7 | 14/14 |

The hard set was used while developing v3, so treat it as a dev set, not final proof.

## How it was improved
- v1 raised false alarms on normal prompts containing words like "ignore", "forget" or "act as".
- v2 added generic instruction data (Dolly/Alpaca) and did not help (10/14 on the hard set).
- v3 added template-based minimal pairs (same trigger words in harmless and malicious use) and removed all false positives on the test set.

## Quickstart
bash
pip install -r requirements.txt
python src/predict.py     # example predictions
python app.py             # local Gradio demo


## Repo layout
- `src/predict.py`: inference function
- `app.py`: Gradio demo
- `notebooks/`: training and evaluation notebook
- `reports/`: confusion matrix, error lists, results

## Limitations
- Evaluated on a random split plus hand-made prompts; generalization to other datasets was not fully tested.
- Weakest on instructions hidden inside a harmless task and on non-English text (German samples were missed).
- Some labels in the source datasets look inconsistent ("act as ..." prompts appear with both labels).
- One layer of defense, not a complete solution.