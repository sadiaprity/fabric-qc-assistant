# AI Fabric Quality Assurance Assistant

An AI system that classifies garment fabric defects from images, explains *where* it's looking via Grad-CAM, and generates grounded quality-control recommendations using a retrieval-augmented generation (RAG) layer built on a custom textile QC knowledge base.

Built as an applied portfolio project combining computer vision (defect classification), interpretability (Grad-CAM), and retrieval-augmented generation — targeting real operational challenges in garment/RMG (ready-made garment) manufacturing quality control.

## What it does

1. **Classifies fabric defects** — a fine-tuned ResNet50 model predicts one of 9 defect categories (hole, stain, broken stitch, needle mark, pinched fabric, and 4 line/bar defect types) plus a defect-free class, from a single fabric image.
2. **Shows its reasoning visually** — Grad-CAM generates a heatmap overlay showing which region of the image drove the prediction, so a human inspector can visually verify the model isn't guessing.
3. **Generates a grounded QC recommendation** — a RAG layer retrieves relevant entries from a hand-written QC knowledge base (defect definitions, severity/AQL tolerance guidance, recommended actions) and an LLM (Gemini 2.5 Flash) turns that into a plain-language note for an inspector.
4. **Routes low-confidence predictions to a human** — predictions below an 85% confidence threshold are automatically flagged for manual review rather than acted on automatically.

## Architecture

```
Fabric image
     │
     ▼
CV classifier (ResNet50, fine-tuned)  ──► predicted defect type + confidence
     │
     ▼
Grad-CAM  ──► heatmap showing where the model looked
     │
     ▼
RAG retrieval (sentence-transformers embeddings over a 29-entry QC knowledge base,
using query decomposition — separate sub-queries for "what is this defect" and
"what should be done about it" rather than one flat search)
     │
     ▼
LLM reasoning (Gemini 2.5 Flash)  ──► plain-language QC note, severity + action
     │
     ▼
Streamlit dashboard  ──► image + heatmap + QC note + human-review flag, on one screen
```

## Results

Trained and evaluated on a 3,077-image, 9-class fabric defect dataset (real production-line images captured under controlled industrial camera conditions).

- **Test accuracy: 96.1%** (weighted F1: 0.96)
- Strong per-class performance on structurally distinct defects (hole, broken stitch, needle mark, pinched fabric)
- **Known limitation:** the model occasionally confuses clean fabric with a minor stain (~4% false-positive rate on defect-free samples) — the safer failure direction for a QC system, since it triggers extra manual review rather than missing a real defect. Documented and handled explicitly in the knowledge base (see `kb026`).
- **Known limitation:** performance degrades on casual phone-camera images, since the model was trained on controlled industrial-camera conditions (fixed distance, consistent lighting). This is expected distribution shift, not a model bug — and the confidence-threshold routing correctly produces low-confidence scores on these out-of-distribution images rather than confidently misclassifying them.

See `assets/confusion_matrix.png` and `assets/gradcam_random_samples.png` for full breakdowns.

## Real world testing

Informally tested on a casual phone-camera photo (outside the training distribution of controlled industrial-camera images). The model correctly produced a low-confidence prediction (36.7%) rather than a falsely confident one, triggering the human-review flag as designed — demonstrating the confidence-threshold safety mechanism working as intended on out-of-distribution input.


## Tech stack

- **CV:** PyTorch, torchvision (ResNet50 transfer learning), Grad-CAM
- **RAG:** sentence-transformers (`all-MiniLM-L6-v2`, local embeddings, no API cost), Gemini 2.5 Flash (generation)
- **Dashboard:** Streamlit
- **Dataset:** [Multi-Class Fabric Defect Detection Dataset](https://www.kaggle.com/datasets/ziya07/multi-class-fabric-defect-detection-dataset) (Kaggle)

## Project structure

```
fabric-qc-assistant/
├── README.md
├── requirements.txt
├── train_fabric_defect_classifier.py   # CV model training (run on Colab/GPU)
├── rag_reasoning_layer.py              # RAG retrieval + Gemini reasoning
├── streamlit_app.py                    # Dashboard tying everything together
├── qc_knowledge_base.json              # 29-entry QC knowledge base
└── assets/
    ├── confusion_matrix.png
    ├── gradcam_random_samples.png
    └── app_demo.png
```

Note: the trained model weights (`fabric_defect_classifier.pt`, ~95MB) are not committed to this repo due to size. Retrain using `train_fabric_defect_classifier.py` on the linked dataset, or contact me for the weights file.

## Running it locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"   # free tier at https://aistudio.google.com/apikey
streamlit run streamlit_app.py
```

## Honest scope notes

This is a portfolio-scale project, not a production deployment — trained on a public dataset, tested by me, not validated on an actual factory floor. The value demonstrated is the *combination*: perception (CV), interpretability (Grad-CAM), and grounded reasoning (RAG) wired into one coherent pipeline for a real industry problem, not a novel technique in any individual component. See fabric_qc_training_notebook.ipynb for the full training run and evaluation outputs.

## Author

Sadia Akhter
— [LinkedIn](https://www.linkedin.com/in/sadia-akhter-prity/)
