# Fruits_Vegetables_Classification_DL
# Fruits & Vegetables Deep Learning Classifier

A deep learning image classification system that identifies **36 fruits and vegetable classes** using fine-tuned convolutional neural networks. Built as a university deep learning project by a 3-person team, each training a different backbone architecture on the same dataset.

---

## Team

| Name | Model | Architecture |
|------|-------|-------------|
| Noha | MobileNetV2 | Lightweight, mobile-optimized |
| Rawan | ResNet50 | Deep residual network |
| Hams | EfficientNetB0 | Compound-scaled, state-of-the-art |

---

## Project Structure

```
Fruits_and_Vegetable_Deep_Learning_Classifier/
│
├── 00_preprocessing.ipynb       # Run this first — shared by all 3
├── 01_mobilenet_noha.ipynb      # Noha — MobileNetV2
├── 02_resnet_rawan.ipynb        # Rawan — ResNet50
├── 03_efficientnet_hams.ipynb   # Hams  — EfficientNetB0
├── 04_comparison.ipynb          # Final comparison of all 3 models
│
└── README.md
```

---

## Dataset

**Source:** [Fruit and Vegetable Image Recognition — Kaggle](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)

```
dataset/
├── train/
│   ├── fruit/
│   │   ├── apple/
│   │   ├── banana/
│   │   └── ... (12 fruit classes)
│   └── vegetable/
│       ├── carrot/
│       ├── beetroot/
│       └── ... (24 vegetable classes)
├── test/
│   ├── fruit/
│   └── vegetable/
└── validation/     ← not used
```

**36 Classes:**

| Fruits (12) | Vegetables (24) |
|-------------|-----------------|
| Apple, Banana, Grapes, Kiwi, Lemon, Mango, Orange, Paprika, Pear, Pineapple, Pomegranate, Watermelon | Beetroot, Bell Pepper, Cabbage, Capsicum, Carrot, Cauliflower, Chilli Pepper, Corn, Cucumber, Eggplant, Garlic, Ginger, Jalepeno, Lettuce, Onion, Peas, Potato, Raddish, Soy Beans, Spinach, Sweetcorn, Sweetpotato, Tomato, Turnip |

**Split strategy:** `validation_split=0.2` on the train folder. The separate validation folder is not used.

---

## Architecture Overview

All 3 models follow the same single-stage classification approach:

```
Input Image (224×224×3)
        ↓
  Pretrained Backbone
  (ImageNet weights,
   partially unfrozen)
        ↓
  Global Average Pooling
        ↓
  Dense Head
  (model-specific)
        ↓
  Dense(36, softmax)
        ↓
  Predicted Class
  e.g. "mango" — 94.3%
```

This is a **single-stage** classifier. The model outputs one of 36 classes directly — no intermediate fruit/vegetable binary step. The network learns categorical boundaries on its own through fine-tuning.

> This is a deliberate upgrade from our previous ML project which used a 2-stage pipeline. That approach was necessary for classical models like Decision Trees and KNN which struggle with many classes. Deep learning does not have this limitation.

---

## Model Details

### MobileNetV2 — Noha
- **Pretrained on:** ImageNet
- **Frozen layers:** first 100 of 154
- **Head:** GAP → Dense(256, relu) → Dropout(0.4) → Dense(36, softmax)
- **Learning rate:** 1e-4
- **Parameters:** ~3.4M
- **Strength:** Fastest inference, smallest size, suitable for mobile deployment

### ResNet50 — Rawan
- **Pretrained on:** ImageNet
- **Frozen layers:** first 125 of 175
- **Head:** GAP → Dense(512, relu) → Dropout(0.4) → Dense(256, relu) → Dropout(0.3) → Dense(36, softmax)
- **Learning rate:** 1e-4
- **Parameters:** ~25.6M
- **Strength:** Residual connections prevent vanishing gradients, strong generalization
- **Note:** Uses BGR channel ordering internally — never add manual `/255` rescaling

### EfficientNetB0 — Hams
- **Pretrained on:** ImageNet
- **Frozen layers:** first 200 of 237
- **Head:** GAP → Dense(512, relu) → Dropout(0.5) → Dense(256, relu) → Dropout(0.3) → Dense(36, softmax)
- **Learning rate:** 1e-5 ← lower than others, EfficientNet diverges with higher LR
- **Parameters:** ~5.3M
- **Strength:** Highest accuracy per parameter, compound scaling across width/depth/resolution

---

## Preprocessing

Each backbone requires its own normalization function. Using the wrong one silently destroys accuracy.

| Model | Import | What it does |
|-------|--------|-------------|
| MobileNetV2 | `from tensorflow.keras.applications.mobilenet_v2 import preprocess_input` | Scales [0,255] → [-1, 1] |
| ResNet50 | `from tensorflow.keras.applications.resnet50 import preprocess_input` | Converts RGB→BGR, subtracts ImageNet channel means |
| EfficientNetB0 | `from tensorflow.keras.applications.efficientnet import preprocess_input` | Scales [0,255] → [-1, 1], normalization also built into model layers |

**Never use `rescale=1./255` with any of these models.**

### Augmentation (training only)

```python
rotation_range     = 20
horizontal_flip    = True
zoom_range         = 0.15
width_shift_range  = 0.1
height_shift_range = 0.1
brightness_range   = [0.8, 1.2]
fill_mode          = 'nearest'
```

---

## Training Config

```python
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32          # drop to 16 if Colab runs out of memory
EPOCHS      = 30          # EarlyStopping will stop early if needed
SEED        = 42
NUM_CLASSES = 36
```

### Callbacks

```python
EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
ModelCheckpoint(save_best_only=True, monitor='val_accuracy')
```

---

## How to Run

### Prerequisites
- Google Colab account (free T4 GPU)
- Google Drive with dataset uploaded

### Dataset Setup on Drive
Upload the dataset to Google Drive with this structure:
```
MyDrive/
└── fruit_veg_classifier/
    ├── train/
    │   ├── fruit/
    │   └── vegetable/
    └── test/
        ├── fruit/
        └── vegetable/
```

### Execution Order

**Step 1** — Open `00_preprocessing.ipynb` on Colab
- Enable GPU: Runtime → Change runtime type → T4 GPU
- Run all cells
- Verify dataset structure and class distribution
- Saves `class_info.json` to Drive

**Step 2** — Each person opens their notebook on Colab independently
```
Noha  → 01_mobilenet_noha.ipynb
Rawan → 02_resnet_rawan.ipynb
Hams  → 03_efficientnet_hams.ipynb
```
- All 3 can run in parallel on separate Colab sessions
- Each saves their trained model to Drive on completion

**Step 3** — After all 3 models are saved, open `04_comparison.ipynb`
- Loads all 3 models
- Runs full comparison on test set
- Generates final results

---

## Outputs

Each model notebook saves to Google Drive:

```
MyDrive/fruit_veg_classifier/
├── class_info.json              ← from preprocessing
├── mobilenet_classifier.h5      ← Noha's model
├── resnet_classifier.h5         ← Rawan's model
├── efficientnet_classifier.h5   ← Hams's model
└── plots/
    ├── class_distribution.png
    ├── sample_images_raw.png
    ├── preprocessing_comparison.png
    ├── augmentation_examples.png
    ├── mobilenet_training_curves.png
    ├── mobilenet_confusion_matrix.png
    ├── resnet_training_curves.png
    ├── resnet_confusion_matrix.png
    ├── efficientnet_training_curves.png
    ├── efficientnet_confusion_matrix.png
    └── comparison_final.png
```

---

## Evaluation Metrics

Each model is evaluated on the held-out test set:

- Overall accuracy
- Per-class precision, recall, F1-score
- Confusion matrix
- Top-3 accuracy
- Inference time per image
- Grad-CAM visualizations

---

## Context

This project is the deep learning continuation of a machine learning project completed in the previous semester. The ML project compared 6 classical classifiers (Decision Tree, Random Forest, XGBoost, KNN, SVM, ANN) on frozen MobileNetV2 features using a 2-stage hierarchical pipeline.

This project asks the next question: **what happens when we stop treating feature extraction and classification as separate problems and let the network learn end-to-end?**

| | ML Project | DL Project |
|--|-----------|-----------|
| Approach | Frozen features + classical classifier | End-to-end fine-tuning |
| Pipeline | 2-stage (binary → multi-class) | Single-stage (36 classes directly) |
| Models | DT, RF, XGBoost, KNN, SVM, ANN | MobileNetV2, ResNet50, EfficientNetB0 |
| Best accuracy | 98% (KNN) | TBD |

---

## Dependencies

```
tensorflow >= 2.10
numpy
matplotlib
scikit-learn
seaborn
pathlib
```

All pre-installed on Google Colab. No local installation needed.

---

## GPU Requirements

| Model | Without GPU | With Colab T4 |
|-------|-------------|---------------|
| MobileNetV2 | ~2-3 hours | ~15-20 min |
| ResNet50 | ~6-8 hours | ~25-35 min |
| EfficientNetB0 | ~3-4 hours | ~20-25 min |

Enable GPU in Colab: Runtime → Change runtime type → T4 GPU
