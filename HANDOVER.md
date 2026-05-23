# Project Handover - Fruits & Vegetables DL Classifier

## Overview

University deep learning project: classify images into **36 classes** (fruits and vegetables)
using four different pre-trained CNN architectures, then compare their performance.

| Person | Model | Notebook |
|--------|-------|----------|
| Noha | MobileNetV2 | `01_mobilenet.ipynb` |
| Rawan | ResNet50 | `02_resnet.ipynb` |
| Hams | EfficientNetB0 | `03_efficientnet.ipynb` |
| Noha | VGG16 | `05_vgg.ipynb` |
| All | Comparison | `04_comparison.ipynb` |

---

## Environment

**Python version:** 3.10 (required - TF has no wheels for 3.14 which is the Windows default)

**Install once:**
```powershell
py -3.10 -m pip install -r requirements.txt
```

**Run notebooks:**
- In VS Code: select `Python 3.10` as the kernel (top-right of notebook)
- In terminal: `py -3.10 -m jupyter notebook`

---

## Dataset Structure

```
train/
    fruit/        → apple  banana  grapes  kiwi  mango  pineapple  pomegranate  watermelon
    vegetable/    → beetroot  bell pepper  cabbage  capsicum  carrot  cauliflower
                    chilli pepper  corn  cucumber  eggplant  garlic  ginger
                    jalepeno  lemon  lettuce  onion  orange  paprika  pear
                    peas  potato  raddish  soy beans  spinach  sweetcorn
                    sweetpotato  tomato  turnip
test/             → same class names (orange is under fruit/ here, not vegetable/)
validation/       → IGNORED (flat structure, inconsistent with train/test)
```

**Counts:** 3,115 train images | 359 test images | 36 classes
**Balance:** very well balanced (~86 images/class, CV = 10.6%)

**Key quirk:** `orange` lives under `vegetable/` in train but `fruit/` in test.
This does NOT matter because the code uses the leaf folder name as the label,
not the fruit/vegetable grouping. Both map to the same integer index.

---

## Notebook Status

| Notebook | Status | Output |
|----------|--------|--------|
| `00_preprocessing.ipynb` | Done, executed | `class_distribution.png`, `sample_images.png` |
| `01_mobilenet.ipynb` | Done, trained | `mobilenet.keras` - 93.04% test accuracy |
| `02_resnet.ipynb` | Done, trained | `resnet50_rawan.keras` |
| `03_efficientnet.ipynb` | Done, trained | `efficientnet.keras` |
| `04_comparison.ipynb` | Ready to run | Requires all 4 models saved |
| `05_vgg.ipynb` | Done, trained | `vgg.keras` |

---

## Shared Data Loading

These three functions are identical across all notebooks so that `CLASS_NAMES`
and integer label indices are consistent.

```python
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

def discover_classes(root):
    classes = {}
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir(): continue
        for cls_dir in sorted(cat_dir.iterdir()):
            if cls_dir.is_dir():
                classes[cls_dir.name.lower()] = cls_dir
    return classes

def build_file_list(root, class_to_idx):
    paths, labels = [], []
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir(): continue
        for cls_dir in sorted(cat_dir.iterdir()):
            if not cls_dir.is_dir(): continue
            cls = cls_dir.name.lower()
            if cls not in class_to_idx: continue
            for f in cls_dir.iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    paths.append(str(f))
                    labels.append(class_to_idx[cls])
    return paths, labels

def make_tf_dataset(file_paths, int_labels, n_classes,
                    img_size=(224,224), batch_size=32,
                    augment=False, shuffle=True, seed=42):
    def load(path, label):
        img = tf.image.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.cast(tf.image.resize(img, img_size), tf.float32) / 255.0
        return img, tf.one_hot(label, n_classes)
    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.random_saturation(img, 0.85, 1.15)
        return tf.clip_by_value(img, 0, 1), label
    ds = tf.data.Dataset.from_tensor_slices((file_paths, int_labels))
    if shuffle: ds = ds.shuffle(len(file_paths), seed=seed)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment: ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
```

**Validation split** (same seed = same split on every machine):
```python
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state=42
)
```

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Classification | Single-stage, 36 classes directly | Simpler, no error propagation from a binary split |
| Weights | Fine-tune ImageNet (not frozen) | Better accuracy; dataset is small so we unfreeze gradually |
| Input size | 224 x 224 | Default for all architectures |
| Normalisation | / 255 → [0, 1] in pipeline | Consistent; each model adds its own preprocessing layer |
| Val split | 20% of train, stratified, seed=42 | `validation/` folder is flat and inconsistent - ignore it |
| Test set | `test/` folder only | Never used during training, only for final reported accuracy |
| Plots | Saved to `plots/` subfolder | Keeps root directory clean |

---

## Model Specifics

### MobileNetV2 (`01_mobilenet.ipynb`)

- Input: [0, 1] - `Rescaling(scale=2, offset=-1)` inside model converts to [-1, 1]
- Phase 1: freeze base, train head only, Adam(1e-3), up to 10 epochs
- Phase 2: unfreeze layers 100+, Adam(1e-5), up to 20 epochs
- Head: GAP → BN → Dropout(0.3) → Dense(36, softmax)
- Saved as: `mobilenet.keras`
- Test accuracy: **93.04%**

### ResNet50 (`02_resnet.ipynb`)

- Input: [0, 1] - `Rescaling(255)` + `Lambda(resnet50.preprocess_input)` inside model
- Phase 1: freeze base, train head only, Adam(1e-3), up to 10 epochs
- Phase 2: unfreeze layers 140+, Adam(1e-5), up to 20 epochs
- Head: GAP → BN → Dropout(0.3) → Dense(36, softmax)
- Saved as: `resnet50_rawan.keras`

### EfficientNetB0 (`03_efficientnet.ipynb`)

- Input: [0, 255] - EfficientNet has built-in preprocessing, pipeline skips /255
- Phase 1: freeze base, train head only, Adam(1e-3), up to 20 epochs
- Phase 2: unfreeze last 30 layers, Adam(1e-4), up to 30 epochs
- Head: Dropout(0.2) → Dense(256, relu) → Dropout(0.2) → Dense(36, softmax)
- Saved as: `efficientnet.keras`

### VGG16 (`05_vgg.ipynb`)

- Input: [0, 1] - custom `VGG16Preprocess` layer (registered serializable) converts to BGR mean-subtracted
- Phase 1: freeze base, train head only, Adam(1e-3), up to 10 epochs
- Phase 2: unfreeze block5 (layers 15+), Adam(1e-5), up to 20 epochs
- Head: GAP → Dense(256, relu) → BN → Dropout(0.4) → Dense(36, softmax)
- Saved as: `vgg.keras`
- Note: VGG16Preprocess must be registered before loading the model:
  ```python
  import keras
  @keras.saving.register_keras_serializable(package='custom')
  class VGG16Preprocess(tf.keras.layers.Layer):
      def call(self, x):
          return tf.keras.applications.vgg16.preprocess_input(x * 255.0)
  ```

---

## Comparison Notebook (`04_comparison.ipynb`)

Loads all four models, evaluates on the same test set, produces:
- Accuracy + macro F1 bar chart
- Per-class F1 heatmap
- Classification reports
- Summary table

Note: EfficientNet expects [0, 255] input — the comparison notebook builds two test datasets
(`test_ds_norm` for [0,1] and `test_ds_raw` for [0,255]) and routes each model to the correct one.

---

## File Checklist

```
project/
├── 00_preprocessing.ipynb     - done, executed
├── 01_mobilenet.ipynb         - done, trained (93.04%)
├── 02_resnet.ipynb            - done, trained
├── 03_efficientnet.ipynb      - done, trained
├── 04_comparison.ipynb        - ready to run
├── 05_vgg.ipynb               - done, trained
├── class_names.json           - generated
├── mobilenet.keras            - saved
├── resnet50_rawan.keras       - saved
├── efficientnet.keras         - saved
├── vgg.keras                  - saved
├── plots/                     - all output images
├── requirements.txt
└── README.md
```
