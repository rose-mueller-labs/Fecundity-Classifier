# FecundityModel 98K V1

A convolutional neural network (CNN) trained to classify **fruit fly fecundity-related image data** into a maximum of 43 eggs in one tile. This model is part of ongoing research on developing machine learning models to automatically count eggs through images.

See more details about this project [here](https://drive.google.com/file/d/1tDlDDwiDEWAMkyRW2rBC2F8td4dDidTX/view).

In this project we utilize the idea of "tiling" to quantify caps, a recursive method of small object detection.

---

## 📑 Model Details

* **Model Name:** FecundityModel 98K V1
* **Version:** 1.0
* **Architecture:** Sequential CNN
* **Framework:** TensorFlow / Keras
* **Parameters:** \~981K
* **File Size:** \~3.74 MB
* **Date:** 2025-05-31

---

## ⚙️ Model Architecture

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 73, 73, 32)          │             896 │
│ max_pooling2d (MaxPooling2D)         │ (None, 36, 36, 32)          │               0 │
│ conv2d_1 (Conv2D)                    │ (None, 34, 34, 64)          │          18,496 │
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 17, 17, 64)          │               0 │
│ conv2d_2 (Conv2D)                    │ (None, 15, 15, 64)          │          36,928 │
│ flatten (Flatten)                    │ (None, 14400)               │               0 │
│ dense (Dense)                        │ (None, 64)                  │         921,664 │
│ dense_1 (Dense)                      │ (None, 43)                  │           2,795 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 980,781
 Trainable params: 980,779
 Non-trainable params: 0
 Optimizer params: 2
```

---

## 📊 Training Data

* **Dataset size:** \~98K labeled images
* **Input shape:** (75 × 75 × 3) RGB images
* **Output classes:** 43 categorical labels (fecundity-related categories)
* **Split:** Training / Validation / Test (exact ratios TBD)

---

## 🚀 Intended Use

* **Research use only**: Designed for experimental evaluation of image-based classification of fecundity in fruit flies.
* **Applications**:

---

## ⚠️ Limitations

* Not intended for clinical or production use.
* Accuracy and generalizability depend heavily on dataset quality and labeling consistency.
* May not transfer well to other organisms, imaging conditions, or unrelated biological classification tasks.
* Biases may exist due to dataset collection methods.

---

## 📈 Performance

* **Baseline CNN classifier** — see ./Tests[X] and ./Plots

<img width="1200" height="500" alt="image" src="https://github.com/user-attachments/assets/54b07a9a-ff06-441c-bc7d-2e08229cbcc8" />

---

## 📌 Citation

---

## ✅ Future Work
* Further testing and new ideas for removing background

