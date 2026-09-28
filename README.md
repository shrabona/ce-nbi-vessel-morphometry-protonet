# Vessel Feature Extraction & Classification
**CE-NBI Laryngeal Lesion Assessment — Shrabona Mukherjee**

This notebook covers vessel feature extraction from CE-NBI images, statistical analysis of vessel-based biomarkers, graph kernel classification, wavelet scattering classification, and few-shot prototypical network experiments.

---

## Dataset Structure

```
<ROOT_DIR>/
├── Training/
│   ├── Benign/
│   │   ├── Polyp/
│   │   │   └── <patient_id>/
│   │   │       └── *.png / *.jpg
│   │   ├── Reinkes_edema/
│   │   └── Low_grade_dysplasia/
│   └── Malignant/
│       ├── High_grade_dysplasia/
│       ├── Carcinoma_in_situ/
│       └── SCC/
└── Testing/
    └── ... (same layout)
```

Update `base_path` at the top of the notebook to point to your local dataset root.

* **Zenodo Dataset:** [10.5281/zenodo.6674034](https://doi.org/10.5281/zenodo.6674034)
* **Paper:** Esmaeili et al. (2023), *Scientific Data*, 10(1), 733.

*Note: Pre-extracted feature tables (`train_features.csv`, `test_features.csv`) are included directly in this repository for immediate benchmark reproduction.*


---

## Class Hierarchy

| Group | Classes |
|---|---|
| **Benign** | Polyp · Reinkes_edema · Low_grade_dysplasia |
| **Malignant** | High_grade_dysplasia · Carcinoma_in_situ · SCC |

---

## Notebook Contents

| Section | Description |
|---|---|
| **Data Loading & Exploration** | Parse dataset into DataFrame, class distribution, image size analysis |
| **Vessel Preprocessing Pipeline** | BGR→RGB, green channel extraction, top-hat transform, Sato filter, skeletonization |
| **Vessel Feature Extraction** | 62 features per image across 3 groups (see below) |
| **Classical ML Baseline** | Random Forest / SVM / XGBoost on vessel features, binary only |
| **Graph Kernel Classification** | Skeleton→graph, categorized edge labels, EdgeHistogram kernel + SVM |
| **Wavelet Scattering + SVM** | Sato-filtered 16×16 images, J=2 scattering, 81 features, 10-fold patient CV |
| **Prototypical Networks** | 5-channel (RGB+Sato+Skeleton) EfficientNetV2 backbone, patient-level prototypes, hierarchical evaluation |

---

## Vessel Feature Groups

| Group | Features | Count |
|---|---|---|
| Basic | vessel_density, skeleton_length, total_vessel_pixels | 3 |
| Skan branches | branch_length, tortuosity, angle, euclidean (×7 stats each) + n_branches, total_branch_length | 30 |
| Regionprops | thickness, vessel_length, eccentricity, orientation (×7 stats each) + n_vessel_segments | 29 |
| **Total** | | **62 per image** |

Each distribution feature extracts: mean · std · min · max · p25 · p75 · p90

---

## Preprocessing Pipeline

Every image goes through this pipeline before feature extraction:

1. **BGR→RGB** — OpenCV default channel order correction
2. **Resize** — 128×128 px
3. **Green channel extraction** — vessels absorb green light in NBI imaging
4. **Top-hat transform** — remove uneven background lighting (11×11 ellipse kernel)
5. **Sato filter** — vessel enhancement at sigmas=[1,2,3]
6. **Thresholding** — top 20% strongest filter responses → binary mask
7. **Skeletonization** — thin vessels to 1-pixel centerlines
8. **Noise removal** — remove skeleton fragments < 3 pixels (remove_small_objects)

For wavelet scattering, an additional **center crop to square** step removes the circular black border of the endoscope frame before resizing to 16×16.

---

## Key Results

All classification is **image-level** with **patient-level train/validation splits** throughout to prevent data leakage.

| Method | Task | Accuracy |
|---|---|---|
| Vessel features + RF/SVM/XGBoost | Binary | ~0.65 |
| Graph kernels (GraKeL EdgeHistogram + SVM) | Binary | ~0.71 |
| Wavelet Scattering + SVM (10-fold CV) | Binary | 0.67 ± 0.07 |
| Wavelet Scattering + SVM (test set) | Binary | ~0.73 |
| Wavelet Scattering + SVM (10-fold CV) | 6-class | 0.34 ± 0.05 |
| ProtoNet 5-channel | Binary | ~0.77 |
| ProtoNet 5-channel | 6-class | ~0.35 |
| ProtoNet 5-channel | Hierarchical | ~0.35 (see breakdown below) |

---

## Prototypical Network — Hierarchical Evaluation Detail

The ProtoNet was also evaluated in a hierarchical pipeline (binary gate → subclass router):

| Stage | Task | Accuracy |
|---|---|---|
| Stage 1 | Binary (Benign vs Malignant) | 0.82 |
| Stage 2 | Benign subclass (Polyp / Reinke's / LGD) | 0.65 |
| Stage 2 | Malignant subclass (HGD / CIS / SCC) | 0.36 |
| **Final** | **Unified 6-class hierarchical** | **0.35** |

Notable: Polyp (F1=0.00) and Carcinoma in situ (recall=0.14) remain the hardest classes across all approaches due to severe class imbalance and visual similarity to adjacent severity grades.

---

## Graph Kernel Approach

Vessel skeletons are converted to graphs where:
- **Nodes** = branch points (where vessels meet)
- **Edges** = vessel branches between branch points
- **Edge labels** = combined categorization of branch length + tortuosity

Continuous values are discretized for GraKeL compatibility:

| Branch length | Label |
|---|---|
| < 5 px | very_short |
| 5–15 px | short |
| 15–30 px | medium |
| > 30 px | long |

| Tortuosity | Label |
|---|---|
| < 1.2 | straight |
| 1.2–1.5 | slight |
| 1.5–2.0 | curved |
| > 2.0 | very_curved |

Combined label example: `"short_curved"`, `"long_straight"`

---

## Prototypical Network Architecture

- **Backbone**: EfficientNetV2-S (pretrained ImageNet, first conv expanded to 5 channels)
- **Input**: 5-channel tensor — RGB (3ch) + Sato filter response (1ch) + Binary skeleton (1ch)
- **Embedding**: Shared trunk → L2-normalized embedding per image
- **Classification**: Nearest-prototype distance in embedding space
- **Prototypes**: Patient-level averaged embeddings per class (all images per patient averaged first, then class average taken across patients)
- **Validation**: Patient-aware GroupShuffleSplit (80/20 from training set)

---

## Requirements

```
torch
torchvision
opencv-python
scikit-image
scikit-learn
skan
grakel
kymatio
pandas
numpy
matplotlib
seaborn
tqdm
scipy
Pillow
```

Install all with:
```bash
pip install torch torchvision opencv-python scikit-image scikit-learn skan grakel kymatio pandas numpy matplotlib seaborn tqdm scipy Pillow
```

---

## Evaluation Paradigm

- **Train/test split**: Provided by dataset (Training/ and Testing/ folders, different patients in each)
- **Validation during development**: Patient-level GroupShuffleSplit (80/20) from training set
- **Cross-validation**: 10-fold StratifiedGroupKFold (patient-level) for wavelet+SVM
- **Primary metrics**: Accuracy · Macro-F1
- **No patient appears in both train and validation splits** at any stage

---

## References

Baharlouei, Z., Rabbani, H., & Plonka, G. (2023). Wavelet scattering transform application in classification of retinal abnormalities using OCT images. *Scientific Reports, 13*(1), 19013. https://doi.org/10.1038/s41598-023-46200-1

Bruna, J., & Mallat, S. (2013). Invariant scattering convolution networks. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 35*(8), 1872–1886. https://doi.org/10.1109/TPAMI.2012.230

Esmaeili, N., Davaris, N., Boese, A., Illanes, A., Navab, N., Friebe, M., & Arens, C. (2023). Contact endoscopy–narrow band imaging (CE-NBI) data set for laryngeal lesion assessment. *Scientific Data, 10*(1), 733. https://doi.org/10.1038/s41597-023-02645-7

Esmaeili, N., Illanes, A., Boese, A., Davaris, N., Arens, C., & Friebe, M. (2019). Novel automated vessel pattern characterization of larynx contact endoscopic video images. *International Journal of Computer Assisted Radiology and Surgery, 14*(10), 1751–1761. https://doi.org/10.1007/s11548-019-02034-9

Pachetti, E., & Colantonio, S. (2024). A systematic review of few-shot learning in medical imaging. *Artificial Intelligence in Medicine, 156*, 102949. https://doi.org/10.1016/j.artmed.2024.102949

Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. *Advances in Neural Information Processing Systems, 30*. https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html
