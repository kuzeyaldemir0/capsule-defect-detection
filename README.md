# FENG 497 - Defect Detection Group Project

In this project, we're using the Capsule subset of the MVTEC AD dataset and trying to detect defective capsules by using different anomaly detection algorithms.

## Getting Started

1. Install [uv](https://docs.astral.sh/uv/), then install the dependencies (Python 3.13):

   ```bash
   uv sync
   ```

2. Download the **capsule** category of [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it into the repo root:

   ```
   capsule/
   ├── train/good/        # 219 good images
   ├── test/good/         # 23 good images
   ├── test/<defect>/     # 109 defective images in 5 defect types
   └── ground_truth/      # pixel masks (not used yet)
   ```

3. Open a notebook (e.g. `patchcore_capsule.ipynb`), pick the `.venv` kernel and run all cells.

## Repository Structure

| File                      | What it is                              |
| ------------------------- | --------------------------------------- |
| `patchcore_capsule.ipynb` | PatchCore baseline on the capsule data  |
| `pyproject.toml`          | Dependencies (managed with uv)          |

## PatchCore Notebook

PatchCore learns only from good images, so no defect examples are needed for training. The notebook:

1. **Extracts patch features** from an ImageNet-pretrained WideResNet-50 (layer 2 + layer 3). The network is not trained further.
2. **Builds a memory bank** by keeping 10% of the training patches via greedy coreset sampling, which picks patches that cover the feature space as evenly as possible.
3. **Scores images**: each patch's score is the distance to its nearest memory bank patch, and the image score is the highest patch score.
4. **Flags an image as defective** if its score is above the threshold (see [Decision Rule](#3-decision-rule)).

### Why we hold out good training images

We set aside 25% of the good training images (54 of 219) as a **calibration set** and don't put them in the memory bank. Two reasons:

- **The threshold shouldn't come from the test set**, otherwise the test results would be optimistic.
- **The test set has only 23 good images**, so the smallest non-zero FPR it can measure is 1/23 ≈ 4.3%. With 54 calibration images we can aim for a lower FPR (2%).

The cost is a smaller memory bank (165 images instead of 219).

### Current results

| Image AUROC | Recall           | FPR           |
| ----------- | ---------------- | ------------- |
| 0.984       | 0.862 (94 / 109) | 0.043 (1 / 23) |

The test set is small, so treat recall and FPR as rough estimates.

## Evaluation Metrics

### 1. Image-level AUROC (Area Under the Receiver Operating Characteristic Curve):

AUROC is the probability that a randomly chosen anomalous (defective) image receives a higher anomaly score than a randomly chosen normal image. 

It is threshold-independent: because it is computed from raw anomaly scores before any threshold is applied, it summarizes performance across all possible thresholds rather than at one fixed decision point. We therefore use it mainly to assess how well our anomaly detection model separates normal from anomalous images, independent of the threshold chosen later for deployment.

$$
\text{AUROC} = P\big(\text{score}(\text{defective}) > \text{score}(\text{normal})\big)
$$

### 2. Recall, FPR, and Precision

We decided to treat `defective` as the positive class since that's what we're trying to find.

|                    | Model says defective (positive)  | Model says normal (negative)         |
| ------------------ | -------------------------------- | ------------------------------------ |
| Actually defective | True Positive (TP): caught it    | False Negative (FN): missed defect   |
| Actually normal    | False Positive (FP): false alarm | True Negative (TN): correctly passed |

Each metric looks at a different group of images:

- **Recall** looks only at the defective images: what share did we catch?

  ```math
  \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  ```

- **False Positive Rate (FPR)** looks only at the good images: what share did we wrongly flag?

  ```math
  \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
  ```

- **Precision** mixes both groups: of everything flagged, what share is actually defective?

  ```math
  \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  ```

Because precision mixes the two groups, it changes when the ratio between them changes, even if the model stays exactly the same. Recall and FPR don't, because each one stays inside its own group. They still depend on the model, the threshold, and which kinds of defects and good images are in the set.

For example, with recall 0.86 and FPR 0.043, precision is about 0.99 on our test set (83% defective) but only about 0.17 in a factory where 1% of capsules are defective. So we focus on recall and FPR, which show how the model performs on defective and good capsules separately, regardless of how many of each there are.

### 3. Decision Rule

We choose the threshold using **only the held-out good images** (the calibration set), aiming for an FPR of at most 2%:

1. Compute the anomaly score of each of the $n$ calibration images.
2. Allow at most $k = \lfloor 0.02 \cdot n \rfloor$ of them above the threshold. With $n = 54$, $k = 1$.
3. Sort the scores from highest to lowest and set the threshold to the $(k+1)$-th score. Here, that's the second-highest score.
4. At test time, an image with a score **above** the threshold is flagged as defective.

The test set isn't used to choose the threshold. It's only used to measure how well that threshold works on unseen images.