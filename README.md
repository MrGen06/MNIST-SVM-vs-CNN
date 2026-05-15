# MNIST Digit Recognition: Machine Learning vs. Deep Learning

**Course:** Deep Learning (CSTC - 308)  
**Team Members:** Goutam Khandelwal (123110028), Gourav Batra (123110026), Anushika Jindal (123110027), Naman Kumar Gupta (123110042) 

## Project Overview
For this assignment, our team conducted an end-to-end comparative analysis between traditional Machine Learning (Support Vector Machines) and Deep Learning (Convolutional Neural Networks) for handwritten digit classification. 

Our goal was not just to train standard models to get a high accuracy score on clean data. We wanted to mathematically optimize both architectures, overcome hardware limitations through dimensionality reduction, and ultimately subject our models to a real-world stress test to determine which architecture is truly more robust.

---

## How to Run Our Code (Reproducibility)
We developed this project using **Google Colab** to leverage cloud GPUs. We have included our final trained models (`advanced_cnn.keras` and `hybrid_svm_15k.pkl`) in the `models/` directory so you can easily verify our results in `04_evaluation.ipynb`.

Because GitHub has a strict 100MB file limit, we could not upload our pre-processed `data/` directory (~555MB). If you would like to run our notebooks locally or reproduce our training steps, here is our setup:

**Running in Google Colab (Our Setup):**
1. Create a folder named `MNIST-Project` in the root of your Google Drive.
2. Download the `models/` folder from this repository and upload it to your Drive folder.
3. Download our pre-processed PCA and flattened data arrays from our shared Google Drive link here: **https://drive.google.com/drive/folders/18XuEiK5iC3komN4ePe1EqvciGd59FkL6?usp=drive_link**
4. Place those data files into a `data/` folder inside your `MNIST-Project` Drive folder.
5. Upload our `.ipynb` notebooks to Colab. The code is already configured to mount your Drive and locate the files.

*(Note: If you prefer to run this locally in Jupyter Notebook, simply clone this repository, add the `data/` folder to the root directory, and remove the Google Colab `drive.mount()` code blocks from the top of the notebooks.)*

---

## Phase 1: Data Engineering & Dimensionality Reduction
* **Notebook:** `01_preprocessing.ipynb`
* **The Problem:** We initially found that the raw 784-dimensional dataset was computationally unviable for SVM training due to its O(n^2) time and memory complexity. 
* **Our Solution:** We applied **Principal Component Analysis (PCA)** to compress the data, calibrating the algorithm to retain exactly 85% of the cumulative explained variance. This successfully reduced our feature space from 784 down to roughly 59 features. We verified the separability of this compressed data by mapping it onto a 2D plane using a **t-SNE** visualization, preserving the shape of the digits while allowing our SVM to run without hardware failure.

---

## Phase 2: Architectural Optimization

### 1. The Custom Hybrid SVM
* **Notebook:** `02_svm.ipynb`
* **Methodology:** We wanted to break past the standard accuracy plateaus of basic linear models. We engineered a Custom Hybrid Kernel for our SVM, mathematically defined as: `K_hybrid = 0.7 * K_poly + 0.3 * K_rbf`. We hypothesized that heavily weighting the Polynomial kernel (70%) would capture the broad, global shape of the digits, while the RBF kernel (30%) would handle localized, non-linear edge details.
* **Results:** This custom architecture successfully achieved **97.68%** accuracy on our clean test subset, outperforming our standard linear and polynomial baselines.

### 2. The Advanced CNN
* **Notebook:** `03_cnn.ipynb`
* **Methodology:** Our initial baseline CNN suffered from obvious overfitting (a 2% gap between training and validation accuracy). To resolve this, we upgraded the architecture by adding `MaxPooling2D` layers to force spatial invariance, and a 50% `Dropout` layer before the final classification to prevent the network from memorizing pixel noise.
* **Results:** The regularization worked exactly as intended. We closed the overfitting gap and achieved a final test accuracy of **99.29%**.

---

## Phase 3: Robustness Evaluation & Conclusion
* **Notebook:** `04_evaluation.ipynb`
* **Methodology:** To answer our final research question—*which model is better for messy, real-world data?*—we subjected both champion models to a Gaussian Noise stress test (variance = 0.3). This simulated poor camera quality or scanner artifacts.

| Architecture | Clean Test Accuracy | Noisy Test Accuracy | Absolute Drop |
| :--- | :--- | :--- | :--- |
| **Hybrid SVM (15k)** | 97.68% | 81.00% | **-16.68%** |
| **Advanced CNN** | 99.29% | 95.59% | **-3.70%** |

### Final Conclusion
Our stress test definitively proved the architectural superiority of Convolutional Neural Networks over traditional SVMs for raw image processing. 

The SVM suffered a severe 16.68% drop in accuracy. We concluded this happened because the random noise completely distorted the exact pixel variance that our PCA mapping relied upon. Conversely, the CNN suffered only a minor 3.70% drop. Because the CNN utilizes spatial downsampling (Pooling layers), it naturally filtered out the background static and focused on the dominant structural strokes of the digits. Ultimately, the CNN proved to be the vastly more robust production model for this dataset.
