
---

# **Lightweight LGTA for Tourism Time Series – Project Documentation**

## **1. Introduction**

This repository presents our work on the LGTA (Latent Generative Time Series Augmentation) model applied to the Tourism dataset. Our project had two goals: first, to run and understand the original LGTA model, and second, to design a lightweight version that reduces complexity while keeping similar behaviour. We completed this work in two phases: running the base LGTA and implementing our improved LGTA. Each phase is kept in its own folder for clarity and easy navigation.

The LGTA model is designed to generate synthetic time series by learning a latent representation of the data. It uses an encoder–decoder structure where the encoder compresses the input sequence into a latent space and the decoder reconstructs it into a meaningful time series. This allows the model to create augmented data that resembles the structure of the original dataset. Our project analyzes this approach, applies it to the Tourism dataset, and shows how a lighter version of the model can still perform effectively.

---

## **2. Project Objectives**

The main objectives of our project were:

### **2.1 Reproduce the Original LGTA Model**

We cloned the official LGTA repository, prepared the Tourism dataset, and ran the original model exactly as intended. This helped us understand the workflow, architecture, and generation process of LGTA.

### **2.2 Develop a Lightweight Version**

We modified the original LGTA architecture to make it lighter and more efficient. Our goal was to reduce the size and computation without breaking the core idea of latent generative modeling.

### **2.3 Compare Base and Improved Versions**

We compared training behaviour, reconstruction quality, and model complexity between the two versions to show how the improved model performs in practice.

### **2.4 Present Findings in a Research Paper**

All results and explanations are included inside our Research_Paper.pdf.

---

## **3. Repository Structure**

```
/lgta_Project
   /lgta_base
      ├── LGTA_original_base/           → Original LGTA code folder used for the base model
      └── L_GTA_BASE_TOURISM.ipynb      → Notebook running the base LGTA on Tourism dataset

   /lgta_improved
      ├── LGTA_original_improved/       → Modified LGTA folder used for the lightweight version
      └── L_GTA_IMPROVED.ipynb          → Notebook with the improved model

Research_Paper.pdf                     → Final project paper
README.md
.gitattributes (Git LFS setup)
```

This structure keeps the project simple and allows any reviewer to open the required folder and run the notebook directly.

---

## **4. System Requirements**

To run the notebooks, you need a basic machine learning environment. The notebooks work on:

* **Python (3.x)**
* **Jupyter Notebook or Google Colab**
* **TensorFlow or PyTorch** (depending on the LGTA implementation inside each LGTA folder)
* **NumPy**
* **Pandas**
* **Matplotlib**
* **scikit-learn**

### **4.1 Recommended Installation (Local)**

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

or, if PyTorch version is required:

```bash
pip install torch torchvision torchaudio
```

Google Colab already includes most of these libraries preinstalled.

---

## **5. How to Run the Project**

### **5.1 Running the Base LGTA Model**

1. Open the folder:

   ```
   /lgta_base
   ```
2. Make sure the folder contains:

   * `LGTA_original_base/`
   * `L_GTA_BASE_TOURISM.ipynb`
3. Open the notebook in Jupyter or Google Colab.
4. Run the notebook from top to bottom.
5. Along with the notebook access the LGTA folder located beside it.

### **5.2 Running the Improved Lightweight LGTA Model**

1. Open:

   ```
   /lgta_improved
   ```
2. Ensure it contains:

   * `LGTA_original_improved/`
   * `L_GTA_IMPROVED.ipynb`
3. Open the notebook in your preferred environment.
4. Run all cells in order.
5. The notebook will use the modified LGTA code to run the lightweight version.

### **5.3 Dataset Handling**

The notebooks are already configured for the Tourism dataset inside the LGTA folders.
No external downloads or special configurations are required unless you intentionally modify the dataset paths.

---

## **6. What We Did in This Project**

### **6.1 Understanding the LGTA Model**

We studied how LGTA uses latent space representations to generate new time series samples. This included analyzing how the encoder compresses information, how sampling in latent space works, and how the decoder reconstructs sequences.

### **6.2 Running the Original LGTA**

We executed the base LGTA model on the Tourism dataset and observed:

* reconstruction output
* latent behaviour
* training flow
* data augmentation samples

This gave us a clear foundation for improvements.

### **6.3 Building the Lightweight LGTA**

We modified selected components to create a model that:

* trains faster
* uses fewer parameters
* stays close in performance to the original

### **6.4 Comparing the Two Versions**

We compared the two notebooks using visual outputs and qualitative observations.
Our research paper summarizes these findings in detail.

---

## **7. Results Summary**

* The base model successfully reproduced the expected outputs.
* The improved lightweight model produced reliable reconstructions while being faster and simpler.
* Latent space behavior remained consistent across both versions.
* The lighter architecture showed better efficiency without breaking LGTA’s concept.

Detailed explanation is available in **Research_Paper.pdf**.

---

## **8. Project File**

* **Research_Paper.pdf** – our final research paper covering:

  * Abstract
  * Introduction
  * Related Work
  * Methodology (Base L-GTA + Proposed Improvement)
  * Experimental Setup
  * Results and Discussion
  * Comparative Analysis (with Base Paper)
  * Conclusion and Future Work
  * References

---

## **9. Team Members**

This project was completed by:

* **Abdullah – CMSID 62724**
* **Muhammad Mutahar – CMSID 63513**
* **Muhammad Borhan ud Din – CMSID 63206**

We worked together on studying the original LGTA, preparing the dataset, modifying the model, evaluating results, and writing the research paper.

---

## **10. Conclusion**

This project provided us with hands-on experience in latent generative modeling for time series. We successfully ran the original LGTA model and introduced a lighter version that reduces computational load. The repository includes all notebooks, modified LGTA folders, documentation, and the research paper. Our goal was to make the structure simple and easy for anyone—whether a lecturer or a researcher—to understand and reproduce.

---
