# Polymer-Toxicity-Prediction-using-Transfer-Learning-Tox21-datasets-
Polymer Toxicity Prediction using Deep Neural Network and Transfer Learning from Plasticizers and Additives (Tox21 datasets)

# 🔬 PolyToxiQ – Polymer Toxicity Prediction Web App

**PolyToxiQ** is a machine learning–powered toxicity screening tool for polymers. Using deep learning and transfer learning from molecular hazard data (Tox21), the app predicts potential toxicological hazards of user-defined polymer structures via PSMILES strings. Built with Streamlit and powered by a fine-tuned DNN model, PolyToxiQ aims to assist chemists, toxicologists, and regulatory researchers in evaluating polymer safety—quickly and interactively.

# PolyToxiQ: Polymer Toxicity Prediction using Deep Neural Networks and Transfer Learning

---

## 1. Motivation

Polymers and plastics are ubiquitous in modern life, utilized in everything from healthcare to food packaging. However, these materials are rarely pure; they contain a matrix of additives, plasticizers, and residual monomers that are not covalently bound and can migrate into the environment or biological systems[cite: 54, 55].

Traditional toxicity assessment methods (in vivo/in vitro) are time-consuming, costly, and ethically debated. [cite_start]Furthermore, while extensive toxicity data exists for small molecules (e.g., the Tox21 dataset), there is a severe scarcity of labeled toxicity data for polymers[cite: 79, 84]. This "data gap" hinders the development of accurate computational risk assessment tools for polymeric materials.

**The core motivation of this project is to bypass the need for expensive experimental polymer data by leveraging Machine Learning methods to transfer knowledge from the data-rich molecular domain to the data-poor polymer domain.**

---

## 2. Objective

The primary objective of this research is to develop a robust Deep Neural Network (DNN) capable of predicting the toxicity of polymers based solely on their chemical structure (PSMILES).

Key goals include:
* [cite_start]**Bridging the Data Gap:** Implementing Transfer Learning to adapt a model trained on 4,752 Tox21 small molecules to predict hazard properties for polymers[cite: 91].
* [cite_start]**Feature Representation:** Utilizing **PolyBERT**, a chemical language model, to generate unified fingerprints for both molecules and polymers, enabling a shared feature space[cite: 43].
* **Model Optimization:** Comparing various transfer learning strategies (Zero-shot, Few-shot, and Frozen-layer fine-tuning) to identify the most effective method for low-data regimes.
* [cite_start]**Democratization:** Deploying the final model as a user-friendly web application (**PolyToxiQ**) for real-time toxicity screening[cite: 50].

---

## 3. Workflow of the Project

The project follows a systematic "Materials Informatics" pipeline, moving from data curation to deployment.

![Implementation Steps](/home/sunil/am2/poetry-demo/Polytox_Matser_Thesis/Few_shot_learning_with_validation_data_Monomers/Github_Documentation/Images_readme/Implementation steps.png)

### A. Data Collection
* [cite_start]**Source Domain:** A curated subset of **4,752 small molecules** from the **Tox21 10K library**, annotated with four key hazard endpoints: Carcinogenicity, Mutagenicity, Specific Target Organ Toxicity (STOT), and Aquatic Toxicity[cite: 516].
* [cite_start]**Target Domain:** A validation set of **96 polymers** (thermoplastics and thermosets) based on the toxicity of their constituent monomers, serving as a proxy for polymer toxicity[cite: 604].

## 📂 Data Availability

Due to GitHub's file size limitations (100MB), the large pre-processed datasets used for training the Deep Neural Networks are hosted externally.

**Required Data Files:**
To reproduce the training results, please download the following files:
1. `Tox_DNN_data_encoder.csv`
2. `Tox_DNN_encoded__final_data.csv`
3. `Tox_LOC_FP.csv`

**Download Link:** [[INSERT YOUR GOOGLE DRIVE/ZENODO LINK HERE](https://drive.google.com/drive/folders/1ww7MxyuJ-yb1RMRYyRFsP1i_HgRW_mv7?usp=sharing)]

**Setup Instructions:**
After downloading, please place the files in their respective directories:
* Place `Tox_LOC_FP.csv` and `Tox_DNN_data_encoder.csv` inside: `Data/Data_preprocessing_Tox21/`
* Place `Tox_DNN_encoded__final_data.csv` inside: `Base_Tox21_mol_DNN_Model/`

**Setup Instructions:**
After downloading, please place the files in their respective directories:
* Place \`Tox_LOC_FP.csv\` and \`Tox_DNN_data_encoder.csv\` inside: \`Data/Data_preprocessing_Tox21/\`
* Place \`Tox_DNN_encoded__final_data.csv\` inside: \`Base_Tox21_mol_DNN_Model/\`
EOT

### B. Representation (PolyBERT)
To make chemical structures machine-readable, we utilized **PolyBERT**.
* **Molecules:** Encoded as SMILES strings.
* [cite_start]**Polymers:** Encoded as **PSMILES** (Polymer SMILES), which explicitly represent repeating units and connection points (e.g., `[*]CC[*]`)[cite: 250].
* [cite_start]**Embedding:** Both formats were converted into **600-dimensional numerical fingerprints** using the PolyBERT transformer, ensuring that molecules and polymers exist in a compatible mathematical space[cite: 284].

### C. Machine Learning & Optimization
We developed a Deep Neural Network (DNN) using **PyTorch**.
* **Architecture:** A multi-layer perceptron with input dimensions matching the PolyBERT vectors.
* [cite_start]**Hyperparameter Tuning:** **Optuna** was used to optimize learning rates, dropout, weight decay, and activation functions (ReLU, ELU, LeakyReLU) over 1,000 trials[cite: 584].

### D. User Interface
[cite_start]The final model was integrated into a **Streamlit** web application, allowing users to input PSMILES or draw structures to receive instant toxicity predictions and similarity analysis[cite: 1163].

---

## 4. Transfer Learning Strategies

A core contribution of this thesis is the comparative analysis of different transfer learning techniques to handle the scarcity of polymer labels.

![Transfer Learning Pipeline](/home/sunil/am2/poetry-demo/Polytox_Matser_Thesis/Few_shot_learning_with_validation_data_Monomers/Github_Documentation/Images_readme/Tranfer learning Pipeline.png)

### A. Polymer-Only Baseline
* **Method:** A DNN trained *exclusively* on the limited polymer dataset (approx. 60 training samples).
* **Purpose:** To establish a baseline and demonstrate that training from scratch with such small data is insufficient.

### B. Zero-Shot Transfer
* **Method:** The model is trained *only* on the Tox21 small molecule dataset. It is then applied directly to polymers without seeing any polymer training data.
* [cite_start]**Hypothesis:** Due to the shared embedding space (PolyBERT), the model can generalize molecular toxicity rules to polymers[cite: 619].

### C. Few-Shot Fine-Tuning
* **Method:** The pre-trained Tox21 model is taken, and *all* layers are allowed to update (fine-tune) using a very small subset (20%) of labeled polymer data.
* [cite_start]**Goal:** To gently adapt the entire network to the target domain[cite: 653].

### D. Frozen-Layer Fine-Tuning (Best Performance)
* **Method:** The initial layers of the pre-trained Tox21 model are **frozen** (weights locked). These layers act as robust feature extractors for chemical substructures. Only the final output layers are retrained on the polymer subset.
* [cite_start]**Advantage:** This prevents "catastrophic forgetting" of the robust molecular features while adapting the decision boundary for polymers[cite: 1060].

![Model Comparisons](/home/sunil/am2/poetry-demo/Polytox_Matser_Thesis/Few_shot_learning_with_validation_data_Monomers/Github_Documentation/Images_readme/Model_compariions.png)

---

## 5. Results

The performance of the models was evaluated using **Micro-Average ROC-AUC** and **F1 Scores** on a hold-out test set (Noble Test Set).

![Comparison Score](/home/sunil/am2/poetry-demo/Polytox_Matser_Thesis/Few_shot_learning_with_validation_data_Monomers/Github_Documentation/Images_readme/Comparision Score.png)

* [cite_start]**Polymer-Only Baseline (ROC-AUC 0.47):** Performed no better than random guessing, proving that deep learning from scratch is impossible with such small data[cite: 1151].
* [cite_start]**Zero-Shot Transfer (ROC-AUC 0.69):** Showed significant improvement, confirming that PolyBERT fingerprints successfully capture shared structural features between molecules and polymers[cite: 1149].
* [cite_start]**Few-Shot Fine-Tuning (ROC-AUC 0.81):** Further improved accuracy by adapting to the specific distribution of polymer data[cite: 1147].
* **Frozen-Layer Fine-Tuning (ROC-AUC 0.92):** The **best performing strategy**. [cite_start]By preserving the low-level chemical features learned from Tox21 and only updating high-level decision layers, this model achieved state-of-the-art results for this dataset[cite: 1142].

![Finetuning Model Comparison](/home/sunil/am2/poetry-demo/Polytox_Matser_Thesis/Few_shot_learning_with_validation_data_Monomers/Github_Documentation/Images_readme/Finetuning model comparision.png)

---

## 6. Conclusion and Future Works

### Conclusion
This project successfully demonstrated that **Transfer Learning** is a viable and highly effective solution for Polymer Informatics, a field often plagued by data scarcity. [cite_start]By leveraging large-scale molecular datasets (Tox21), we achieved high-accuracy toxicity predictions for polymers (ROC-AUC 0.92) using the **Frozen-Layer Fine-Tuning** strategy[cite: 1177]. The development of the **PolyToxiQ** web app further bridges the gap between complex deep learning models and practical regulatory application.

### Future Scope
* [cite_start]**Expanded Endpoints:** Incorporating additional hazards such as Endocrine Disruption and Reproductive Toxicity[cite: 1183].
* [cite_start]**Physicochemical Features:** Integrating solubility, molecular weight ($M_n$, $M_w$), and dispersity into the input vector to improve robustness[cite: 1184].
* [cite_start]**Data Augmentation:** Expanding the training base using curated polymer databases (e.g., CompTox, PubChem Polymers)[cite: 1186].
* **Uncertainty Quantification:** Implementing Bayesian Neural Networks to provide confidence intervals for predictions, which is crucial for regulatory decision-making.

---
*Developed at the Chair of Material Informatics, University of Bayreuth.*