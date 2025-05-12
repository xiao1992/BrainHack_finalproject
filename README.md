# Opening the Black Box: Interpretable Emotion Recognition from Consumer-based EEG

## Overview
Most EEG emotion recognition systems focus solely on achieving high classification accuracy, often using black-box models with little insight into why certain predictions are made. However, for practical deployment in mental health, education, and user-adaptive systems, interpretability is crucial. This project uses a Kaggle dataset titled "EEG Brainwave Dataset: Feeling Emotions" and introduces an explainable AI pipeline to identify which brainwave features and electrode locations contribute most to different emotional states.

---

## Project Motivation
This project aims to move beyond traditional emotion classification by introducing model interpretability into the analysis of EEG data. The goal is not only to classify emotional states (positive, neutral, negative) but also to visualize and understand the key neural features influencing the prediction. This is practical for building transparent emotion-aware applications.
![Emotion Regulation](https://www.hopebridge.com/wp-content/uploads/2022/05/Understand-Emotion-Tips-Kids-Autism-Hopebridge.jpg)

### Key Innovations:
- Consumer-designed EEG for emotional modeling (Muse Headband).
- Integration of explainable AI (XAI) techniques using SHAP and LIME.
- Identification of feature-level and electrode-level importance in emotion prediction & visualization of global v.s local emotional predictors.
- A reproducible baseline for interpretable EEG-based emotion models.

Classical studies such as Koelstra et al. (2012) with the DEAP dataset laid foundational work by correlating EEG with self-reported valence and arousal. However, many subsequent studies focused on improving classification accuracy without addressing the interpretability of the results. For instance, deep learning approaches, while powerful, often function as black boxes and fail to offer insight into which neural patterns contribute to specific emotional responses. Explainable AI has emerged as a promising solution to bridge the gap. SHAP, introduced by Lundberg and Lee (2017), provides consistent explanations for complex machine learning models, making it an ideal tool for EEG applications where transparency is unclear. In recent EEG-based emotion recognition work, SHAP has been used to identify key frequency bands and electrode contributions, enhancing the credibility of these systems for clinical or user-facing applications.

---

## Dataset: Kaggle "EEG Brainwave Dataset: Feeling Emotions"
This dataset was collected using a Muse EEG headband (TP9, AF7, AF8, TP10) across emotional stimuli (music tracks) designed to trigger positive, neutral, and negative emotional states. Two participants (1 male, 1 female) were recorded for 3 minutes per state. An additional 6 minutes of resting-state EEG was recorded. Data is labeled based on the emotional condition during each recording segment.

### Scope & Constraints
Given the limited 3-week timeframe for the BrainHack school project, we chose a small dataset but with a complete scope of a project that includes compare different ML models and explore a reproducible baseline for interpretable EEG-based emotion models.

---

## Assumptions for Labeling
We treat the labeled segments ("positive", "neutral", "negative") as ground truth emotional states and train classifiers accordingly. EEG features are extracted from these segments using frequency-domain analysis. We assume: 1) band power changes correlate with emotional states (i.e more alpha in relaxed states); 2) certain electrodes (like AF7/AF8) capture emotion-related activity more effectively.

---

## Methodology
### 1. **EEG Preprocessing**
Filter, ICA, normalization; segment into time-locked windows of 2-5s; for each window, we compute power spectral density (PSD) values for all frequency bands at each of the four electrodes (TP9, AF7, AF8, TP10). These PSD values serve as the input features for our machine learning models. To ensure comparability across subjects and sessions, we normalize all features using z-score normalization within each subject.

### 2. **Model Training**
We train classification models to predict the emotional state (positive, neutral, or negative) based on the extracted EEG features. Our baseline models include Random Forest and XGBoost classifiers, which are well-suited for small tabular datasets and provide feature importance metrics. Support Vector Machines (SVM) will also be tested for performance benchmarking. We evaluate each model using k-fold cross-validation, and track classification performance using accuracy, F1-score, and ROC-AUC. The final trained model is selected based on its generalization performance.

### 3. **Apply Explainer**
Once a model is trained, we turn to explainable AI (XAI) methods to interpret the classifier's decision-making process. We apply SHAP (SHapley Additive exPlanations). SHAP allows us to examine both global feature importance (which features contribute most to emotion classification overall) and local explanations for individual EEGs. We generate SHAP plots to visualize the distribution and magnitude of feature contributions and to investigate how features influence predictions toward particular emotional classes. 

### 4. **Evaluation Metrics**
Finally, we visualize the learned EEG feature space using t-SNE or UMAP, which allow us to examine how well the emotional states cluster in the feature space. We also construct heatmaps to represent the importance of each electrode-frequency pair, offering intuitive insights into which brain regions and frequency bands are most relevant for emotion detection.

---

## Relevant Works
*Birdy654. (2022). EEG Brainwave Dataset: Feeling Emotions [Data set]. Kaggle. https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions
*Koelstra, S., Muhl, C., Soleymani, M., Lee, J. S., Yazdani, A., Ebrahimi, T., ... & Patras, I. (2012). DEAP: A Database for Emotion Analysis Using Physiological Signals. IEEE Transactions on Affective Computing, 3(1), 18–31. https://doi.org/10.1109/T-AFFC.2011.15
*Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems, 30, 4765–4774. https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf

---

## Future Work
Incorporate real-time explainability for live feedback applications.
Expand to more subjects using full DEAP datasets.
Combine EEG with peripheral biosignals such as GSR, MEG for multimodal explainability.
Integrate domain-specific knowledge in cognitive neuroscience into feature selection.

![Emotion Regulation](https://bewelltherapygroup.org/wp-content/uploads/2024/03/Untitled-design-67.png)

---
