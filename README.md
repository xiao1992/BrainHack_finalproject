# Modeling Emotion Regulation Patterns with Contrastive Learning on Low-Cost EEG

## Overview
Most EEG emotion recognition focuses on classifying current emotional states but emotion regulation is key for mental health and less studied. We will use a Kaggle dataset called "EEG Brainwave Dataset: Feeling Emotions" that allows a lightweight, low-cost approach using consumer EEG.

---

## Project Motivation
This project aims to go beyond traditional emotion classification by examing an individual's **ability to regulate emotions** from EEG signals during emotional stimulation. Introduce contrastive learning to model changes in EEG during different stimuli. Instead of just classifying "happy vs. sad", the goal is to infer whether the subject shows signs of emotional regulation over time (i.e decreasing arousal, stabilizing valence).
![Emotion Regulation](https://illuminatingyou.com/wp-content/uploads/2023/10/AdobeStock_647250235-scaled.jpeg)

### Key Innovations:
- Consumer-designed EEG for emotional modeling (Muse Headband).
- Contrastive learning on time-windowed features (i.e SimCLR).
- Predict emotion regulation success based on EEG dynamics (i.e does valence/arousal stabilize over time?).
- A practical use case or conclusion: could someone use music to self-soothe or regulate emotion?

Contrastive learning has recently shown strong performance in EEG-based emotion recognition. It functions by teaching models to distinguish between positive pairs (same label or segment) and negative pairs (different labels) of data. Recent studies demonstrate that integrating Graph Convolutional Networks (GCNs) into contrastive learning significantly enhances performance by encoding spatial dependencies between electrodes. For example, a 2022 paper titled "Self-supervised Group Meiosis Contrastive Learning for EEG-Based Emotion Recognition" achieved over 95% accuracy on the DEAP dataset using GCNs and contrastive learning jointly.

Another innovation is PhysioSync, which applies temporal and cross-modal contrastive learning to synchronize EEG with physiological signals like GSR or EMG, furthure improving emotional representation learning across time and modality in a 2025 study titled "PhysioSync: Temporal and Cross-Modal Contrastive Learning Inspired by Physiological Synchronization for EEG-Based Emotion Recognition".

---

## Dataset: Kaggle "EEG Brainwave Dataset: Feeling Emotions"
This dataset, hosted on Kaggle, was collected using a Muse EEG headband while 15 participants listened to audio tracks designed to evoke different emotional responses. Participants listened to 40 music clips designed to span a range of valence (pleasant–unpleasant) and arousal (calm–exciting) emotional states. EEG signals were recorded during each session using the Muse 2 headband, which provides 4 EEG channels: TP9, AF7, AF8, and TP10 (temporal and frontal).

After listening to each music clip, participants rated their emotional responses using self-assessment scales:
Valence: How pleasant/unpleasant they felt during the music (scale: 0 to 1, where 0 = unpleasant, 1 = pleasant).
Arousal: How activated or stimulated they felt (scale: 0 to 1, where 0 = calm, 1 = highly aroused).

---

## Assumptions for Labeling
We hypothesize emotion regulation based on patterns in self-report and EEG:
-	Low arousal with high valence after a high-arousal stimulus → possible regulation
-	Compare early vs. late EEG segments to infer downregulation over time

---

## Scope & Constraints
Given the limited 3-week timeframe for the BrainHack school project, I will narrow down the scope to: 1) 5 participants (instead of 15) with complete and high-quality recordings, 2) 10 songs per participant (intead of 40) with clearer valence or arousal emotional labels, 3) a simplified contrastive learning framework such as SimCLR, 4) lightweight MLP classifiers will then be trained on the learned embeddings. 

The final goal is to demonstrate that contrastive learning can enhance EEG feature representations for emotion-related tasks, providing a proof-of-concept for further scaling.

---

## Methodology
### 1. **EEG Preprocessing**
Filter, ICA, normalization; segment into time-locked windows of 2-5s; Get Power Spectral Density (PSD) for each segment and apply augmentation methods for contrastive training. 

### 2. **Contrastive Learning**
Framework: SimCLR
Loss: NT-Xent (contrastive loss)
Apply contrastive learning (SimCLR) to learn emotionally discriminative EEG features. Learn embeddings via contrastive loss; predict with simple classifier. The contrastive model uses a small neural network backbone (i.e a few convolutional or dense layers) trained with NT-Xent loss. After pretraining, the learned embeddings are fed into a simple MLP classifier that predicts the participant's emotional regulation.

### 3. **Classifier / Prediction**
Use the embedding results to train a regression or classification model to predict emotional regulation performance. We will use 1) MLP on Contrastive Embeddings, 2) Logistic/Linear Regression for the baseline (set a performance floor to beat with deep models), 3) SVM on embeddings. For classification, we use the assumptions to categorize regulation success.

**Emotion Regulation Score**
Based on self-report + EEG mismatch / time dynamics.

---

## 4. **Evaluation Metrics**
Evaluation of the models include using F1 score, accuracy, ROC-AUC. Will also conduct visualizations such as t-SNE/UMAP.

---

## Relevant Works
*X. Shen, X. Liu, X. Hu, D. Zhang and S. Song, "Contrastive Learning of Subject-Invariant EEG Representations for Cross-Subject Emotion Recognition," in IEEE Transactions on Affective Computing, vol. 14, no. 3, pp. 2496-2511, 1 July-Sept. 2023, doi: 10.1109/TAFFC.2022.3164516.
*Zhang, Hong, et al. "PhysioSync: Temporal and Cross-Modal Contrastive Learning Inspired by Physiological Synchronization for EEG-Based Emotion Recognition." 2025. arXiv:2504.17163.

---

## Future Work
Incorporate peripheral modalities (GSR, EMG) into contrastive framework;
Expand to all 32 subjects and full stimulus set (DEAP dataset);
Include GCNs and PhsioSync (cross-modal & temporal contrastive learning) in the model training;
Explore methods (such as meditaiton, journaling) to help with emotion awareness.
![Emotion Regulation](https://bewelltherapygroup.org/wp-content/uploads/2024/03/Untitled-design-67.png)

---
