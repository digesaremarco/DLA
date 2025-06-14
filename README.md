# Deep Learning Applications
This repository contains three practical labs developed for the _Deep Learning Applications_ course, aimed at introducing key concepts of modern machine learning, from CNNs to Transformers and Reinforcement Learning.

---

## 🔬 Lab 1 – Convolutional Neural Networks (CNNs) for Image Classification

**Description:**  
The first lab focuses on exploring and implementing different neural architectures for image classification, using both **fully connected (MLP)** and **convolutional (CNN)** models. The main experiments were conducted on **CIFAR-10**, with **MNIST** included for comparison. **Transfer learning** techniques were also tested from CIFAR-10 to CIFAR-100.

### ✅ Main Objectives
- Understand and compare MLP, CNN, and their **residual** variants
- Visualize and interpret intermediate representations learned by the models
- Evaluate the impact of **residual connections** on performance and stability
- Experiment with **transfer learning** using **feature extraction** from a pretrained CNN

### 🧠 MLP and Residual MLP on MNIST
- **Dataset:** MNIST (28x28 grayscale images, 10 digits)
- **Implemented architectures:**
  - **Standard MLP:** two fully connected layers with ReLU activation
  - **Residual MLP:** skip connections between layers to improve gradient flow

### 🧠 CNN and Residual CNN on CIFAR-10
- **Dataset:** CIFAR-10 (32x32 RGB images, 10 classes)
- **Implemented architectures:**
  - **Standard CNN:** 3 convolutional blocks + ReLU + BatchNorm + MaxPooling + FC
  - **Residual CNN:** includes **residual blocks** similar to ResNet (skip connections)

### 🔁 Transfer Learning: CIFAR-10 ➝ CIFAR-100
- **Feature extraction:**
  - A **residual CNN pretrained on CIFAR-10** was used to extract intermediate representations (feature maps) from CIFAR-100 images.
- **Fine-tuning strategies evaluated:**
  - Freezing convolutional layers and training only the classification head
  - Unfreezing selected layers
- **Results:**
  - Extracted features proved informative
  - Transfer learning led to performance improvements compared to the SVM baseline

---

## 🎮 Lab 2 – Reinforcement Learning: REINFORCE and DQN in Gymnasium Environments

**Description:**  
The second lab explores Reinforcement Learning in simulated Gymnasium environments, implementing and comparing two algorithms: **REINFORCE** (policy gradient) and **DQN** (value-based).

### ✅ Main Objectives
- Understand the differences between policy-based and value-based methods  
- Assess the effect of baselines on variance in REINFORCE  
- Test agents on CartPole-v1 and LunarLander-v3

### 📌 Activities
- **CLI configuration:** choose algorithm (`--agent`), environment (`--env_name`), baseline (`--baseline`), hyperparameters (`gamma`, `lr`, `episodes`, etc.)
- **REINFORCE:**  
  - Stochastic policy network and optional value network  
  - Baselines: none, standardized, or learned  
  - Episode-end updates (policy gradient theorem)
- **DQN:**  
  - Replay buffer, epsilon-greedy, Q-network  
  - MSE loss between Q-target and predicted Q  
- **Experiment logging with wandb**  
  - Cumulative rewards, policy loss, baseline behavior  
  - Optional agent visualization (`--visualize`)

### ⚙️ Tested Environments
- `CartPole-v1`: classic control  
- `LunarLander-v3`: more complex environment  

### 📈 Results
- REINFORCE improves significantly with learned baseline (value network)  
- DQN showed greater stability and efficiency  
- wandb helped directly compare policy- and value-based agents

---

## 🤖 Lab 3 – Transformers: Feature Extraction and Fine-tuning with BERT

**File:** `Lab3_Transformers.ipynb`  
**Description:**  
The third lab focuses on using **pretrained Transformers** (such as DistilBERT and BERT) for NLP tasks, particularly sentiment analysis, using the **Hugging Face Transformers** library.

### ✅ Main Objectives
- Understand Transformer structure and how BERT works  
- Use pretrained models for feature extraction and fine-tuning  
- Evaluate performance on text datasets

### 📌 Activities
- **Feature Extraction with DistilBERT:**  
  - Dataset: Rotten Tomatoes (Cornell Movie Review)  
  - Tokenization with `distilbert-base-uncased`  
  - Feature extraction and training of a classifier (e.g., SVM)  
- **BERT Fine-tuning:**  
  - Model: `BertForSequenceClassification`  
  - Training with `AdamW`, linear scheduler, GPU-optimized batching  
  - Evaluation using multiple metrics

### 📈 Results
- Feature extraction + SVM yielded solid results with a simple pipeline  
- Fine-tuning reached **>90% accuracy** on the validation set in a few epochs  

---

## 📎 General Notes
- All labs are implemented in **PyTorch**  
- Notebooks are compatible with **Google Colab**  
- The project shows a gradual progression from traditional CNN architectures to modern Transformers like BERT

---

## 💡 AI Tools Used

- **ChatGPT** was used to resolve minor bugs, clarify implementation doubts, and assist in formatting this README.
- **GitHub Copilot** was used to speed up code writing by providing intelligent code completions and suggestions.
