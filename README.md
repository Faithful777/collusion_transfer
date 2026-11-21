NB: A copy of the Bertrand dataset is availiable here as zip. it can be used to run the notebook

# Collusion Detection: Current-Step Transfer Study

This project implements and evaluates collusion monitors (classifiers) on multi-agent interactions across two distinct economic environments:
1. **SALT-NLP Agent Collusion (AC):** A 2-player task setting with explicit behavioral labeling.
2. **Bertrand AI (v2):** A 3-seller classic pricing game with economic-proxy labeling.

## Methodology & Architecture
- **Model Scope:** Evaluates a quantized `Qwen/Qwen2.5-3B-Instruct` base model alongside two finetuned low-rank adapters:
  - **CE-Only (Cross-Entropy):** Optimized for direct text-label predictions.
  - **CE + Axis-Margin:** Optimized on a joint loss comprising categorical cross-entropy and a custom activation margin calculated from layer-specific state representations.
- **Retrospective Current-Step Transfer:** Monitors evaluate complete transaction histories (prefix contexts) up to the current completed step, ensuring zero exposure to future steps. Classifiers are fitted on **AC training/validation data only** and evaluated on out-of-domain Bertrand data (exploratory cross-domain transfer).
- **Probing Readouts:** Classifies using four separate decision methodologies:
  1. Activation Axis Projection
  2. Supervised Linear Probe
  3. Qwen YES/NO Output Margins
  4. Lexical TF-IDF Baselines

## Technical Features & Setup
- **Compute Engine:** Optimized for memory-constrained T4 environments using memory-efficient Scaled Dot Product Attention (SDPA) with explicit grouped Key-Value (KV) repetition.
- **Loss Scale Management:** Outfitted with an FP16 bounded backoff optimizer loop to guard against numeric gradients overflow.
- **Auditing & Metrics:** Outputs cumulative metrics over complete, non-truncated trajectories alongside custom run tracking charts.
