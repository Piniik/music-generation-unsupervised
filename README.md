# Unsupervised Multi-Genre Music Generation

Course project for CSE425/EEE474 Neural Networks.

## Project Goal
Build an unsupervised neural network model for symbolic multi-genre music generation using MIDI data.

## Planned Scope
- Task 2: Variational Autoencoder (VAE) for multi-genre music generation
- Dataset: Lakh MIDI Dataset
- Baselines:
  - Random Note Generator
  - Markov Chain Music Model

If time permits, we may extend to:

- **Task 3:** Transformer-based long-sequence music generation

## Repository Structure

```text
music-generation-unsupervised/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── notebooks/
├── src/
├── outputs/
└── report/
```
## Team Workflow
- 24341268: preprocessing, baselines, evaluation
- 22141022: VAE, training, generation

## Setup
```bash
pip install -r requirements.txt
