# EEG Imagined Speech Neuro-Signal Processing

This repository provides code associated with the study:

**"EEG Imagined Speech Neuro-Signal Preprocessing and Deep Learning Classification"**

The repository is being progressively developed to provide reproducible implementations of the deep learning architectures and preprocessing components proposed in the study.

## Current Release

The current release provides the implementation of the proposed **CNN-2-Bi-LSTM** architecture, which achieved the best overall performance among the evaluated CNN-LSTM architectures.

The complete preprocessing, training, validation, and evaluation pipeline will be added progressively.

## CNN-2-Bi-LSTM Architecture

The CNN-2-Bi-LSTM model consists of:

- Input layer
- Batch Normalization
- 1D Convolutional layer
- Batch Normalization
- Max Pooling
- First Bidirectional LSTM layer
- Dropout
- Second Bidirectional LSTM layer
- Batch Normalization
- Fully Connected layer
- Dropout
- Softmax output layer

### Default Configuration

| Parameter | Value |
|---|---:|
| Input shape | (32, 14) |
| Convolutional filters | 128 |
| Kernel size | 10 |
| First Bi-LSTM units | 256 |
| Second Bi-LSTM units | 256 |
| Fully connected units | 128 |
| Dropout rate | 0.5 |
| Number of classes | 30 |

The implementation uses TensorFlow/Keras.

## Dataset

The experiments were conducted using the publicly available **Kumar EEG Imagined Speech Dataset**, comprising 30 imagined-speech classes across characters, digits, and objects.

## Preprocessing

The study proposes an **ICA-Assisted Frequency-Domain Filtering (FD-F)** preprocessing framework.

The proposed pipeline integrates Independent Component Analysis (ICA) with frequency-domain filtering and temporal segmentation.

The preprocessing implementation and additional preprocessing configurations investigated in related work are available in the following repository:

https://github.com/Fatma3598/EEG-Imagined-Speech-Preprocessing-Generalization

## Publication

Elwasify, F., Shaaban, E., & Abdelmoneem, R. M. (2026).

**EEG imagined speech neuro-signal preprocessing and deep learning classification.**

Scientific Reports, 16, 10604.

https://doi.org/10.1038/s41598-026-39395-6

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{Elwasify2026EEGImaginedSpeech,
  author  = {Elwasify, Fatma and Shaaban, Eman and Abdelmoneem, Randa M.},
  title   = {EEG imagined speech neuro-signal preprocessing and deep learning classification},
  journal = {Scientific Reports},
  volume  = {16},
  pages   = {10604},
  year    = {2026},
  doi     = {10.1038/s41598-026-39395-6}
}
