# ASR with Real-Time Speech Processing

## Overview
This repository contains the implementation details of an Automatic Speech Recognition (ASR) system designed to process real-time audio input for recognizing specific commands. The system employs Mel-Frequency Cepstral Coefficients (MFCC) for feature extraction and Gaussian Mixture Model (GMM) based Hidden Markov Models (HMMs) for sequence modeling. This project was developed as part of EE 516 at the University of Washington under the guidance of Professor J. Bilmes.

## Table of Contents
- Introduction
- Data Loading and Preprocessing
- Feature Extraction
- Gaussian Mixture Model (GMM) based HMM
  - Components of GMM-HMM
  - Training GMM-HMM
    - Expectation Step
    - Maximization Step
- Speech Detection
  - Start of Speech Detection
  - End of Speech Detection
  - Validation of Detected Speech Segments
  - Real-Time Operation
- Flow
- Results and Evaluation
- Conclusion
- Acknowledgments

## Introduction
This project aims to develop an ASR system that can recognize specific voice commands and execute corresponding actions in real-time. The system uses MFCC for feature extraction from audio signals and GMM-based HMMs for modeling the temporal sequences of these features.

## Data Loading and Preprocessing
Audio files are loaded from the file system, with each command category having 30 distinct sets of files. The audio data is loaded using the `librosa` library, and the filenames are organized based on a specific naming pattern.

## Feature Extraction
MFCC features and their delta features are computed for each audio signal. These features are then concatenated to form the final feature set used for training the HMM.

## Gaussian Mixture Model (GMM) based HMM
In many real-world applications, the emission probabilities are better modeled by Gaussian Mixture Models (GMMs) instead of single Gaussian distributions. A GMM can capture the multimodal nature of the data by combining several Gaussian components.

### Components of GMM-HMM
1. **States (S):** A finite set of states \{s1, s2, ..., sN\}.
2. **Observations (O):** The observations are generated from a mixture of Gaussians.
3. **Initial State Probabilities (π):** The initial state distribution remains the same.
4. **Transition Probabilities (A):** The state transition probabilities remain unchanged.
5. **Emission Probabilities (B):** Each state’s emission probability is modeled as a GMM, defined by a set of parameters \{wk, µk, Σk\} for each component k, where wk are the mixture weights, µk are the means, and Σk are the covariance matrices.

### Training GMM-HMM
Training a GMM-HMM involves estimating the parameters λ = (π, A, B) using the Expectation-Maximization (EM) algorithm.

#### Expectation Step
In the E-step, we calculate the responsibilities of each Gaussian component for each observation.
\[
\gamma_t(i) = P(q_t = s_i | O, \lambda)
\]
\[
\gamma_t(i, k) = \frac{w_{ik} N(o_t | \mu_{ik}, \Sigma_{ik})}{b_i(o_t)}
\]

#### Maximization Step
In the M-step, we update the parameters to maximize the expected log-likelihood of the data.
\[
w_{ik} = \frac{\sum_{t=1}^T \gamma_t(i, k)}{\sum_{t=1}^T \gamma_t(i)}
\]
\[
\mu_{ik} = \frac{\sum_{t=1}^T \gamma_t(i, k) o_t}{\sum_{t=1}^T \gamma_t(i, k)}
\]
\[
\Sigma_{ik} = \frac{\sum_{t=1}^T \gamma_t(i, k) (o_t - \mu_{ik})(o_t - \mu_{ik})^T}{\sum_{t=1}^T \gamma_t(i, k)}
\]

## Speech Detection
The speech detection process utilizes a state machine with states including ‘silence’, ‘possible start’, ‘speech’, and ‘possible end’. The transitions between these states are governed by energy thresholds ITL and ITU, and the zero-crossing threshold IZCT.

### Start of Speech Detection
1. **Initial Search:** The algorithm starts at the beginning of the speech signal and searches toward the center for the first frame where the energy exceeds ITL and then ITU without dropping below ITL again.
2. **Refinement Using Zero-Crossings:** Checks 25 frames backward and adjusts the starting point if necessary.

### End of Speech Detection
1. **Initial Detection:** From the state ‘speech’, the system transitions to ‘possible end’ when the energy falls below ITU and then ITL.
2. **Refinement Using Zero-Crossings:** Checks 25 frames forward and adjusts the endpoint if necessary.

### Validation of Detected Speech Segments
Segments are validated based on duration and peak amplitude criteria.

### Real-Time Operation
The system continuously captures and processes audio data. Detected valid speech segments are classified using trained GMM-HMMs based on the highest log-likelihood.

## Flow
1. Load training audio filenames and compute MFCC and delta MFCC features.
2. Train GMMs and HMMs using the EM algorithm.
3. Implement real-time speech detection and classification.

## Specific Commands
The ASR system is trained to recognize and respond to the following commands:

1. "Odessa": Wake word to activate the ASR system.
2. "turn on": Command to turn on a device or system.
3. "turn off": Command to turn off a device or system.
4. "play music": Command to start playing music.
5. "stop music": Command to stop playing music.
6. "time": Command to query the current time.

## Results and Evaluation
The models are evaluated on a validation set, with accuracy computed for each command category. The overall goal is to maximize the likelihood of the observations.

## Conclusion
This ASR system effectively processes real-time audio input to recognize specific commands using MFCC feature extraction and HMMs. The system demonstrates accurate recognition performance on the evaluated dataset and provides real-time responses to recognized commands.

## Acknowledgments
This implementation was developed as part of EE 516 at the University of Washington under the guidance of Professor J. Bilmes. We thank Professor Bilmes for his invaluable guidance and support throughout the project.
