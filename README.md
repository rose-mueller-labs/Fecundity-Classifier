# **[title TBD] [#Params] V1 Caps**

<style>
img {
 display: inline;
}
</style>

[![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--TDT-blue#model-badge)](#model-architecture)
| [![Model size](https://img.shields.io/badge/Params-0.6B-green#model-badge)](#model-architecture)
| [![Language](https://img.shields.io/badge/Language-en-orange#model-badge)](#datasets)


## <span style="color:#466f00;">Description:</span>

`parakeet-tdt-0.6b-v2` is a 600-million-parameter automatic speech recognition (ASR) model designed for high-quality English transcription, featuring support for punctuation, capitalization, and accurate timestamp prediction. Try Demo here: https://huggingface.co/spaces/nvidia/parakeet-tdt-0.6b-v2 

This XL variant of the FastConformer [1] architecture integrates the TDT [2] decoder and is trained with full attention, enabling efficient transcription of audio segments up to 24 minutes in a single pass. The model achieves an RTFx of 3380 on the HF-Open-ASR leaderboard with a batch size of 128. Note: *RTFx Performance may vary depending on dataset audio duration and batch size.*  

**Key Features**
- Accurate word-level timestamp predictions  
- Automatic punctuation and capitalization  
- Robust performance on spoken numbers, and song lyrics transcription 

For more information, refer to the [Model Architecture](#model-architecture) section and the [NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer).

This model is ready for commercial/non-commercial use.


## <span style="color:#466f00;">License/Terms of Use:</span>

GOVERNING TERMS: Use of this model is governed by the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en) license.


### <span style="color:#466f00;">Deployment Geography:</span>
Global


### <span style="color:#466f00;">Use Case:</span>

This model serves developers, researchers, academics, and industries building applications that require speech-to-text capabilities, including but not limited to: conversational AI, voice assistants, transcription services, subtitle generation, and voice analytics platforms.


### <span style="color:#466f00;">Release Date:</span>

05/01/2025

### <span style="color:#466f00;">Model Architecture:</span>

**Architecture Type**: 

FastConformer-TDT

**Network Architecture**:

* This model was developed based on [FastConformer encoder](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer) architecture[1] and TDT decoder[2]
* This model has 600 million model parameters.

### <span style="color:#466f00;">Input:</span>
- **Input Type(s):** 16kHz Audio
- **Input Format(s):** `.wav` and `.flac` audio formats
- **Input Parameters:** 1D (audio signal)
- **Other Properties Related to Input:**  Monochannel audio

### <span style="color:#466f00;">Output:</span>
- **Output Type(s):**  Text
- **Output Format:**  String
- **Output Parameters:**  1D (text)
- **Other Properties Related to Output:** Punctuations and Capitalizations included.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. 

## <span style="color:#466f00;">How to Use this Model:</span>

#### Model Version

Current version: parakeet-tdt-0.6b-v2. Previous versions can be [accessed](https://huggingface.co/collections/nvidia/parakeet-659711f49d1469e51546e021) here. 

## <span style="color:#466f00;">Training and Evaluation Datasets:</span>

### <span style="color:#466f00;">Training</span>

This model was trained using the NeMo toolkit [3], following the strategies below:

- Initialized from a wav2vec SSL checkpoint pretrained on the LibriLight dataset[7].  
- Trained for 150,000 steps on 128 A100 GPUs. 
- Dataset corpora were balanced using a temperature sampling value of 0.5.  
- Stage 2 fine-tuning was performed for 2,500 steps on 4 A100 GPUs using approximately 500 hours of high-quality, human-transcribed data of NeMo ASR Set 3.0.  

Training was conducted using this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py) and [TDT configuration](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_tdt_ctc_bpe.yaml).

The tokenizer was constructed from the training set transcripts using this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).

### <span style="color:#466f00;">Training Dataset</span>
The model was trained on the Granary dataset, consisting of approximately 120,000 hours of English speech data:

- 10,000 hours from human-transcribed NeMo ASR Set 3.0, including:
  - LibriSpeech (960 hours)
  - Fisher Corpus
  - National Speech Corpus Part 1 
  - VCTK
  - VoxPopuli (English)
  - Europarl-ASR (English)
  - Multilingual LibriSpeech (MLS English) – 2,000-hour subset
  - Mozilla Common Voice (v7.0)
  - AMI

- 110,000 hours of pseudo-labeled data from:
  - YTC (YouTube-Commons) dataset[4]
  - YODAS dataset [5]
  - Librilight [7]

All transcriptions preserve punctuation and capitalization. The Granary dataset will be made publicly available after presentation at Interspeech 2025.

**Data Collection Method by dataset**

* Hybrid: Automated, Human

**Labeling Method by dataset**

* Hybrid: Synthetic, Human 

**Properties:**

* Noise robust data from various sources
* Single channel, 16kHz sampled data

#### Evaluation Dataset

Huggingface Open ASR Leaderboard datasets are used to evaluate the performance of this model. 

**Data Collection Method by dataset**
* Human

**Labeling Method by dataset**
* Human
