English–Urdu Neural Machine Translation

Vanilla RNN Encoder–Decoder for English-to-Urdu Translation

A deep learning project implementing a word-level English-to-Urdu Neural Machine Translation (NMT) system using a classical Vanilla RNN encoder–decoder architecture.

The project focuses on building the complete sequence-to-sequence pipeline from data preprocessing and tokenization to model training, greedy decoding, evaluation, and qualitative translation analysis.

Course: Generative Artificial Intelligence — Assignment #1
Department: Software Engineering, FAST — National University of Computer and Emerging Sciences (NUCES), Islamabad
Authors: Shah Faisal (23I-0058), Hamad Khan (23I-3095)

Overview

The system translates English sentences into Urdu using a classical sequence-to-sequence architecture inspired by the original encoder–decoder approach.

Unlike modern Transformer-based translation systems, this implementation intentionally uses a Vanilla RNN as required by the assignment.

The pipeline includes:

Robust bilingual dataset loading

Unicode-aware English and Urdu text normalization

Duplicate and missing-data removal

Sentence-length analysis

Reproducible train/validation/test splitting

Word-level tokenization

Vocabulary construction

Sequence encoding and padding

Padding-aware loss masking

Vanilla RNN encoder–decoder training

Greedy autoregressive decoding

Test-set loss and perplexity evaluation

Qualitative comparison of reference and generated Urdu translations

Dataset

The project uses an English–Urdu parallel translation dataset containing paired sentences.

The dataset is downloaded automatically using KaggleHub.

Each training example consists of:

English sentence → Urdu sentence

The dataset is divided into:

80% Training

10% Validation

10% Testing

The split is performed reproducibly using a fixed random seed.

An explicit exact-pair overlap check is used to reduce the risk of data leakage between the splits.

Text Preprocessing

The preprocessing pipeline is designed to handle both Latin-script English and Arabic-script Urdu.

English normalization

Unicode normalization using NFKC

Curly-quote standardization

Whitespace normalization

Removal of missing and duplicate entries

Urdu normalization

Unicode normalization using NFC

Zero-width joiner/non-joiner handling

Whitespace normalization

Removal of missing and duplicate entries

Sentence-length distributions are also analyzed before training to understand the characteristics of the bilingual dataset.

Tokenization

The system uses word-level tokenization.

A Unicode-aware regular expression preserves:

Latin/English words

Urdu words written in Arabic script

Punctuation as separate tokens

The vocabulary is constructed using the training split only.

Special tokens are reserved:

<PAD>   Padding token
<BOS>   Beginning-of-sentence token
<EOS>   End-of-sentence token
<UNK>   Unknown-word token

This prevents information from the validation and test sets from influencing vocabulary construction.

Sequence Preparation

The English source sentence is converted into an encoder sequence.

For the Urdu target, separate sequences are prepared:

Decoder Input:
<BOS> Urdu tokens

Target Output:
Urdu tokens <EOS>

Sequences are padded to fixed lengths for batch processing.

A Boolean padding mask ensures that padded positions do not contribute to the training loss.

The resulting datasets are batched and optimized using TensorFlow's tf.data pipeline with shuffling and prefetching.

Model Architecture

Vanilla RNN Encoder–Decoder

The model is implemented as a subclassed TensorFlow/Keras model named:

VanillaRNMT

Encoder

English tokens
      ↓
Embedding
      ↓
SimpleRNN (256 units)
      ↓
Final hidden state

The final encoder hidden state represents the source sentence and is passed to the decoder.

Decoder

<BOS> + Urdu tokens
      ↓
Embedding
      ↓
SimpleRNN (256 units)
      ↓
Dense / Softmax
      ↓
Urdu vocabulary probabilities

The decoder generates the Urdu sequence autoregressively.

Architecture constraint

The implementation deliberately contains:

SimpleRNN

Encoder–decoder structure

Softmax output

It does not use:

LSTM

GRU

Attention

Transformer

Transformer encoder/decoder

This preserves the classical Vanilla RNN requirement of the assignment.

Training

The model uses:

Loss: Masked Sparse Categorical Cross-Entropy

Optimizer: Adam

Initial learning rate: 1e-3

Gradient clipping: Global norm = 1.0

Batch size: 64

Checkpointing: Best validation loss

Early stopping: Patience of 4 epochs

Padding positions are masked during loss calculation so that they do not artificially affect the training objective.

Decoding

The trained model uses greedy decoding on held-out test sentences.

At every decoding step:

The decoder receives the previously generated token.

It predicts a probability distribution over the Urdu vocabulary.

The token with the highest probability is selected.

The generated token is fed back into the decoder.

Generation continues until <EOS> is produced or the maximum sequence length is reached.

Conceptually:

English sentence
      ↓
RNN Encoder
      ↓
Context / Hidden State
      ↓
RNN Decoder
      ↓
Urdu token 1
      ↓
Urdu token 2
      ↓
...
      ↓
<EOS>

Evaluation

The project evaluates the trained translation model using:

Quantitative evaluation

Training loss

Validation loss

Test loss

Perplexity

Qualitative evaluation

Representative held-out examples are presented in a table containing:

English Source

Reference Urdu

Model Output

Source sentence

Human translation

Generated translation

This allows common translation errors and generation behavior to be inspected directly.

Key Outputs

The notebook generates:

English sentence-length distribution

Urdu sentence-length distribution

Train/validation/test split summary

Vocabulary statistics

Model architecture information

Training/validation loss curve

Final test loss

Test perplexity

Representative English → Urdu translations

Reproducibility

The project fixes:

SEED = 42

The seed is used for reproducible data splitting and relevant random operations.

The pipeline also performs an explicit leakage check to ensure that exact English–Urdu sentence pairs do not overlap across the train, validation, and test splits.

GPU execution can still introduce minor numerical differences between runs.

Technology Stack

Python

TensorFlow / Keras

NumPy

Pandas

Scikit-learn

KaggleHub

Matplotlib

Kaggle Notebooks

Tesla T4 GPU

Repository Structure

English-Urdu-Neural-Machine-Translation/
│
├── Compiled_Question02.ipynb
├── main.tex
├── figs/
└── README.md

Main Notebook

Compiled_Question02.ipynb

The notebook contains the complete English-to-Urdu NMT pipeline, including preprocessing, tokenization, vocabulary creation, sequence preparation, Vanilla RNN training, greedy decoding, evaluation, and translation examples.

How to Run

The project was designed for execution in Kaggle Notebooks.

Requirements

A GPU is recommended for training.

Main dependencies include:

Python 3.12
TensorFlow / Keras
NumPy
Pandas
Scikit-learn
Matplotlib
KaggleHub

The notebook installs kagglehub automatically where required and downloads the dataset programmatically.

For execution outside Kaggle, configure a valid Kaggle API credential before running the dataset-download section.

Limitations

The system intentionally uses a Vanilla RNN rather than more modern architectures.

Potential limitations include:

Difficulty modeling long-range dependencies

Limited contextual representation compared with Transformers

Greedy decoding can produce suboptimal sequences

Word-level vocabulary can lead to <UNK> tokens

Translation quality depends heavily on dataset size and domain

The dataset domain may not represent general-purpose English–Urdu conversation

Future Work

Possible extensions include:

BLEU and ROUGE evaluation

Beam-search decoding

Subword tokenization such as BPE

Larger embedding and hidden dimensions

Attention-based encoder–decoder models

LSTM/GRU comparison

Transformer-based English–Urdu translation

Larger and more diverse parallel corpora

Academic Context

Generative Artificial Intelligence — Assignment #1

This project represents the English→Urdu Neural Machine Translation component of the assignment.

The same assignment also contains a separate Deep CNNs for Chest X-Ray Pneumonia Classification component.

Authors

Shah Faisal — 23I-0058
Hamad Khan — 23I-3095

Department of Software Engineering
FAST — National University of Computer and Emerging Sciences (NUCES), Islamabad
