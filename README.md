# Neural Machine Translation model with Attention

## Introduction
This is a simple implementation of a Seq2Seq neural machine translation model from English to German (though languages can be exchanged with whatever one would like) with the following key features:

* Encoder-Decoder architecture.
* Two stacked (unidirectional) LSTM-layers for the Encoder.
* Two stacked (unidirectional) LSTM-layers for the Decoder including an Attention-Mechanism as proposed by Luong (2015).
* Attention uses the 'concat' score function as proposed by Luong.
* Decoder uses Teacher Forcing during Training Time.

For more details, see section **Model Details** down below.

## Requirements
This model requires Python 3.8 or greater. Additionally, one needs to install the following packages by simply running the command `pip3 install -r requirements.txt` or install the following packages manually:

```
numpy >= 1.19.1
tensorflow >= 2.2.0
sentencepiece >= 0.1.91
matplotlib >= 3.3.0
bleu >= 0.3
rouge >= 1.0.0
scikit-learn >= 0.23.1
```

## Project structure

```
Files:
attentionLayer.py (Definition of the Attention Layer)
greedy_approach.py (Definition of Greedy Implementation)
hyperparameters.py (Definition of Hyperparameters)
model.py (Definition of the NMT model)
preprocessing.py (functions for preprocessing a dataset)
test.py (used for inference/translation)
train_evaluate_main.py (loading data, training the model)

Folders:
corpus (download of files below required)
|----europarl-v7.de-en.de
|----europarl-v7.de-en.en
sentencepiece (pretrained tokenizer on both languages, will be created during training)
|----tokenizer_src.model
|----tokenizer_src.vocab
|----tokenizer_trg.model
|----tokenizer_trg.vocab
saved_models (will hold trained model weights, will be created during training)
|----best_model_acc.h5 (model with best val accuracy)
```


## Usage

### Training the model
First clone the model `git clone https://github.com/xflashxx/seq2seq_nmt_att.git`, then change your current directory to the project directory `cd seq2seq_nmt_att`.

Prerequisites:

* English-German Europarl dataset.

Download the English-German Europarl Corpus from here: [Europarl-Dataset](https://www.statmt.org/europarl/v7/de-en.tgz), extract it and move the folder `corpus` to the project folder.

Alternatively, you can use another parallel Europarl Corpus as well (for example the English and Spanish parallel Corpus, EN-ES). Simply adjust the filepaths in the file `hyperparameters.py`. 

To train the model on the English-German Europarl Corpus, simply run:

```bash
python3 train_evaluate_main.py
```
Alternatively, you can open the file `train_evaluate_main.py` in an IDE of your choice and run it there.





### Translating with a pretrained model
In order to translate a sentence from the source into the target language, we need a pretrained model.
To make translations using your pretrained model, run it from the command line:

```bash
python3 test.py
```
You will then be able to enter a sentence in your source language, and the model will translate it into the target language.


### Define custom parameters
The most important parameters can be set in the class `K` in `hyperparameters.py` and can be changed. However, they must be changed before training the model (except the Translation parameters). Those parameters are:


**Data Parameters**:

* `FILE_SRC (default: "corpus/europarl-v7.de-en.en"`): file path to the source parallel corpus.
* `FILE_TRG (default: "corpus/europarl-v7.de-en.de"`): file path to the target parallel corpus.
* `NUMBER_SENTENCES (default: 1,000,000)`: The maximum number of sentences that should be read from the Europarl Corpus. If all sentences should be read, set it to `None`. Important: this parameter must be that high such that the validation and test datasets have at least `BATCH_SIZE` observations.
* `SEQUENCE_LENGTH (default: 70)`: The number of tokens one sequence/sentence will hold. Applies for both languages.
* `DO_LOWER_CASE (default: True)`: Whether to convert all text to lower-case. If changed, you must also retrain the SentencePiece tokenizer for the target language (simply delete the SentencePiece folder).
* `VOCAB_SIZE_SRC (default: 37,000)`: Size of the vocabulary for the source language.
* `VOCAB_SIZE_TRG (default: 37,000)`: Size of the vocabulary for the target language.
* `TRAIN_SIZE (default: 0.8)`: Fraction of all sentences to be used for training the model.
* `VAL_SIZE (default: 0.1)`: Fraction of all sentences used for evaluating model performance during training (will not be used for training the model).
* `TOKENIZER_SRC_PATH`: Path to the pretrained SentencePiece tokenizer for the Source Language (a pretrained tokenizer will be stored at the specified path).
* `TOKENIZER_TRG_PATH`: Path to the pretrained SentencePiece tokenizer for the Target Language (a pretrained tokenizer will be stored at the specified path).

**Model Parameters**:

* `EMBEDDING_DIM (default: 512)`: Dimension for the Embedding layer in the encoder and decoder.
* `ATTENTION_DIM (default: 512)`: Number of Attention Units.
* `LSTM_UNITS (default: 512)`: Number of LSTM units per Layer in both the Encoder and the Decoder.
* `DROPOUT (default: 0.2)`: Fraction of Input Units to be dropped on all LSTM-layers.
* `BATCH_SIZE (default: 64)`: Number of sentences that should be given to the model during training per step.
* `EPOCHS (default: 50)`: Number of epochs the model should be trained with.
* `PATIENCE (default: 10)`: Number of epochs to wait before canceling the training process when no improvement is observed.

**Translation Parameters**:

* `ADVANCED_METRICS (default: False)`: Whether to compute Accuracy, BLEU and ROUGE score.
* `NUMBER_TEST_SENTENCES (default: 10000)`: Number of test sentences to use to compute the performance metrics Accuracy, BLEU and ROUGE. Takes the next number of sentences specified here after the number of training sentences. If None, simply take the last specified number of sentences at the end of the entire corpus.


#### FAQ

**Can I use another dataset, not just EuroParl?**

Yes, as long as the sentences in both languages are in different files and are alligned (meaning: the n-th sentence in document A corresponds to the translated n-th sentence in document B). There are parallel corpora where a sentence from both languages are in one file, separated by a tab `\t`, but you will have to modify the code such that it takes that into account or convert your tabbed corpus.

## Model Details
This Neural Machine Translation model is a Sequence-to-Sequence (Seq2Seq) model, consisting of an Encoder and a Decoder implementing LSTMs. The Encoder processes the Input (a sentence, i.e. a fixed-length sequence of token ids) and returns its hidden and cell states.
The decoder takes the hidden sequences from the encoder, processes it, and outputs a probability distribution over each possible token (of size `K.VOCAB_SIZE_TRG`).

Overview of the entire NMT model:

![model](graphics/nmt.pdf?raw=true "The whole Model")

### Encoder
The Encoder requires as input:

* Token IDs: token ids of a tokenized sentence in the source language.

![encoder](graphics/encoder.pdf?raw=true "The Encoder")

First, word embeddings are computed, then these are processed by two stacked forward LSTM-layers, each outputting its last hidden and cell state. The last LSTM-layer additionally outputs all hidden states for every token in the sequence.
Thus, the encoder produces three outputs:

* all hidden states (over the entire sequence) from LSTM Layer 2 (*Encoder Output*)
* the last hidden states from LSTM Layer 1 and 2 (*enc_h1* and *enc_h2*)
* the last cell states from LSTM Layer 1 and 2 (*enc_c1* and *enc_c2*)

### Decoder
During training, we use teacher forcing. Thus we require an additional input for the decoder (not shown in the diagram below):

* Token IDs: ids of a tokenized (sub-)word in the target language, with the last token id of each sequence being removed.
* Tokens IDs (for teacher forcing): ids of a tokenized (sub-)word in the target language, with the first token id of each sequence being removed.
* all hidden states from the encoder (*Encoder Output*)
* the last hidden states from the encoder (*enc_h1* and *enc_h2*)
* the last cell states from the encoder (*enc_c1* and *enc_c2*)

![decoder](graphics/decoder.pdf?raw=true "The Decoder")

The Embedding layer takes as input the Token IDs (2x, again for teacher forcing) and outputs a word embedding for each token. Then the word embeddings are fed to two LSTM-layers, with initial states for LSTM Layer 1 initialized from the Encoders LSTM 1 Layer (*enc_h1* and *enc_c2*) and LSTM Layer 2 initialized from the Encoders LSTM 2 Layer (*enc_h2* and *enc_c2*).
The output of the last LSTM-layer (*Decoder_Output*) and all hidden states from the encoder are processed by an Attention Layer layer (Luong) and outputs a so called *Context vector*.
We then concatenate the context vector and the decoders output and apply a Dense layer (Wc Layer) to it, which yields a *attentional hidden state* vector. Finally, we apply a second Dense layer, producing logits for the prediction (of size `VOCAB_SIZE_TRG`). (Note: the last Dense Layer does not use an Activation function. Normally, we would use the *softmax* activation, however doing so would drastically increase both computation time and memory consumption when computing on GPUs).

The function `get_models()` defined in `model.py` will return three models:

* the whole translation model (encoder + decoder), used for training
* an encoder model (used for translation)
* a decoder model (used for translation)

The `translation_model` is used for training the whole model and incorporates both the encoder and the decoder.
The `encoder` and `decoder` models are simply just the seperate encoder and decoder from the `translation_model`.
We need the encoder and the decoder seperated from each other during inference/translation. Doing it this way, the encoder and decoder share the same (trained) weights as the whole translation model used for training it, since it shares its layers. Thus, we do not have to transfer the trained weights from the translation model to the encoder and decoder manually.

## Ressources
[M.-T. Luong, H. Pham, and C. D. Manning, “Effective Approaches to Attention-based Neural Machine Translation” 2015.](https://arxiv.org/abs/1508.04025)

[P. Koehn, “Europarl: A Parallel Corpus for Statistical Machine Translation,” in Conference Proceedings: the tenth Machine Translation Summit, AAMT. Phuket, Thailand: AAMT, 2005, pp. 79–86.](http://homepages.inf.ed.ac.uk/pkoehn/publications/europarl-mtsummit05.pdf)

[SentencePiece Tokenizer](https://github.com/google/sentencepiece)