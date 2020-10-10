'''Collection of all functions required for loading and preprocessing the language documents.
'''

import os
import re
import sys
import matplotlib.pyplot as plt
import numpy
from hyperparameters import K
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from tensorflow.data import Dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_doc(filepath: str, number_sentences: int, do_lower_case: bool = True) -> [str]:
    '''Loads a language file from the Europarl Dataset.

    Args:
        filepath: path to file that which content should be read.
        number_sentences: number of sentences that should be returned.
        do_lower_case: whether to convert every character to lower case.
    Returns:
        sentences: list containing strings from input file.
    '''
    # check if file exists, then read it line by line.
    if not os.path.isfile(filepath):
        print("Either the path is wrong or you forgot to put the file in its specified location.")
        print(f"Missing file: {filepath}")
        print("Please download the Europarl dataset manually and put it in the appropriate file path.")
        sys.exit()

    with open(filepath, mode='r', encoding="utf-8") as reader:
        sentences = reader.read().splitlines()

    if number_sentences is not None:
        # from all read sentences, take the 'number_sentences' first ones
        sentences = sentences[:number_sentences]
    # convert each string to lower case.
    if do_lower_case:
        sentences = [sentence.lower() for sentence in sentences]

    print(f"Loaded {len(sentences):,} sentences from file: {filepath}")
    return sentences


###############################
#    PREPARATION FUNCTIONS    #
###############################

def fix_punctuation(sentence: str) -> str:
    '''Cleans the sentences.

    Args:
        sentence: one string containing the sentence.
    Returns:
        sentence: the cleaned sentence.
    '''
    # remove german umlaute
    sentence = sentence.replace('ä', 'ae').replace('Ä', 'Ae')
    sentence = sentence.replace('ü', 'ue').replace('Ü', 'Ue')
    sentence = sentence.replace('ö', 'oe').replace('Ö', 'Oe')
    sentence = sentence.replace('ß', 'ss')

    # remove (FR), (EN), (DE), ...
    sentence = re.sub(r'\([A-Z]{2}\)', '', sentence)
    # remove clock time (13.20 a.m. or 13.20 Uhr)
    sentence = re.sub(r'\d*\.\d{1,2}\s+', '', sentence)
    # remove dates (e.g. 30.6.2001)
    sentence = re.sub(r'\d+\.\d+\.\d+', '', sentence)
    # replace '' and "" with nothing
    sentence = re.sub(r'([\'"])', '', sentence)
    # replace every character with space except: a-z, A-Z and the characters , ? ! .
    sentence = re.sub(r'[^a-zA-Z,?!.]+', ' ', sentence)
    # add white space before and after these characters: , ? ! .
    sentence = re.sub(r'([,?!.])', r' \1 ', sentence)
    # replace several space characters by one space
    sentence = re.sub(r'\s+', r' ', sentence)

    return sentence


def train_sentencepiece_tokenizer(sentences: list, vocab_size: int, folder_name: str = "sentencepiece",
                                  model_name: str = "tokenizer_de") -> None:
    '''Trains a sentencepiece tokenizer on a given corpus.

    Args:
        sentences: contains all sentences of a corpus.
        vocab_size: maximum number of (sub-)words in the vocabulary of the tokenizer.
        folder_name: name of the folder where the trained tokenizer will be placed in.
        model_name: filename of the trained sentencepiece tokenizer.
    '''
    temp_file = "sentences.txt"  # this file will be deleted after training of the tokenizer is done.

    if folder_name != "":
        output_file = folder_name + "/" + model_name
    else:
        output_file = model_name

    # write all sentences to a temporary file
    with open(temp_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + "\n")

    parameters = f"--input={temp_file} \
                --model_prefix={output_file} \
                --vocab_size={vocab_size} \
                --bos_id=2 \
                --eos_id=3 \
                --unk_id=1 \
                --pad_id=0 \
                --bos_piece=<s> \
                --eos_piece=</s> \
                --hard_vocab_limit=false"
    # train tokenizer on our corpus
    SentencePieceTrainer.train(parameters)
    # delete temp_file
    os.remove(temp_file)


def load_sentencepiece_tokenizer(tokenizer_path: str) -> SentencePieceProcessor:
    ''' Loads an already pretrained sentencepiece tokenizer.

    Args:
        tokenizer_path: path to the files of the pretrained sentencepiece tokenizer.
    Returns:
        tokenizer: pretrained sentencepiece tokenizer.
    '''
    if not os.path.isfile(tokenizer_path):
        print("SentencePiece tokenizer not found!")
        sys.exit()

    tokenizer = SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)
    # enable inserting <s> and </s> tags automatically at start/end of a sentence.
    tokenizer.set_encode_extra_options('bos:eos')
    return tokenizer


def tokenize_sentences(sentences: list, tokenizer_path: str, sequence_length: int,
                       vocab_size: int, teacher_forcing: bool) -> [numpy.ndarray]:
    '''Tokenization of the Source/Target Language using the sentencepiece tokenizer.

    Note:
        In the Seq2Seq model, we are using Teacher forcing for the Decoder
        At the beginning the decoder gets as input the first word and must then predict the second word (its output),
        in the next step, it gets as input the second word and must predict the third word.
        Therefore: the input and output of the decoder are shifted by one observation.

    Args:
        tokenizer_path: path to the pretrained Sentencepiece tokenizer.
        sentences: list containing the sentences as a string.
        sequence_length: maximum number of tokens within a sequence (sentence).
        vocab_size: the maximum number of words to keep, based on word frequency.
                    Only the most common `num_words-1` words will be kept.
        teacher_forcing: whether to prepare sequences for teacher forcing.
    Returns:
        (If teacher_forcing == False):
            encoder_input: ids for each token in a sentence.
        (If teacher_forcing == True):
            decoder_input: ids for each token in a sentence. (Last token removed)
            decoder_target_output: ids for each token in a sentence. (First token removed)
    '''

    # if no pretrained tokenizer is found, train one
    if not os.path.isfile(tokenizer_path):
        foldername, file = os.path.split(tokenizer_path)
        filename = os.path.splitext(file)[0]
        if not os.path.isdir(foldername):
            os.mkdir(foldername)
        if vocab_size is not None:
            print("No pretrained tokenizer found. Begin training...")
            train_sentencepiece_tokenizer(sentences, vocab_size, foldername, filename)
        else:
            print("Please train a model first. Exiting.")
            sys.exit()

    # load pretrained tokenizer
    tokenizer = load_sentencepiece_tokenizer(tokenizer_path)

    # Convert tokens to ids for every sentence in the whole corpus.
    token_ids = tokenizer.encode(sentences)

    if teacher_forcing == True:
        # pad the created token_ids
        # hint: maxlen = sequence_length + 1, as we are going to cut of the first/last token id for teacher forcing
        padded_tokens = pad_sequences(token_ids, padding="post", maxlen=sequence_length + 1)

        ## Prepare token_ids such that Teacher Forcing for the decoder actually works.

        # Remove the last token id from the decoder input (what is the actual input to the decoder)
        decoder_input = [token_seq[:-1]for token_seq in padded_tokens]
        decoder_input = numpy.vstack(decoder_input)

        # Remove first token id from the decoder target output (what the predicted output of the decoder should be)
        decoder_target_output = [token_seq[1:]for token_seq in padded_tokens]
        decoder_target_output = numpy.vstack(decoder_target_output)

        return([decoder_input, decoder_target_output])

    else:
        # pad the created token_ids
        encoder_input = pad_sequences(token_ids, padding="post", maxlen=sequence_length)
        return([encoder_input])


def remove_multiple_sentences(source_sentences: [str], target_sentences: [str]) -> [str]:
    '''Removes strings which empty strings in source and target language.

    Problem: Some strings (either in the source or in the target language) might contain empty strings.
    For a real example: see line 22 in both files contained in the folder "de-en".
    Example:
        english_sentence1 = ""  german_sentence1 = "Frau Präsidentin!"
        english_sentence2 = "Miss President. How are you?" german_sentence2 = "Wie geht es Ihnen?"

    To maximize translation performance, we will remove such line numbers, including the preceeding and succeeding line.

    Args:
        source_sentences: list of strings in the source language.
        target_lang: list of strings in the target language.
    Returns:
        prep_source_sentences: containing strings with only one sentence in the source language.
        prep_target_sentences: containing strings with only one sentence in the target language.
    '''

    indices_to_drop = list()

    for sentence_index in range(len(source_sentences)):
        source_string = source_sentences[sentence_index]
        target_string = target_sentences[sentence_index]

        if source_string == "" or target_string == "":
            indices = [sentence_index - 1, sentence_index, sentence_index + 1]
            indices_to_drop.extend(indices)

    indices_to_drop = list(set(indices_to_drop))
    # account for the case that the first or last sentence might be empty. The list of inidices to drop then would contain
    # indices -1 and max_length+1, which would create an out-of-bounds error. Thus, remove those indices.
    indices_to_drop = [x for x in indices_to_drop if x > 0 and x < len(source_sentences)]

    prep_source_sentences = [source_sentences[ind] for ind in range(len(source_sentences)) if ind not in indices_to_drop]
    prep_target_sentences = [target_sentences[ind] for ind in range(len(target_sentences)) if ind not in indices_to_drop]

    print(f"Removed {len(indices_to_drop):,} strings.")
    print(f"Remaining sentences: {len(prep_source_sentences):,}.\n")
    return prep_source_sentences, prep_target_sentences


def generate_tf_data(enc_input: list, dec_input: list, batch_size: int, train_size: int, val_size: int) -> [Dataset]:
    '''Generates a tensorflow data set, splits it in train, test and validation sets.

    Problem: Feeding in three arrays containing almost two million sequences each, requires too much main memory.
    Solution: We use the Tensorflow Dataset, where we can feed the model with slices of the whole dataset.

    Also: shuffles the observations.

    Args:
        enc_input: encoder input ids, token ids for each word and each sentence
        dec_input: used for teacher forcing. Token ids for each word and each sentence in target lang.
            More specific:
                - decoder input, token sequences (index 0 in dec_input)
                - decoder target output, token sequences (for teacher forcing, index 1 in dec_input)
        batch_size: Number of observation passed to the Seq2Seq model during training time.
        train_size: Fraction of all observations to be reserved for training the model.
        val_size: Fraction of all observations to be reserved for evaluating the model performance during training.
    Returns:
        train_data: contains encoder_input, decoder_input, decoder_target_output for training the model.
        val_data: contains encoder_input, decoder_input, decoder_target_output for evaluating the model.
    '''

    assert train_size + val_size == 1, "Train, Validation and Test size doesn't sum up to 1!"

    data_size = enc_input[0].shape[0]

    # Summarize the source language token ids and the decoder input as: model_input
    model_input = Dataset.from_tensor_slices((enc_input[0], dec_input[0]))
    #                                         enc_token_ids dec_token_ids

    # convert decoder_target_output to TF.Dataset
    decoder_target_output = Dataset.from_tensor_slices((dec_input[1]))
    #                                            dec_token_ids used as target output (shifted by one observation)

    # Combine the model_input and the decoder_target_output to a full TF.Dataset, shuffle it
    full_data = Dataset.zip((model_input, decoder_target_output)).shuffle(data_size)

    # Train Val split
    train_size = int(train_size * data_size)
    val_size = int(val_size * data_size)

    train_data = full_data.take(train_size)
    val_data = full_data.skip(train_size)

    train_data = train_data.batch(batch_size, drop_remainder=True)
    val_data = val_data.batch(batch_size, drop_remainder=True)

    return train_data, val_data


def remove_long_sequences(source_tokens: [numpy.ndarray], target_tokens: [numpy.ndarray]) -> [numpy.ndarray]:
    ''' Removes sequences which exceeds the specified max. number of tokens.

    Idea:
        In order to improve the dataset quality, we will filter out all sequences which
        exceed the maximum sequence length.
        Given a padded sequence, we first look where the end-of-sentence </s> indicator token is located at
        (the </s> token with token id 3). Since we are padding the sequences, the only tokens ids following
        the EOS id 3 are the padding token ids 0.
        We loop through all sequences and look at which index the </s> token is located at.
        Since very long sequences are simply cut off, they will have the </s>-token id at the last index.
        Thus, we will remove all sequences where the </s>-token id 3 is at the last index.
        (Info: there might be sequences which just might be exactly 'SEQUENCE_LENGTH' indices long. We still remove them,
         as we cannot know whether the encoded sentence is complete or is exceeding the max. specified length.)

    Args:
        source_tokens: contains token ids of all sequences of source language sentences.
        target_tokens: contains token ids of all sequences of target language sentences.
    Returns:
        source_tokens: all source language sequences not exceeding the maximum sequence length.
        target_tokens: all target language sequences not exceeding the maximum sequence length.
    '''
    indices_to_drop = list()

    src_indices = numpy.where(numpy.where(source_tokens[0] == 3)[1] == 69)
    indices_to_drop.extend(list(src_indices[0]))

    # target_tokens[1] = (target) token ids of encoded target language sentences
    trg_indices = numpy.where(numpy.where(target_tokens[1] == 3)[1] == 69)
    indices_to_drop.extend(list(trg_indices[0]))

    indices_to_drop = list(set(indices_to_drop)) # removes duplicate indices from list

    source_tokens = [numpy.delete(x, indices_to_drop, axis=0) for x in source_tokens]
    target_tokens = [numpy.delete(x, indices_to_drop, axis=0) for x in target_tokens]

    print(f"Removed {len(indices_to_drop):,} sentences (probably) exceeding maximum specified length.")
    print(f"Remaining sentences: {len(source_tokens[0]):,}\n")

    return source_tokens, target_tokens


def prepare_languages(source_lang_path: str, target_lang_path: str, number_sentences: int = None,
                      sequence_length: int = None, vocab_size_src: int = 30000, vocab_size_trg: int = 30000,
                      tokenizer_src_path: str = None, tokenizer_trg_path: str = None, do_lower_case: bool = True,
                      batch_size: int = 64, train_size: float = 0.8, val_size: float = 0.1) -> [Dataset]:
    '''Wrapper to load both source and language files, to trim, preprocess and to convert them for the translation model.

    Args:
        source_lang_path: path to the source language file containing strings.
        target_lang_path: path to the target language file containing strings.
        number_sentences: maximum number of sentences that should be read.
        sequence_length: maximum sequence length of token ids that the input to the encoder/decoder can have.
        vocab_size_src: maximum number of unique token ids in the source language.
        vocab_size_trg: maximum number of unique token ids in the target language.
        tokenizer_src_path: path to pretrained sentencepiece tokenizer on the source corpus.
        tokenizer_trg_path: path to pretrained sentencepiece tokenizer on the target corpus.
        do_lower_case: whether to convert all sentences into lower-case.
        batch_size: batch size to be used during training.
        train_size: fraction of sequences used during training.
        val_size: fraction of sequences used for validating model performance during training.

    Returns:
        train_data: tf.data; contains data for training the model
        val_data: tf.data; contains data for validating the model's accuracy after each epoch
        test_data: tf.data; contains data for testing the final models accuracy.
    '''
    # load raw language files as strings
    source_sentences = load_doc(source_lang_path, number_sentences, do_lower_case)
    target_sentences = load_doc(target_lang_path, number_sentences, do_lower_case)

    assert len(source_sentences) == len(target_sentences), "Source and Target file do not contain same amount of strings.\n"

    print("Removing strings with empty sentences...")
    source_sentences, target_sentences = remove_multiple_sentences(source_sentences, target_sentences)
    assert len(source_sentences) > 0 and len(
        target_sentences) > 0, "The number of sentences in the config file was selected to small!"

    print("Fixing Punctuation...")
    source_sentences = [fix_punctuation(s) for s in source_sentences]
    target_sentences = [fix_punctuation(s) for s in target_sentences]

    print("Tokenize Source Sentences...\n")
    source_tokens = tokenize_sentences(source_sentences, tokenizer_src_path, sequence_length, vocab_size_src, False)

    print("Tokenize Target Sentences...\n")
    target_tokens = tokenize_sentences(target_sentences, tokenizer_trg_path, sequence_length, vocab_size_trg, True)

    print("Removing sentences exceeding maximum sequence length...\n")
    source_tokens, target_tokens = remove_long_sequences(source_tokens, target_tokens)

    print("Converting to TF.Data...")
    train_data, val_data = generate_tf_data(source_tokens, target_tokens, batch_size, train_size, val_size)
    print("Data Preparation done!\n")
    return train_data, val_data
