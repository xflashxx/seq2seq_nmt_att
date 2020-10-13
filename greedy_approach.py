import numpy
import tensorflow as tf
from preprocessing import (load_sentencepiece_tokenizer, fix_punctuation, tokenize_sentences)


# Greedy Translation
def greedy_translation(sentence: str, sequence_length: int, tokenizer_src_path: str,
                       tokenizer_trg_path: str, return_sequence: bool, encoder, decoder):
    '''Greedy Translation approach.

    Args:
        sentene: Sentence in Source language to be translated into Target Language.
        sequence_length: Maximum sequence length used when training the Seq2Seq Model.
        tokenizer_src_path: Path to the pretrained sentencepiece tokenizer for the source language.
        tokenizer_trg_path: Path to the pretrained sentencepiece tokenizer for the target language.
        return_sequence: If True, the generated tokens will be returned and not decoded into text.
        encoder: Keras Model, Encoder from the Translation model.
        decoder: Keras Model, Decoder from the Translation model.
    Returns:
        output_sentence: decoded token id sequence (into words) created by the decoder OR
        output_sequence: token id sequence created by the decoder.
    '''
    # Prepare input sentence
    sentence = fix_punctuation(sentence.lower())
    enc_input = tokenize_sentences(sentences = [sentence], tokenizer_path = tokenizer_src_path,
                                   sequence_length = sequence_length, vocab_size = None, teacher_forcing = False)

    # compute initial state for the decoder, given from the encoder
    enc_output, enc_hidden1, enc_hidden2 = encoder.predict(enc_input)

    # prepare first decoder input: convert "<s>" to its token id.
    dec_input = numpy.zeros((1, 1))
    # load target language (sentencepiece) tokenizer
    target_tokenizer = load_sentencepiece_tokenizer(tokenizer_trg_path)
    # get <s> (begin of sentence) and </s> (end of sentence) token id
    bos_id = target_tokenizer.bos_id()
    eos_id = target_tokenizer.eos_id()

    # set initial input to decoder as start token
    dec_input[0, 0] = bos_id
    # set hidden states from both LSTM layers from decoder to final hidden states from LSTM layers from encoder
    dec_hidden1, dec_hidden2 = enc_hidden1, enc_hidden2
    # create output sequence list, will hold predicted token ids
    output_sequence = [bos_id]

    while True:
        logits, dec_hidden1, dec_hidden2 = decoder.predict([dec_input, dec_hidden1, dec_hidden2, enc_output])
        # get id with max. probability from the predicted decoder output
        pred_token_id = int(tf.argmax(logits, -1).numpy()[0][0])
        dec_input[0, 0] = pred_token_id
        # add predicted token id to the list of all predicted token ids so far
        output_sequence.append(pred_token_id)
        # check if sequence exceeds max. seq. length or if EOS-tag was predicted, then break
        if len(output_sequence) >= sequence_length or pred_token_id == eos_id:
            break
    # given predicted token sequence, convert it into a sentence
    output_sentence = target_tokenizer.decode_ids(output_sequence)

    if return_sequence == True:
        return output_sequence
    else:
        return output_sentence
