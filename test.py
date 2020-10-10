'''
File to import and to test a pretrained Seq2Seq model.
'''
import os
import pickle
import sys
import numpy
from model import get_models
from hyperparameters import K
from preprocessing import (load_sentencepiece_tokenizer, load_doc, remove_multiple_sentences, fix_punctuation)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from greedy_approach import greedy_translation
from bleu import list_bleu
from rouge import Rouge
from sklearn.metrics import accuracy_score


checkpoint_file = 'saved_models/best_model_acc.h5'

if not os.path.isfile(checkpoint_file):
    print("Pretraining of the model required. Aborting...")
    sys.exit()


if __name__ == "__main__":

    # Define file paths to source and target corpus, pretrained sentencepiece tokenizer
    file_src = "de-en/europarl-v7.de-en.en"
    file_trg = "de-en/europarl-v7.de-en.de"

    # Load pretrained model
    translation_model, encoder, decoder = get_models(for_training=False)
    translation_model.load_weights(checkpoint_file)

    # compute Scores
    if K.ADVANCED_METRICS:
        ''' To compute the accuracy, we need: target_ref_sequences = tokenized sentences from original sentences
                                              target_hyp_sequences = predicted tokens for each sentence

            To compute BLEU and ROUGE, we need: target_ref_sentences = original sentences
                                                target_hyp_sentences = decoded predicted sentences
        '''
        print("Computing BLEU, ROUGE and Accuracy scores...")

        # from all sentences (1.920.209), use the next x sentences not used for training the model to evaluate
        # the model performance, with x = 'K.NUMBER_TEST_SENTENCES'
        if K.NUMBER_SENTENCES is not None:
            if (1920209 - (K.NUMBER_SENTENCES + K.NUMBER_TEST_SENTENCES)) > 0:
                total_sentences = K.NUMBER_SENTENCES + K.NUMBER_TEST_SENTENCES
            else:
                print("Please select a smaller number for sentences to test.")
                sys.exit()
        else:
            total_sentences = None

        # load raw language files as strings
        source_sentences = load_doc(file_src, total_sentences, do_lower_case = True)[-K.NUMBER_TEST_SENTENCES:]
        target_ref_sentences = load_doc(file_trg, total_sentences, do_lower_case = True)[-K.NUMBER_TEST_SENTENCES:]

        # Remove strings with empty sentences
        source_sentences, target_ref_sentences = remove_multiple_sentences(source_sentences, target_ref_sentences)

        # Fix Punctuation (for source sentences, this is done in the greedy function)
        target_ref_sentences = [fix_punctuation(s) for s in target_ref_sentences]

        # To tokenize the original sentences, load pretrained tokenizer
        target_tokenizer = load_sentencepiece_tokenizer(K.TOKENIZER_TRG_PATH)
        # Tokenize target sentences and pad them
        target_ref_sequences = target_tokenizer.encode(target_ref_sentences)
        target_ref_sequences = pad_sequences(target_ref_sequences, padding="post", maxlen=K.SEQUENCE_LENGTH)

        target_hyp_sequences = [] # will hold predicted target sequences

        print("Predicting sentences. This might take a while.")
        # predict taget sequences given the original source sentences
        for sentence in source_sentences:
            hyp = greedy_translation(sentence, K.SEQUENCE_LENGTH, K.TOKENIZER_SRC_PATH, K.TOKENIZER_TRG_PATH,
                                     True, encoder, decoder)
            target_hyp_sequences.append(hyp)

        # pad predicted target sequences
        target_hyp_sequences = pad_sequences(target_hyp_sequences, padding="post", maxlen=K.SEQUENCE_LENGTH)
        target_hyp_sentences = [target_tokenizer.decode_ids(s.tolist()) for s in target_hyp_sequences]

        print("Computing Scores.")
        # compute BLEU score
        bleu_score = list_bleu([target_ref_sentences], target_hyp_sentences)

        # compute average ROUGE score
        rouge = Rouge()
        rouge_scores = rouge.get_scores(target_hyp_sentences, target_ref_sentences, avg=True)

        # compute average Accuracy score
        # important: padding tokens in both sequences would yield a higher accuracy score.
        # to avoid this, for each reference and hypotheses pair, we compute the inidices of the first occurence
        # of a padding token id, and then take the higher index. Then we truncate the sequence pairs according to
        # the computed index, such that we will minimize the influence of padding tokens affecting the acc score.

        # truncate both sequences
        # will store sequences with removed padding tokens
        list_target_ref_seq = list()
        list_target_hyp_seq = list()

        for idx in range(target_hyp_sequences.shape[0]):
            hyp_found_indices = numpy.where(target_hyp_sequences[idx] == 3)[0]
            # might be the case that there is no padding token, so we have to check it
            if not hyp_found_indices.size == 0:
                hyp_index = hyp_found_indices[0]
            else:
                hyp_index = 0

            ref_found_indices = numpy.where(target_ref_sequences[idx] == 3)[0]
            if not ref_found_indices.size == 0:
                ref_index = ref_found_indices[0]
            else:
                ref_index = 0

            idx_to_cut_at = max(hyp_index, ref_index)

            truncated_hyp = target_hyp_sequences[idx, :idx_to_cut_at]
            truncated_ref = target_ref_sequences[idx, :idx_to_cut_at]
            list_target_hyp_seq.append(truncated_hyp)
            list_target_ref_seq.append(truncated_ref)

        accuracy_avg_score = 0.0
        for ref, hyp in zip(list_target_ref_seq, list_target_hyp_seq):
            accuracy_avg_score += accuracy_score(ref, hyp)
        accuracy_avg_score = accuracy_avg_score/len(list_target_hyp_seq)
        accuracy_avg_score

        print(f"BLEU Score: {bleu_score}\nAccuracy Score: {accuracy_avg_score}\nROUGE Scores: {rouge_scores}")

    # Translate a user sentence
    while True:
        user_input = input(">>> What do you want to translate? (q to Quit):")
        user_input = user_input.lower()
        if user_input == "q":
            break

        translation = greedy_translation(user_input, K.SEQUENCE_LENGTH, K.TOKENIZER_SRC_PATH, K.TOKENIZER_TRG_PATH,
                                         False, encoder, decoder)
        print(f"Greedy Translation: {translation}\n")
