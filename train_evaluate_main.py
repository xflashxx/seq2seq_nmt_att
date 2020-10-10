'''Main script.

Used to load the prepared dataset and to train the model.

'''
import os
import pickle

import tensorflow as tf
from model import get_models
from hyperparameters import K
from preprocessing import prepare_languages
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Avoid size constraints on GPU memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


if __name__ == "__main__":

    # LOADING AND PREPARING THE DATA

    # Define file paths to english and german corpus, pretrained sentencepiece tokenizer
    file_src = "de-en/europarl-v7.de-en.en"
    file_trg = "de-en/europarl-v7.de-en.de"

    # Prepare Inputs for model
    train_data, val_data = prepare_languages(file_src, file_trg, K.NUMBER_SENTENCES, K.SEQUENCE_LENGTH,
                                                        K.VOCAB_SIZE_SRC, K.VOCAB_SIZE_TRG, K.TOKENIZER_SRC_PATH,
                                                        K.TOKENIZER_TRG_PATH, K.DO_LOWER_CASE, K.BATCH_SIZE,
                                                        K.TRAIN_SIZE, K.VAL_SIZE)


    ## MODEL TRAINING TIME

    # get translation model, its encoder and its decoder
    translation_model, encoder, decoder = get_models(for_training=True)
    translation_model.summary()
    # define model directory where best models will be saved into.
    # if it doesn't exist, create it.
    if not os.path.isdir("saved_models/"):
        os.mkdir("saved_models/")

    # define callbacks for model training
    checkpoint_file = 'saved_models/best_model_acc.h5'
    # callback for early stopping
    callbacks = [
        EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max', patience=K.PATIENCE, verbose=1),
        ModelCheckpoint(checkpoint_file, monitor='val_sparse_categorical_accuracy',
                        mode='max', save_best_only=True, save_weights_only=True, verbose=1)]

    print("Begin training... this might take a lot of time.")
    history = translation_model.fit(x=train_data, epochs=K.EPOCHS, batch_size=K.BATCH_SIZE,
                                    validation_data=val_data, verbose=1, callbacks=callbacks)
