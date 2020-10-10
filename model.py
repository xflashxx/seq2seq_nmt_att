'''
Definition of Seq2Seq Attention model.

Implemention after Luong(2015) with score function: "concat"
'''
import tensorflow as tf
from tensorflow.keras.layers import (LSTM, Concatenate, Dense, Embedding, Input, Layer)
from tensorflow.keras.models import Model
from hyperparameters import K
from attentionLayer import AttentionLayer



## Definition of NMT model with Attention

# define encoder of NMT
def get_encoder() -> Model:
    # define Inputs to the Encoder
    token_ids = Input(shape=(K.SEQUENCE_LENGTH), name='Encoder_Input_IDs', dtype=tf.int32)

    # define all required Layers (Embeddings-Layer, 2x LSTM-Layer)
    embedding_layer = Embedding(K.VOCAB_SIZE_SRC, output_dim=(K.EMBEDDING_DIM), mask_zero=True, name="Enc_Embedding_Layer")
    # 2x LSTM Layers
    lstm_layer1 = LSTM(K.LSTM_UNITS, return_sequences=True, return_state=True, dropout=K.DROPOUT, name="Encoder_Layer1")
    lstm_layer2 = LSTM(K.LSTM_UNITS, return_sequences=True, return_state=True, dropout=K.DROPOUT, name="Encoder_Layer2")

    # compute contextual word embeddings
    word_embeddings = embedding_layer(token_ids)
    # compute output of LSTM Layers
    encoder_output, enc_h1, enc_c1 = lstm_layer1(word_embeddings)
    encoder_output, enc_h2, enc_c2 = lstm_layer2(encoder_output)
    # create the Encoder model
    encoder = tf.keras.Model(inputs=[token_ids],
                             outputs=[encoder_output, [enc_h1, enc_c1], [enc_h2, enc_c2]])
    return encoder

# define decoder of NMT
def get_decoder(for_training: bool = True) -> Model:
    # Define Inputs to the Decoder
    if for_training == True:
        # input will be a whole sequence
        token_ids = Input(shape=(K.SEQUENCE_LENGTH), dtype=tf.int32, name="Decoder_Input_IDs")
    else:
        # input will only be one token at a time
        token_ids = Input(shape=(1), dtype=tf.int32, name="Decoder_Input_IDs")

    # Addtional Input: last (hidden and cell) states from the Encoder
    enc_h1 = Input(shape=(K.LSTM_UNITS,), dtype=tf.float32, name="Encoder_Hidden_State_1")
    enc_c1 = Input(shape=(K.LSTM_UNITS,), dtype=tf.float32, name="Encoder_Cell_State_1")
    enc_h2 = Input(shape=(K.LSTM_UNITS,), dtype=tf.float32, name="Encoder_Hidden_State_2")
    enc_c2 = Input(shape=(K.LSTM_UNITS,), dtype=tf.float32, name="Encoder_Cell_State_2")

    # Additional Input: Whole Encoder Output Sequence
    encoder_output = Input(shape=(K.SEQUENCE_LENGTH, K.LSTM_UNITS), dtype=tf.float32, name="Encoder_Hidden_States")

    # Define all required Layers (Embedding, 2x LSTM, Attention, 2x Dense Layer)
    embedding_layer = Embedding(K.VOCAB_SIZE_TRG, output_dim=(K.EMBEDDING_DIM), mask_zero=True, name="Dec_Embedding_Layer")
    # 2x LSTM Layer
    lstm_layer1 = LSTM(K.LSTM_UNITS, return_sequences=True, return_state=True, dropout=K.DROPOUT, name="Decoder_Layer1")
    lstm_layer2 = LSTM(K.LSTM_UNITS, return_sequences=True, return_state=True, dropout=K.DROPOUT, name="Decoder_Layer2")
    # Attention Layer
    attention_layer = AttentionLayer(K.ATTENTION_DIM)
    # 2x Dense Layer (1 to compute attentional vector, 1 to compute word prediction)
    wc_layer = Dense(K.LSTM_UNITS, activation="tanh")
    # for the dense layer we do not use activation = softmax, as doing so will increase computation time drastically
    dense_layer = Dense(K.VOCAB_SIZE_TRG, name="Dense_Output_Layer")


    # Compute Word Embeddings
    word_embeddings = embedding_layer(token_ids)
    # Compute output of LSTM Layers
    decoder_output, dec_h1, dec_c1 = lstm_layer1(word_embeddings, initial_state=[enc_h1, enc_c1])
    decoder_output, dec_h2, dec_c2 = lstm_layer2(decoder_output, initial_state=[enc_h2, enc_c2])
    # Compute Context Vector
    context_vector, alignment_vector = attention_layer(decoder_output, encoder_output)
    # Concatenation of context_vector and hidden state of decoder
    decoder_output = Concatenate(name="Output_With_Context")([context_vector, decoder_output])
    # Compute Attentional Vector
    attentional_hidden_state = wc_layer(decoder_output)
    # Dense (prediction) layer
    logits = dense_layer(attentional_hidden_state)

    # Define Decoder model
    decoder = tf.keras.Model(inputs=[token_ids, [enc_h1, enc_c1], [enc_h2, enc_c2], encoder_output],
                             outputs=[logits, [dec_h1, dec_c1], [dec_h2, dec_c2]])
    return decoder


# define NMT model (combine encoder and decoder, then return all three models)
def get_models(for_training: bool = True) -> [Model]:

    # get encoder and decoder
    encoder = get_encoder()
    decoder = get_decoder(for_training)

    # define Inputs (for the Encoder and Decoder)
    encoder_token_ids = Input(shape=(K.SEQUENCE_LENGTH,), name='Encoder_Input_IDs', dtype=tf.int32)

    if for_training == True:
        decoder_token_ids = Input(shape=(K.SEQUENCE_LENGTH), dtype=tf.int32, name="I_Dec_Input_IDs")
    else:
        decoder_token_ids = Input(shape=(1), dtype=tf.int32, name="I_Dec_Input_IDs")

    encoder_output, enc_hidden1, enc_hidden2 = encoder([encoder_token_ids])
    logits, dec_hidden1, dec_hidden2 = decoder([decoder_token_ids, enc_hidden1, enc_hidden2, encoder_output])

    # Define Translation Model
    translation_model = tf.keras.Model(inputs=[encoder_token_ids, decoder_token_ids],
                                       outputs=[logits])
    adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
    translation_model.compile(optimizer=adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['sparse_categorical_accuracy'])

    return translation_model, encoder, decoder
