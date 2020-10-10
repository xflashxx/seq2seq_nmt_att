import tensorflow as tf
from tensorflow.keras.layers import (Dense, Layer)
from hyperparameters import K


# define custom Attention Layer (Luong)
class AttentionLayer(Layer):
    def __init__(self, attention_dim):
        super(AttentionLayer, self).__init__()
        self.Wa = Dense(attention_dim)
        self.va = Dense(1)

    @tf.function
    def call(self, decoder_state, encoder_states):
        '''
        Score Function: Concat
        decoder state at time u: d_u
        encoder state at time t: e_t
        then score(d_u, e_t) = va * tanh(Wa[d_u, e_t])

        Steps:  1) first concatenate d_u and e_t (using tiling, since we have all hidden states from the encoder,
                   not just e_t, but e_1, ... , e_T)
                2) apply Dense layer to concatenated states
                3) Apply tanh function
                4) apply Dense(1) layer

        decoder_state shape:  (batch, 1, lstm_units)
        encoder_states shape: (batch, sequence_length, lstm_units)
        -> modify decoder_state such that: (batch, sequence_length, lstm_units)
        '''
        decoder_sequence_length = decoder_state.shape[1]
        if not decoder_sequence_length == encoder_states.shape[1]:
            decoder_state = tf.tile(decoder_state, [1, encoder_states.shape[1], 1])
        score = self.va(
                    tf.nn.tanh(
                        self.Wa(
                            tf.concat((decoder_state, encoder_states), axis=-1))))

        # score shape: (batch, sequence_length, 1)
        alignment_vector = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(
            tf.math.multiply(alignment_vector, encoder_states), axis=1)
        context_vector = tf.expand_dims(context_vector, axis=1)
        context_vector = tf.tile(context_vector, [1, decoder_sequence_length, 1])
        return context_vector, alignment_vector
