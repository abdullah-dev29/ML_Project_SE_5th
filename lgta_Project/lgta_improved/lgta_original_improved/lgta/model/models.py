import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    Input, Bidirectional, Concatenate, LSTM, Dense, MultiHeadAttention,
    LayerNormalization, Dropout, RepeatVector, BatchNormalization,
    TimeDistributed, Flatten, Reshape, Conv1D, MaxPooling1D, GRU,
)
from keras.regularizers import l2
from keras.models import Model
from .helper import Sampling


class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, max_seq_len, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            shape=(self.max_seq_len, self.embed_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="pos_embedding",
        )
        super(PositionalEmbedding, self).build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos_encoding = self.pos_embedding[:seq_len, :]
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)
        return inputs + pos_encoding

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({"max_seq_len": self.max_seq_len, "embed_dim": self.embed_dim})
        return config


class TransformerBlockWithPosEncoding(tf.keras.layers.Layer):
    def __init__(self, num_heads, ff_dim, max_seq_len, rate=0.3, **kwargs):
        super(TransformerBlockWithPosEncoding, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.rate = rate

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.pos_embed_layer = PositionalEmbedding(self.max_seq_len, embed_dim)
        self.mhatt = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=embed_dim, name="multi_head_attention"
        )
        self.ffn = tf.keras.Sequential(
            [
                Dense(self.ff_dim, activation="relu", kernel_initializer="glorot_uniform"),
                Dense(embed_dim, kernel_initializer="glorot_uniform"),
            ],
            name="ffn",
        )

        self.layernorm1 = LayerNormalization(epsilon=1e-6, name="layer_norm_1")
        self.layernorm2 = LayerNormalization(epsilon=1e-6, name="layer_norm_2")
        self.dropout1 = Dropout(self.rate, name="dropout_1")
        self.dropout2 = Dropout(self.rate, name="dropout_2")
        super(TransformerBlockWithPosEncoding, self).build(input_shape)

    def call(self, inputs, training=False):
        x = self.pos_embed_layer(inputs)
        attn_output = self.mhatt(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        context_vector = tf.reduce_mean(out2, axis=1)
        return context_vector

    def get_config(self):
        config = super(TransformerBlockWithPosEncoding, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "max_seq_len": self.max_seq_len,
            "rate": self.rate,
        })
        return config


class CVAE(keras.Model):
    def __init__(self, encoder, decoder, window_size, kl_weight=1.0, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.window_size = window_size
        self.kl_weight = kl_weight

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=None, mask=None):
        dynamic_features, inp_data = inputs
        z_mean, z_log_var, z = self.encoder([dynamic_features, inp_data])
        pred = self.decoder([z, dynamic_features])
        return pred

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        dynamic_features, inp_data = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([dynamic_features, inp_data])
            pred = self.decoder([z, dynamic_features])

            reconstruction_loss = tf.reduce_mean(tf.square(inp_data - pred))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + self.kl_weight * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def get_CVAE(window_size, n_main_features, n_dyn_features, latent_dim, num_heads=8, ff_dim=256):
    # ENCODER: CNN-GRU (IMPROVED)
    inp = Input(shape=(window_size, n_main_features), name="input_main_CVAE")
    dyn_inp = Input(shape=(window_size, n_dyn_features), name="input_dyn_CVAE")

    enc = Concatenate(name="concat_encoder_inputs_CVAE")([dyn_inp, inp])

    # CNN Layer 1
    enc = Conv1D(64, 7, padding="same", activation="relu", kernel_initializer="glorot_uniform", name="conv1d_1")(enc)
    enc = BatchNormalization(name="batch_norm_1")(enc)
    enc = MaxPooling1D(2, name="maxpool_1")(enc)
    enc = Dropout(0.2, name="dropout_1")(enc)

    # CNN Layer 2
    enc = Conv1D(128, 5, padding="same", activation="relu", kernel_initializer="glorot_uniform", name="conv1d_2")(enc)
    enc = BatchNormalization(name="batch_norm_2")(enc)
    enc = MaxPooling1D(2, name="maxpool_2")(enc)
    enc = Dropout(0.2, name="dropout_2")(enc)

    # CNN Layer 3
    enc = Conv1D(256, 3, padding="same", activation="relu", kernel_initializer="glorot_uniform", name="conv1d_3")(enc)
    enc = BatchNormalization(name="batch_norm_3")(enc)
    enc = Dropout(0.3, name="dropout_3")(enc)

    # GRU Layer 1
    enc = GRU(128, return_sequences=True, dropout=0.2, kernel_regularizer=l2(0.001), kernel_initializer="glorot_uniform", name="gru_1")(enc)

    # GRU Layer 2
    enc = GRU(128, return_sequences=False, dropout=0.2, kernel_regularizer=l2(0.001), kernel_initializer="glorot_uniform", name="gru_2")(enc)

    enc = Dense(latent_dim, activation="relu", kernel_regularizer=l2(0.001), kernel_initializer="glorot_uniform", name="dense_latent_CVAE")(enc)
    enc = BatchNormalization(name="batch_norm_latent")(enc)

    z_mean = Dense(latent_dim, kernel_initializer="glorot_uniform", name="z_mean")(enc)
    z_log_var = Dense(latent_dim, kernel_initializer="glorot_uniform", name="z_log_var")(enc)
    z = Sampling(name="sampling_layer")([z_mean, z_log_var])

    encoder = Model([dyn_inp, inp], [z_mean, z_log_var, z], name="encoder_CVAE")

    # DECODER (original)
    inp_z = Input(shape=(latent_dim,), name="input_latent_CVAE")
    dec_dyn_inp = Input(shape=(window_size, n_dyn_features), name="input_dyn_decoder_CVAE")

    dec = RepeatVector(window_size, name="repeat_vector_CVAE")(inp_z)
    dec = Concatenate(name="concat_decoder_inputs_CVAE")([dec, dec_dyn_inp])

    dec = Bidirectional(
        LSTM(n_main_features, kernel_initializer="glorot_uniform",
             return_sequences=True, dropout=0.3, kernel_regularizer=l2(0.001), name="bidirectional_lstm_decoder_CVAE"),
        merge_mode="ave",
    )(dec)

    out = Flatten(name="flatten_decoder_output_CVAE")(dec)
    out = Dense(window_size * n_main_features, kernel_regularizer=l2(0.001), name="dense_output_CVAE")(out)
    out = Reshape((window_size, n_main_features), name="reshape_final_output_CVAE")(out)

    decoder = Model([inp_z, dec_dyn_inp], out, name="decoder_CVAE")

    return encoder, decoder
