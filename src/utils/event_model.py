import tensorflow as tf
from tensorflow.keras import layers, Model


def build_event_model(
    vocab_size,
    emb_dim=256,
    num_heads=8,
    ff_dim=512,
    num_layers=4,
    max_pos_emb=10000
):
    """
    Transformer-based decoder for event-token prediction.
    """
    inputs = layers.Input(shape=(None,), dtype='int32', name='event_seq')
    # token embedding + positional embedding
    tok_emb = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, mask_zero=True)(inputs)
    positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
    pos_emb = layers.Embedding(input_dim=max_pos_emb, output_dim=emb_dim)(positions)
    x = tok_emb + pos_emb

    for _ in range(num_layers):
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=emb_dim)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
        ff = layers.Dense(ff_dim, activation='relu')(x)
        ff = layers.Dense(emb_dim)(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    logits = layers.Dense(vocab_size, name='event_logits')(x)
    return Model(inputs=inputs, outputs=logits, name='event_model')