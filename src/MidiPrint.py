import os
import glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from midi_tools import midi_to_note_state_matrix, note_state_matrix_to_midi

# —————————————————————————————————————————————
# CONFIGURATION
# —————————————————————————————————————————————
LOWER_BOUND   = 24
UPPER_BOUND   = 102
NOTE_RANGE    = UPPER_BOUND - LOWER_BOUND
NUM_TIMESTEPS = 120
N_VISIBLE     = 2 * NOTE_RANGE * NUM_TIMESTEPS
N_HIDDEN      = 50

BATCH_SIZE    = 100
EPOCHS        = 300
LEARNING_RATE = 0.005
NUM_SAMPLES   = 50  # how many rollout examples to generate

# Seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# —————————————————————————————————————————————
# GENRE & MOOD SETUP (for reporting)
# —————————————————————————————————————————————
import GetGM as gm
import Report as r

genres = gm.Genres()
moods  = gm.Moods()
r.updateData(genres, moods)

# —————————————————————————————————————————————
# DATA LOADING
# —————————————————————————————————————————————

def get_songs(path: str) -> list[np.ndarray]:
    """Load MIDI files, convert to note‐state matrices, filter short sequences."""
    files = glob.glob(os.path.join(path, '*.mid*'))
    songs = []
    for f in tqdm(files, desc="Loading MIDI"):
        mat = np.array(midi_to_note_state_matrix(f, LOWER_BOUND, UPPER_BOUND))
        if mat.shape[0] > NUM_TIMESTEPS:
            songs.append(mat)
    return songs


def songs_to_windows(songs: list[np.ndarray]) -> np.ndarray:
    """
    Break each song into non-overlapping windows of length NUM_TIMESTEPS,
    flatten to vectors of size N_VISIBLE.
    """
    windows = []
    for mat in songs:
        total_steps = (mat.shape[0] // NUM_TIMESTEPS) * NUM_TIMESTEPS
        trimmed = mat[:total_steps]
        reshaped = trimmed.reshape(-1, NOTE_RANGE * 2 * NUM_TIMESTEPS)
        windows.append(reshaped)
    return np.vstack(windows)

# —————————————————————————————————————————————
# RBM MODEL
# —————————————————————————————————————————————

class RBM(tf.Module):
    def __init__(self, n_visible: int, n_hidden: int, lr: float):
        super().__init__()
        self.W  = tf.Variable(tf.random.normal([n_visible, n_hidden], stddev=0.01), name="W")
        self.bv = tf.Variable(tf.zeros([n_visible]), name="bv")
        self.bh = tf.Variable(tf.zeros([n_hidden]), name="bh")
        self.lr = lr

    @staticmethod
    def _sample(probs: tf.Tensor) -> tf.Tensor:
        return tf.floor(probs + tf.random.uniform(tf.shape(probs)))

    @tf.function
    def gibbs_sample(self, v0: tf.Tensor, k: int = 1) -> tf.Tensor:
        v = v0
        for _ in tf.range(k):
            h = RBM._sample(tf.sigmoid(tf.matmul(v, self.W) + self.bh))
            v = RBM._sample(tf.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.bv))
        return tf.stop_gradient(v)

    @tf.function
    def train_step(self, v0: tf.Tensor) -> tf.Tensor:
        h0 = RBM._sample(tf.sigmoid(tf.matmul(v0, self.W) + self.bh))
        vk = self.gibbs_sample(v0, k=1)
        hk = RBM._sample(tf.sigmoid(tf.matmul(vk, self.W) + self.bh))

        batch_size = tf.cast(tf.shape(v0)[0], tf.float32)
        dW  = (tf.matmul(tf.transpose(v0), h0) - tf.matmul(tf.transpose(vk), hk)) * (self.lr / batch_size)
        dbv = tf.reduce_mean(v0 - vk, axis=0) * self.lr
        dbh = tf.reduce_mean(h0 - hk, axis=0) * self.lr

        self.W.assign_add(dW)
        self.bv.assign_add(dbv)
        self.bh.assign_add(dbh)

        return tf.reduce_mean(tf.square(v0 - vk))

# —————————————————————————————————————————————
# TRAIN & GENERATE
# —————————————————————————————————————————————

def main():
    songs = get_songs('Rock_Music_Midi')
    print(f"{len(songs)} songs processed")
    windows = songs_to_windows(songs)

    dataset = (
        tf.data.Dataset.from_tensor_slices(windows.astype(np.float32))
        .shuffle(buffer_size=10 * BATCH_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    rbm = RBM(n_visible=N_VISIBLE, n_hidden=N_HIDDEN, lr=LEARNING_RATE)

    loss_history = []
    for epoch in range(1, EPOCHS + 1):
        epoch_loss, batches = 0.0, 0
        for batch in dataset:
            loss = rbm.train_step(batch)
            epoch_loss += loss
            batches += 1
        avg_loss = epoch_loss / tf.cast(batches, tf.float32)
        loss_history.append(avg_loss.numpy())
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{EPOCHS}, Recon Loss: {avg_loss:.6f}")

    zeros = tf.zeros([NUM_SAMPLES, N_VISIBLE], dtype=tf.float32)
    samples = rbm.gibbs_sample(zeros, k=1).numpy()

    for i, vec in enumerate(samples):
        if not np.any(vec):
            continue
        S = vec.reshape(NUM_TIMESTEPS, NOTE_RANGE * 2)
        note_state_matrix_to_midi(S, name=f"generated_melody_{i}", lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND)

    for _ in tqdm(range(len(samples)), desc="Generating music"):
        pass

    plt.plot(loss_history)
    plt.title("RBM Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()

if __name__ == "__main__":
    main()
