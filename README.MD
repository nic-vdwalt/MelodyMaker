# Melody Maker

**Melody Maker** is a machine-learning driven music generation toolkit developed by Waltworks. It leverages RNN models to learn patterns from MIDI datasets and generate new melodies and chord progressions.

---

## Table of Contents

1. [Features](#features)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Configuration](#configuration)  
5. [Usage](#usage)  
   - [Training](#training)  
   - [Generation](#generation)  
6. [Project Structure](#project-structure)  
7. [Contributing](#contributing)  
8. [License](#license)  

---

## Features

- **Data preprocessing**: Convert MIDI files to note-state matrices.  
- **Model training**: Train melody and chord RNN models on your MIDI dataset.  
- **Music generation**: Produce new MIDI files from trained models.  
- **Configurable**: Easily adjust sequence length, model hyperparameters, and data directories.
- **Native training studio**: Configure, launch, stop, and monitor training in an Ingot desktop client with live dataset and loss progress.

---

## Prerequisites

- Python 3.8+  
- `make` (GNU Make)  
- Virtual environment tool (e.g., `venv`, `conda`)  

### Python Libraries

```bash
numpy
mido
tensorflow>=2.6
scikit-learn
tqdm
```

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-org/melody-maker.git
   cd melody-maker
   ```

2. **Create and activate a virtual environment**  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

All configurable parameters live in `src/config.py`. Key settings include:

| Variable             | Description                              | Default            |
|----------------------|------------------------------------------|--------------------|
| `MIDI_DIR`           | Path to your raw MIDI training files     | `Rock_Music_Midi`  |
| `LOWER_BOUND`        | Lowest MIDI pitch to include             | `24`               |
| `UPPER_BOUND`        | Highest MIDI pitch to include            | `102`              |
| `SEQ_LEN`            | Sequence length (timesteps)              | `60`               |
| `BATCH_SIZE`         | Training batch size                      | `64`               |
| `EPOCHS`             | Number of training epochs                | `20`               |
| `MELODY_MODEL_PATH`  | Output path for melody model checkpoint  | `melody_model.h5`  |
| `CHORD_MODEL_PATH`   | Output path for chord model checkpoint   | `chord_model.h5`   |

---

## Usage

### Training

> **Note:** You only need to run training once per dataset.

Command line:

```bash
make train
```

Native Ingot training studio:

```bash
make client
./client-odin/dist/melody-trainer
```

The client expects Ingot at `../ingot` by default; set `INGOT=/path/to/ingot` to override it. It stores local form settings, launches `src/train.py` without a shell, and displays live scan, epoch, loss, and validation-loss progress. TensorFlow must be installed in `.venv` before a run can start.

Training recursively discovers `.mid` and `.midi` files, skips macOS resource-fork files, streams windows through `tf.data`, and saves the event model and seed to the configured paths.

### Generation

You can regenerate music at any time once models are trained:

```bash
make generate
```

This will:

1. Load the trained models.  
2. Generate a new melody and chord sequence.  
3. Export a MIDI file named `output_generated.mid`.

---

## Project Structure

```
.
├── Makefile              # Training, generation, and client targets
├── client-odin           # Native Ingot training studio
├── requirements.txt      # Python dependencies
├── README.md             # This document
└── src
    ├── config.py         # Configuration parameters
    ├── data_utils.py     # Encoders and scalers
    ├── midi_tools.py     # MIDI ↔ state-matrix conversion
    ├── rnn.py            # Model definitions & generation logic
    └── generate.py       # Entry point for training & generation
```

---

## Contributing

We welcome contributions! Please:

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes (`git commit -m "Add your feature"`)  
4. Push to the branch (`git push origin feature/YourFeature`)  
5. Open a Pull Request  

Please adhere to the existing code style and include tests for new functionality.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
