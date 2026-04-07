from pathlib import Path

# -----------------------------
# Project root and paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_MIDI_DIR = DATA_DIR / "raw_midi"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "train_test_split"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

SRC_DIR = PROJECT_ROOT / "src"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GENERATED_MIDI_DIR = OUTPUTS_DIR / "generated_midis"
PLOTS_DIR = OUTPUTS_DIR / "plots"
SURVEY_RESULTS_DIR = OUTPUTS_DIR / "survey_results"

REPORT_DIR = PROJECT_ROOT / "report"
ARCHITECTURE_DIAGRAMS_DIR = REPORT_DIR / "architecture_diagrams"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# -----------------------------
# Dataset / preprocessing
# -----------------------------
DATASET_NAME = "lakh_midi"

TIME_RESOLUTION = 16          # steps per bar or quantization unit
SEQUENCE_LENGTH = 128         # fixed-length token windows
MIN_NOTES_PER_FILE = 10
MAX_FILES_DEBUG = 200         # useful for early testing on a small subset

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42

# -----------------------------
# Tokenization
# -----------------------------
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# -----------------------------
# Model config: VAE
# -----------------------------
EMBED_DIM = 128
HIDDEN_DIM = 256
LATENT_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.2

# -----------------------------
# Training config
# -----------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3

BETA_START = 0.0
BETA_END = 0.1
BETA_ANNEAL_EPOCHS = 10

DEVICE = "cuda"  # fallback to cpu in code if unavailable

# -----------------------------
# Generation config
# -----------------------------
NUM_GENERATED_SAMPLES = 8
SAMPLING_TEMPERATURE = 1.0
MAX_GENERATION_LENGTH = 128


def ensure_directories() -> None:
    """
    Create all required project directories if they do not already exist.
    Safe to call at startup.
    """
    directories = [
        DATA_DIR,
        RAW_MIDI_DIR,
        PROCESSED_DIR,
        SPLIT_DIR,
        OUTPUTS_DIR,
        GENERATED_MIDI_DIR,
        PLOTS_DIR,
        SURVEY_RESULTS_DIR,
        REPORT_DIR,
        ARCHITECTURE_DIAGRAMS_DIR,
        CHECKPOINT_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("Project directories are ready.")