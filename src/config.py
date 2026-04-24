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

TIME_RESOLUTION = 16          # steps per bar
BEATS_PER_BAR = 4
STEPS_PER_BEAT = TIME_RESOLUTION // BEATS_PER_BAR   # 4
STEP_IN_BEATS = 1.0 / STEPS_PER_BEAT

SEQUENCE_LENGTH = 512       # fixed-length token windows
WINDOW_STRIDE = 256            # overlapping windows
MIN_NOTES_PER_FILE = 20
MAX_FILES_DEBUG = 1000

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

MAX_TIME_SHIFT_STEPS = 128     # up to 4 bars at 16 steps/bar
MAX_DURATION_STEPS = 128
VELOCITY_BINS = [16, 32, 48, 64, 80, 96, 112, 127]

# -----------------------------
# Model config: VAE
# -----------------------------
EMBED_DIM = 64
HIDDEN_DIM = 128
LATENT_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.3

# -----------------------------
# Training config
# -----------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3

BETA_START = 0.02
BETA_END = 0.5
BETA_ANNEAL_EPOCHS = 10       # must be <= NUM_EPOCHS
WORD_DROPOUT = 0.10

NUM_WORKERS = 4
PIN_MEMORY = True
USE_AMP = True

DEVICE = "cuda"

# -----------------------------
# Generation config
# -----------------------------
NUM_GENERATED_SAMPLES = 8
SAMPLING_TEMPERATURE = 1.0
TOP_K = 16
MAX_GENERATION_LENGTH = 1536  # in tokens, corresponds to 4 bars at 16 steps/bar
DEFAULT_BPM = 120


# -----------------------------
# Common filenames
# -----------------------------
TOKENIZED_DATASET_PATH = PROCESSED_DIR / "tokenized_dataset_debug.json"
WINDOWED_DATASET_PATH = PROCESSED_DIR / "windowed_dataset_debug.json"
VOCAB_PATH = PROCESSED_DIR / "vocab_debug.json"

ENCODED_DATASET_PATH = SPLIT_DIR / "encoded_dataset_debug.json"
TRAIN_PATH = SPLIT_DIR / "train_debug.json"
VAL_PATH = SPLIT_DIR / "val_debug.json"
TEST_PATH = SPLIT_DIR / "test_debug.json"

BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "vae_debug_best.pt"
HISTORY_PATH = CHECKPOINT_DIR / "vae_debug_history.json"

# -----------------------------
# Model config: Transformer
# -----------------------------
TRANSFORMER_D_MODEL = 256
TRANSFORMER_NHEAD = 8
TRANSFORMER_NUM_LAYERS = 6
TRANSFORMER_FF_DIM = 1024
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_MAX_SEQ_LEN = 512

# -----------------------------
# Task 4: RLHF config
# -----------------------------
RLHF_DIR = OUTPUTS_DIR / "rlhf"
RLHF_SURVEY_MIDI_DIR = RLHF_DIR / "survey_midis"
RLHF_TOKEN_DIR = RLHF_DIR / "tokens"
RLHF_SCORE_CSV = RLHF_DIR / "human_scores.csv"

REWARD_MODEL_CHECKPOINT = CHECKPOINT_DIR / "reward_model_best.pt"
RLHF_TRANSFORMER_CHECKPOINT = CHECKPOINT_DIR / "transformer_rlhf_best.pt"

RLHF_NUM_SURVEY_SAMPLES = 20
RLHF_MAX_NEW_TOKENS = 512
RLHF_PRIMER_LENGTH = 16

REWARD_EMBED_DIM = 128
REWARD_HIDDEN_DIM = 256
REWARD_NUM_EPOCHS = 20
REWARD_BATCH_SIZE = 16
REWARD_LEARNING_RATE = 1e-3

RLHF_ITERATIONS = 200
RLHF_BATCH_SIZE = 4
RLHF_LEARNING_RATE = 1e-5
RLHF_KL_COEF = 0.01

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