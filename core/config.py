from pathlib import Path

BASE_DIR        = Path(__file__).resolve().parent.parent

DATA_DIR        = BASE_DIR / "data"
GENUINE_DIR     = DATA_DIR / "genuine"
TAMPERED_DIR    = DATA_DIR / "tampered"
MASKS_DIR       = DATA_DIR / "masks"
CHECKPOINTS_DIR = BASE_DIR / "model" / "checkpoints"

MAX_IMAGE_SIDE = 2000
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".pdf"}
MAX_FILE_MB    = 20

ELA_QUALITY   = 95
ELA_AMPLIFY   = 15
ELA_THRESHOLD = 0.12

NOISE_WINDOW    = 16
NOISE_THRESHOLD = 0.15

CM_BLOCK_SIZE   = 16
CM_STRIDE       = 8
CM_MATCH_THRESH = 0.95

DCT_HIST_BINS   = 64

FONT_BASELINE_TOL = 3

FUSION_WEIGHTS = {
    'ela':          0.13,
    'noise':        0.13,
    'copy_move':    0.13,
    'double_jpeg':  0.08,
    'font':         0.13,
    'metadata':     0.08,
    'ai_generated': 0.12,
    'model':        0.20,
}

TAMPER_THRESHOLD = 0.50

MODEL_INPUT_SIZE    = 256
BATCH_SIZE          = 16
LEARNING_RATE       = 1e-4
MAX_EPOCHS          = 50
EARLY_STOP_PATIENCE = 7