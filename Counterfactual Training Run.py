
"""# SETUP AND CONFIGURATION"""

# Cell 1.1: Base Imports
# =====================================
# Base Library Imports
# =====================================

import os
import sys
import warnings
import logging
import traceback
from typing import Dict, List, Tuple, Optional

# Data processing
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Ensure imblearn is installed: pip install imbalanced-learn
try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    print("imblearn not found. Please install it: pip install imbalanced-learn")
    # You might want to raise an error or exit if it's critical
    # raise ImportError("Please install imbalanced-learn: pip install imbalanced-learn")

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR, LinearLR, SequentialLR # Added more schedulers
# Ensure transformers is installed: pip install transformers
try:
    from transformers import RobertaTokenizer, RobertaModel
except ImportError:
    print("transformers not found. Please install it: pip install transformers")
    # raise ImportError("Please install transformers: pip install transformers")

# Metrics and Evaluation
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, precision_recall_curve, roc_auc_score # Added roc_auc_score
)

# Visualization
# Ensure matplotlib, seaborn, plotly are installed: pip install matplotlib seaborn plotly
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    print("Visualization libraries (matplotlib, seaborn, plotly) not found. Please install them.")

# Progress Tracking
# Ensure tqdm is installed: pip install tqdm ipywidgets
try:
    from tqdm.notebook import tqdm
    # Ensure ipywidgets is enabled for tqdm in notebooks
    # In Jupyter Lab: jupyter labextension install @jupyter-widgets/jupyterlab-manager
    # In Classic Notebook: jupyter nbextension enable --py widgetsnbextension
except ImportError:
    print("tqdm not found. Please install it: pip install tqdm ipywidgets")

# Ensure IPython is available for display
try:
    from IPython.display import clear_output, display
except ImportError:
    print("IPython not found. Display features might not work.")


# --- Added Print Statement ---
print("-" * 50)
print("Cell 1.1: Base Imports completed successfully!")
print("-" * 50)

# Cell 1.2: Configuration and Environment Setup
# =====================================
# System Configuration and Environment Setup
# =====================================

# <<< --- Explicit Dataset Selection --- >>>
# Choose the dataset to run: "MELD" or "IEMOCAP"
# This setting will determine which paths and potentially model parameters are used.
SELECTED_DATASET = "IEMOCAP" # <<< --- CHANGE THIS LINE TO "MELD" or "IEMOCAP" --- >>>
# <<< ---------------------------------- >>>


class Config:
    def __init__(self):
        print("Initializing Configuration...")
        # --- Path Settings ---
        # Ensure these paths are exactly correct in your Google Drive
        self.IEMOCAP_EXCEL_PATH = "path file"
        self.IEMOCAP_RESULT_DIR = "path file"

        self.MELD_EXCEL_PATH = "path file" # Corrected path as per user request
        self.MELD_RESULT_DIR = "path file" # Corrected path as per user request

        # --- Dynamic Settings (will be set by set_dataset) ---
        self.CURRENT_DATASET = None # Initialize as None
        self.EXCEL_PATH = None
        self.RESULT_DIR = None
        self.NUM_EMOTIONS = None # Will be determined later from data

        # --- System Settings ---
        self.SEED = 42
        self.EARLY_STOP = "off"  # Options: "on", "off". Controls early stopping based on validation loss.
        self.DEBUG_MODE = False  # Set to True for more verbose debugging prints (if needed later)

        # --- Model Architecture Parameters ---
        self.ROBERTA_MODEL_NAME = 'roberta-base'
        self.WAV2VEC2_MODEL_NAME = 'facebook/wav2vec2-base-960h' # Used for V1 pre-extraction
        self.TEXT_OUTPUT_DIM = 768 # RoBERTa base output
        self.AUDIO_V1_DIM = 512 # Wav2Vec2 output (used in AudioProcessor)
        self.AUDIO_V3_DIM = 25 # MFCC
        self.AUDIO_V4_DIM = 25 # Spectral
        self.VISUAL_A2_DIM = 512 # Output of ResNet+LSTM (used in VisualProcessor)

        self.TEXT_PROJECTION_DIM = 512 # Dimension after projecting RoBERTa output
        self.AUDIO_PROJECTION_DIM = 256 # Dimension for fused audio features before BiLSTM
        self.VISUAL_PROJECTION_DIM = 256 # Dimension for visual features before BiLSTM

        self.PROGRESSIVE_BILSTM_HIDDEN_DIM = 256 # Hidden dim per direction for Progressive BiLSTM
        self.CROSS_ATTENTION_HEADS = 8 # Number of heads for cross-modal attention
        self.STANDARDIZED_FEATURE_DIM = 256 # Dimension after cross-attention and standardization
        self.FEATURE_STACK_MLP_DIMS = [768, 1024, 512, 256] # Input/hidden/output dims for final MLP

        self.DROPOUT_RATE = 0.3 # General dropout rate
        self.FEATURE_STACK_DROPOUT = 0.2 # Dropout in the final MLP stack

        # --- Text Processing ---
        self.MAX_LEN = 128 # Max sequence length for RoBERTa tokenizer

        # --- Training Parameters ---
        self.BATCH_SIZE = 16 # Adjust based on T4 memory (16 is a reasonable start)
        self.ACCUMULATION_STEPS = 4  # Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS (64)
        self.LEARNING_RATE = 5e-5 # Initial learning rate for AdamW
        self.WEIGHT_DECAY = 0.01 # Weight decay for AdamW
        self.ADAM_EPSILON = 1e-8 # Epsilon for AdamW
        self.MAX_GRAD_NORM = 1.0 # Gradient clipping threshold

        # Phase-specific Epochs (Reduced for faster testing, adjust as needed)
        self.PHASE1_EPOCHS = 
        self.PHASE2_EPOCHS = 
        self.PHASE3_EPOCHS = 
        self.EARLY_STOPPING_PATIENCE = 7 # Patience for early stopping (if enabled)

        # --- Schedulers & Warmup ---
        self.WARMUP_EPOCHS = { 1: 1, 2: 1, 3: 1 } # Reduced warmup epochs for faster testing
        # self.WARMUP_EPOCHS = { 1: 3, 2: 4, 3: 5 } # Original warmup epochs
        self.LR_SCHEDULE_TYPE = { 1: 'plateau', 2: 'cosine', 3: 'one_cycle' } # Scheduler type per phase
        self.WARMUP_FACTOR = 0.1 # Start LR = LEARNING_RATE * WARMUP_FACTOR
        self.MIN_LR = 1e-7 # Minimum learning rate
        self.LR_SCHEDULER_FACTOR = 0.5 # Factor for ReduceLROnPlateau
        self.LR_SCHEDULER_PATIENCE = 2 # Patience for ReduceLROnPlateau
        self.COSINE_T_0 = 5 # T_0 for CosineAnnealingWarmRestarts
        self.COSINE_T_MULT = 2 # T_mult for CosineAnnealingWarmRestarts
        self.ONE_CYCLE_PCT_START = 0.3 # pct_start for OneCycleLR

        # --- Loss Function Parameters ---
        self.FOCAL_GAMMA = 2.0
        self.ALIGNMENT_TEMP = 0.1
        self.ALIGNMENT_MARGIN = 0.5
        self.INTENTION_CONFIDENCE_THRESHOLD = 0.7
        self.LABEL_SMOOTHING = 0.1 # Applied in CrossEntropyLoss for intention

        # Loss Weights (Scheduled)
        self.INITIAL_WEIGHTS = {'focal': 1.0, 'alignment': 0.0, 'intention': 0.0}
        self.PHASE2_ALIGNMENT_WEIGHT_SCHEDULE = lambda epoch: min(0.1, 0.02 * (epoch + 1)) # epoch is 0-indexed
        self.PHASE3_ALIGNMENT_WEIGHT = 0.1 # Fixed in Phase 3
        self.PHASE3_INTENTION_WEIGHT_SCHEDULE = lambda epoch: min(0.05, 0.01 * (epoch + 1)) # epoch is 0-indexed

        # --- System Optimization ---
        self.USE_AMP = True # Use Automatic Mixed Precision (FP16) - Good for T4
        self.NUM_WORKERS = 2 # Number of workers for DataLoader (adjust based on Colab performance)
        self.PIN_MEMORY = True # Pin memory for faster data transfer to GPU

        print("Config Initialized.")

    def set_dataset(self, dataset_name: str):
        """Set the current dataset and update paths"""
        print(f"\nAttempting to set dataset to: {dataset_name}")
        if dataset_name == "MELD":
            self.CURRENT_DATASET = "MELD"
            self.EXCEL_PATH = self.MELD_EXCEL_PATH
            self.RESULT_DIR = self.MELD_RESULT_DIR
            print(f"Dataset successfully set to: MELD")
        elif dataset_name == "IEMOCAP":
            self.CURRENT_DATASET = "IEMOCAP"
            self.EXCEL_PATH = self.IEMOCAP_EXCEL_PATH
            self.RESULT_DIR = self.IEMOCAP_RESULT_DIR
            print(f"Dataset successfully set to: IEMOCAP")
        else:
            print(f"Error: Unknown dataset '{dataset_name}'. Please use 'MELD' or 'IEMOCAP'.")
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"  - Excel Path set to: {self.EXCEL_PATH}")
        print(f"  - Result Dir set to: {self.RESULT_DIR}")
        # Ensure result directory exists
        os.makedirs(self.RESULT_DIR, exist_ok=True)
        print(f"  - Ensured Result Dir exists: {self.RESULT_DIR}")

def setup_environment(config: Config) -> torch.device:
    """Initialize environment with given configuration"""
    print("\nSetting up Environment...")

    # Set random seeds for reproducibility
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random_state_set = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Set deterministic algorithms (can impact performance, use cautiously)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False # Turn off benchmark if deterministic
        torch.backends.cudnn.benchmark = True # Usually better for speed if input sizes are consistent
        random_state_set = True

    print(f"- Random seeds set (Seed: {seed}). Reproducibility state set: {random_state_set}")

    # CUDA settings for optimization
    if torch.cuda.is_available():
        # Set environment variables for CUDA (optional, can sometimes help debugging)
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Use for debugging CUDA errors
        # os.environ['TORCH_USE_CUDA_DSA'] = '1' # Might be default in newer PyTorch
        pass

    # Suppress warnings (use cautiously, might hide important info)
    warnings.filterwarnings('ignore', category=UserWarning) # Suppress common UserWarnings
    print("- Warnings suppressed (UserWarning category).")

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"- Device selected: {device}")

    print("\nEnvironment Setup Summary:")
    print(f"- PyTorch Version: {torch.__version__}")
    print(f"- CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"- GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"- CuDNN Enabled: {torch.backends.cudnn.enabled}")
        print(f"- CuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        # print(f"- CuDNN Deterministic: {torch.backends.cudnn.deterministic}") # Only if set above
    print(f"- Seed: {config.SEED}")
    print(f"- Early Stopping: {config.EARLY_STOP.upper()}")
    print(f"- Use AMP (FP16): {config.USE_AMP}")
    print(f"- Current Dataset Target: {config.CURRENT_DATASET}") # Shows the target set by set_dataset
    print("-" * 50)

    return device

# --- Execution ---
print("--- Starting Cell 1.2 Execution ---")

# 1. Initialize configuration object
config = Config()

# 2. Set the dataset based on the selection at the top
config.set_dataset(SELECTED_DATASET)

# 3. Setup the environment (device, seeds, etc.)
device = setup_environment(config)

print("--- Finished Cell 1.2 Execution ---")
print("-" * 50)

# Cell 1.3: GPU Setup and Memory Optimization
# =====================================
# GPU Memory Management and Optimization
# =====================================

def optimize_gpu_memory():
    """Check GPU status and clear cache."""
    print("\n--- Starting Cell 1.3: GPU Memory Check ---")
    if torch.cuda.is_available():
        try:
            gpu_index = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_index)
            print(f"GPU Device: {gpu_name} (Index: {gpu_index})")

            # Empty CUDA cache before starting
            torch.cuda.empty_cache()
            print("- Emptied CUDA cache.")

            # Optional: Set memory allocation settings - Often not needed, PyTorch handles it well.
            # Can sometimes cause issues if set too high. Let's disable it for now.
            # torch.cuda.set_per_process_memory_fraction(0.95) # Use 95% of available memory
            # print("- Attempted to set memory fraction (Note: This is often unnecessary).")

            # Ensure benchmark is set (redundant if set in 1.2, but safe)
            if not torch.backends.cudnn.benchmark:
                 torch.backends.cudnn.benchmark = True
                 print("- Enabled CuDNN benchmark mode for potential speedup.")
            else:
                 print("- CuDNN benchmark mode already enabled.")

            # Get current memory information AFTER clearing cache
            total_memory = torch.cuda.get_device_properties(gpu_index).total_memory
            allocated_memory = torch.cuda.memory_allocated(gpu_index)
            reserved_memory = torch.cuda.memory_reserved(gpu_index) # PyTorch reserves memory pool
            free_memory_inside_reserved = reserved_memory - allocated_memory
            # More accurate 'free' from nvidia-smi perspective (approximate)
            # Note: torch.cuda.mem_get_info() gives (free, total) directly in bytes
            free_global_memory, total_global_memory = torch.cuda.mem_get_info(gpu_index)


            print("\nGPU Memory Status:")
            print(f"- Total Global Memory: {total_global_memory / 1e9:.2f} GB")
            print(f"- Free Global Memory:  {free_global_memory / 1e9:.2f} GB")
            print(f"- PyTorch Allocated:   {allocated_memory / 1e9:.2f} GB")
            print(f"- PyTorch Reserved:    {reserved_memory / 1e9:.2f} GB")
            print(f"- Free within Reserved:{free_memory_inside_reserved / 1e9:.2f} GB")
            print("-" * 50)

            return True
        except Exception as e:
            print(f"Error during GPU memory optimization/check: {str(e)}")
            print(traceback.format_exc())
            print("-" * 50)
            return False
    else:
        print("No GPU available. Running on CPU.")
        print("-" * 50)
        return False

# --- Execution ---
# Check/Optimize GPU memory settings
gpu_ready = optimize_gpu_memory()
print(f"GPU Ready Status: {gpu_ready}")
print("--- Finished Cell 1.3 Execution ---")
print("-" * 50)

# Cell 1.4: Helper Functions and Logging Setup
# =====================================
# Logging Configuration and Utility Functions
# =====================================

import logging
import os
import sys
import time # Import time for potential timestamping if needed
from tqdm.notebook import tqdm
from IPython.display import display, clear_output
import torch # Added torch import for tensor checking
import json # Added for saving history
import numpy as np # Import numpy
import pandas as pd # Import pandas

# --- Logging Setup ---
# [Keep the setup_logging function as it was in the previous correct version]
def setup_logging(result_dir: str, debug_mode: bool = False) -> logging.Logger:
    """Configure logging for both file and console output."""
    print(f"\n--- Setting up Logging ---")
    print(f"- Target Result Directory: {result_dir}")

    # Create result directory if it doesn't exist
    try:
        os.makedirs(result_dir, exist_ok=True)
        print(f"- Ensured result directory exists.")
    except OSError as e:
        print(f"Error creating directory {result_dir}: {e}")
        # Fallback to a default log file name if directory creation fails
        log_file = 'training_fallback.log'
    else:
        log_file = os.path.join(result_dir, 'training.log')

    # Configure logging format
    log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Create logger instance
    logger = logging.getLogger('EmotionRecognitionLogger') # Use a specific name

    # Set logging level
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger.setLevel(log_level)
    print(f"- Logger level set to: {logging.getLevelName(logger.level)}")

    # Avoid adding multiple handlers if re-running the cell
    if logger.hasHandlers():
        print("- Logger already has handlers. Clearing existing ones.")
        logger.handlers.clear()

    # File handler
    try:
        file_handler = logging.FileHandler(log_file, mode='a') # Append mode
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        print(f"- File handler configured for: {log_file}")
    except Exception as e:
        print(f"Error setting up file handler for {log_file}: {e}")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout) # Explicitly use stdout
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    print("- Console handler configured.")

    # Prevent propagation to root logger (optional, but good practice)
    logger.propagate = False

    print("--- Logging Setup Complete ---")
    return logger


# --- Utility to Log and Print ---
# [Keep the log_and_print function as it was]
def log_and_print(logger_instance: logging.Logger, message: str, level: int = logging.INFO):
    """Utility function to log message and print to console."""
    if logger_instance:
        logger_instance.log(level, message)
    # Always print to console regardless of logger level for notebook visibility
    print(message)


# --- Training Progress Bar Class ---
# [Keep the TrainingProgress class as it was]
class TrainingProgress:
    """Utility class for tracking and displaying training progress via tqdm."""
    def __init__(self, total_epochs: int, total_batches_train: int, total_batches_val: int):
        self.progress_bar_train = None
        self.progress_bar_val = None
        self.total_epochs = total_epochs
        self.total_batches_train = total_batches_train
        self.total_batches_val = total_batches_val
        self.current_metrics = {}
        print(f"TrainingProgress initialized: Epochs={total_epochs}, TrainBatches={total_batches_train}, ValBatches={total_batches_val}") # Added print

    def init_epoch(self, epoch: int, phase_num: int, mode: str = 'train'):
        """Initialize progress bar for a new epoch (train or val)."""
        total_batches = self.total_batches_train if mode == 'train' else self.total_batches_val
        desc = f'Phase {phase_num} - Epoch {epoch}/{self.total_epochs} [{mode.upper()}]'
        # print(f"Initializing progress bar: {desc}") # Can be noisy
        if mode == 'train':
            self.progress_bar_train = tqdm(
                total=total_batches,
                desc=desc,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                leave=False # Keep the bar until closed
            )
        else:
            self.progress_bar_val = tqdm(
                total=total_batches,
                desc=desc,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                leave=False
            )

    def update(self, batch_metrics: Dict[str, float], mode: str = 'train'):
        """Update progress bar with current batch metrics."""
        # Format metrics for display
        formatted_metrics = {k: f"{v:.4f}" for k, v in batch_metrics.items()}
        self.current_metrics.update(formatted_metrics)

        if mode == 'train' and self.progress_bar_train:
            self.progress_bar_train.set_postfix(self.current_metrics)
            self.progress_bar_train.update(1)
        elif mode == 'val' and self.progress_bar_val:
            self.progress_bar_val.set_postfix(self.current_metrics)
            self.progress_bar_val.update(1)

    def close_epoch(self, mode: str = 'train'):
        """Close progress bar for the completed epoch."""
        # print(f"Closing progress bar for epoch [{mode.upper()}].") # Can be noisy
        if mode == 'train' and self.progress_bar_train:
            self.progress_bar_train.close()
            self.progress_bar_train = None # Reset for next epoch
        elif mode == 'val' and self.progress_bar_val:
            self.progress_bar_val.close()
            self.progress_bar_val = None # Reset for next epoch
        # Display final epoch metrics after closing the bar
        metrics_str = " | ".join([f"{k}: {v}" for k, v in self.current_metrics.items()])
        # print(f"Epoch [{mode.upper()}] completed. Final Batch Metrics: {metrics_str}") # Can be noisy
        self.current_metrics = {} # Reset metrics for next epoch


# --- Metrics Tracking Class ---
class MetricsTracker:
    """Track and store training/validation metrics across epochs and phases."""
    def __init__(self):
        # Store metrics per phase and mode (train/val)
        self.history = {} # E.g., history['phase1']['train']['loss'] = [...]
        self.best_val_metrics = {
            'loss': float('inf'),
            'accuracy': 0.0,
            'f1_weighted': 0.0, # Use weighted F1 for best model tracking
            'epoch': -1,
            'phase': -1
        }
        # Define expected metrics to initialize lists
        self.metric_keys = [
            'loss', 'accuracy',
            'precision_macro', 'precision_weighted', 'precision_micro',
            'recall_macro', 'recall_weighted', 'recall_micro',
            'f1_macro', 'f1_weighted', 'f1_micro'
        ]
        print("MetricsTracker initialized.")
        print(f" - Will track metrics: {self.metric_keys}")
        print(f" - Best validation metric tracked: f1_weighted")

    def _ensure_phase_mode_exists(self, phase: int, mode: str):
        """Initialize dictionary structure if it doesn't exist."""
        phase_key = f'phase{phase}'
        if phase_key not in self.history:
            self.history[phase_key] = {'train': {}, 'val': {}}
            print(f"MetricsTracker: Initialized structure for {phase_key}")
        if mode not in self.history[phase_key]:
             self.history[phase_key][mode] = {key: [] for key in self.metric_keys}
             print(f"MetricsTracker: Initialized metric lists for {phase_key} -> {mode}")
        # Ensure all metric keys exist even if added later
        for key in self.metric_keys:
            if key not in self.history[phase_key][mode]:
                self.history[phase_key][mode][key] = []
                print(f"MetricsTracker: Added missing metric list for {phase_key} -> {mode} -> {key}")


    def update_epoch_metrics(self, phase: int, epoch: int, mode: str, metrics: Dict[str, float]):
        """Update metrics for a completed epoch."""
        self._ensure_phase_mode_exists(phase, mode)
        phase_key = f'phase{phase}'

        print(f"MetricsTracker: Updating metrics for Phase {phase}, Epoch {epoch}, Mode {mode.upper()}")
        # Print received metrics for debugging
        formatted_metrics = {k: f"{v:.4f}" for k, v in metrics.items()}
        print(f"  - Received metrics: {formatted_metrics}")


        for key, value in metrics.items():
            if key in self.history[phase_key][mode]:
                # Ensure we don't add duplicate entries if called multiple times for same epoch
                if len(self.history[phase_key][mode][key]) == epoch:
                     self.history[phase_key][mode][key].append(value)
                     # print(f"    - Appended {key}: {value:.4f}") # Verbose
                elif len(self.history[phase_key][mode][key]) > epoch:
                     self.history[phase_key][mode][key][epoch] = value # Overwrite if needed
                     # print(f"    - Overwrote {key} at epoch {epoch}: {value:.4f}") # Verbose
                else:
                     # Handle case where epochs are skipped? Fill with NaN or raise error?
                     print(f"    - Warning: Attempting to update metrics for future epoch {epoch}. Current length for {key}: {len(self.history[phase_key][mode][key])}. Appending.")
                     # For now, append, assuming epochs are sequential
                     self.history[phase_key][mode][key].append(value)
            else:
                 print(f"    - Warning: Received metric '{key}' not in predefined metric_keys. Ignoring.")

        # Update best validation metrics based on f1_weighted
        if mode == 'val':
            is_best = False
            current_f1_weighted = metrics.get('f1_weighted', 0.0)
            current_loss = metrics.get('loss', float('inf'))

            # Check if current validation F1 (weighted) is the best so far
            if current_f1_weighted > self.best_val_metrics['f1_weighted']:
                print(f"  - New best validation F1_WEIGHTED found: {current_f1_weighted:.4f} (Previous: {self.best_val_metrics['f1_weighted']:.4f}) at Phase {phase}, Epoch {epoch}")
                self.best_val_metrics['loss'] = current_loss
                self.best_val_metrics['accuracy'] = metrics.get('accuracy', 0.0)
                self.best_val_metrics['f1_weighted'] = current_f1_weighted
                self.best_val_metrics['epoch'] = epoch
                self.best_val_metrics['phase'] = phase
                is_best = True
            # Tie-breaking using validation loss if F1 is equal
            elif current_f1_weighted == self.best_val_metrics['f1_weighted'] and current_loss < self.best_val_metrics['loss']:
                 print(f"  - New best validation Loss found (with equal F1_WEIGHTED): {current_loss:.4f} (Previous: {self.best_val_metrics['loss']:.4f}) at Phase {phase}, Epoch {epoch}")
                 self.best_val_metrics['loss'] = current_loss
                 self.best_val_metrics['accuracy'] = metrics.get('accuracy', 0.0)
                 # F1 is already equal
                 self.best_val_metrics['epoch'] = epoch
                 self.best_val_metrics['phase'] = phase
                 is_best = True

            print(f"  - Is best epoch so far? {is_best}")
            return is_best # Return flag indicating if this was the best epoch so far

        return False # Not applicable for 'train' mode

    def get_best_validation_metrics(self) -> Dict[str, float]:
        """Return the best validation metrics recorded."""
        print(f"MetricsTracker: Retrieving best validation metrics: {self.best_val_metrics}")
        return self.best_val_metrics

    def get_history(self) -> Dict:
        """Return the complete metrics history."""
        # print("MetricsTracker: Retrieving full history.") # Can be verbose
        return self.history

    def save_history(self, filepath: str):
        """Saves the metrics history dictionary to a JSON file."""
        print(f"MetricsTracker: Attempting to save history to {filepath}")
        try:
            # Convert potential numpy floats/ints to standard Python types for JSON serialization
            serializable_history = self._convert_to_serializable(self.history)
            with open(filepath, 'w') as f:
                json.dump(serializable_history, f, indent=4)
            print(f"MetricsTracker: History successfully saved to {filepath}")
        except Exception as e:
            print(f"MetricsTracker: Error saving history to {filepath}: {e}")
            log_and_print(logger, f"Error saving metrics history: {e}", logging.ERROR) # Also log it

    def _convert_to_serializable(self, obj):
        """Recursively converts numpy types in nested dicts/lists to Python types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(elem) for elem in obj]
        # Use isinstance with base classes np.integer and np.floating
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
             # Handle potential NaN values during conversion
             if np.isnan(obj):
                 return None # Or 'NaN' as a string, depending on preference
             return float(obj)
        elif isinstance(obj, np.ndarray): # Convert arrays to lists
            # Recursively apply conversion to array elements
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj): # Handle potential pandas NaNs
             return None
        # Add check for torch tensors if they might appear in history (unlikely but safe)
        elif isinstance(obj, torch.Tensor):
             return self._convert_to_serializable(obj.cpu().numpy()) # Convert tensor to numpy array first
        return obj


# --- Execution ---
# [Keep the Execution block as it was in the previous correct version]
print("\n--- Starting Cell 1.4 Execution ---")

# Initialize logging using the config object from Cell 1.2
# Ensure config and config.RESULT_DIR are accessible
if 'config' in locals() and hasattr(config, 'RESULT_DIR') and config.RESULT_DIR:
    logger = setup_logging(config.RESULT_DIR, config.DEBUG_MODE)
    log_and_print(logger, "Logger Initialized using config.")

    # Example usage of logger and log_and_print
    log_and_print(logger, "Example INFO log message.", logging.INFO)
    if config.DEBUG_MODE:
        log_and_print(logger, "Example DEBUG log message.", logging.DEBUG)

    # Initialize MetricsTracker (can be done here or later)
    metrics_tracker = MetricsTracker()
    log_and_print(logger, "MetricsTracker Initialized.")

    # --- Example Usage of MetricsTracker (for demonstration) ---
    print("\n--- MetricsTracker Example Usage ---")
    example_metrics_epoch0 = {
        'loss': 1.5, 'accuracy': 50.0,
        'precision_macro': 0.45, 'precision_weighted': 0.48, 'precision_micro': 0.50,
        'recall_macro': 0.46, 'recall_weighted': 0.50, 'recall_micro': 0.50,
        'f1_macro': 0.45, 'f1_weighted': 0.49, 'f1_micro': 0.50
    }
    example_metrics_epoch1 = {
        'loss': 1.2, 'accuracy': 60.0,
        'precision_macro': 0.55, 'precision_weighted': 0.58, 'precision_micro': 0.60,
        'recall_macro': 0.56, 'recall_weighted': 0.60, 'recall_micro': 0.60,
        'f1_macro': 0.55, 'f1_weighted': 0.59, 'f1_micro': 0.60
    }
    is_best_e0 = metrics_tracker.update_epoch_metrics(phase=1, epoch=0, mode='val', metrics=example_metrics_epoch0)
    is_best_e1 = metrics_tracker.update_epoch_metrics(phase=1, epoch=1, mode='val', metrics=example_metrics_epoch1)
    print("Example Best Metrics after 2 epochs:", metrics_tracker.get_best_validation_metrics())
    # Example save
    example_save_path = os.path.join(config.RESULT_DIR, 'example_history.json')
    metrics_tracker.save_history(example_save_path)
    print(f"Example history saved to {example_save_path}")
    print("--- End MetricsTracker Example ---")


else:
    print("Error: 'config' object not found or RESULT_DIR not set. Cannot initialize logger or MetricsTracker.")
    logger = None # Set logger to None if setup fails
    metrics_tracker = None # Set tracker to None

print("\n--- Finished Cell 1.4 Execution ---")
print("-" * 50)

"""#  DATA PROCESSING"""

# Cell 2.1: Dataset Loading Functions
# =====================================
# Dataset Loading and Initial Processing Functions
# =====================================
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import os
import traceback

# Assuming 'config' and 'logger', 'log_and_print' are defined in previous cells

def verify_dataset_type(df: pd.DataFrame, expected_type: str) -> Tuple[str, int]:
    """
    Verify dataset type (MELD/IEMOCAP) based on simple heuristics and return num_emotions.
    """
    print("\n--- Verifying Dataset Type ---")
    num_emotions = df['emotion'].nunique() # Use consistent 'emotion' column name
    total_samples = len(df)
    print(f"- Found {total_samples} samples and {num_emotions} unique emotions.")

    # Simple heuristic based on size (can be refined)
    detected_type = "MELD" if total_samples > 4000 else "IEMOCAP" # Adjusted threshold slightly
    print(f"- Detected type based on size: {detected_type}")
    print(f"- Expected type from config: {expected_type}")

    if detected_type != expected_type:
        log_and_print(logger, f"Warning: Detected dataset type ({detected_type}) does not match expected type ({expected_type}). Proceeding with expected type.", logging.WARNING)
        # You might want to raise an error here if strict matching is required
        # raise ValueError(f"Dataset type mismatch: Expected {expected_type}, detected {detected_type}")
    else:
        log_and_print(logger, f"Dataset type verification successful: {expected_type}")

    return expected_type, num_emotions


def load_and_preprocess_data(config: Config) -> pd.DataFrame:
    """
    Load and preprocess the dataset specified in the config.
    Renames columns to a consistent format.
    """
    print("\n--- Starting Cell 2.1: Data Loading and Preprocessing ---")
    try:
        excel_path = config.EXCEL_PATH
        dataset_type = config.CURRENT_DATASET
        result_dir = config.RESULT_DIR

        log_and_print(logger, f"Loading dataset: {dataset_type} from: {excel_path}")
        df = pd.read_excel(excel_path)
        log_and_print(logger, f"Initial DataFrame shape: {df.shape}")
        log_and_print(logger, f"Initial columns: {df.columns.tolist()}")
        print("\nDataFrame Head (Initial):")
        display(df.head())

        # --- Column Renaming for Consistency ---
        if dataset_type == "MELD":
            # Expected MELD columns: "Sr No.", "Filename", "Utterance", "Emotion", "Sentiment", "A2", "V1", "V3", "V4", "StartTime", "EndTime"
            rename_map = {
                "Utterance": "utterance",
                "Emotion": "emotion",
                "V1": "v1",
                "V3": "v3",
                "V4": "v4",
                "A2": "a2"
                # Add other columns if needed, e.g., "Filename": "filename"
            }
            required_cols = ["Utterance", "Emotion", "V1", "V3", "V4", "A2"]
        elif dataset_type == "IEMOCAP":
            # Expected IEMOCAP columns: "file_name", "start_time", "end_time", "Utterance", "Emotion", "V1", "V3", "V4", "V2", "A2"
            # Note: The 'V2' column in IEMOCAP excel seems misplaced based on docs, assuming it's not the text feature V2.
            rename_map = {
                "Utterance": "utterance",
                "Emotion": "emotion",
                "V1": "v1",
                "V3": "v3",
                "V4": "v4",
                "A2": "a2"
                # Add other columns if needed, e.g., "file_name": "filename"
            }
            required_cols = ["Utterance", "Emotion", "V1", "V3", "V4", "A2"]
        else:
            log_and_print(logger, f"Error: Unknown dataset type '{dataset_type}' for column mapping.", logging.ERROR)
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # Check if required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            log_and_print(logger, f"Error: Missing required columns in {excel_path}: {missing_cols}", logging.ERROR)
            raise ValueError(f"Missing columns: {missing_cols}")

        # Apply renaming
        df.rename(columns=rename_map, inplace=True)
        consistent_cols = list(rename_map.values())
        log_and_print(logger, f"Columns renamed for consistency. Using: {consistent_cols}")
        print("\nDataFrame Head (After Renaming):")
        display(df[consistent_cols].head()) # Display only the relevant columns

        # Keep only necessary columns
        df = df[consistent_cols]
        log_and_print(logger, f"Filtered DataFrame shape: {df.shape}")

        # --- Feature Parsing ---
        # Define feature columns that need parsing
        feature_columns = ['v1', 'v3', 'v4', 'a2']
        log_and_print(logger, f"Parsing feature columns: {feature_columns}")
        parsing_errors = 0
        total_rows = len(df)
        for col in feature_columns:
            if col in df.columns:
                print(f"  - Parsing column '{col}'...")
                # Check type before applying literal_eval
                if df[col].dtype == 'object': # Only apply if it's a string
                     # Use tqdm for progress bar during parsing
                    parsed_col = []
                    for item in tqdm(df[col], desc=f"Parsing {col}", leave=False):
                        try:
                            # Handle potential NaN or non-string values gracefully
                            if pd.isna(item):
                                parsed_col.append(None) # Or handle as needed, e.g., np.nan, empty list
                            elif isinstance(item, str):
                                parsed_col.append(ast.literal_eval(item))
                            else:
                                parsed_col.append(item) # Assume already parsed if not string
                        except (ValueError, SyntaxError, TypeError) as e:
                            # print(f"    Warning: Error parsing value in column '{col}': {item}. Error: {e}")
                            parsed_col.append(None) # Or some default value
                            parsing_errors += 1
                    df[col] = parsed_col
                else:
                     print(f"    Column '{col}' is not of type 'object', skipping ast.literal_eval.")
            else:
                log_and_print(logger, f"Warning: Feature column '{col}' not found after renaming.", logging.WARNING)

        if parsing_errors > 0:
             log_and_print(logger, f"Warning: Encountered {parsing_errors} errors during feature parsing.", logging.WARNING)

        # --- Data Cleaning & Type Conversion ---
        # Drop rows where parsing failed (resulted in None)
        initial_len = len(df)
        df.dropna(subset=feature_columns, inplace=True)
        dropped_rows = initial_len - len(df)
        if dropped_rows > 0:
            log_and_print(logger, f"Dropped {dropped_rows} rows due to parsing errors or missing features.")

        # Convert utterances to strings
        df['utterance'] = df['utterance'].astype(str)
        log_and_print(logger, "Converted 'utterance' column to string type.")

        # --- Dataset Verification & Emotion Distribution ---
        verified_type, num_emotions = verify_dataset_type(df, dataset_type)
        config.NUM_EMOTIONS = num_emotions # Store number of emotions in config

        log_and_print(logger, f"Final number of samples after cleaning: {len(df)}")
        log_and_print(logger, f"Number of unique emotions found: {num_emotions}")

        # Display and plot emotion distribution
        emotion_dist = df['emotion'].value_counts()
        log_and_print(logger, "\nEmotion Distribution (After Cleaning):")
        print(emotion_dist.to_string()) # Print distribution neatly

        plt.figure(figsize=(10, 5)) # Adjusted size
        sns.barplot(x=emotion_dist.index, y=emotion_dist.values, palette='viridis') # Use a different palette
        plt.title(f'Emotion Distribution - {verified_type} ({len(df)} samples)', fontsize=14)
        plt.xlabel('Emotion', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save plot with high DPI
        plot_path = os.path.join(result_dir, 'visualizations', 'emotion_distribution.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True) # Ensure viz dir exists
        plt.savefig(plot_path, dpi=1200)
        log_and_print(logger, f"Emotion distribution plot saved to: {plot_path}")
        # plt.show() # Display plot in notebook
        plt.close() # Close plot to free memory

        print("\nDataFrame Info (After Preprocessing):")
        df.info(memory_usage='deep')

        print("\n--- Finished Cell 2.1 ---")
        print("-" * 50)
        return df

    except Exception as e:
        log_and_print(logger, f"Error in data loading/preprocessing: {str(e)}", logging.ERROR)
        log_and_print(logger, traceback.format_exc(), logging.ERROR)
        print("\n--- Cell 2.1 Failed ---")
        print("-" * 50)
        raise # Re-raise the exception after logging

# --- Execution ---
# Ensure config, logger, log_and_print are available from previous cells
if 'config' in locals() and 'logger' in locals():
    df_processed = load_and_preprocess_data(config)
    # Optional: Display final processed dataframe head
    if df_processed is not None:
        print("\nFinal Processed DataFrame Head:")
        display(df_processed.head())
else:
    print("Error: 'config' or 'logger' not defined. Cannot execute Cell 2.1.")
    df_processed = None

# Cell 2.2: Data Preparation and Splitting
# =====================================
# Data Preparation, Encoding and Splitting
# =====================================

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# Ensure imblearn is imported from Cell 1.1
try:
    from imblearn.over_sampling import RandomOverSampler
except ImportError:
    log_and_print(logger, "Error: imblearn.over_sampling.RandomOverSampler not available. Please install imbalanced-learn.", logging.ERROR)
    # Depending on requirements, you might raise an error here
    # raise ImportError("RandomOverSampler not found. Install imbalanced-learn.")
import torch
import numpy as np
import pandas as pd
import traceback

# Assuming df_processed, config, logger, log_and_print are available

def prepare_data(df: pd.DataFrame, config: Config) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], LabelEncoder, int]:
    """
    Prepare data for training: encode labels, split data, handle class imbalance, and convert to tensors.
    Returns train_data, val_data, label_encoder, num_classes.
    """
    print("\n--- Starting Cell 2.2: Data Preparation and Splitting ---")
    if df is None or df.empty:
        log_and_print(logger, "Error: Input DataFrame is empty or None.", logging.ERROR)
        raise ValueError("Input DataFrame is empty.")

    try:
        # --- 1. Label Encoding ---
        log_and_print(logger, "Encoding emotion labels...")
        le = LabelEncoder()
        df['emotion_encoded'] = le.fit_transform(df['emotion'])
        num_classes = len(le.classes_)
        config.NUM_EMOTIONS = num_classes # Update config again just in case
        log_and_print(logger, f"Found {num_classes} unique classes: {le.classes_}")
        print(f"Label mapping: {dict(zip(le.classes_, range(num_classes)))}")

        # --- 2. Prepare Features (X) and Labels (y) ---
        # Use the consistent column names
        feature_cols = ['utterance', 'v1', 'v3', 'v4', 'a2']
        label_col = 'emotion_encoded'
        log_and_print(logger, f"Selecting features: {feature_cols} and label: {label_col}")

        X = df[feature_cols]
        y = df[label_col]
        log_and_print(logger, f"Shape of features X: {X.shape}")
        log_and_print(logger, f"Shape of labels y: {y.shape}")

        # Display class distribution before oversampling
        log_and_print(logger, "\nClass distribution BEFORE oversampling:")
        print(y.value_counts().sort_index().to_string())
        print(pd.Series(le.inverse_transform(y.value_counts().sort_index().index), index=y.value_counts().sort_index().values).to_string())


        # --- 3. Handle Class Imbalance (Random Oversampling) ---
        log_and_print(logger, "Applying Random Over-Sampling...")
        try:
            ros = RandomOverSampler(random_state=config.SEED)
            # Oversample based on index to avoid duplicating large feature objects in memory during resample
            X_index = pd.DataFrame(X.index, columns=['original_index'])
            X_resampled_index, y_resampled = ros.fit_resample(X_index, y)
            log_and_print(logger, f"Oversampling complete. Original samples: {len(y)}, Resampled samples: {len(y_resampled)}")

            # Retrieve the actual features using the resampled indices
            X_resampled = X.loc[X_resampled_index['original_index']].reset_index(drop=True)
            y_resampled = y_resampled.reset_index(drop=True) # Ensure y_resampled index matches X_resampled

            log_and_print(logger, f"Shape of X after resampling: {X_resampled.shape}")
            log_and_print(logger, f"Shape of y after resampling: {y_resampled.shape}")

            # Display class distribution after oversampling
            log_and_print(logger, "\nClass distribution AFTER oversampling:")
            resampled_counts = y_resampled.value_counts().sort_index()
            print(resampled_counts.to_string())
            print(pd.Series(le.inverse_transform(resampled_counts.index), index=resampled_counts.values).to_string())

        except Exception as e:
            log_and_print(logger, f"Error during oversampling: {e}. Proceeding without oversampling.", logging.WARNING)
            X_resampled, y_resampled = X.reset_index(drop=True), y.reset_index(drop=True) # Use original data if oversampling fails


        # --- 4. Split Data into Train and Validation Sets ---
        log_and_print(logger, "Splitting data into training (80%) and validation (20%) sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_resampled, y_resampled,
            test_size=0.2,
            random_state=config.SEED,
            stratify=y_resampled # Stratify based on the resampled labels
        )
        log_and_print(logger, f"Training set size: {len(y_train)}")
        log_and_print(logger, f"Validation set size: {len(y_val)}")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

        # --- 5. Convert to Dictionary and Tensors ---
        log_and_print(logger, "Converting data splits to Tensors...")

        def create_tensor_dict(X_split, y_split):
            try:
                data_dict = {
                    # Keep utterances as pandas Series/list for tokenizer later
                    'utterances': X_split['utterance'].tolist(),
                    # Stack numerical features into tensors
                    'v1': torch.tensor(np.stack(X_split['v1'].values), dtype=torch.float32),
                    'v3': torch.tensor(np.stack(X_split['v3'].values), dtype=torch.float32),
                    'v4': torch.tensor(np.stack(X_split['v4'].values), dtype=torch.float32),
                    'a2': torch.tensor(np.stack(X_split['a2'].values), dtype=torch.float32),
                    'labels': torch.tensor(y_split.values, dtype=torch.long)
                }
                # Print shapes for verification
                print(f"  - Utterances: {len(data_dict['utterances'])} samples")
                for key in ['v1', 'v3', 'v4', 'a2', 'labels']:
                    print(f"  - {key} shape: {data_dict[key].shape}, dtype: {data_dict[key].dtype}")
                return data_dict
            except Exception as e:
                 log_and_print(logger, f"Error converting split to tensors: {e}", logging.ERROR)
                 log_and_print(logger, traceback.format_exc(), logging.ERROR)
                 return None # Return None if conversion fails

        print("\nConverting Training Data:")
        train_data = create_tensor_dict(X_train, y_train)
        print("\nConverting Validation Data:")
        val_data = create_tensor_dict(X_val, y_val)

        if train_data is None or val_data is None:
            raise RuntimeError("Failed to convert data splits to tensors.")

        log_and_print(logger, "Data successfully prepared and converted to tensors.")
        print("\n--- Finished Cell 2.2 ---")
        print("-" * 50)

        return train_data, val_data, le, num_classes

    except Exception as e:
        log_and_print(logger, f"Error in data preparation/splitting: {str(e)}", logging.ERROR)
        log_and_print(logger, traceback.format_exc(), logging.ERROR)
        print("\n--- Cell 2.2 Failed ---")
        print("-" * 50)
        # Return None or empty structures to indicate failure
        return None, None, None, 0

# --- Execution ---
# Ensure df_processed, config, logger, log_and_print are available
if 'df_processed' in locals() and df_processed is not None and 'config' in locals() and 'logger' in locals():
    train_data, val_data, label_encoder, num_classes = prepare_data(df_processed, config)

    # Verify output
    if train_data and val_data and label_encoder:
        log_and_print(logger, f"Data preparation successful. Num classes: {num_classes}")
        print(f"Train data keys: {train_data.keys()}")
        print(f"Validation data keys: {val_data.keys()}")
        print(f"Label Encoder Classes: {label_encoder.classes_}")
    else:
        log_and_print(logger, "Data preparation failed.", logging.ERROR)
else:
    print("Error: 'df_processed', 'config', or 'logger' not available. Cannot execute Cell 2.2.")
    train_data, val_data, label_encoder, num_classes = None, None, None, 0

# Cell 2.3: Dataset Class Definition
# =====================================
# Custom Dataset Class Implementation
# =====================================

import torch
from torch.utils.data import Dataset
# Ensure transformers is imported from Cell 1.1
try:
    from transformers import RobertaTokenizer
except ImportError:
    log_and_print(logger, "Error: transformers.RobertaTokenizer not available.", logging.ERROR)
    # raise ImportError("RobertaTokenizer not found.")
import traceback

# Assuming config, logger, log_and_print, train_data, val_data are available

class EmotionDataset(Dataset):
    """
    Custom Dataset class for multimodal emotion recognition data.
    Handles tokenization of text and returns features as tensors.
    """
    def __init__(self, data: Dict, tokenizer: RobertaTokenizer, max_len: int, dataset_name: str = "Dataset"):
        """
        Args:
            data (Dict): Dictionary containing 'utterances', 'v1', 'v3', 'v4', 'a2', 'labels'.
                         Features should already be torch tensors, 'utterances' a list of strings.
            tokenizer (RobertaTokenizer): Pre-initialized RoBERTa tokenizer.
            max_len (int): Maximum sequence length for tokenization.
            dataset_name (str): Name for logging purposes (e.g., "Training", "Validation").
        """
        print(f"\n--- Initializing EmotionDataset for {dataset_name} ---")
        self.dataset_name = dataset_name
        try:
            self.utterances = data['utterances']
            self.v1 = data['v1']
            self.v3 = data['v3']
            self.v4 = data['v4']
            self.a2 = data['a2']
            self.labels = data['labels']
            self.tokenizer = tokenizer
            self.max_len = max_len

            # --- Verification ---
            self.num_samples = len(self.labels)
            if not all(len(lst) == self.num_samples for lst in [self.utterances, self.v1, self.v3, self.v4, self.a2]):
                log_and_print(logger, f"Warning: Mismatch in feature list lengths for {dataset_name} dataset!", logging.WARNING)
                # Find the minimum length to avoid index errors
                self.num_samples = min(len(self.labels), len(self.utterances), len(self.v1), len(self.v3), len(self.v4), len(self.a2))
                log_and_print(logger, f"Adjusting number of samples to minimum length: {self.num_samples}", logging.WARNING)

            log_and_print(logger, f"EmotionDataset '{dataset_name}' initialized successfully with {self.num_samples} samples.")
            print(f" - Max sequence length for tokenizer: {self.max_len}")
            print(f"--- Finished Initializing EmotionDataset for {dataset_name} ---")

        except KeyError as e:
            log_and_print(logger, f"Error initializing EmotionDataset '{dataset_name}': Missing key {e}", logging.ERROR)
            raise
        except Exception as e:
            log_and_print(logger, f"Error initializing EmotionDataset '{dataset_name}': {e}", logging.ERROR)
            log_and_print(logger, traceback.format_exc(), logging.ERROR)
            raise

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset, including tokenized text.
        """
        # print(f"[{self.dataset_name}] Getting item at index: {idx}") # Can be very verbose
        if idx >= self.num_samples:
             raise IndexError(f"Index {idx} out of bounds for dataset with size {self.num_samples}")

        try:
            utterance = str(self.utterances[idx]) # Ensure utterance is string

            # --- Tokenize Text ---
            # print(f"  - Tokenizing utterance: '{utterance[:50]}...'") # Verbose
            encoding = self.tokenizer(
                text=utterance,
                add_special_tokens=True,       # Add [CLS] and [SEP]
                max_length=self.max_len,       # Pad/truncate
                padding='max_length',          # Pad to max_len
                truncation=True,               # Truncate if longer
                return_attention_mask=True,    # Return attention mask
                return_tensors='pt'            # Return PyTorch tensors
            )
            # .squeeze(0) removes the batch dimension added by return_tensors='pt'
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            # print(f"  - input_ids shape: {input_ids.shape}") # Verbose
            # print(f"  - attention_mask shape: {attention_mask.shape}") # Verbose

            # --- Prepare Sample Dictionary ---
            sample = {
                'input_ids': input_ids,             # Shape: [max_len]
                'attention_mask': attention_mask,   # Shape: [max_len]
                'v1': self.v1[idx],                 # Shape: [512]
                'v3': self.v3[idx],                 # Shape: [25]
                'v4': self.v4[idx],                 # Shape: [25]
                'a2': self.a2[idx],                 # Shape: [512]
                'label': self.labels[idx]           # Shape: [] (scalar)
            }

            # --- Print Shapes for Verification (for the first few samples) ---
            # if idx < 3: # Limit printing to first few indices
            #     print(f"  - Sample {idx} shapes:")
            #     for key, value in sample.items():
            #          if isinstance(value, torch.Tensor):
            #              print(f"    {key}: {value.shape}")
            #          else:
            #              print(f"    {key}: Type {type(value)}")

            return sample

        except Exception as e:
            log_and_print(logger, f"Error getting item at index {idx} for {self.dataset_name}: {e}", logging.ERROR)
            log_and_print(logger, f"Problematic utterance: {self.utterances[idx] if idx < len(self.utterances) else 'Index out of bounds'}", logging.ERROR)
            log_and_print(logger, traceback.format_exc(), logging.ERROR)
            # Return None or raise error? Returning None might cause issues in DataLoader.
            # Let's try returning a dummy sample, but this should be investigated.
            # Alternatively, filter out problematic indices beforehand.
            # For now, re-raise to halt execution and identify the root cause.
            raise


def verify_sample_dimensions(dataset: EmotionDataset, sample_idx: int = 0):
    """Verify dimensions and types of a single sample from the dataset."""
    print(f"\n--- Verifying Sample Dimensions (Index: {sample_idx}) for {dataset.dataset_name} ---")
    try:
        if len(dataset) > sample_idx:
            sample = dataset[sample_idx]
            print("Sample details:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"- {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                else:
                    print(f"- {key}: type={type(value)}, value='{str(value)[:50]}...'") # Print truncated value for non-tensors
        else:
            print(f"Cannot verify sample {sample_idx}, dataset size is {len(dataset)}.")
        print("--- Finished Sample Verification ---")
    except Exception as e:
        log_and_print(logger, f"Error during sample verification for index {sample_idx}: {e}", logging.ERROR)
        print("--- Sample Verification Failed ---")

# --- Execution ---
print("\n--- Starting Cell 2.3 Execution ---")

# Ensure config, logger, log_and_print, train_data, val_data are available
if 'config' in locals() and 'logger' in locals() and 'train_data' in locals() and train_data:
    try:
        # Initialize tokenizer
        tokenizer_name = config.ROBERTA_MODEL_NAME
        log_and_print(logger, f"Initializing RoBERTa tokenizer: {tokenizer_name}")
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        log_and_print(logger, "RoBERTa tokenizer initialized successfully.")

        # Create Dataset instances (can be done here or in Cell 2.4)
        # We'll create them here for verification purposes
        print("\nCreating Training Dataset instance...")
        train_dataset = EmotionDataset(train_data, tokenizer, config.MAX_LEN, "Training")

        print("\nCreating Validation Dataset instance...")
        val_dataset = EmotionDataset(val_data, tokenizer, config.MAX_LEN, "Validation")

        # Verify dimensions of one sample from each dataset
        verify_sample_dimensions(train_dataset, 0)
        verify_sample_dimensions(val_dataset, 0)

    except Exception as e:
        log_and_print(logger, f"Error during Cell 2.3 execution: {e}", logging.ERROR)
        log_and_print(logger, traceback.format_exc(), logging.ERROR)
        tokenizer = None
        train_dataset = None
        val_dataset = None
else:
    print("Error: Required variables (config, logger, train_data) not available. Cannot execute Cell 2.3.")
    tokenizer = None
    train_dataset = None
    val_dataset = None


print("\n--- Finished Cell 2.3 Execution ---")
print("-" * 50)

# Cell 2.4: DataLoader Setup
# =====================================
# DataLoader Configuration and Initialization
# =====================================

import torch
from torch.utils.data import DataLoader
import traceback

# Assuming config, logger, log_and_print, train_dataset, val_dataset, device are available

def create_dataloaders(
    train_dataset: Dataset, # Modified to accept Dataset objects
    val_dataset: Dataset,   # Modified to accept Dataset objects
    config: Config
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from Dataset objects.

    Args:
        train_dataset (Dataset): The training dataset instance.
        val_dataset (Dataset): The validation dataset instance.
        config (Config): The configuration object.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation dataloaders.
    """
    print("\n--- Starting Cell 2.4: DataLoader Setup ---")
    if not train_dataset or not val_dataset:
         log_and_print(logger, "Error: Train or validation dataset is None. Cannot create DataLoaders.", logging.ERROR)
         raise ValueError("Input datasets cannot be None.")

    try:
        log_and_print(logger, "Creating DataLoaders...")
        print(f"- Batch Size: {config.BATCH_SIZE}")
        print(f"- Num Workers: {config.NUM_WORKERS}")
        print(f"- Pin Memory: {config.PIN_MEMORY}")

        # Create training DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,               # Shuffle training data each epoch
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY, # Speeds up CPU-to-GPU transfer if True
            drop_last=True              # Drop last incomplete batch in training
        )
        log_and_print(logger, f"Training DataLoader created with {len(train_loader)} batches.")

        # Create validation DataLoader
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE, # Can use same or different batch size for validation
            shuffle=False,              # No need to shuffle validation data
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=False             # Keep last batch in validation
        )
        log_and_print(logger, f"Validation DataLoader created with {len(val_loader)} batches.")

        # --- Verify Batch Dimensions ---
        log_and_print(logger, "Verifying batch dimensions using one sample batch from train_loader...")
        try:
            train_batch = next(iter(train_loader))
            print("Sample Training Batch Details:")
            for key, value in train_batch.items():
                if isinstance(value, torch.Tensor):
                    # Check if tensors will be on GPU if pin_memory=True (they won't be yet, happens in training loop)
                    print(f"- {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                elif isinstance(value, list):
                     print(f"- {key}: type=list, length={len(value)}, example='{str(value[0])[:50]}...'") # Show list length and example
                else:
                     print(f"- {key}: type={type(value)}")
        except StopIteration:
            log_and_print(logger, "Warning: train_loader is empty, cannot verify batch dimensions.", logging.WARNING)
        except Exception as e_batch:
            log_and_print(logger, f"Error verifying batch dimensions: {e_batch}", logging.ERROR)


        print("\n--- Finished Cell 2.4 ---")
        print("-" * 50)
        return train_loader, val_loader

    except Exception as e:
        log_and_print(logger, f"Error in DataLoader creation: {str(e)}", logging.ERROR)
        log_and_print(logger, traceback.format_exc(), logging.ERROR)
        print("\n--- Cell 2.4 Failed ---")
        print("-" * 50)
        # Return None to indicate failure
        return None, None

# --- Execution ---
# Ensure train_dataset, val_dataset, config, logger, log_and_print are available
if 'train_dataset' in locals() and train_dataset and 'val_dataset' in locals() and val_dataset and 'config' in locals() and 'logger' in locals():
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)

    # Verify loaders were created
    if train_loader and val_loader:
        log_and_print(logger, "DataLoaders created successfully.")
    else:
        log_and_print(logger, "DataLoader creation failed.", logging.ERROR)
else:
    print("Error: Required variables (train_dataset, val_dataset, config, logger) not available. Cannot execute Cell 2.4.")
    train_loader, val_loader = None, None

"""# MODEL ARCHITECTURE"""

# Cell 3.1: Progressive BiLSTM Implementation
# =====================================
# Progressive BiLSTM with Multi-Scale Learning
# =====================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict # Added Dict
import traceback

# Assuming config, logger, log_and_print, device are available

class ProgressiveBiLSTM(nn.Module):
    """
    Implements a multi-layer Bidirectional LSTM with internal multi-head self-attention
    and adaptive residual gating, as described in the methodology (Sec III.D.3).
    Returns outputs from ALL intermediate blocks.
    """
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 3, dropout: float = 0.3, attention_heads: int = 8):
        """
        Args:
            input_size (int): Dimension of the input features.
            hidden_size (int): Hidden dimension for each LSTM direction (default: 256).
            num_layers (int): Number of stacked BiLSTM layers (default: 3).
            dropout (float): Dropout rate for internal layers (default: 0.3).
            attention_heads (int): Number of heads for internal self-attention (default: 8).
        """
        super(ProgressiveBiLSTM, self).__init__()
        print(f"\n--- Initializing ProgressiveBiLSTM (Modified to return all block outputs) ---") # Modified print

        self.input_size = input_size
        self.hidden_size = hidden_size # Per direction
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.attention_heads = attention_heads
        self.lstm_output_size = hidden_size * 2 # Output dim of bidirectional LSTM

        # --- Layer Components ---
        self.lstm_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.transform_layers = nn.ModuleList()
        self.residual_gates = nn.ModuleList()

        current_input_size = input_size
        for i in range(num_layers):
            # 1. BiLSTM Layer
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=current_input_size,
                    hidden_size=hidden_size,
                    num_layers=1, # Each layer in ModuleList is a single LSTM layer
                    bidirectional=True,
                    batch_first=True # Input/output format: (batch, seq, feature)
                )
            )
            print(f"  - Added LSTM Layer {i+1}: Input={current_input_size}, Hidden={hidden_size} (bidirectional)")

            # 2. Multi-Head Self-Attention Layer
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=self.lstm_output_size, # Input/output dim for attention
                    num_heads=attention_heads,
                    dropout=dropout,
                    batch_first=True
                )
            )
            print(f"  - Added Attention Layer {i+1}: Embed={self.lstm_output_size}, Heads={attention_heads}")

            # 3. Feature Transformation Layer (FFN)
            self.transform_layers.append(
                nn.Sequential(
                    # Methodology just shows Linear(H_attn), let's match that simply first
                    # Can add more complexity if needed later
                    nn.Linear(self.lstm_output_size, self.lstm_output_size),
                    nn.LayerNorm(self.lstm_output_size), # LayerNorm is crucial for stability
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            print(f"  - Added Transform Layer {i+1}: Dim={self.lstm_output_size}")

            # 4. Adaptive Residual Gating Layer (for layers > 0)
            if i > 0:
                # Gate input is concatenation of current transformed features (lstm_output_size)
                # and the output features from the previous block (also lstm_output_size)
                gate_input_dim = self.lstm_output_size * 2
                self.residual_gates.append(
                    nn.Sequential(
                        nn.Linear(gate_input_dim, 1),
                        nn.Sigmoid()
                    )
                )
                print(f"  - Added Residual Gate {i}: Input Dim={gate_input_dim}")

            # Input size for the next LSTM layer is the output size of the current block
            current_input_size = self.lstm_output_size

        log_and_print(logger, f"Progressive BiLSTM initialized: Input={input_size}, Hidden={hidden_size}, Layers={num_layers}, Heads={attention_heads}")
        print(f"--- Finished Initializing ProgressiveBiLSTM ---")


    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through Progressive BiLSTM.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, seq_length, input_size).
            mask (Optional[torch.Tensor]): Boolean tensor indicating padding locations.
                                           Shape: (batch_size, seq_length). `True` means ignore, `False` means attend.
                                           If input mask has 1 for valid and 0 for padding, it needs inversion.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - final_output (torch.Tensor): Output from the last time step of the final layer. Shape: (batch_size, hidden_size * 2).
                - layer_block_outputs (List[torch.Tensor]): List containing output tensors from each *block* (LSTM+Attn+Transform+Gate).
                                                            Shape of each: (batch_size, seq_length, hidden_size * 2).
        """
        # print(f"\nProgressiveBiLSTM Forward Pass:") # Verbose
        # print(f"- Input x shape: {x.shape}") # Verbose
        # if mask is not None: print(f"- Input mask shape: {mask.shape}") # Verbose

        layer_block_outputs = [] # Renamed from layer_outputs
        current_features = x # Input to the first layer

        # Prepare attention mask for PyTorch MultiheadAttention
        # It expects True for padded tokens (key_padding_mask)
        attention_mask = mask # Assume input mask is already boolean with True for padding

        for i in range(self.num_layers):
            # print(f"  - Processing Layer {i+1}/{self.num_layers}") # Verbose

            # 1. BiLSTM Processing (Eq 12)
            # lstm_out: (batch, seq_len, hidden_size * 2)
            lstm_out, _ = self.lstm_layers[i](current_features)
            # print(f"    - LSTM output shape: {lstm_out.shape}") # Verbose

            # 2. Multi-Head Self-Attention (Eq 13)
            # Input: query, key, value are all lstm_out
            # attention_mask: (batch, seq_len) - True for positions to ignore
            try:
                attention_out, _ = self.attention_layers[i](
                    query=lstm_out,
                    key=lstm_out,
                    value=lstm_out,
                    key_padding_mask=attention_mask # Use the prepared mask
                )
                # print(f"    - Attention output shape: {attention_out.shape}") # Verbose
            except Exception as attn_err:
                 log_and_print(logger, f"Error in Attention Layer {i+1}: {attn_err}", logging.ERROR)
                 # If attention fails, maybe just pass lstm_out? Or re-raise?
                 # For now, let's pass lstm_out to potentially recover
                 attention_out = lstm_out
                 # raise attn_err # Option to halt execution

            # 3. Feature Transformation (Eq 14)
            transformed = self.transform_layers[i](attention_out)
            # print(f"    - Transformed output shape: {transformed.shape}") # Verbose

            # 4. Adaptive Residual Gating (Eq 15, 16) - for layers i > 0 (index > 0)
            if i > 0:
                # Concatenate transformed features and the input features to this layer
                # Note: current_features is the output of the previous block (i-1)
                gate_input = torch.cat([transformed, current_features], dim=-1)
                # print(f"    - Gate input shape: {gate_input.shape}") # Verbose
                gate = self.residual_gates[i-1](gate_input) # Use gate index i-1
                # print(f"    - Gate shape: {gate.shape}") # Verbose
                gated_output = gate * transformed + (1 - gate) * current_features
                # print(f"    - Gated output shape: {gated_output.shape}") # Verbose
            else:
                # No gating for the first layer
                gated_output = transformed

            current_features = gated_output # Output of this block becomes input for the next
            layer_block_outputs.append(current_features) # Store the output of this block

        # 5. Final Output (Eq: F_final = F(3)[:, -1, :])
        # Take the output of the last layer, last time step
        # Assuming sequence length > 0
        if current_features.shape[1] > 0:
             # --- Get the output of the *last time step* for the final classification path ---
             final_output_last_step = current_features[:, -1, :]
        else:
             # Handle empty sequence case if necessary
             log_and_print(logger, "Warning: Sequence length is 0 in ProgressiveBiLSTM output.", logging.WARNING)
             final_output_last_step = torch.zeros((current_features.shape[0], self.lstm_output_size), device=x.device)

        # print(f"- Final output shape (last time step): {final_output_last_step.shape}") # Verbose
        # print(f"- Number of intermediate layer block outputs: {len(layer_block_outputs)}") # Verbose
        # print("ProgressiveBiLSTM Forward Pass Complete.") # Verbose

        # Return the final output (last time step) AND the list of all block outputs (full sequences)
        return final_output_last_step, layer_block_outputs

    def get_config(self) -> Dict:
        """Return layer configuration."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout_rate,
            'attention_heads': self.attention_heads,
            'lstm_output_size': self.lstm_output_size
        }

# --- Test Function ---
def test_progressive_bilstm(config: Config, device: torch.device):
    """Test Progressive BiLSTM implementation."""
    print("\n--- Starting Cell 3.1: Progressive BiLSTM Test ---")
    try:
        # Test parameters (using config where possible)
        batch_size = config.BATCH_SIZE # 16
        # Use a representative input size (e.g., text projection dim)
        input_size = config.TEXT_PROJECTION_DIM # 512
        seq_length = config.MAX_LEN # 128
        hidden_size = config.PROGRESSIVE_BILSTM_HIDDEN_DIM # 256
        num_layers = 3
        dropout = config.DROPOUT_RATE # 0.3
        attention_heads = 8 # Matching methodology

        log_and_print(logger, f"Testing ProgressiveBiLSTM with: Batch={batch_size}, SeqLen={seq_length}, Input={input_size}, Hidden={hidden_size}, Layers={num_layers}")

        # Create model instance
        model = ProgressiveBiLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            attention_heads=attention_heads
        ).to(device)
        print("\nModel Instantiated:")
        # print(model) # Print model structure

        # Create test input tensor
        test_input = torch.randn(batch_size, seq_length, input_size).to(device)
        print(f"\nTest Input shape: {test_input.shape}")

        # Create test mask (Example: last 10 tokens are padding)
        # Mask should be True for padding, False otherwise
        test_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool).to(device)
        if seq_length > 10:
            test_mask[:, -10:] = True
        print(f"Test Mask shape: {test_mask.shape}")
        # print(f"Example Mask (first batch): {test_mask[0]}") # Verbose

        # --- Forward pass ---
        model.eval() # Set model to evaluation mode for testing
        with torch.no_grad(): # Disable gradient calculation
            final_output, layer_block_outputs = model(test_input, test_mask) # Capture both outputs

        # --- Verification ---
        log_and_print(logger, "\nProgressive BiLSTM Test Results:")
        print(f"- Input shape: {test_input.shape}")
        print(f"- Mask shape: {test_mask.shape}")

        # Check final output shape (last time step)
        expected_final_dim = hidden_size * 2 # 512
        print(f"- Final Output (last step) shape: {final_output.shape}")
        assert final_output.shape == (batch_size, expected_final_dim), f"Expected final output shape {(batch_size, expected_final_dim)}, but got {final_output.shape}"
        print(f"  (Matches expected: {(batch_size, expected_final_dim)})")

        # Check number and shape of layer block outputs (full sequences)
        print(f"- Number of layer block outputs: {len(layer_block_outputs)}")
        assert len(layer_block_outputs) == num_layers, f"Expected {num_layers} layer block outputs, but got {len(layer_block_outputs)}"
        for i, layer_out in enumerate(layer_block_outputs):
            print(f"  - Block {i+1} output shape: {layer_out.shape}")
            expected_layer_dim = hidden_size * 2
            # Shape should be (batch, seq_len, hidden*2)
            assert layer_out.shape == (batch_size, seq_length, expected_layer_dim), f"Expected block {i+1} output shape {(batch_size, seq_length, expected_layer_dim)}, but got {layer_out.shape}"
            print(f"  (Matches expected: {(batch_size, seq_length, expected_layer_dim)})")


        log_and_print(logger, "Progressive BiLSTM Test successful!")
        print("\n--- Finished Cell 3.1 Test ---")
        print("-" * 50)
        return True

    except Exception as e:
        log_and_print(logger, f"Error in Progressive BiLSTM test: {str(e)}", logging.ERROR)
        log_and_print(logger, traceback.format_exc(), logging.ERROR)
        print("\n--- Cell 3.1 Test Failed ---")
        print("-" * 50)
        return False

# --- Execution ---
# Ensure config, logger, log_and_print, device are available
if 'config' in locals() and 'logger' in locals() and 'device' in locals():
    test_successful_bilstm = test_progressive_bilstm(config, device)
else:
    print("Error: Required variables (config, logger, device) not available. Cannot execute Cell 3.1.")
    test_successful_bilstm = False

# Cell 3.2: Text, Audio, and Visual Processing Modules
# =====================================
# Modality-Specific Processing Modules
# =====================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List # Added List
import traceback

# Ensure transformers, ProgressiveBiLSTM are imported/defined
# Assuming config, logger, log_and_print, device, ProgressiveBiLSTM are available

class TextProcessor(nn.Module):
    """
    Processes tokenized text using RoBERTa and Progressive BiLSTM.
    Returns final output and intermediate BiLSTM block outputs.
    """
    def __init__(self, config: Config):
        super(TextProcessor, self).__init__()
        print("\n--- Initializing TextProcessor ---")
        self.config = config
        hidden_dim = config.PROGRESSIVE_BILSTM_HIDDEN_DIM # 256
        dropout = config.DROPOUT_RATE # 0.3

        # RoBERTa base model
        try:
            self.roberta = RobertaModel.from_pretrained(config.ROBERTA_MODEL_NAME)
            print(f"- RoBERTa model ('{config.ROBERTA_MODEL_NAME}') loaded.")
        except Exception as e:
            log_and_print(logger, f"Failed to load RoBERTa model: {e}", logging.ERROR)
            raise

        # Feature converter (matches Eq 2)
        self.text_fc = nn.Sequential(
            nn.Linear(config.TEXT_OUTPUT_DIM, config.TEXT_PROJECTION_DIM), # 768 -> 512
            nn.LayerNorm(config.TEXT_PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        print(f"- Text FC layer created: {config.TEXT_OUTPUT_DIM} -> {config.TEXT_PROJECTION_DIM}")

        # Progressive BiLSTM for text (matches Methodology Sec III.D.3)
        self.text_bilstm = ProgressiveBiLSTM(
            input_size=config.TEXT_PROJECTION_DIM, # 512
            hidden_size=hidden_dim, # 256
            num_layers=3,
            dropout=dropout,
            attention_heads=8 # As per methodology
        )
        print(f"- Text ProgressiveBiLSTM created (Input: {config.TEXT_PROJECTION_DIM})")
        print(f"--- Finished Initializing TextProcessor ---")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for text processing.

        Args:
            input_ids (torch.Tensor): Token IDs. Shape: (batch_size, seq_length).
            attention_mask (torch.Tensor): Attention mask (1 for real tokens, 0 for padding). Shape: (batch_size, seq_length).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - final_text_features (torch.Tensor): Final text feature representation from BiLSTM (last time step). Shape: (batch_size, hidden_dim * 2).
                - text_bilstm_block_outputs (List[torch.Tensor]): List of outputs from each BiLSTM block (full sequences).
        """
        # print("\nTextProcessor Forward Pass:") # Verbose
        # print(f"- input_ids shape: {input_ids.shape}") # Verbose
        # print(f"- attention_mask shape: {attention_mask.shape}") # Verbose
        try:
            # 1. RoBERTa Encoding (Eq 1)
            # roberta_outputs.last_hidden_state shape: (batch, seq_len, 768)
            roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = roberta_outputs.last_hidden_state
            # print(f"  - RoBERTa output shape: {sequence_output.shape}") # Verbose

            # 2. Feature Conversion (Eq 2)
            # text_features shape: (batch, seq_len, 512)
            text_features = self.text_fc(sequence_output)
            # print(f"  - Text FC output shape: {text_features.shape}") # Verbose

            # 3. Progressive BiLSTM (Methodology Sec III.D.3)
            # Prepare mask for BiLSTM attention (True for padding)
            # Input attention_mask is 1 for valid, 0 for padding. Need to invert.
            bilstm_mask = (attention_mask == 0) # True where attention_mask is 0 (padding)
            # print(f"  - BiLSTM mask shape (True for padding): {bilstm_mask.shape}") # Verbose

            # final_text_features shape: (batch, hidden_dim * 2 = 512) - last time step
            # text_bilstm_block_outputs: List of 3 tensors, each (batch, seq_len, 512)
            final_text_features, text_bilstm_block_outputs = self.text_bilstm(text_features, bilstm_mask)
            # print(f"  - Final BiLSTM output shape: {final_text_features.shape}") # Verbose
            # print(f"  - BiLSTM block outputs: {[o.shape for o in text_bilstm_block_outputs]}") # Verbose

            return final_text_features, text_bilstm_block_outputs

        except Exception as e:
            log_and_print(logger, f"Error in TextProcessor forward pass: {str(e)}", logging.ERROR)
            log_and_print(logger, traceback.format_exc(), logging.ERROR)
            # Return zero tensors of the expected shapes to avoid crashing downstream
            dummy_final = torch.zeros((input_ids.size(0), self.config.PROGRESSIVE_BILSTM_HIDDEN_DIM * 2), device=input_ids.device)
            dummy_blocks = [torch.zeros((input_ids.size(0), self.config.MAX_LEN, self.config.PROGRESSIVE_BILSTM_HIDDEN_DIM * 2), device=input_ids.device)] * 3 # Assuming 3 layers
            return dummy_final, dummy_blocks


class AudioProcessor(nn.Module):
    """
    Processes audio features (V1, V3, V4) using FC layers, attention, and Progressive BiLSTM.
    Returns final output and intermediate BiLSTM block outputs.
    """
    def __init__(self, config: Config):
        super(AudioProcessor, self).__init__()
        print("\n--- Initializing AudioProcessor ---")
        self.config = config
        hidden_dim = config.PROGRESSIVE_BILSTM_HIDDEN_DIM # 256
        dropout = config.DROPOUT_RATE # 0.3

        # V1 (Wav2Vec2 Embeddings) processing (Eq 7)
        self.v1_fc = nn.Sequential(
            nn.Linear(config.AUDIO_V1_DIM, 256), # 512 -> 256
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        print(f"- V1 FC layer created: {config.AUDIO_V1_DIM} -> 256")

        # V3 (MFCC) + V4 (Spectral) processing (Eq 10)
        v3_v4_input_dim = config.AUDIO_V3_DIM + config.AUDIO_V4_DIM # 25 + 25 = 50
        self.v34_fc = nn.Sequential(
            nn.Linear(v3_v4_input_dim, 128), # 50 -> 128
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        print(f"- V3+V4 FC layer created: {v3_v4_input_dim} -> 128")

        # Self-Attention on processed V1 features (Eq 8, 9) - Methodology uses 4 heads
        self.v1_attention = nn.MultiheadAttention(
            embed_dim=256, # Dimension of processed v1
            num_heads=4,   # Methodology specifies 4 heads here
            dropout=dropout,
            batch_first=True
        )
        print(f"- V1 Self-Attention created: Embed=256, Heads=4")

        # Combined features processing (Eq 11)
        # Input is concat(attended_v1 [256], processed_v34 [128]) = 384
        self.combined_fc = nn.Sequential(
            nn.Linear(256 + 128, config.AUDIO_PROJECTION_DIM), # 384 -> 256
            nn.LayerNorm(config.AUDIO_PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        print(f"- Combined Audio FC layer created: 384 -> {config.AUDIO_PROJECTION_DIM}")

        # Progressive BiLSTM for audio (Methodology Sec III.D.3)
        self.audio_bilstm = ProgressiveBiLSTM(
            input_size=config.AUDIO_PROJECTION_DIM, # 256
            hidden_size=hidden_dim, # 256
            num_layers=3,
            dropout=dropout,
            attention_heads=8 # Internal BiLSTM attention heads
        )
        print(f"- Audio ProgressiveBiLSTM created (Input: {config.AUDIO_PROJECTION_DIM})")
        print(f"--- Finished Initializing AudioProcessor ---")

    def forward(self, v1: torch.Tensor, v3: torch.Tensor, v4: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for audio processing.

        Args:
            v1 (torch.Tensor): Voice embeddings. Shape: (batch_size, 512).
            v3 (torch.Tensor): MFCC features. Shape: (batch_size, 25).
            v4 (torch.Tensor): Spectral features. Shape: (batch_size, 25).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - final_audio_features (torch.Tensor): Final audio feature representation from BiLSTM (last time step). Shape: (batch_size, hidden_dim * 2).
                - audio_bilstm_block_outputs (List[torch.Tensor]): List of outputs from each BiLSTM block (full sequences).
        """
        # print("\nAudioProcessor Forward Pass:") # Verbose
        # print(f"- v1 shape: {v1.shape}, v3 shape: {v3.shape}, v4 shape: {v4.shape}") # Verbose
        try:
            # 1. Process V1 (Eq 7)
            v1_features = self.v1_fc(v1) # Shape: (batch, 256)
            # print(f"  - Processed V1 shape: {v1_features.shape}") # Verbose

            # 2. Apply Self-Attention to V1 (Eq 8, 9)
            # Add sequence dimension for attention: (batch, 1, 256)
            v1_features_seq = v1_features.unsqueeze(1)
            # print(f"  - V1 Seq shape for Attn: {v1_features_seq.shape}") # Verbose
            v1_attended, _ = self.v1_attention(v1_features_seq, v1_features_seq, v1_features_seq)
            # Remove sequence dimension: (batch, 256)
            v1_attended = v1_attended.squeeze(1)
            # print(f"  - Attended V1 shape: {v1_attended.shape}") # Verbose

            # 3. Process V3 and V4 (Eq 10)
            v3_v4_combined = torch.cat([v3, v4], dim=-1) # Shape: (batch, 50)
            v34_features = self.v34_fc(v3_v4_combined) # Shape: (batch, 128)
            # print(f"  - Processed V3+V4 shape: {v34_features.shape}") # Verbose

            # 4. Combine and Fuse Attended V1 and Processed V34 (Eq 11)
            combined_audio = torch.cat([v1_attended, v34_features], dim=-1) # Shape: (batch, 384)
            fused_audio = self.combined_fc(combined_audio) # Shape: (batch, 256)
            # print(f"  - Fused Audio shape (Pre-BiLSTM): {fused_audio.shape}") # Verbose

            # 5. Progressive BiLSTM (Methodology Sec III.D.3)
            # Add sequence dimension of 1 as per methodology note D.2
            fused_audio_seq = fused_audio.unsqueeze(1) # Shape: (batch, 1, 256)
            # No mask needed for sequence length 1
            # final_audio_features shape: (batch, hidden_dim*2 = 512) - last time step
            # audio_bilstm_block_outputs: List of 3 tensors, each (batch, 1, 512)
            final_audio_features, audio_bilstm_block_outputs = self.audio_bilstm(fused_audio_seq, mask=None)
            # print(f"  - Final BiLSTM output shape: {final_audio_features.shape}") # Verbose
            # print(f"  - BiLSTM block outputs: {[o.shape for o in audio_bilstm_block_outputs]}") # Verbose

            return final_audio_features, audio_bilstm_block_outputs

        except Exception as e:
            log_and_print(logger, f"Error in AudioProcessor forward pass: {str(e)}", logging.ERROR)
            log_and_print(logger, traceback.format_exc(), logging.ERROR)
            # Return zero tensors of the expected shapes
            dummy_final = torch.zeros((v1.size(0), self.config.PROGRESSIVE_BILSTM_HIDDEN_DIM * 2), device=v1.device)
            dummy_blocks = [torch.zeros((v1.size(0), 1, self.config.PROGRESSIVE_BILSTM_HIDDEN_DIM * 2), device=v1.device)] * 3 # Assuming 3 layers
            return dummy_final, dummy_blocks


class VisualProcessor(nn.Module):
    """
    Processes visual features (A2) using FC layers, attention, and Progressive BiLSTM.
    Returns final output and intermediate BiLSTM block outputs.
    """
    def __init__(self, config: Config):
        super(VisualProcessor, self).__init__()
        print("\n--- Initializing VisualProcessor ---")
        self.config = config
        hidden_dim = config.PROGRESSIVE_BILSTM_HIDDEN_DIM # 256
        dropout = config.DROPOUT_RATE # 0.3

        # Multi-scale feature processing (Eq 3, 4)
        self.visual_fc1 = nn.Sequential(
            nn.Linear(config.VISUAL_A2_DIM, 384), # 512 -> 384
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        print(f"- Visual FC1 layer created: {config.VISUAL_A2_DIM} -> 384")

        self.visual_fc2 = nn.Sequential(
            nn.Linear(384, config.VISUAL_PROJECTION_DIM), # 384 -> 256
            nn.LayerNorm(config.VISUAL_PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        print(f"- Visual FC2 layer created: 384 -> {config.VISUAL_PROJECTION_DIM}")

        # Self-Attention on processed visual features (Eq 5, 6) - Methodology uses 4 heads
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.VISUAL_PROJECTION_DIM, # 256
            num_heads=4, # Methodology specifies 4 heads here
            dropout=dropout,
            batch_first=True
        )
        print(f"- Visual Self-Attention created: Embed=256, Heads=4")

        # Progressive BiLSTM for visual (Methodology Sec III.D.3)
        self.visual_bilstm = ProgressiveBiLSTM(
            input_size=config.VISUAL_PROJECTION_DIM, # 256
            hidden_size=hidden_dim, # 256
            num_layers=3,
            dropout=dropout,
            attention_heads=8 # Internal BiLSTM attention heads
        )
        print(f"- Visual ProgressiveBiLSTM created (Input: {config.VISUAL_PROJECTION_DIM})")
        print(f"--- Finished Initializing VisualProcessor ---")

    def forward(self, a2: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for visual processing.

        Args:
            a2 (torch.Tensor): Visual features from CNN-LSTM. Shape: (batch_size, 512).

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - final_visual_features (torch.Tensor): Final visual feature representation from BiLSTM (last time step). Shape: (batch_size, hidden_dim * 2).
                - visual_bilstm_block_outputs (List[torch.Tensor]): List of outputs from each BiLSTM block (full sequences).
        """
        # print("\nVisualProcessor Forward Pass:") # Verbose
        # print(f"- a2 shape: {a2.shape}") # Verbose
        try:
            # 1. Multi-scale Feature Processing (Eq 3, 4)
            visual_features_1 = self.visual_fc1(a2) # Shape: (batch, 384)
            visual_features_2 = self.visual_fc2(visual_features_1) # Shape: (batch, 256)
            # print(f"  - Processed Visual (FC2) shape: {visual_features_2.shape}") # Verbose

            # 2. Self-Attention (Eq 5, 6)
            # Add sequence dimension for attention: (batch, 1, 256)
            visual_features_seq = visual_features_2.unsqueeze(1)
            # print(f"  - Visual Seq shape for Attn: {visual_features_seq.shape}") # Verbose
            attended_features, _ = self.self_attention(
                visual_features_seq, visual_features_seq, visual_features_seq
            ) # Shape: (batch, 1, 256)
            # print(f"  - Attended Visual shape: {attended_features.shape}") # Verbose

            # 3. Progressive BiLSTM (Methodology Sec III.D.3)
            # Input is attended_features (batch, 1, 256) - as per methodology note D.2
            # No mask needed for sequence length 1
            # final_visual_features shape: (batch, hidden_dim*2 = 512) - last time step
            # visual_bilstm_block_outputs: List of 3 tensors, each (batch, 1, 512)
            final_visual_features, visual_bilstm_block_outputs = self.visual_bilstm(attended_features, mask=None)
            # print(f"  - Final BiLSTM output shape: {final_visual_features.shape}") # Verbose
            # print(f"  - BiLSTM block outputs: {[o.shape for o in visual_bilstm_block_outputs]}") # Verbose

            return final_visual_features, visual_bilstm_block_outputs

        except Exception as e:
            log_and_print(logger, f"Error in VisualProcessor forward pass: {str(e)}", logging.ERROR)
            log_and_print(logger, traceback.format_exc(), logging.ERROR)
            # Return zero tensors of the expected shapes
            dummy_final = torch.zeros((a2.size(0), self.config.PROGRESSIVE_BILSTM_HIDDEN_DIM * 2), device=a2.device)
            dummy_blocks = [torch.zeros((a2.size(0), 1, self.config.PROGRESSIVE_BILSTM_HIDDEN_DIM * 2), device=a2.device)] * 3 # Assuming 3 layers
            return dummy_final, dummy_blocks

# --- Test Function ---
def test_processors(config: Config, device: torch.device):
    """Test all processing modules with expected input/output shapes."""
    print("\n--- Starting Cell 3.2: Processors Test ---")
    try:
        # Create test inputs matching expected dimensions
        batch_size = config.BATCH_SIZE # 16
        seq_length = config.MAX_LEN # 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
        # Attention mask: 1 for valid tokens, 0 for padding
        attention_mask = torch.ones(batch_size, seq_length).to(device)
        if seq_length > 10: attention_mask[:, -10:] = 0 # Example padding

        v1 = torch.randn(batch_size, config.AUDIO_V1_DIM).to(device) # 512
        v3 = torch.randn(batch_size, config.AUDIO_V3_DIM).to(device) # 25
        v4 = torch.randn(batch_size, config.AUDIO_V4_DIM).to(device) # 25
        a2 = torch.randn(batch_size, config.VISUAL_A2_DIM).to(device) # 512

        log_and_print(logger, "Initializing processors for testing...")
        # Initialize processors
        text_processor = TextProcessor(config).to(device)
        audio_processor = AudioProcessor(config).to(device)
        visual_processor = VisualProcessor(config).to(device)
        log_and_print(logger, "Processors initialized.")

        # --- Test each processor ---
        log_and_print(logger, "\nTesting processor forward passes:")
        text_processor.eval()
        audio_processor.eval()
        visual_processor.eval()

        with torch.no_grad():
            # Test text processor
            print("\nTesting Text Processor...")
            text_out, text_blocks = text_processor(input_ids, attention_mask)
            print(f"- Text Processor Final Output shape: {text_out.shape}")
            expected_shape = (batch_size, config.PROGRESSIVE_BILSTM_HIDDEN_DIM * 2)
            assert text_out.shape == expected_shape, f"Text output shape mismatch! Expected {expected_shape}, Got {text_out.shape}"
            print("  (Final Shape OK)")
            assert isinstance(text_blocks, list) and len(text_blocks) == 3, "Text blocks should be a list of 3 tensors"
            assert text_blocks[0].shape == (batch_size, seq_length, expected_shape[1]), f"Text block 0 shape mismatch! Expected {(batch_size, seq_length, expected_shape[1])}, Got {text_blocks[0].shape}"
            print(f"- Text Processor Block Outputs: List of {len(text_blocks)} tensors, e.g., Block 0 shape {text_blocks[0].shape} (OK)")


            # Test audio processor
            print("\nTesting Audio Processor...")
            audio_out, audio_blocks = audio_processor(v1, v3, v4)
            print(f"- Audio Processor Final Output shape: {audio_out.shape}")
            assert audio_out.shape == expected_shape, f"Audio output shape mismatch! Expected {expected_shape}, Got {audio_out.shape}"
            print("  (Final Shape OK)")
            assert isinstance(audio_blocks, list) and len(audio_blocks) == 3, "Audio blocks should be a list of 3 tensors"
            assert audio_blocks[0].shape == (batch_size, 1, expected_shape[1]), f"Audio block 0 shape mismatch! Expected {(batch_size, 1, expected_shape[1])}, Got {audio_blocks[0].shape}" # Seq len is 1
            print(f"- Audio Processor Block Outputs: List of {len(audio_blocks)} tensors, e.g., Block 0 shape {audio_blocks[0].shape} (OK)")


            # Test visual processor
            print("\nTesting Visual Processor...")
            visual_out, visual_blocks = visual_processor(a2)
            print(f"- Visual Processor Final Output shape: {visual_out.shape}")
            assert visual_out.shape == expected_shape, f"Visual output shape mismatch! Expected {expected_shape}, Got {visual_out.shape}"
            print("  (Final Shape OK)")
            assert isinstance(visual_blocks, list) and len(visual_blocks) == 3, "Visual blocks should be a list of 3 tensors"
            assert visual_blocks[0].shape == (batch_size, 1, expected_shape[1]), f"Visual block 0 shape mismatch! Expected {(batch_size, 1, expected_shape[1])}, Got {visual_blocks[0].shape}" # Seq len is 1
            print(f"- Visual Processor Block Outputs: List of {len(visual_blocks)} tensors, e.g., Block 0 shape {visual_blocks[0].shape} (OK)")


        log_and_print(logger, "\nAll processors tested successfully!")
        print("\n--- Finished Cell 3.2 Test ---")
        print("-" * 50)
        return True

    except Exception as e:
        log_and_print(logger, f"Error in processor tests: {str(e)}", logging.ERROR)
        log_and_print(logger, traceback.format_exc(), logging.ERROR)
        print("\n--- Cell 3.2 Test Failed ---")
        print("-" * 50)
        return False

# --- Execution ---
# Ensure config, logger, log_and_print, device, ProgressiveBiLSTM are available
if 'config' in locals() and 'logger' in locals() and 'device' in locals() and 'ProgressiveBiLSTM' in locals():
    test_successful_processors = test_processors(config, device)
else:
    print("Error: Required variables/classes not available. Cannot execute Cell 3.2.")
    test_successful_processors = False

# Cell 3.3: Main Model Architecture (Including Counterfactual Components)
# =====================================
# Multimodal Emotion Recognition Model with Cross-Attention Fusion
# =====================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List # Added List
import traceback

# Assuming config, logger, log_and_print, device,
# TextProcessor, AudioProcessor, VisualProcessor, ProgressiveBiLSTM are available/defined
# CounterfactualGenerator and IntentionPredictor definitions should be here

# --- Definitions from original Cell 3.4 (ensure they are here) ---

class CounterfactualGenerator(nn.Module):
    """Generates counterfactual features using modality-specific layers and adaptive scaling."""
    # Note: Dimensions adjusted to work with standardized features (256-dim)
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        print("\n--- Initializing CounterfactualGenerator ---")
        self.num_emotions = None # Set via set_num_emotions
        self.hidden_dim = hidden_dim # Should be config.STANDARDIZED_FEATURE_DIM (256)

        # Modality-specific transformation layers
        self.feature_layers = nn.ModuleDict({
            'text': nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(), # GELU often works well in transformer-like blocks
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim) # Keep dim same
            ),
            'audio': nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            'visual': nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        })
        print(f"- CF Feature Layers created for text, audio, visual (Dim: {hidden_dim})")

        # Adaptive scaling parameters (one per modality)
        self.scales = nn.ParameterDict({
            'text': nn.Parameter(torch.tensor(0.0)), # Initialize near zero for small initial change
            'audio': nn.Parameter(torch.tensor(0.0)),
            'visual': nn.Parameter(torch.tensor(0.0))
        })
        print("- CF Adaptive Scaling parameters created.")

        # Feature fusion layer (combines intermediate CF features)
        # Input is concatenation of 3 * hidden_dim
        fusion_input_dim = hidden_dim * 3
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, hidden_dim), # Fuse back to hidden_dim
            nn.GELU(),
            nn.Dropout(dropout),
            # Removed extra linear layer from original code for simplicity
            # nn.Linear(hidden_dim, hidden_dim)
        )
        print(f"- CF Fusion layer created: Input={fusion_input_dim}, Output={hidden_dim}")
        print("--- Finished Initializing CounterfactualGenerator ---")


    def process_modality(self, features: torch.Tensor, modality: str) -> torch.Tensor:
        """Applies transformation and adaptive scaling to a single modality."""
        # print(f"  CF Gen - Processing {modality}: Input shape {features.shape}") # Verbose
        transformed = self.feature_layers[modality](features)
        # print(f"  CF Gen - Transformed {modality} shape: {transformed.shape}") # Verbose
        # Use sigmoid on scale parameter to keep it between 0 and 1
        scale = torch.sigmoid(self.scales[modality])
        # print(f"  CF Gen - Scale for {modality}: {scale.item():.4f}") # Verbose
        # Interpolate between original and transformed based on scale (Eq 21)
        intermediate_cf = scale * transformed + (1 - scale) * features
        # print(f"  CF Gen - Intermediate CF {modality} shape: {intermediate_cf.shape}") # Verbose
        return intermediate_cf

    def generate_counterfactuals(self, std_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Generates counterfactual standardized features for all modalities.

        Args:
            std_features (Dict[str, torch.Tensor]): Dictionary containing the original
                                                    standardized features ('text', 'audio', 'visual'),
                                                    each of shape (batch, hidden_dim=256).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the final counterfactual
                                     standardized features, each of shape (batch, hidden_dim=256).
        """
        # print("\n  CF Gen - Generating Counterfactuals...") # Verbose
        cf_features_intermediate = {}
        modalities_present = list(std_features.keys())

        # 1. Generate intermediate CF features per modality (Eq 20, 21)
        for modality in modalities_present:
            if modality in self.feature_layers: # Check if we have layers for this modality
                cf_features_intermediate[modality] = self.process_modality(std_features[modality], modality)
            else:
                cf_features_intermediate[modality] = std_features[modality] # Pass through if no layers defined

        # print(f"  CF Gen - Intermediate CF shapes: {[f.shape for f in cf_features_intermediate.values()]}") # Verbose

        # 2. Fuse intermediate CF features (Implicit step in methodology Fig 2 / text)
        # Ensure order for consistent concatenation
        ordered_intermediate_features = [cf_features_intermediate[m] for m in ['text', 'audio', 'visual'] if m in cf_features_intermediate]

        if len(ordered_intermediate_features) > 1: # Only fuse if multiple modalities exist
            fused_intermediate = torch.cat(ordered_intermediate_features, dim=-1)
            # print(f"  CF Gen - Concat Intermediate CF shape (Fusion input): {fused_intermediate.shape}") # Verbose
            fused_output = self.fusion(fused_intermediate) # Fuses back to hidden_dim (256)
            # print(f"  CF Gen - Fused Output shape: {fused_output.shape}") # Verbose
        elif len(ordered_intermediate_features) == 1:
             # If only one modality, fusion output is just that modality's intermediate feature
             fused_output = ordered_intermediate_features[0]
             # print(f"  CF Gen - Skipping fusion for single modality.") # Verbose
        else:
             log_and_print(logger, "Warning: No features found to fuse in CounterfactualGenerator.", logging.WARNING)
             return std_features # Return original if no features

        # 3. Refine final CF features using fused info (Optional step, adds complexity, let's match simpler Eq 20/21 first)
        # The methodology text implies Tm and the adaptive scaling (Eq 20/21) produce the final CF features.
        # The diagram shows fusion, but the text description is simpler. Let's stick to the simpler text version first.
        # The intermediate features ARE the final CF features based on Eq 21.
        final_cf_features = cf_features_intermediate

        # --- Optional Refinement Step (Closer to original code's potential intent, but not Eq 21) ---
        # final_cf_features = {}
        # for modality in modalities_present:
        #     if modality in cf_features_intermediate:
        #         # Blend fused info back into each modality's intermediate CF
        #         scale = torch.sigmoid(self.scales[modality]) # Reuse scale? Or new scale? Let's reuse.
        #         final_cf_features[modality] = scale * fused_output + (1 - scale) * cf_features_intermediate[modality]
        #     else:
        #         final_cf_features[modality] = std_features[modality] # Should not happen if modalities_present is correct
        # print(f"  CF Gen - Final Refined CF shapes: {[f.shape for f in final_cf_features.values()]}") # Verbose
        # --- End Optional Refinement ---


        # 4. Final Stability Check (Clamp values)
        for modality in final_cf_features:
            # Clamp values to prevent potential instability in loss calculations
            final_cf_features[modality] = torch.clamp(final_cf_features[modality], -10.0, 10.0) # Increased range slightly

        # print("  CF Gen - Counterfactual Generation Complete.") # Verbose
        return final_cf_features

    def set_num_emotions(self, num_emotions: int):
        """Set number of emotions (might be used by internal layers if designed differently)."""
        self.num_emotions = num_emotions
        # print(f"CounterfactualGenerator: num_emotions set to {num_emotions}") # Verbose

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass alias for generate_counterfactuals."""
        return self.generate_counterfactuals(x)

class IntentionPredictor(nn.Module):
    """
    Predicts emotion based on the difference between original and counterfactual features.
    Matches Methodology Sec III.E.2 and Eq 22.
    """
    def __init__(self, feature_dim: int, num_emotions: int, dropout: float = 0.3):
        """
        Args:
            feature_dim (int): Dimension of the input features (output of Feature Stack MLP, e.g., 256).
            num_emotions (int): Number of output emotion classes.
            dropout (float): Dropout rate.
        """
        super().__init__()
        print("\n--- Initializing IntentionPredictor ---")
        self.feature_dim = feature_dim
        self.num_emotions = num_emotions

        # MLP structure based on Methodology Figure/Text (Page 8)
        # Input is concat(h_orig, h_cf - h_orig) -> 2 * feature_dim
        input_concat_dim = feature_dim * 2

        # Define the MLP layers
        self.mlp = nn.Sequential(
            # Layer 1: 512 -> 256 (as per figure h_mlp1)
            nn.Linear(input_concat_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Layer 2: 256 -> 128 (as per figure h_mlp2)
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
             # Layer 3: 128 -> num_emotions (Classifier)
            nn.Linear(feature_dim // 2, num_emotions)
        )
        print(f"- Intention MLP created: Input={input_concat_dim}, Output={num_emotions}")
        print(f"--- Finished Initializing IntentionPredictor ---")

    def forward(self, original_features: torch.Tensor, counterfactual_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for intention prediction.

        Args:
            original_features (torch.Tensor): Fused features from original input. Shape: (batch, feature_dim).
            counterfactual_features (torch.Tensor): Fused features from counterfactual input. Shape: (batch, feature_dim).

        Returns:
            torch.Tensor: Logits for intention prediction. Shape: (batch, num_emotions).
        """
        # print("\nIntentionPredictor Forward Pass:") # Verbose
        # print(f"- Original Features shape: {original_features.shape}") # Verbose
        # print(f"- Counterfactual Features shape: {counterfactual_features.shape}") # Verbose

        # 1. Calculate feature difference (h_diff_only = h_cf - h_orig)
        feature_diff_only = counterfactual_features - original_features
        # print(f"  - Feature Difference shape: {feature_diff_only.shape}") # Verbose

        # 2. Concatenate (h_diff = concat(h_orig, h_diff_only)) (Eq 22)
        combined_diff_features = torch.cat([original_features, feature_diff_only], dim=-1)
        # print(f"  - Concatenated Diff Features shape: {combined_diff_features.shape}") # Verbose

        # 3. Apply MLP (Eq 22)
        intention_logits = self.mlp(combined_diff_features)
        # print(f"  - Intention Logits shape: {intention_logits.shape}") # Verbose

        return intention_logits


# --- Main Model Definition ---

class MultimodalEmotionRecognition(nn.Module):
    """
    Main Multimodal Emotion Recognition model implementing the methodology's
    cross-attention fusion and three-phase training structure.
    """
    def __init__(self, config: Config, num_emotions: int):
        """
        Initialize Multimodal Emotion Recognition Model.

        Args:
            config (Config): Configuration object.
            num_emotions (int): Number of emotion classes.
        """
        super(MultimodalEmotionRecognition, self).__init__()
        print("\n--- Initializing MultimodalEmotionRecognition Model ---")
        self.config = config
        self.num_emotions = num_emotions
        self.hidden_dim = config.STANDARDIZED_FEATURE_DIM # 256 (Target dim after standardization)
        self.bilstm_output_dim = config.PROGRESSIVE_BILSTM_HIDDEN_DIM * 2 # 512
        self.cross_attn_dim = self.hidden_dim # Use 256 for cross-attention Q,K,V base
        # Corrected enhanced dim based on Eq: Fm + A_other1->m + A_other2->m = 256 + 256 + 256 = 768
        self.enhanced_feature_dim = self.cross_attn_dim * 3
        dropout = config.DROPOUT_RATE

        log_and_print(logger, f"Initializing Multimodal Model for {num_emotions} emotions.")
        print(f"- Target Standardized Dim (hidden_dim): {self.hidden_dim}")
        print(f"- BiLSTM Output Dim: {self.bilstm_output_dim}")
        print(f"- Cross-Attention Base Dim: {self.cross_attn_dim}")
        print(f"- Enhanced Feature Dim (Pre-Standardization): {self.enhanced_feature_dim}")

        # --- 1. Modality-Specific Processors ---
        self.text_processor = TextProcessor(config)
        self.audio_processor = AudioProcessor(config)
        self.visual_processor = VisualProcessor(config)
        print("\n- Modality Processors Initialized.")

        # --- 2. Projection Layer (Post-BiLSTM, Pre-Cross-Attention) ---
        # Project 512-dim BiLSTM outputs to 256-dim for cross-attention base
        self.projection_layers = nn.ModuleDict({
            'text': nn.Linear(self.bilstm_output_dim, self.cross_attn_dim), # 512 -> 256
            'audio': nn.Linear(self.bilstm_output_dim, self.cross_attn_dim), # 512 -> 256
            'visual': nn.Linear(self.bilstm_output_dim, self.cross_attn_dim) # 512 -> 256
        })
        print(f"- Projection Layers (BiLSTM Output -> CrossAttn Input) Initialized: {self.bilstm_output_dim} -> {self.cross_attn_dim}")

        # --- 3. Cross-Attention Mechanism (Methodology Sec III.D.4.a) ---
        num_attn_heads = config.CROSS_ATTENTION_HEADS # 8
        self.cross_attention = nn.ModuleDict()
        modalities = ['text', 'audio', 'visual']
        for m1 in modalities:
            for m2 in modalities:
                if m1 != m2:
                    # Attention where m1 attends to m2 (m1 is Query, m2 is Key/Value)
                    attn_key = f'{m1}_attends_{m2}'
                    self.cross_attention[attn_key] = nn.MultiheadAttention(
                        embed_dim=self.cross_attn_dim, # 256
                        num_heads=num_attn_heads,
                        dropout=dropout,
                        batch_first=True # Expect (batch, seq, feature)
                    )
        print(f"- Cross-Attention Layers Initialized ({len(self.cross_attention)} pairs, Heads={num_attn_heads}, Dim={self.cross_attn_dim})")

        # --- 4. Standardization Layer (Methodology Sec III.D.4.b) ---
        # Takes 768-dim Enhanced features -> 256-dim Standardized features
        self.standardization_layers = nn.ModuleDict({
            'text': nn.Sequential(
                nn.LayerNorm(self.enhanced_feature_dim), # Input 768
                nn.Linear(self.enhanced_feature_dim, self.hidden_dim), # 768 -> 256
                nn.LayerNorm(self.hidden_dim), # Output 256
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            'audio': nn.Sequential(
                nn.LayerNorm(self.enhanced_feature_dim),
                nn.Linear(self.enhanced_feature_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            'visual': nn.Sequential(
                nn.LayerNorm(self.enhanced_feature_dim),
                nn.Linear(self.enhanced_feature_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        })
        print(f"- Standardization Layers Initialized: {self.enhanced_feature_dim} -> {self.hidden_dim}")

        # --- 5. Feature Stack MLP (Methodology Sec III.D.4.c) ---
        mlp_dims = config.FEATURE_STACK_MLP_DIMS # [768, 1024, 512, 256]
        mlp_layers = []
        input_dim = mlp_dims[0] # 768 (Concat of 3 * 256 standardized features)
        for output_dim in mlp_dims[1:]:
            mlp_layers.append(nn.Linear(input_dim, output_dim))
            mlp_layers.append(nn.LayerNorm(output_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(config.FEATURE_STACK_DROPOUT)) # Use specific dropout
            input_dim = output_dim
        # Remove final LayerNorm/ReLU/Dropout after the last linear layer
        self.feature_stack = nn.Sequential(*mlp_layers[:-3])
        self.feature_stack_output_dim = mlp_dims[-1] # 256
        print(f"- Feature Stack MLP Initialized: Input={mlp_dims[0]}, Output={self.feature_stack_output_dim}")

        # --- 6. Counterfactual Generator ---
        self.cf_generator = CounterfactualGenerator(
            hidden_dim=self.hidden_dim, # Operates on 256-dim standardized features
            dropout=dropout
        )
        self.cf_generator.set_num_emotions(num_emotions)
        print("- Counterfactual Generator Initialized.")

        # --- 7. Intention Predictor ---
        self.intention_predictor = IntentionPredictor(
            feature_dim=self.feature_stack_output_dim, # Operates on 256-dim MLP output
            num_emotions=num_emotions,
            dropout=dropout # Pass dropout from config
        )
        print("- Intention Predictor Initialized.")

        # --- 8. Phase Classifiers ---
        self.phase_classifiers = nn.ModuleDict({
            'phase1': nn.Linear(self.feature_stack_output_dim, num_emotions),
            'phase2': nn.Linear(self.feature_stack_output_dim, num_emotions),
            'phase3': nn.Linear(self.feature_stack_output_dim, num_emotions)
        })
        print(f"- Phase Classifiers Initialized (Input: {self.feature_stack_output_dim}, Output: {num_emotions})")

        log_and_print(logger, "MultimodalEmotionRecognition Model components initialized successfully.")
        print("--- Finished Initializing MultimodalEmotionRecognition Model ---")

    # --- Helper Methods (Internal Flow) ---
    def _apply_cross_attention(self, projected_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Applies the all-pairs cross-attention mechanism."""
        # print("  - Applying Cross-Attention...") # Verbose
        modalities = list(projected_features.keys())
        attention_results = {m: [] for m in modalities} # Store results where m is the Query

        # Add sequence dimension for MultiheadAttention (batch, 1, feature)
        seq_features = {m: feat.unsqueeze(1) for m, feat in projected_features.items()}
        # print(f"    - Input shapes to attention: {[f.shape for f in seq_features.values()]}") # Verbose

        for m1 in modalities: # m1 is the Query modality
            for m2 in modalities: # m2 is the Key/Value modality
                if m1 != m2:
                    attn_key = f'{m1}_attends_{m2}'
                    try:
                        # Query=m1, Key=m2, Value=m2
                        attn_output, _ = self.cross_attention[attn_key](
                            query=seq_features[m1],
                            key=seq_features[m2],
                            value=seq_features[m2],
                            key_padding_mask=None # No padding for seq_len=1
                        )
                        # Result has shape (batch, 1, cross_attn_dim), squeeze seq dim
                        attention_results[m1].append(attn_output.squeeze(1))
                        # print(f"    - Ran {attn_key}, output shape: {attn_output.squeeze(1).shape}") # Verbose
                    except Exception as e:
                         log_and_print(logger, f"Error during cross-attention {attn_key}: {e}", logging.ERROR)
                         attention_results[m1].append(torch.zeros_like(projected_features[m1]))

        # Create Enhanced features (Eq on Page 7: Enhanced_m = Concat(Fm, A_other1->m, A_other2->m))
        enhanced_features = {}
        for m in modalities:
            original_feat = projected_features[m] # Shape: (batch, 256)
            attn_res = attention_results[m] # List of 2 tensors, each (batch, 256)
            if len(attn_res) == 2: # Ensure we got both attention results
                enhanced_features[m] = torch.cat([original_feat] + attn_res, dim=-1) # Shape: (batch, 256 + 256 + 256 = 768)
            else:
                log_and_print(logger, f"Warning: Missing attention results for modality {m}. Using zeros.", logging.WARNING)
                zeros = torch.zeros_like(original_feat)
                enhanced_features[m] = torch.cat([original_feat, zeros, zeros], dim=-1) # Fallback
            # print(f"    - Enhanced {m} shape: {enhanced_features[m].shape}") # Verbose

        # print("  - Cross-Attention complete.") # Verbose
        return enhanced_features

    def _standardize_features(self, enhanced_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Applies the standardization layer to enhanced features."""
        # print("  - Applying Standardization...") # Verbose
        std_features = {}
        for m, feat in enhanced_features.items():
            std_features[m] = self.standardization_layers[m](feat) # 768 -> 256
            # print(f"    - Standardized {m} shape: {std_features[m].shape}") # Verbose
        # print("  - Standardization complete.") # Verbose
        return std_features

    def _fuse_features(self, std_features: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Concatenates standardized features and passes through MLP Feature Stack."""
        # print("  - Applying Feature Stack MLP...") # Verbose
        modalities_present = list(std_features.keys())
        if not modalities_present:
             log_and_print(logger, "Warning: No standardized features to fuse.", logging.WARNING)
             return None

        # Ensure order for consistent concatenation
        ordered_features = []
        for m in ['text', 'audio', 'visual']:
             if m in std_features:
                 ordered_features.append(std_features[m])
             else:
                 # If a modality is missing, maybe add zeros? Or error? Let's add zeros.
                 log_and_print(logger, f"Warning: Standardized feature for '{m}' missing during fusion. Using zeros.", logging.WARNING)
                 # Need device info if creating zeros
                 device = list(std_features.values())[0].device
                 batch_size = list(std_features.values())[0].size(0) # Get batch size from another feature
                 zeros = torch.zeros((batch_size, self.hidden_dim), device=device)
                 ordered_features.append(zeros)

        if len(ordered_features) != 3:
             log_and_print(logger, f"Error: Incorrect number of features ({len(ordered_features)}) before MLP stack concatenation.", logging.ERROR)
             return None

        combined = torch.cat(ordered_features, dim=-1) # Should be 3 * 256 = 768
        # print(f"    - Concatenated shape (MLP input): {combined.shape}") # Verbose

        # Process through MLP stack (768 -> ... -> 256)
        processed = self.feature_stack(combined)
        # print(f"    - Processed shape (MLP output): {processed.shape}") # Verbose
        # print("  - Feature Stack MLP complete.") # Verbose
        return processed

    # --- Forward Methods for Phases ---

    def forward_shared_steps(self, batch: Dict[str, torch.Tensor]) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[str, List[torch.Tensor]]]]:
        """
        Performs the initial steps common to all phases.
        Returns standardized features and intermediate BiLSTM block outputs.
        """
        # print("Running forward_shared_steps...") # Verbose
        bilstm_block_outputs = {'text': None, 'audio': None, 'visual': None} # Store intermediate outputs

        # 1. Modality Processors -> BiLSTM Output (512-dim final, List[3 x (B, S, 512)] blocks)
        try:
            text_bilstm_out, text_blocks = self.text_processor(batch['input_ids'], batch['attention_mask'])
            audio_bilstm_out, audio_blocks = self.audio_processor(batch['v1'], batch['v3'], batch['v4'])
            visual_bilstm_out, visual_blocks = self.visual_processor(batch['a2'])
            # print(f"  - BiLSTM Final Outputs Shapes: T={text_bilstm_out.shape}, A={audio_bilstm_out.shape}, V={visual_bilstm_out.shape}") # Verbose
            bilstm_block_outputs['text'] = text_blocks
            bilstm_block_outputs['audio'] = audio_blocks
            bilstm_block_outputs['visual'] = visual_blocks
        except Exception as e:
             log_and_print(logger, f"Error in modality processors during forward pass: {e}", logging.ERROR)
             return None, None

        # 2. Projection (512 -> 256) - Applied to the *final* BiLSTM output (last time step)
        projected_features = {
            'text': self.projection_layers['text'](text_bilstm_out),
            'audio': self.projection_layers['audio'](audio_bilstm_out),
            'visual': self.projection_layers['visual'](visual_bilstm_out)
        }
        # print(f"  - Projected Shapes: {[f.shape for f in projected_features.values()]}") # Verbose

        # 3. Cross-Attention -> Enhanced Features (768-dim)
        enhanced_features = self._apply_cross_attention(projected_features)

        # 4. Standardization (768 -> 256)
        std_features = self._standardize_features(enhanced_features)

        # Return standardized features (for fusion/CF) and the intermediate BiLSTM block outputs (for t-SNE)
        return std_features, bilstm_block_outputs

    def forward_phase1(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # print("\n--- Running Forward Phase 1 ---") # Verbose
        std_features, bilstm_block_outputs = self.forward_shared_steps(batch)

        if std_features is None:
             # Need device info if creating zeros
             device = batch.get('input_ids', batch.get('v1')).device
             batch_size = batch.get('input_ids', batch.get('v1')).size(0)
             return {'final_logits': torch.zeros((batch_size, self.num_emotions), device=device)}

        # 5. Feature Stack MLP (Fusion)
        processed_orig = self._fuse_features(std_features)
        if processed_orig is None:
             device = batch.get('input_ids', batch.get('v1')).device
             batch_size = batch.get('input_ids', batch.get('v1')).size(0)
             return {'final_logits': torch.zeros((batch_size, self.num_emotions), device=device)}

        # 6. Classifier
        logits = self.phase_classifiers['phase1'](processed_orig)
        # print(f"- Phase 1 Logits shape: {logits.shape}") # Verbose

        return {
            'final_logits': logits,
            'processed_features': processed_orig,
            'std_features': std_features,
            'bilstm_block_outputs': bilstm_block_outputs # Pass through BiLSTM block outputs
        }

    def forward_phase2(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # print("\n--- Running Forward Phase 2 ---") # Verbose
        std_features, bilstm_block_outputs = self.forward_shared_steps(batch)

        if std_features is None:
             device = batch.get('input_ids', batch.get('v1')).device
             batch_size = batch.get('input_ids', batch.get('v1')).size(0)
             return {'final_logits': torch.zeros((batch_size, self.num_emotions), device=device)}

        # 5. Feature Stack MLP (Fusion) - Original Features
        processed_orig = self._fuse_features(std_features)
        if processed_orig is None:
             device = batch.get('input_ids', batch.get('v1')).device
             batch_size = batch.get('input_ids', batch.get('v1')).size(0)
             return {'final_logits': torch.zeros((batch_size, self.num_emotions), device=device)}

        # 6. Classifier (Original Features)
        logits = self.phase_classifiers['phase2'](processed_orig)
        # print(f"- Phase 2 Logits shape: {logits.shape}") # Verbose

        # --- Counterfactual Path ---
        # 7. Generate Counterfactual Standardized Features
        cf_std_features = self.cf_generator(std_features)
        # print(f"  - CF Std Shapes: {[f.shape for f in cf_std_features.values()]}") # Verbose

        # 8. Feature Stack MLP (Fusion) - Counterfactual Features
        processed_cf = self._fuse_features(cf_std_features)
        if processed_cf is None:
             log_and_print(logger, "Warning: Fusion failed for counterfactual features in Phase 2.", logging.WARNING)
             processed_cf = torch.zeros_like(processed_orig)

        # print(f"  - Processed CF shape: {processed_cf.shape}") # Verbose

        return {
            'final_logits': logits,
            'processed_features': processed_orig,
            'counterfactual_features': processed_cf,
            'std_features': std_features,
            'cf_std_features': cf_std_features,
            'bilstm_block_outputs': bilstm_block_outputs # Pass through BiLSTM block outputs
        }

    def forward_phase3(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # print("\n--- Running Forward Phase 3 ---") # Verbose
        phase2_outputs = self.forward_phase2(batch) # Reuse phase 2 logic to get all features

        # Check for errors from phase 2 forward pass
        if phase2_outputs.get('processed_features') is None:
             device = batch.get('input_ids', batch.get('v1')).device
             batch_size = batch.get('input_ids', batch.get('v1')).size(0)
             return {
                 'final_logits': torch.zeros((batch_size, self.num_emotions), device=device),
                 'intention_logits': torch.zeros((batch_size, self.num_emotions), device=device)
             }

        processed_orig = phase2_outputs['processed_features']
        processed_cf = phase2_outputs['counterfactual_features']

        # 1. Classifier (Original Features) - Use Phase 3 classifier
        logits = self.phase_classifiers['phase3'](processed_orig)
        # print(f"- Phase 3 Logits shape: {logits.shape}") # Verbose

        # 2. Intention Predictor
        intention_logits = self.intention_predictor(processed_orig, processed_cf)
        # print(f"- Intention Logits shape: {intention_logits.shape}") # Verbose

        # Return all necessary outputs, including those from phase 2 dict
        return {
            **phase2_outputs, # Include all outputs from phase 2 call
            'final_logits': logits, # Overwrite with phase 3 logits
            'intention_logits': intention_logits,
        }

    def forward(self, batch: Dict[str, torch.Tensor], phase: int = 1) -> Dict[str, torch.Tensor]:
        """Main forward pass dispatcher."""
        if phase == 1:
            return self.forward_phase1(batch)
        elif phase == 2:
            return self.forward_phase2(batch)
        elif phase == 3:
            return self.forward_phase3(batch)
        else:
            log_and_print(logger, f"Error: Invalid phase number {phase} requested.", logging.ERROR)
            device = batch.get('input_ids', batch.get('v1')).device
            batch_size = batch.get('input_ids', batch.get('v1')).size(0)
            # Return minimal dict on error
            return {'final_logits': torch.zeros((batch_size, self.num_emotions), device=device)}

    def set_num_emotions(self, num_emotions: int):
        """Update model's final layers for a specific number of emotions."""
        # This method might be called after __init__ if num_emotions wasn't known initially
        if hasattr(self, 'phase_classifiers'): # Check if layers are initialized
             print(f"\n--- Updating Model for {num_emotions} Emotions ---")
             self.num_emotions = num_emotions
             current_device = next(self.parameters()).device

             classifier_input_dim = self.feature_stack_output_dim
             for phase_name in ['phase1', 'phase2', 'phase3']:
                 self.phase_classifiers[phase_name] = nn.Linear(
                     classifier_input_dim, num_emotions
                 ).to(current_device)
             print(f"- Phase classifiers updated (Input: {classifier_input_dim}, Output: {num_emotions})")

             intention_input_dim = self.feature_stack_output_dim
             self.intention_predictor = IntentionPredictor(
                 feature_dim=intention_input_dim,
                 num_emotions=num_emotions,
                 dropout=self.config.DROPOUT_RATE # Ensure dropout is passed
             ).to(current_device)
             print(f"- Intention predictor updated (Input Dim: {intention_input_dim}, Output: {num_emotions})")

             if hasattr(self.cf_generator, 'set_num_emotions'):
                  self.cf_generator.set_num_emotions(num_emotions)
                  print("- CF Generator num_emotions updated.")

             log_and_print(logger, f"Model updated successfully for {num_emotions} emotions.")
             print("--- Finished Updating Model ---")
        else:
             # This case might happen if called before __init__ finishes, though unlikely here.
             log_and_print(logger, "Warning: Attempted to set num_emotions before model fully initialized.", logging.WARNING)
             self.num_emotions = num_emotions # Store it for __init__


# --- Test Function ---
def test_model_architecture(config: Config, device: torch.device, num_emotions: int):
    """Test the main model architecture for all phases."""
    print("\n--- Starting Cell 3.3: Main Model Architecture Test ---")
    # Ensure CounterfactualGenerator and IntentionPredictor are defined before this call
    if 'CounterfactualGenerator' not in globals() or 'IntentionPredictor' not in globals():
         print("Error: CounterfactualGenerator or IntentionPredictor class not defined before testing.")
         return False
    try:
        batch_size = config.BATCH_SIZE # 16
        seq_length = config.MAX_LEN # 128
        log_and_print(logger, f"Testing model architecture with Batch={batch_size}, NumEmotions={num_emotions}")

        # Create a dummy batch
        test_batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)).to(device),
            'attention_mask': torch.ones(batch_size, seq_length).to(device),
            'v1': torch.randn(batch_size, config.AUDIO_V1_DIM).to(device),
            'v3': torch.randn(batch_size, config.AUDIO_V3_DIM).to(device),
            'v4': torch.randn(batch_size, config.AUDIO_V4_DIM).to(device),
            'a2': torch.randn(batch_size, config.VISUAL_A2_DIM).to(device),
            'label': torch.randint(0, num_emotions, (batch_size,)).to(device)
        }
        print("\nDummy Batch Created:")
        for k, v in test_batch.items(): print(f"- {k}: {v.shape}")

        # Initialize model
        model = MultimodalEmotionRecognition(config=config, num_emotions=num_emotions).to(device)
        model.eval()

        # --- Test each phase ---
        all_phases_passed = True
        for phase in [1, 2, 3]:
            print(f"\n--- Testing Forward Phase {phase} ---")
            try:
                with torch.no_grad():
                    outputs = model(test_batch, phase=phase)

                print(f"Phase {phase} Output Keys: {list(outputs.keys())}")

                # Check essential output shapes
                assert 'final_logits' in outputs, f"Phase {phase} missing 'final_logits'"
                assert outputs['final_logits'].shape == (batch_size, num_emotions), \
                    f"Phase {phase} 'final_logits' shape mismatch! Expected {(batch_size, num_emotions)}, Got {outputs['final_logits'].shape}"
                print(f"- final_logits shape: {outputs['final_logits'].shape} (OK)")

                assert 'processed_features' in outputs, f"Phase {phase} missing 'processed_features'"
                assert outputs['processed_features'].shape == (batch_size, config.FEATURE_STACK_MLP_DIMS[-1]), \
                    f"Phase {phase} 'processed_features' shape mismatch! Expected {(batch_size, config.FEATURE_STACK_MLP_DIMS[-1])}, Got {outputs['processed_features'].shape}"
                print(f"- processed_features shape: {outputs['processed_features'].shape} (OK)")

                assert 'std_features' in outputs and isinstance(outputs['std_features'], dict), f"Phase {phase} missing 'std_features' dict"
                print("- std_features keys:", list(outputs['std_features'].keys()))
                for m, feat in outputs['std_features'].items():
                     assert feat.shape == (batch_size, config.STANDARDIZED_FEATURE_DIM), \
                        f"Phase {phase} std_features['{m}'] shape mismatch! Expected {(batch_size, config.STANDARDIZED_FEATURE_DIM)}, Got {feat.shape}"
                print(f"- std_features shapes: OK")

                # Check BiLSTM block outputs
                assert 'bilstm_block_outputs' in outputs and isinstance(outputs['bilstm_block_outputs'], dict), f"Phase {phase} missing 'bilstm_block_outputs' dict"
                print("- bilstm_block_outputs keys:", list(outputs['bilstm_block_outputs'].keys()))
                for mod, blocks in outputs['bilstm_block_outputs'].items():
                     assert isinstance(blocks, list) and len(blocks) == 3, f"Expected 3 blocks for {mod}, got {len(blocks)}"
                     expected_block_shape_part = (batch_size, -1, config.PROGRESSIVE_BILSTM_HIDDEN_DIM * 2) # Seq len varies
                     # Check shape of first block (adjust seq len check based on modality)
                     seq_len_mod = seq_length if mod == 'text' else 1
                     assert blocks[0].shape == (batch_size, seq_len_mod, expected_block_shape_part[2]), \
                         f"Phase {phase} bilstm_block_outputs['{mod}'][0] shape mismatch! Expected {(batch_size, seq_len_mod, expected_block_shape_part[2])}, Got {blocks[0].shape}"
                print(f"- bilstm_block_outputs shapes: OK")


                if phase >= 2:
                    assert 'counterfactual_features' in outputs, f"Phase {phase} missing 'counterfactual_features'"
                    assert outputs['counterfactual_features'].shape == (batch_size, config.FEATURE_STACK_MLP_DIMS[-1]), \
                        f"Phase {phase} 'counterfactual_features' shape mismatch! Expected {(batch_size, config.FEATURE_STACK_MLP_DIMS[-1])}, Got {outputs['counterfactual_features'].shape}"
                    print(f"- counterfactual_features shape: {outputs['counterfactual_features'].shape} (OK)")

                    assert 'cf_std_features' in outputs and isinstance(outputs['cf_std_features'], dict), f"Phase {phase} missing 'cf_std_features' dict"
                    print("- cf_std_features keys:", list(outputs['cf_std_features'].keys()))
                    for m, feat in outputs['cf_std_features'].items():
                         assert feat.shape == (batch_size, config.STANDARDIZED_FEATURE_DIM), \
                            f"Phase {phase} cf_std_features['{m}'] shape mismatch! Expected {(batch_size, config.STANDARDIZED_FEATURE_DIM)}, Got {feat.shape}"
                    print(f"- cf_std_features shapes: OK")


                if phase == 3:
                    assert 'intention_logits' in outputs, f"Phase {phase} missing 'intention_logits'"
                    assert outputs['intention_logits'].shape == (batch_size, num_emotions), \
                        f"Phase {phase} 'intention_logits' shape mismatch! Expected {(batch_size, num_emotions)}, Got {outputs['intention_logits'].shape}"
                    print(f"- intention_logits shape: {outputs['intention_logits'].shape} (OK)")

                print(f"--- Phase {phase} Test Passed ---")

            except Exception as phase_e:
                log_and_print(logger, f"Error during Phase {phase} test: {phase_e}", logging.ERROR)
                log_and_print(logger, traceback.format_exc(), logging.ERROR)
                all_phases_passed = False
                print(f"--- Phase {phase} Test Failed ---")

        if all_phases_passed:
            log_and_print(logger, "\nModel architecture test completed successfully for all phases!")
            print("\n--- Finished Cell 3.3 Test ---")
            print("-" * 50)
            return True
        else:
            log_and_print(logger, "\nModel architecture test failed for one or more phases.", logging.ERROR)
            print("\n--- Cell 3.3 Test Failed ---")
            print("-" * 50)
            return False

    except Exception as e:
        log_and_print(logger, f"Error in model architecture test setup: {str(e)}", logging.ERROR)
        log_and_print(logger, traceback.format_exc(), logging.ERROR)
        print("\n--- Cell 3.3 Test Failed ---")
        print("-" * 50)
        return False

# --- Execution ---
# Ensure config, logger, device, and processor classes are available
# Use config.NUM_EMOTIONS which should have been set in Cell 2.2
if 'config' in locals() and hasattr(config, 'NUM_EMOTIONS') and config.NUM_EMOTIONS and config.NUM_EMOTIONS > 0 \
    and 'logger' in locals() and 'device' in locals() \
    and 'TextProcessor' in locals() and 'AudioProcessor' in locals() and 'VisualProcessor' in locals() \
    and 'CounterfactualGenerator' in locals() and 'IntentionPredictor' in locals() \
    and 'ProgressiveBiLSTM' in locals():

    print(f"Executing test with NUM_EMOTIONS from config: {config.NUM_EMOTIONS}")
    # Assign the model instance to a variable for potential use in later cells
    # Re-initialize here for testing, the actual training uses initialize_model in PhaseTrainer
    model_instance_test = MultimodalEmotionRecognition(config=config, num_emotions=config.NUM_EMOTIONS).to(device)
    test_successful_model = test_model_architecture(config, device, config.NUM_EMOTIONS)
    # Keep the model instance from training available if it exists
    if 'model_instance' not in locals(): model_instance = None

else:
    print("Error: Required variables/classes not available or config.NUM_EMOTIONS not set. Cannot execute Cell 3.3.")
    # Print status of required components for debugging
    print(f"config exists: {'config' in locals()}")
    if 'config' in locals():
        print(f"config.NUM_EMOTIONS exists and > 0: {hasattr(config, 'NUM_EMOTIONS') and config.NUM_EMOTIONS and config.NUM_EMOTIONS > 0}")
        if hasattr(config, 'NUM_EMOTIONS'): print(f"config.NUM_EMOTIONS value: {config.NUM_EMOTIONS}")
    print(f"logger exists: {'logger' in locals()}")
    print(f"device exists: {'device' in locals()}")
    print(f"TextProcessor exists: {'TextProcessor' in locals()}")
    print(f"AudioProcessor exists: {'AudioProcessor' in locals()}")
    print(f"VisualProcessor exists: {'VisualProcessor' in locals()}")
    print(f"CounterfactualGenerator exists: {'CounterfactualGenerator' in locals()}")
    print(f"IntentionPredictor exists: {'IntentionPredictor' in locals()}")
    print(f"ProgressiveBiLSTM exists: {'ProgressiveBiLSTM' in locals()}")

    test_successful_model = False
    model_instance = None # Ensure model_instance is None if setup fails
    model_instance_test = None

# Cell 3.5: Loss Functions
# =====================================
# Custom Loss Functions Implementation
# =====================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import traceback

# Assuming config, logger, log_and_print, device are available

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    gamma: Focusing parameter (default: 2.0 from config).
    alpha: Class weights (Optional).
    """
    def __init__(self, gamma: float, alpha: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        print(f"\n--- Initializing FocalLoss ---")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, list): # Convert list to tensor
                self.alpha = torch.tensor(alpha)
            # Ensure alpha is on the correct device later if needed
            # self.alpha = self.alpha.to(device) # Move to device in forward pass if needed
            print(f"- Alpha (class weights) provided: {self.alpha.shape}")
        else:
            print("- No alpha (class weights) provided.")
        print(f"- Gamma (focusing parameter): {self.gamma}")
        print(f"- Reduction: {self.reduction}")
        print(f"--- Finished Initializing FocalLoss ---")


    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Logits from model (N, C).
            targets (torch.Tensor): Ground truth labels (N,).
        Returns:
            torch.Tensor: Calculated focal loss.
        """
        # Use cross_entropy with reduction='none' to get per-sample loss
        # Pass alpha directly to cross_entropy if available and on the correct device
        current_device = inputs.device
        alpha_on_device = self.alpha.to(current_device) if self.alpha is not None else None

        try:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=alpha_on_device)
            # Calculate pt = exp(-ce_loss) which is the probability of the true class
            pt = torch.exp(-ce_loss)
            # Calculate focal loss: alpha * (1-pt)^gamma * ce_loss
            focal_loss = (1 - pt) ** self.gamma * ce_loss

            # Apply reduction
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else: # 'none'
                return focal_loss
        except Exception as e:
             log_and_print(logger, f"Error in FocalLoss forward: {e}", logging.ERROR)
             # Return zero loss on error? Or raise? Let's return zero.
             return torch.tensor(0.0, device=current_device, requires_grad=True)


# Note: AlignmentLoss and IntentionLoss are implemented within CombinedLoss

class CombinedLoss(nn.Module):
    """
    Calculates the combined loss based on the training phase,
    including Focal Loss, Alignment Loss, and Intention Loss.
    Implements weight scheduling as per methodology.
    """
    def __init__(
        self,
        config: Config,
        num_classes: int,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        print(f"\n--- Initializing CombinedLoss ---")
        self.config = config
        self.num_classes = num_classes

        # --- Initialize Loss Components ---
        # 1. Focal Loss (Used in all phases)
        self.focal_loss_fn = FocalLoss(gamma=config.FOCAL_GAMMA, alpha=class_weights)
        print("- FocalLoss component initialized.")

        # 2. Alignment Loss Parameters (Used in Phase 2 & 3)
        self.alignment_temp = config.ALIGNMENT_TEMP
        self.alignment_margin = config.ALIGNMENT_MARGIN
        self.eps = 1e-8 # Epsilon for numerical stability
        print(f"- Alignment Loss params: Temp={self.alignment_temp}, Margin={self.alignment_margin}")

        # 3. Intention Loss (Used in Phase 3)
        # Using CrossEntropyLoss with label smoothing as per methodology text
        self.intention_loss_fn = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        self.intention_confidence_threshold = config.INTENTION_CONFIDENCE_THRESHOLD
        print(f"- Intention Loss component (CE w/ Smoothing={config.LABEL_SMOOTHING}) initialized.")
        print(f"- Intention Confidence Threshold: {self.intention_confidence_threshold}")

        # --- Loss Weights ---
        # Store base weights and schedules
        self.base_weights = config.INITIAL_WEIGHTS.copy() # {'focal': 1.0, 'alignment': 0.0, 'intention': 0.0}
        self.schedules = {
            'alignment': config.PHASE2_ALIGNMENT_WEIGHT_SCHEDULE, # Phase 2 schedule
            'intention': config.PHASE3_INTENTION_WEIGHT_SCHEDULE  # Phase 3 schedule
        }
        self.phase3_fixed_alignment_weight = config.PHASE3_ALIGNMENT_WEIGHT # 0.1

        print(f"- Initial Loss Weights: {self.base_weights}")
        print(f"--- Finished Initializing CombinedLoss ---")

    def _get_current_weights(self, phase: int, epoch: int) -> Dict[str, float]:
        """Calculates loss weights for the current phase and epoch."""
        current_weights = self.base_weights.copy() # Start with base weights

        if phase == 2:
            # Apply alignment schedule
            current_weights['alignment'] = self.schedules['alignment'](epoch)
            # Intention remains 0
        elif phase == 3:
            # Use fixed alignment weight
            current_weights['alignment'] = self.phase3_fixed_alignment_weight
            # Apply intention schedule
            current_weights['intention'] = self.schedules['intention'](epoch)

        # print(f"  - Weights for Phase {phase}, Epoch {epoch}: {current_weights}") # Verbose
        return current_weights

    def _compute_alignment_loss(
        self,
        original_fused: torch.Tensor, # Output of Feature Stack MLP (e.g., 256-dim)
        counterfactual_fused: torch.Tensor, # Output of Feature Stack MLP (e.g., 256-dim)
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Computes contrastive alignment loss (Methodology Page 10)."""
        try:
            # 1. Normalize features
            orig_norm = F.normalize(original_fused, p=2, dim=-1)
            cf_norm = F.normalize(counterfactual_fused, p=2, dim=-1)

            # 2. Compute similarity matrix (Eq 22, adjusted) S = (F_orig_norm @ F_cf_norm.T) / tau
            # Shape: (batch, batch)
            similarity_matrix = torch.matmul(orig_norm, cf_norm.transpose(0, 1)) / self.alignment_temp

            # 3. Create positive mask (samples with same label)
            # Shape: (batch, batch)
            pos_mask = (targets.unsqueeze(0) == targets.unsqueeze(1)).float()
            # Ensure diagonal is part of positive mask if comparing sample i with its own cf version
            # If orig[i] should align with cf[i], diagonal should be 1.
            # If orig[i] should align with cf[j] where label[i]==label[j], non-diagonal is 1.
            # The current pos_mask handles the label matching correctly.

            # 4. Create negative mask
            neg_mask = 1.0 - pos_mask

            # --- InfoNCE-like Loss Calculation (Alternative formulation from Page 10, Step 8) ---
            # Numerator: Similarity of positive pairs (e.g., sample i with cf sample i)
            # For simplicity, let's focus on aligning sample i with its own counterfactual i
            # This requires orig_norm and cf_norm to be aligned row-wise.
            # If using matrix, we need the diagonal elements for self-alignment.
            # Let's use the formulation from Eq 23 directly.

            # exp(Sum_pos S_ij) term: For sample i, sum similarities with all cf samples j where label(i)==label(j)
            exp_pos_sum = torch.exp(similarity_matrix) * pos_mask
            sum_exp_pos = torch.sum(exp_pos_sum, dim=1) # Sum across cf samples j for each original i

            # exp(Sum_neg [max(S_ik + m, 0)]) term: For sample i, sum max(0, S_ik + m) for all cf samples k where label(i)!=label(k)
            neg_similarities_margin = torch.clamp(similarity_matrix + self.alignment_margin, min=0.0) * neg_mask
            sum_exp_neg = torch.sum(torch.exp(neg_similarities_margin), dim=1) # Sum across cf samples k

            # Denominator = Sum_pos + Sum_neg
            denominator = sum_exp_pos + sum_exp_neg

            # Loss = -log(Sum_pos / Denominator) = -log(Sum_pos) + log(Denominator)
            # Add epsilon for stability
            loss_i = -torch.log(sum_exp_pos + self.eps) + torch.log(denominator + self.eps)

            # Mean loss over batch
            alignment_loss = loss_i.mean()

            # Clamp loss for stability
            alignment_loss = torch.clamp(alignment_loss, min=0.0, max=20.0) # Increased max clamp slightly

            return alignment_loss

        except Exception as e:
             log_and_print(logger, f"Error computing alignment loss: {e}", logging.ERROR)
             return torch.tensor(0.0, device=original_fused.device, requires_grad=True)


    def _compute_intention_loss(
        self,
        intention_logits: torch.Tensor,
        targets: torch.Tensor,
        original_logits: torch.Tensor # Needed for confidence calculation
    ) -> torch.Tensor:
        """Computes confidence-weighted intention loss (Methodology Page 11, Eq 24)."""
        try:
            # Calculate confidence scores from the *original* prediction logits
            with torch.no_grad(): # Don't track gradients for confidence calculation
                original_probs = F.softmax(original_logits, dim=1)
                confidence_scores, _ = torch.max(original_probs, dim=1)
                # Create weights based on confidence threshold
                weights = (confidence_scores > self.intention_confidence_threshold).float()

            # Calculate intention loss (CE with label smoothing) per sample
            intention_loss_per_sample = self.intention_loss_fn(intention_logits, targets) # Uses reduction='mean' by default

            # Apply confidence weighting (if any samples meet threshold)
            # Note: CE loss is already averaged. To weight correctly, we need per-sample loss.
            # Re-calculate with reduction='none'
            intention_loss_none = nn.CrossEntropyLoss(label_smoothing=self.config.LABEL_SMOOTHING, reduction='none')(intention_logits, targets)
            weighted_loss = intention_loss_none * weights

            # Calculate the mean loss over the *weighted* samples
            # Avoid division by zero if no samples meet the threshold
            num_weighted_samples = weights.sum()
            if num_weighted_samples > 0:
                intention_loss = weighted_loss.sum() / num_weighted_samples
            else:
                intention_loss = torch.tensor(0.0, device=intention_logits.device, requires_grad=True) # No loss if no samples meet threshold

            # Clamp loss for stability
            intention_loss = torch.clamp(intention_loss, min=0.0, max=10.0)

            return intention_loss

        except Exception as e:
             log_and_print(logger, f"Error computing intention loss: {e}", logging.ERROR)
             return torch.tensor(0.0, device=intention_logits.device, requires_grad=True)


    def forward(
        self,
        outputs: Dict[str, torch.Tensor], # Model outputs for original features
        targets: torch.Tensor,
        phase: int,
        epoch: int, # Current epoch number (0-indexed)
        # counterfactual_outputs: Optional[Dict[str, torch.Tensor]] = None # No longer needed, CF features are in 'outputs' for phase >= 2
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate the combined loss based on the phase and epoch.

        Args:
            outputs (Dict): Dictionary of tensors from the model's forward pass.
                            Expected keys vary by phase:
                            - Phase 1: 'final_logits', 'processed_features'
                            - Phase 2: 'final_logits', 'processed_features', 'counterfactual_features'
                            - Phase 3: 'final_logits', 'intention_logits', 'processed_features', 'counterfactual_features'
            targets (torch.Tensor): Ground truth labels.
            phase (int): Current training phase (1, 2, or 3).
            epoch (int): Current epoch number (0-indexed) for weight scheduling.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - total_loss: The final weighted loss tensor.
                - loss_dict: Dictionary containing individual loss components.
        """
        # print(f"\nCombinedLoss Forward: Phase={phase}, Epoch={epoch}") # Verbose
        # print(f"- Output keys: {list(outputs.keys())}") # Verbose
        current_device = targets.device
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=current_device) # Ensure total_loss requires grad if components do

        # Get weights for the current phase/epoch
        current_weights = self._get_current_weights(phase, epoch)

        # --- Calculate Individual Losses ---
        # 1. Focal Loss (Always calculated on original predictions)
        if 'final_logits' in outputs:
            focal_loss = self.focal_loss_fn(outputs['final_logits'], targets)
            loss_dict['focal'] = focal_loss
            total_loss = total_loss + current_weights['focal'] * focal_loss
            # print(f"  - Focal Loss: {focal_loss.item():.4f}") # Verbose
        else:
            log_and_print(logger, "Warning: 'final_logits' not found in model outputs for Focal Loss.", logging.WARNING)
            loss_dict['focal'] = torch.tensor(0.0, device=current_device)

        # 2. Alignment Loss (Phase 2 and 3)
        if phase >= 2 and current_weights['alignment'] > 0:
            if 'processed_features' in outputs and 'counterfactual_features' in outputs:
                alignment_loss = self._compute_alignment_loss(
                    outputs['processed_features'],
                    outputs['counterfactual_features'],
                    targets
                )
                loss_dict['alignment'] = alignment_loss
                total_loss = total_loss + current_weights['alignment'] * alignment_loss
                # print(f"  - Alignment Loss: {alignment_loss.item():.4f} (Weight: {current_weights['alignment']:.3f})") # Verbose
            else:
                log_and_print(logger, "Warning: Features for Alignment Loss not found in model outputs.", logging.WARNING)
                loss_dict['alignment'] = torch.tensor(0.0, device=current_device)
        else:
             loss_dict['alignment'] = torch.tensor(0.0, device=current_device) # Log zero if not active

        # 3. Intention Loss (Phase 3 only)
        if phase == 3 and current_weights['intention'] > 0:
            if 'intention_logits' in outputs and 'final_logits' in outputs:
                intention_loss = self._compute_intention_loss(
                    outputs['intention_logits'],
                    targets,
                    outputs['final_logits'] # Pass original logits for confidence calculation
                )
                loss_dict['intention'] = intention_loss
                total_loss = total_loss + current_weights['intention'] * intention_loss
                # print(f"  - Intention Loss: {intention_loss.item():.4f} (Weight: {current_weights['intention']:.3f})") # Verbose
            else:
                log_and_print(logger, "Warning: Logits for Intention Loss not found in model outputs.", logging.WARNING)
                loss_dict['intention'] = torch.tensor(0.0, device=current_device)
        else:
             loss_dict['intention'] = torch.tensor(0.0, device=current_device) # Log zero if not active


        loss_dict['total'] = total_loss
        # print(f"  - Total Loss: {total_loss.item():.4f}") # Verbose
        return total_loss, loss_dict

# --- Test Function ---
def test_loss_functions(config: Config, device: torch.device, num_classes: int):
    """Test the CombinedLoss function for all phases."""
    print("\n--- Starting Cell 3.5: Loss Functions Test ---")
    try:
        batch_size = config.BATCH_SIZE # 16
        feature_dim = config.FEATURE_STACK_MLP_DIMS[-1] # 256 (Output of MLP stack)
        log_and_print(logger, f"Testing loss functions with Batch={batch_size}, NumClasses={num_classes}, FeatureDim={feature_dim}")

        # Create dummy model outputs matching expected structure for each phase
        dummy_outputs_phase1 = {
            'final_logits': torch.randn(batch_size, num_classes).to(device),
            'processed_features': torch.randn(batch_size, feature_dim).to(device),
            'std_features': {'text': torch.randn(batch_size, config.STANDARDIZED_FEATURE_DIM).to(device)} # Example std feature
        }
        dummy_outputs_phase2 = {
            **dummy_outputs_phase1, # Include phase 1 outputs
            'counterfactual_features': torch.randn(batch_size, feature_dim).to(device),
            'cf_std_features': {'text': torch.randn(batch_size, config.STANDARDIZED_FEATURE_DIM).to(device)} # Example cf std feature
        }
        dummy_outputs_phase3 = {
            **dummy_outputs_phase2, # Include phase 2 outputs
            'intention_logits': torch.randn(batch_size, num_classes).to(device)
        }
        all_dummy_outputs = [dummy_outputs_phase1, dummy_outputs_phase2, dummy_outputs_phase3]

        # Create dummy targets
        targets = torch.randint(0, num_classes, (batch_size,)).to(device)
        print(f"\nDummy Targets shape: {targets.shape}")

        # Initialize CombinedLoss
        # Create dummy class weights if needed for testing FocalLoss alpha
        # class_weights = torch.rand(num_classes).to(device)
        class_weights = None # Test without weights first
        combined_loss_fn = CombinedLoss(config=config, num_classes=num_classes, class_weights=class_weights)

        # --- Test each phase ---
        all_phases_passed = True
        test_epoch = 5 # Use a sample epoch number for scheduling test
        for phase in [1, 2, 3]:
            print(f"\n--- Testing Loss Calculation for Phase {phase}, Epoch {test_epoch} ---")
            dummy_outputs = all_dummy_outputs[phase-1]
            print(f"Input keys for Phase {phase}: {list(dummy_outputs.keys())}")

            try:
                total_loss, loss_dict = combined_loss_fn(
                    dummy_outputs,
                    targets,
                    phase=phase,
                    epoch=test_epoch
                )

                print(f"Phase {phase} Calculated Losses:")
                for loss_name, loss_value in loss_dict.items():
                    print(f"- {loss_name}: {loss_value.item():.4f}")
                    assert not torch.isnan(loss_value), f"{loss_name} is NaN!"
                    assert not torch.isinf(loss_value), f"{loss_name} is Inf!"
                    assert loss_value >= 0.0, f"{loss_name} is negative!" # Losses should be non-negative

                print(f"--- Phase {phase} Loss Test Passed ---")

            except Exception as phase_e:
                log_and_print(logger, f"Error during Phase {phase} loss test: {phase_e}", logging.ERROR)
                log_and_print(logger, traceback.format_exc(), logging.ERROR)
                all_phases_passed = False
                print(f"--- Phase {phase} Loss Test Failed ---")

        if all_phases_passed:
            log_and_print(logger, "\nLoss function tests completed successfully for all phases!")
            print("\n--- Finished Cell 3.5 Test ---")
            print("-" * 50)
            return True
        else:
            log_and_print(logger, "\nLoss function tests failed for one or more phases.", logging.ERROR)
            print("\n--- Cell 3.5 Test Failed ---")
            print("-" * 50)
            return False

    except Exception as e:
        log_and_print(logger, f"Error in loss function test setup: {str(e)}", logging.ERROR)
        log_and_print(logger, traceback.format_exc(), logging.ERROR)
        print("\n--- Cell 3.5 Test Failed ---")
        print("-" * 50)
        return False

# --- Execution ---
# Ensure config, logger, device, num_classes are available
if 'config' in locals() and 'logger' in locals() and 'device' in locals() and 'num_classes' in locals() and num_classes > 0:
    test_successful_losses = test_loss_functions(config, device, num_classes)
else:
    print("Error: Required variables not available or num_classes not set. Cannot execute Cell 3.5.")
    test_successful_losses = False

# Cell 3.6: Training Pipeline (REMOVED)
# =====================================
# Training and Validation Functions (Moved to PhaseTrainer in Cell 3.7)
# =====================================

# --- Class Definitions Removed ---
# The CustomSchedulerManager and TrainingPipeline classes previously defined
# in this cell have been removed.
# Their logic will be integrated into the PhaseTrainer class in Cell 3.7
# for better organization and simplification.

# --- Execution ---
print("\n--- Starting Cell 3.6 Execution ---")
print("Cell 3.6: CustomSchedulerManager and TrainingPipeline classes removed.")
print("Training loop logic will be handled by PhaseTrainer in Cell 3.7.")
print("--- Finished Cell 3.6 Execution ---")
print("-" * 50)

# No objects or functions defined in this cell anymore.
# We proceed directly to Cell 3.7 where the training logic resides.

# Cell 3.7: Model Initialization and Training Execution
# =====================================
# Training Execution and Phase Management
# =====================================

import torch
import torch.optim as optim
# Schedulers already imported in Cell 1.1
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
# Import specific metrics needed here
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.notebook import tqdm
import numpy as np
import os
import time
import traceback
import copy # For saving best model state
import json # Added for saving history

# Assuming config, logger, log_and_print, device, train_loader, val_loader,
# MultimodalEmotionRecognition, CombinedLoss, MetricsTracker are available/defined

def initialize_model(config: Config, num_emotions: int, device: torch.device) -> nn.Module: # Return type hint nn.Module
    """Initializes a new instance of the main model."""
    log_and_print(logger, f"\n--- Initializing New Model Instance ---")
    log_and_print(logger, f"- Dataset: {config.CURRENT_DATASET}")
    log_and_print(logger, f"- Number of emotions: {num_emotions}")

    # Ensure the model class is defined (it should be from Cell 3.3)
    if 'MultimodalEmotionRecognition' not in globals():
         log_and_print(logger, "Error: MultimodalEmotionRecognition class not found.", logging.ERROR)
         raise NameError("MultimodalEmotionRecognition class not defined.")

    # Ensure model class is callable
    if not callable(MultimodalEmotionRecognition):
         log_and_print(logger, "Error: MultimodalEmotionRecognition is not callable.", logging.ERROR)
         raise TypeError("MultimodalEmotionRecognition is not a class or callable.")


    model = MultimodalEmotionRecognition(config=config, num_emotions=num_emotions).to(device)

    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_and_print(logger, f"Model parameters:")
    log_and_print(logger, f"- Total parameters: {total_params:,}")
    log_and_print(logger, f"- Trainable parameters: {trainable_params:,}")
    print("--- Finished Initializing Model Instance ---")
    # print(model) # Optionally print model structure
    return model

class PhaseTrainer:
    """Manages the training process across different phases."""
    def __init__(self, config: Config, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
        print("\n--- Initializing PhaseTrainer ---")
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.result_dir = config.RESULT_DIR
        # Ensure metrics_tracker is initialized (should be from Cell 1.4)
        if 'metrics_tracker' in globals() and metrics_tracker is not None:
             self.metrics_tracker = metrics_tracker
             print("- Using existing MetricsTracker instance.")
        else:
             log_and_print(logger, "Warning: Global 'metrics_tracker' not found or is None. Initializing a new one.", logging.WARNING)
             # Need to ensure MetricsTracker class is available if we initialize here
             if 'MetricsTracker' in globals():
                 self.metrics_tracker = MetricsTracker()
             else:
                 log_and_print(logger, "Error: MetricsTracker class definition not found. Cannot initialize.", logging.ERROR)
                 raise NameError("MetricsTracker class not defined.")


        if not all([self.train_loader, self.val_loader]):
             log_and_print(logger, "Error: DataLoaders not provided to PhaseTrainer.", logging.ERROR)
             raise ValueError("DataLoaders are required.")

        log_and_print(logger, "PhaseTrainer initialized.")
        print(f"- Result Directory: {self.result_dir}")
        print(f"- Early Stopping: {config.EARLY_STOP.upper()}")
        print(f"- AMP Enabled: {config.USE_AMP}")
        print(f"--- Finished Initializing PhaseTrainer ---")


    def save_checkpoint(self, phase: int, epoch: int, model: nn.Module, optimizer: optim.Optimizer, best_metric: float, is_best: bool, filename_prefix: str):
        """Saves model checkpoint."""
        # Ensure model state dict is valid
        try:
            model_state = model.state_dict()
        except Exception as e:
            log_and_print(logger, f"Error getting model state_dict: {e}. Skipping checkpoint save.", logging.ERROR)
            return

        state = {
            'phase': phase,
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': best_metric, # Store the metric value that triggered the save
            'metrics_history': self.metrics_tracker.get_history(), # Save full history
            'num_emotions': getattr(model, 'num_emotions', self.config.NUM_EMOTIONS) # Save num_emotions used
        }
        filename = f"{filename_prefix}_phase{phase}_latest.pth" # Add _latest suffix
        save_path = os.path.join(self.result_dir, filename)
        try:
            torch.save(state, save_path)
            log_and_print(logger, f"Saved LATEST checkpoint to {save_path} (Epoch: {epoch}, Val F1w: {best_metric:.4f})") # Use best_metric which is val f1w
        except Exception as e:
             log_and_print(logger, f"Error saving LATEST checkpoint to {save_path}: {e}", logging.ERROR)


        # Keep a separate copy if it's the best model so far for this phase
        if is_best:
            best_filename = f"BEST_{filename_prefix}_phase{phase}.pth"
            best_save_path = os.path.join(self.result_dir, best_filename)
            try:
                torch.save(state, best_save_path)
                log_and_print(logger, f"Saved BEST checkpoint to {best_save_path}")
            except Exception as e:
                 log_and_print(logger, f"Error saving BEST checkpoint to {best_save_path}: {e}", logging.ERROR)


    def _calculate_epoch_metrics(self, all_labels: List, all_preds: List, total_loss: float, num_samples: int) -> Dict[str, float]:
        """Helper function to calculate standard epoch metrics."""
        if num_samples == 0:
            # Return zeros or default values if no samples were processed
            return {
                'loss': float('inf'), 'accuracy': 0.0,
                'precision_macro': 0.0, 'precision_weighted': 0.0, 'precision_micro': 0.0,
                'recall_macro': 0.0, 'recall_weighted': 0.0, 'recall_micro': 0.0,
                'f1_macro': 0.0, 'f1_weighted': 0.0, 'f1_micro': 0.0
            }

        epoch_loss = total_loss / num_samples
        # Convert lists to numpy arrays for sklearn metrics
        labels_np = np.array(all_labels)
        preds_np = np.array(all_preds)

        # Calculate metrics
        accuracy = accuracy_score(labels_np, preds_np) * 100
        # Use zero_division=0 to avoid warnings when a class has no predictions/labels
        precision_macro = precision_score(labels_np, preds_np, average='macro', zero_division=0) * 100
        precision_weighted = precision_score(labels_np, preds_np, average='weighted', zero_division=0) * 100
        precision_micro = precision_score(labels_np, preds_np, average='micro', zero_division=0) * 100
        recall_macro = recall_score(labels_np, preds_np, average='macro', zero_division=0) * 100
        recall_weighted = recall_score(labels_np, preds_np, average='weighted', zero_division=0) * 100
        recall_micro = recall_score(labels_np, preds_np, average='micro', zero_division=0) * 100
        f1_macro = f1_score(labels_np, preds_np, average='macro', zero_division=0) * 100
        f1_weighted = f1_score(labels_np, preds_np, average='weighted', zero_division=0) * 100
        f1_micro = f1_score(labels_np, preds_np, average='micro', zero_division=0) * 100

        return {
            'loss': epoch_loss,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'precision_micro': precision_micro,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'recall_micro': recall_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_micro': f1_micro
        }

    def train_epoch(self, epoch: int, phase: int, model: nn.Module, loss_handler: CombinedLoss, optimizer: optim.Optimizer, scheduler, scaler: GradScaler, progress_manager: TrainingProgress) -> Dict[str, float]:
        """Trains the model for one epoch."""
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        num_samples = 0

        progress_manager.init_epoch(epoch + 1, phase, mode='train') # epoch is 0-indexed

        optimizer.zero_grad() # Zero gradients at the beginning of the epoch accumulation cycle

        # --- Add counter for OneCycleLR steps ---
        one_cycle_step_count = 0
        is_one_cycle = isinstance(scheduler, OneCycleLR)
        steps_per_epoch_sched = 0 # Initialize
        total_steps_sched = 0 # Initialize
        if is_one_cycle:
            # Calculate total steps expected by OneCycleLR for this epoch
            # Note: scheduler.total_steps might be None initially, calculate manually
            steps_per_epoch_sched = len(self.train_loader) // self.config.ACCUMULATION_STEPS
            # Ensure steps_per_epoch_sched is at least 1 if train_loader is smaller than accumulation steps
            if steps_per_epoch_sched == 0 and len(self.train_loader) > 0:
                steps_per_epoch_sched = 1
            # Get total epochs from scheduler if possible, else from config
            num_epochs_sched = getattr(scheduler, 'epochs', getattr(self.config, f'PHASE{phase}_EPOCHS'))
            total_steps_sched = steps_per_epoch_sched * num_epochs_sched
            # Handle case where total_steps might be None in scheduler if not properly initialized
            if scheduler.total_steps is None:
                 scheduler.total_steps = total_steps_sched # Manually set if needed
                 log_and_print(logger, f"Warning: Manually setting scheduler.total_steps to {total_steps_sched}", logging.WARNING)

            print(f"  [OneCycleLR] Epoch {epoch+1}: Expecting {steps_per_epoch_sched} steps this epoch. Total configured steps: {scheduler.total_steps}")


        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Move batch to device
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                targets = batch['label']
                batch_size = targets.size(0)

                # Forward pass with AMP context
                with autocast(enabled=self.config.USE_AMP):
                    outputs = model(batch, phase=phase)
                    # Ensure outputs needed for loss are present
                    if 'final_logits' not in outputs:
                         log_and_print(logger, f"Warning: 'final_logits' missing in model output for batch {batch_idx}, phase {phase}. Skipping batch.", logging.WARNING)
                         continue
                    loss, loss_dict = loss_handler(outputs, targets, phase=phase, epoch=epoch)
                    # Normalize loss for accumulation
                    loss = loss / self.config.ACCUMULATION_STEPS

                # Check for valid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    log_and_print(logger, f"Warning: NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}. Skipping backward for this batch.", logging.WARNING)
                    # Don't zero gradients here, let accumulation handle it
                    continue

                # Backward pass & Optimization
                # Scale loss for AMP
                scaler.scale(loss).backward()

                # Gradient accumulation & Optimizer step
                # Ensure step happens even if last batch doesn't fill accumulation cycle fully
                is_last_batch = (batch_idx + 1) == len(self.train_loader)
                if (batch_idx + 1) % self.config.ACCUMULATION_STEPS == 0 or is_last_batch:
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.MAX_GRAD_NORM)
                    # Optimizer step
                    scaler.step(optimizer)
                    # Update scaler
                    scaler.update()
                    # Zero gradients for the next accumulation cycle
                    optimizer.zero_grad()

                    # --- Updated Scheduler Step Logic ---
                    if is_one_cycle:
                        # Check if scheduler's internal step count is less than total steps
                        # scheduler._step_count starts at 1
                        # Use scheduler._step_count directly as it tracks total steps across epochs
                        if scheduler.total_steps is None:
                             log_and_print(logger, "Warning: scheduler.total_steps is None, cannot reliably check step count for OneCycleLR.", logging.WARNING)
                             # Fallback: step anyway but might error
                             scheduler.step()
                             one_cycle_step_count += 1
                        elif scheduler._step_count <= scheduler.total_steps:
                             # print(f"  Stepping OneCycleLR (Step {scheduler._step_count})") # Debug print
                             scheduler.step()
                             one_cycle_step_count += 1
                        else:
                             log_and_print(logger, f"Warning: Reached total OneCycleLR steps ({scheduler.total_steps}). Skipping scheduler.step() for batch {batch_idx+1}.", logging.WARNING)


                # --- Metrics Calculation (on CPU to save GPU memory) ---
                # Use the un-normalized loss for tracking total loss
                total_loss += loss.item() * self.config.ACCUMULATION_STEPS * batch_size # Adjust for normalization and batch size
                num_samples += batch_size
                with torch.no_grad():
                    preds = outputs['final_logits'].argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(targets.cpu().numpy())

                # Update progress bar
                batch_metrics = {
                    'loss': loss.item() * self.config.ACCUMULATION_STEPS, # Show un-normalized loss
                    'lr': optimizer.param_groups[0]['lr'] # Get current LR
                }
                # Add individual losses if available
                batch_metrics.update({k: v.item() for k, v in loss_dict.items() if k != 'total'})
                progress_manager.update(batch_metrics, mode='train')

            except Exception as e:
                # Catch the specific ValueError from OneCycleLR if the check above fails
                if isinstance(e, ValueError) and "Tried to step" in str(e):
                     log_and_print(logger, f"Caught OneCycleLR step error at epoch {epoch}, batch {batch_idx}: {e}. Continuing epoch.", logging.ERROR)
                     # Don't re-raise, just log it, as the check should prevent it mostly
                else:
                     log_and_print(logger, f"Error during training epoch {epoch}, batch {batch_idx}: {e}", logging.ERROR)
                     log_and_print(logger, traceback.format_exc(), logging.ERROR)
                continue # Skip to next batch

        progress_manager.close_epoch(mode='train')

        # Calculate epoch metrics using helper function
        epoch_metrics = self._calculate_epoch_metrics(all_labels, all_preds, total_loss, num_samples)
        print(f"Epoch {epoch+1} [Train] Metrics Calculated: Loss={epoch_metrics['loss']:.4f}, Acc={epoch_metrics['accuracy']:.2f}%, F1w={epoch_metrics['f1_weighted']:.2f}%") # Added print
        if is_one_cycle: print(f"  [OneCycleLR] Actual steps taken this epoch: {one_cycle_step_count}")
        return epoch_metrics


    def validate(self, epoch: int, phase: int, model: nn.Module, loss_handler: CombinedLoss, progress_manager: TrainingProgress) -> Dict[str, float]:
        """Validates the model for one epoch."""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        num_samples = 0

        progress_manager.init_epoch(epoch + 1, phase, mode='val')

        with torch.no_grad(): # Disable gradient calculations
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    targets = batch['label']
                    batch_size = targets.size(0)

                    # Forward pass with AMP context (recommended even for validation)
                    with autocast(enabled=self.config.USE_AMP):
                        outputs = model(batch, phase=phase)
                        if 'final_logits' not in outputs:
                             log_and_print(logger, f"Warning: 'final_logits' missing in validation output for batch {batch_idx}, phase {phase}. Skipping batch.", logging.WARNING)
                             continue
                        # Calculate loss (optional but useful for ReduceLROnPlateau)
                        loss, loss_dict = loss_handler(outputs, targets, phase=phase, epoch=epoch) # Pass epoch for consistency

                    # Accumulate loss and predictions
                    total_loss += loss.item() * batch_size
                    num_samples += batch_size
                    preds = outputs['final_logits'].argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(targets.cpu().numpy())

                    # Update progress bar
                    batch_metrics = {'loss': loss.item()}
                    batch_metrics.update({k: v.item() for k, v in loss_dict.items() if k != 'total'})
                    progress_manager.update(batch_metrics, mode='val')

                except Exception as e:
                    log_and_print(logger, f"Error during validation epoch {epoch}, batch {batch_idx}: {e}", logging.ERROR)
                    log_and_print(logger, traceback.format_exc(), logging.ERROR)
                    continue # Skip to next batch

        progress_manager.close_epoch(mode='val')

        # Calculate epoch metrics using helper function
        epoch_metrics = self._calculate_epoch_metrics(all_labels, all_preds, total_loss, num_samples)
        print(f"Epoch {epoch+1} [Val] Metrics Calculated: Loss={epoch_metrics['loss']:.4f}, Acc={epoch_metrics['accuracy']:.2f}%, F1w={epoch_metrics['f1_weighted']:.2f}%") # Added print
        return epoch_metrics


    def train_phase(self, phase: int) -> bool:
        """Manages the training process for a single phase."""
        try:
            log_and_print(logger, f"\n{'='*20} Starting Phase {phase} Training {'='*20}")
            start_time_phase = time.time()

            # --- Initialization ---
            num_epochs = getattr(self.config, f'PHASE{phase}_EPOCHS')
            log_and_print(logger, f"Target epochs for this phase: {num_epochs}")

            # 1. Initialize Model
            model = initialize_model(self.config, self.config.NUM_EMOTIONS, self.device)

            # 2. Load Weights from Previous Phase (if applicable)
            if phase > 1:
                prev_model_path = os.path.join(self.result_dir, f'BEST_model_phase{phase-1}.pth') # Load BEST from previous
                if os.path.exists(prev_model_path):
                    log_and_print(logger, f"Loading BEST weights from Phase {phase-1}: {prev_model_path}")
                    try:
                        checkpoint = torch.load(prev_model_path, map_location=self.device)
                        # Check for num_emotions mismatch before loading
                        ckpt_num_emotions = checkpoint.get('num_emotions', -1)
                        if ckpt_num_emotions != -1 and ckpt_num_emotions != model.num_emotions:
                             log_and_print(logger, f"Warning: Checkpoint num_emotions ({ckpt_num_emotions}) differs from current model ({model.num_emotions}). Final layers might not load correctly.", logging.WARNING)
                        # Load state dict (allow missing/unexpected keys for flexibility between phases if needed)
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False) # Use strict=False
                        log_and_print(logger, f"Successfully loaded weights from epoch {checkpoint.get('epoch', 'N/A')}.")
                    except Exception as e:
                         log_and_print(logger, f"Error loading weights from {prev_model_path}: {e}. Starting phase {phase} from scratch.", logging.ERROR)
                else:
                    log_and_print(logger, f"No BEST checkpoint found from Phase {phase-1} at {prev_model_path}. Starting phase {phase} from scratch.", logging.WARNING)

            # 3. Initialize Optimizer
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                eps=self.config.ADAM_EPSILON
            )
            log_and_print(logger, f"Optimizer AdamW initialized (LR={self.config.LEARNING_RATE}, WD={self.config.WEIGHT_DECAY}).")

            # 4. Initialize Scheduler
            scheduler_type = self.config.LR_SCHEDULE_TYPE.get(phase, 'plateau') # Default to plateau
            if scheduler_type == 'plateau':
                # Step based on f1_weighted
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=self.config.LR_SCHEDULER_FACTOR, patience=self.config.LR_SCHEDULER_PATIENCE, verbose=True, min_lr=self.config.MIN_LR)
                log_and_print(logger, f"Scheduler initialized: ReduceLROnPlateau (Mode=max on Val F1w, Factor={self.config.LR_SCHEDULER_FACTOR}, Patience={self.config.LR_SCHEDULER_PATIENCE})")
            elif scheduler_type == 'cosine':
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.config.COSINE_T_0, T_mult=self.config.COSINE_T_MULT, eta_min=self.config.MIN_LR)
                log_and_print(logger, f"Scheduler initialized: CosineAnnealingWarmRestarts (T_0={self.config.COSINE_T_0}, T_mult={self.config.COSINE_T_MULT})")
            elif scheduler_type == 'one_cycle':
                # steps_per_epoch needs to be calculated carefully if using accumulation
                steps_per_epoch = len(self.train_loader) // self.config.ACCUMULATION_STEPS
                if steps_per_epoch == 0 and len(self.train_loader) > 0: steps_per_epoch = 1 # Ensure at least 1 step
                scheduler = OneCycleLR(optimizer, max_lr=self.config.LEARNING_RATE, epochs=num_epochs, steps_per_epoch=steps_per_epoch, pct_start=self.config.ONE_CYCLE_PCT_START, div_factor=10.0, final_div_factor=1e4)
                log_and_print(logger, f"Scheduler initialized: OneCycleLR (MaxLR={self.config.LEARNING_RATE}, StepsPerEpoch={steps_per_epoch}, PctStart={self.config.ONE_CYCLE_PCT_START})")
            else:
                 log_and_print(logger, f"Warning: Unknown scheduler type '{scheduler_type}'. Defaulting to ReduceLROnPlateau.", logging.WARNING)
                 scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

            # 5. Initialize Loss Handler
            # Ensure CombinedLoss class is available
            if 'CombinedLoss' not in globals():
                 log_and_print(logger, "Error: CombinedLoss class definition not found.", logging.ERROR)
                 raise NameError("CombinedLoss class not defined.")
            class_weights = None # Placeholder
            loss_handler = CombinedLoss(config=self.config, num_classes=model.num_emotions, class_weights=class_weights)
            log_and_print(logger, "CombinedLoss handler initialized.")

            # 6. Initialize AMP GradScaler
            scaler = GradScaler(enabled=self.config.USE_AMP)
            log_and_print(logger, f"AMP GradScaler initialized (Enabled: {self.config.USE_AMP}).")

            # 7. Initialize Progress Manager
            progress_manager = TrainingProgress(num_epochs, len(self.train_loader), len(self.val_loader))

            # --- Training Loop ---
            best_val_metric_phase = -1.0 # Track best F1w within this phase
            patience_counter = 0

            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                log_and_print(logger, f"\n--- Phase {phase}, Epoch {epoch+1}/{num_epochs} ---")

                # Train one epoch
                train_metrics = self.train_epoch(epoch, phase, model, loss_handler, optimizer, scheduler, scaler, progress_manager)

                # Validate one epoch
                val_metrics = self.validate(epoch, phase, model, loss_handler, progress_manager)

                # Log epoch metrics (more detailed)
                log_and_print(logger, f"Epoch {epoch+1} Train Metrics: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.2f}%, F1w={train_metrics['f1_weighted']:.2f}%, F1m={train_metrics['f1_macro']:.2f}%")
                log_and_print(logger, f"Epoch {epoch+1} Val Metrics  : Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.2f}%, F1w={val_metrics['f1_weighted']:.2f}%, F1m={val_metrics['f1_macro']:.2f}%")

                # Update metrics tracker and check if current validation is best OVERALL
                # Note: is_best_overall tracks the best across ALL phases run so far
                _ = self.metrics_tracker.update_epoch_metrics(phase, epoch, 'train', train_metrics) # Returns False
                is_best_overall = self.metrics_tracker.update_epoch_metrics(phase, epoch, 'val', val_metrics) # Returns True if best overall

                current_val_f1w = val_metrics['f1_weighted'] # Use F1 weighted for scheduler and early stopping

                # Check if current epoch is the best *within this phase* for saving BEST_phaseX checkpoint
                is_best_this_phase = False
                if current_val_f1w > best_val_metric_phase:
                     best_val_metric_phase = current_val_f1w
                     is_best_this_phase = True
                     patience_counter = 0 # Reset patience if phase best improves
                     log_and_print(logger, f"Best validation F1w *for this phase* improved to {best_val_metric_phase:.2f}%.")
                else:
                     patience_counter += 1
                     log_and_print(logger, f"Validation F1w did not improve *for this phase*. Patience: {patience_counter}/{self.config.EARLY_STOPPING_PATIENCE}")


                # Scheduler Step (Only step epoch-based schedulers here)
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(current_val_f1w) # Step based on validation F1 weighted
                elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                    scheduler.step() # Step every epoch
                # OneCycleLR is stepped per batch inside train_epoch

                # Save checkpoint (last and best *for this phase*)
                self.save_checkpoint(phase, epoch, model, optimizer, current_val_f1w, is_best_this_phase, filename_prefix="model")

                epoch_end_time = time.time()
                log_and_print(logger, f"Epoch {epoch+1} completed in {epoch_end_time - epoch_start_time:.2f} seconds.")

                # Early Stopping Check (based on phase best)
                if self.config.EARLY_STOP == "on" and patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    log_and_print(logger, f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation F1w for this phase.")
                    break # Exit epoch loop

            phase_end_time = time.time()
            log_and_print(logger, f"\n{'='*20} Phase {phase} Training Completed {'='*20}")
            log_and_print(logger, f"Phase {phase} duration: {phase_end_time - start_time_phase:.2f} seconds.")
            # Find best epoch metrics for this specific phase from history
            phase_key = f'phase{phase}'
            best_epoch_this_phase = -1
            best_f1w_this_phase = -1.0
            if phase_key in self.metrics_tracker.history and 'val' in self.metrics_tracker.history[phase_key] and 'f1_weighted' in self.metrics_tracker.history[phase_key]['val']:
                 val_f1w_history = self.metrics_tracker.history[phase_key]['val']['f1_weighted']
                 if val_f1w_history:
                     best_epoch_this_phase = np.argmax(val_f1w_history)
                     best_f1w_this_phase = val_f1w_history[best_epoch_this_phase]
            log_and_print(logger, f"Best Validation F1w for Phase {phase}: {best_f1w_this_phase:.2f}% at Epoch {best_epoch_this_phase + 1}") # +1 for 1-based epoch display

            # Save history after phase completion
            history_save_path = os.path.join(self.result_dir, 'training_history.json')
            self.metrics_tracker.save_history(history_save_path)

            return True # Indicate phase completed successfully

        except Exception as e:
            log_and_print(logger, f"Critical Error in Training Phase {phase}: {str(e)}", logging.CRITICAL)
            log_and_print(logger, traceback.format_exc(), logging.CRITICAL)
            return False # Indicate phase failed


# --- Execution Block ---
# Ensure config, loaders, device, logger, etc. are available
# Check necessary classes and the global metrics_tracker instance
if 'config' in locals() and 'train_loader' in locals() and 'val_loader' in locals() and 'device' in locals() and 'logger' in locals() \
    and 'MultimodalEmotionRecognition' in locals() and 'CombinedLoss' in locals() and 'MetricsTracker' in globals() and 'metrics_tracker' in globals() and metrics_tracker is not None:

    # Initialize the trainer (it will use the global metrics_tracker)
    trainer = PhaseTrainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # Run training phases sequentially
    overall_success = True
    for phase_num in range(1, 4): # Phases 1, 2, 3
        success = trainer.train_phase(phase_num)
        if not success:
            log_and_print(logger, f"Training stopped due to failure in Phase {phase_num}.", logging.ERROR)
            overall_success = False
            break # Stop if a phase fails

    if overall_success:
        log_and_print(logger, "\nAll training phases completed successfully!")
    else:
        log_and_print(logger, "\nTraining process finished with errors.")

    # Store final metrics history for later use (e.g., visualization)
    final_metrics_history = trainer.metrics_tracker.get_history()

    # Save history one last time after all phases
    final_history_save_path = os.path.join(config.RESULT_DIR, 'training_history_final.json')
    trainer.metrics_tracker.save_history(final_history_save_path)
    log_and_print(logger, f"Final training history saved to {final_history_save_path}")

else:
    print("Error: Required variables/classes not defined. Cannot execute Cell 3.7 (PhaseTrainer).")
    final_metrics_history = None
    # Print status of required components for debugging
    print(f"config exists: {'config' in locals()}")
    print(f"train_loader exists: {'train_loader' in locals()}")
    print(f"val_loader exists: {'val_loader' in locals()}")
    print(f"device exists: {'device' in locals()}")
    print(f"logger exists: {'logger' in locals()}")
    print(f"MultimodalEmotionRecognition exists: {'MultimodalEmotionRecognition' in locals()}")
    print(f"CombinedLoss exists: {'CombinedLoss' in locals()}")
    print(f"MetricsTracker class exists in globals: {'MetricsTracker' in globals()}")
    if 'MetricsTracker' in globals():
        print(f"metrics_tracker instance exists and is not None: {'metrics_tracker' in globals() and metrics_tracker is not None}")

# Cell 3.8: Visualization Functions Definition
# =====================================
# Defines the VisualizationManager class for creating plots later.
# Actual plotting requires data generated during/after training and evaluation.
# =====================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import traceback # Import traceback
import json # For loading history if needed, and saving per-class metrics
from itertools import cycle # Import cycle for palette generation if needed

# Ensure sklearn is installed for metrics/t-SNE: pip install scikit-learn
try:
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    from sklearn.manifold import TSNE
except ImportError:
    print("Warning: scikit-learn not found. Visualization functions requiring it (confusion_matrix, roc_curve, TSNE) might fail later.")
    # Define dummy functions if needed, or just let it fail later if called without sklearn
    confusion_matrix = None
    roc_curve = None
    auc = None
    TSNE = None
    pass
from typing import Dict, List, Optional # Added Optional

# Assuming logger, log_and_print are available from Cell 1.4
# Assuming config is available for SEED access

class VisualizationManager:
    """Handles creation and saving of various visualizations."""
    def __init__(self, result_dir: str):
        print(f"\n--- Initializing VisualizationManager ---")
        self.result_dir = result_dir
        # Use a style that generally looks good
        try:
            plt.style.use('seaborn-v0_8-whitegrid') # Try a seaborn style
        except:
            plt.style.use('default') # Fallback
            print("Seaborn style not found, using default.")


        self.viz_dir = os.path.join(result_dir, 'visualizations')
        try:
            os.makedirs(self.viz_dir, exist_ok=True)
            print(f"- Visualization directory ensured: {self.viz_dir}")
        except OSError as e:
            print(f"Error creating directory {self.viz_dir}: {e}")
            # Fallback if needed? For now, just print error.

        self.dpi = 1200 # Keep high DPI
        # Update rcParams for better aesthetics
        plt.rcParams.update({
            'figure.dpi': self.dpi, 'savefig.dpi': self.dpi,
            'figure.figsize': (12, 7), # Slightly larger default size
            'figure.autolayout': True, # Often helps, but can sometimes conflict with tight_layout
            'font.size': 11, 'axes.titlesize': 16, 'axes.labelsize': 13,
            'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 10,
            'lines.linewidth': 1.8, 'lines.markersize': 5,
            'axes.grid': True, # Ensure grid is on by default
            'grid.linestyle': '--',
            'grid.alpha': 0.6
        })
        print(f"- Matplotlib style set, DPI set to {self.dpi}.")
        log_and_print(logger, f"VisualizationManager initialized. Plots will be saved in: {self.viz_dir}")
        print(f"--- Finished Initializing VisualizationManager ---")

    # --- Plotting Methods (to be called later with data) ---

    def plot_learning_curves(self, metrics_history: Dict):
        """Plots training and validation metrics over epochs for all phases."""
        log_and_print(logger, "Attempting to generate learning curve plots...")
        print("\n--- Generating Learning Curve Plots ---")
        if not metrics_history:
            log_and_print(logger, "Metrics history is empty, cannot plot learning curves.", logging.WARNING)
            print("--- Finished Learning Curve Plots (No Data) ---")
            return

        # Define metrics to plot and their display names/units
        metrics_to_plot = {
            'loss': ('Loss', 'Loss'),
            'accuracy': ('Accuracy', 'Accuracy (%)'),
            'f1_weighted': ('F1 Score (Weighted)', 'F1 Score (%)'),
            'f1_macro': ('F1 Score (Macro)', 'F1 Score (%)'),
            'f1_micro': ('F1 Score (Micro)', 'F1 Score (%)'),
            'precision_weighted': ('Precision (Weighted)', 'Precision (%)'),
            'precision_macro': ('Precision (Macro)', 'Precision (%)'),
            'precision_micro': ('Precision (Micro)', 'Precision (%)'),
            'recall_weighted': ('Recall (Weighted)', 'Recall (%)'),
            'recall_macro': ('Recall (Macro)', 'Recall (%)'),
            'recall_micro': ('Recall (Micro)', 'Recall (%)'),
        }

        all_train_metrics = {m: [] for m in metrics_to_plot}
        all_val_metrics = {m: [] for m in metrics_to_plot}
        phase_boundaries = [0]
        total_epochs = 0

        # Consolidate data across phases
        phase_keys = sorted([p for p in metrics_history.keys() if p.startswith('phase')])
        if not phase_keys:
             log_and_print(logger, "No phase data found in metrics history.", logging.WARNING)
             print("--- Finished Learning Curve Plots (No Phase Data) ---")
             return

        print(f"Found phase keys: {phase_keys}")
        for phase_key in phase_keys:
             phase_num = int(phase_key.replace('phase',''))
             phase_data = metrics_history[phase_key]
             print(f"Processing {phase_key}...")
             if 'train' in phase_data and 'val' in phase_data:
                 min_epochs_phase = float('inf')
                 valid_phase = True
                 # Check if *all* required metrics are present and have same length
                 required_metrics_present = all(m in phase_data['train'] and m in phase_data['val'] for m in metrics_to_plot)
                 if not required_metrics_present:
                      log_and_print(logger, f"Warning: Missing one or more required metrics for {phase_key}. Skipping phase.", logging.WARNING)
                      print(f"  Skipping {phase_key} due to missing metrics.")
                      continue

                 # Check lengths
                 first_metric = list(metrics_to_plot.keys())[0]
                 train_len = len(phase_data['train'][first_metric])
                 val_len = len(phase_data['val'][first_metric])

                 if train_len == 0 or val_len == 0:
                      log_and_print(logger, f"Warning: Zero epochs found for {phase_key}. Skipping phase.", logging.WARNING)
                      print(f"  Skipping {phase_key} due to zero epochs.")
                      continue

                 if train_len != val_len:
                      log_and_print(logger, f"Warning: Mismatched train ({train_len}) and val ({val_len}) epoch counts for {phase_key}. Taking minimum.", logging.WARNING)
                      print(f"  Taking min epochs ({min(train_len, val_len)}) for {phase_key}.")
                      min_epochs_phase = min(train_len, val_len)
                 else:
                      min_epochs_phase = train_len

                 # Check all metrics have this length
                 for metric in metrics_to_plot:
                      if len(phase_data['train'][metric]) < min_epochs_phase or len(phase_data['val'][metric]) < min_epochs_phase:
                           log_and_print(logger, f"Warning: Metric '{metric}' has insufficient length for {phase_key}. Skipping phase.", logging.WARNING)
                           print(f"  Skipping {phase_key} due to insufficient length for {metric}.")
                           valid_phase = False
                           break
                 if not valid_phase: continue

                 # Append data
                 print(f"  Appending {min_epochs_phase} epochs from {phase_key}.")
                 for metric in metrics_to_plot:
                      all_train_metrics[metric].extend(phase_data['train'][metric][:min_epochs_phase])
                      all_val_metrics[metric].extend(phase_data['val'][metric][:min_epochs_phase])

                 total_epochs += min_epochs_phase
                 phase_boundaries.append(total_epochs)
             else:
                 log_and_print(logger, f"Warning: Missing 'train' or 'val' data for {phase_key}. Skipping phase.", logging.WARNING)
                 print(f"  Skipping {phase_key} due to missing train/val keys.")


        if total_epochs == 0:
             log_and_print(logger, "No valid epoch data found across phases. Cannot plot learning curves.", logging.WARNING)
             print("--- Finished Learning Curve Plots (No Epoch Data) ---")
             return

        epochs_range = range(1, total_epochs + 1)
        print(f"Plotting curves over {total_epochs} total epochs.")
        print(f"Phase boundaries (epoch end): {phase_boundaries}") # Print boundaries for debugging

        for metric, (title, ylabel) in metrics_to_plot.items():
             # Check if data exists for this metric before plotting
             if not all_train_metrics[metric] or not all_val_metrics[metric]:
                  log_and_print(logger, f"Skipping plot for '{metric}' due to missing data.", logging.WARNING)
                  continue
             try:
                 plt.figure(figsize=(12, 6)) # Reset figure size for each plot
                 plt.plot(epochs_range, all_train_metrics[metric], 'o-', label=f'Train {title}', markersize=4, linewidth=1.5)
                 plt.plot(epochs_range, all_val_metrics[metric], 's--', label=f'Validation {title}', markersize=4, linewidth=1.5)

                 # --- *** CORRECTED PHASE LABELING *** ---
                 # Iterate through the segments defined by the boundaries
                 for i in range(len(phase_boundaries) - 1):
                     start_epoch = phase_boundaries[i]
                     end_epoch = phase_boundaries[i+1]
                     phase_num = i + 1

                     # Calculate midpoint for text placement
                     text_pos_x = start_epoch + (end_epoch - start_epoch) / 2.0
                     # Adjust y position based on plot limits (calculate *after* plotting data)
                     y_min, y_max = plt.ylim()
                     text_pos_y = y_max - 0.03 * (y_max - y_min) # Place slightly below top

                     # Add phase text label
                     plt.text(text_pos_x, text_pos_y, f'Phase {phase_num}',
                              horizontalalignment='center', verticalalignment='top', fontsize=10, alpha=0.9,
                              bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, ec='none'))

                     # Add vertical boundary line *after* the phase (except for the last one)
                     if i < len(phase_boundaries) - 2: # Don't draw line after the last phase
                         plt.axvline(x=end_epoch + 0.5, color='grey', linestyle=':', linewidth=1.5, alpha=0.8)
                 # --- *** END CORRECTION *** ---

                 plt.title(f'Overall {title} Across All Phases', fontsize=16, pad=20)
                 plt.xlabel('Total Epochs', fontsize=13)
                 plt.ylabel(ylabel, fontsize=13)
                 plt.legend(fontsize=10)
                 plt.grid(True, linestyle='--', alpha=0.6) # Ensure grid is visible
                 plt.xlim(0.5, total_epochs + 0.5)
                 plt.xticks(np.arange(0, total_epochs + 1, step=max(1, total_epochs // 10)), fontsize=11) # Adjust x-ticks dynamically
                 plt.yticks(fontsize=11)
                 plt.tight_layout() # Adjust layout AFTER adding elements
                 save_path = os.path.join(self.viz_dir, f'learning_curve_{metric}_overall.png')
                 plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight') # Use bbox_inches='tight'
                 plt.show() # Display plot in notebook
                 plt.close() # Close plot to free memory
                 log_and_print(logger, f"Overall {title} learning curve saved to {save_path}")
             except Exception as e:
                  log_and_print(logger, f"Error plotting overall curve for metric '{metric}': {e}", logging.ERROR)
                  log_and_print(logger, traceback.format_exc(), logging.ERROR) # Log traceback
                  plt.close() # Ensure plot is closed even on error

        print("--- Finished Learning Curve Plots ---")


    def plot_confidence_distribution(self, correct_confidences: List[float], incorrect_confidences: List[float], phase: int):
        """Plots confidence score distributions."""
        log_and_print(logger, f"Attempting to generate confidence distribution plot for Phase {phase}...")
        print(f"\n--- Generating Confidence Distribution Plot for Phase {phase} ---")
        plt.figure(figsize=(10, 6))
        if not correct_confidences and not incorrect_confidences:
             log_and_print(logger, f"No confidence data provided for Phase {phase}. Skipping plot.", logging.WARNING)
             plt.text(0.5, 0.5, 'No confidence data available', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
             print("--- Finished Confidence Distribution Plot (No Data) ---")
        else:
            bins = np.linspace(0, 1, 31)
            if correct_confidences:
                 sns.histplot(correct_confidences, bins=bins, kde=True, stat='density', alpha=0.6, label=f'Correct ({len(correct_confidences)})', color='green')
            if incorrect_confidences:
                 sns.histplot(incorrect_confidences, bins=bins, kde=True, stat='density', alpha=0.6, label=f'Incorrect ({len(incorrect_confidences)})', color='red')

            plt.title(f'Phase {phase} Prediction Confidence Distribution', fontsize=16, pad=20)
            plt.xlabel('Confidence Score', fontsize=13)
            plt.ylabel('Density', fontsize=13)
            plt.legend(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.xlim(0, 1)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt.tight_layout()
            save_path = os.path.join(self.viz_dir, f'confidence_dist_phase{phase}.png')
            try:
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.show() # Display plot in notebook
                plt.close()
                log_and_print(logger, f"Confidence distribution plot saved to {save_path}")
            except Exception as e:
                log_and_print(logger, f"Error saving/showing confidence plot to {save_path}: {e}", logging.ERROR)
                plt.close()
            print("--- Finished Confidence Distribution Plot ---")


    def plot_confusion_matrices(self, true_labels: np.ndarray, predictions: np.ndarray,
                              emotion_names: List[str], phase: int):
        """Plots and saves a confusion matrix, also saves numerical data."""
        log_and_print(logger, f"Attempting to generate confusion matrix plot and data for Phase {phase}...")
        print(f"\n--- Generating Confusion Matrix Plot & Data for Phase {phase} ---")
        if confusion_matrix is None:
             log_and_print(logger, "Skipping confusion matrix: sklearn.metrics.confusion_matrix not available.", logging.WARNING)
             print("--- Finished Confusion Matrix (Skipped) ---")
             return
        if len(true_labels) == 0 or len(predictions) == 0:
             log_and_print(logger, f"No labels/predictions provided for Phase {phase}. Skipping confusion matrix.", logging.WARNING)
             print("--- Finished Confusion Matrix (No Data) ---")
             return

        # Define save paths
        plot_save_path = os.path.join(self.viz_dir, f'confusion_matrix_phase{phase}.png')
        txt_save_path = os.path.join(self.viz_dir, f'confusion_matrix_phase{phase}.txt')

        try:
            # Calculate CM
            labels_idx = np.arange(len(emotion_names))
            cm = confusion_matrix(true_labels, predictions, labels=labels_idx)

            # Save numerical data
            try:
                # Add row/column headers for clarity in text file
                cm_df = pd.DataFrame(cm, index=emotion_names, columns=emotion_names)
                cm_df.to_csv(txt_save_path, sep='\t') # Use tab separation for better alignment
                # np.savetxt(txt_save_path, cm, fmt='%d', delimiter=',', header=','.join(emotion_names), comments='') # Old way
                log_and_print(logger, f"Numerical confusion matrix saved to {txt_save_path}")
            except Exception as e_save:
                log_and_print(logger, f"Error saving numerical confusion matrix to {txt_save_path}: {e_save}", logging.ERROR)

            # Plotting
            plt.figure(figsize=(max(8, len(emotion_names)*0.9), max(7, len(emotion_names)*0.8))) # Dynamic sizing
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized) # Handle division by zero if a class has no true samples

            sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                        xticklabels=emotion_names, yticklabels=emotion_names,
                        annot_kws={'size': 9 if len(emotion_names) <= 10 else 7},
                        linewidths=0.5, linecolor='lightgray', cbar=True, square=True, cbar_kws={'shrink': .7})

            plt.title(f'Phase {phase} Confusion Matrix (Row Normalized, Counts Annot.)', fontsize=16, pad=20)
            plt.xlabel('Predicted Label', fontsize=13)
            plt.ylabel('True Label', fontsize=13)
            plt.xticks(rotation=45, ha='right', fontsize=11)
            plt.yticks(rotation=0, fontsize=11)
            plt.tight_layout()
            plt.savefig(plot_save_path, dpi=self.dpi, bbox_inches='tight')
            plt.show() # Display plot in notebook
            plt.close()
            log_and_print(logger, f"Confusion matrix plot saved to {plot_save_path}")

        except Exception as e:
             log_and_print(logger, f"Error generating confusion matrix plot/data for Phase {phase}: {e}", logging.ERROR)
             log_and_print(logger, traceback.format_exc(), logging.ERROR)
             plt.close()
        print("--- Finished Confusion Matrix Plot & Data ---")


    def plot_roc_curves(self, true_labels: np.ndarray, pred_probs: np.ndarray,
                      emotion_names: List[str], phase: int):
        """Plots and saves ROC curves for each class, also saves numerical data."""
        log_and_print(logger, f"Attempting to generate ROC curve plot and data for Phase {phase}...")
        print(f"\n--- Generating ROC Curve Plot & Data for Phase {phase} ---")
        if roc_curve is None or auc is None:
             log_and_print(logger, "Skipping ROC curves: sklearn.metrics.roc_curve/auc not available.", logging.WARNING)
             print("--- Finished ROC Curve (Skipped) ---")
             return
        if len(true_labels) == 0 or len(pred_probs) == 0:
             log_and_print(logger, f"No labels/probabilities provided for Phase {phase}. Skipping ROC curves.", logging.WARNING)
             print("--- Finished ROC Curve (No Data) ---")
             return

        n_classes = len(emotion_names)
        if pred_probs.shape[1] != n_classes:
             log_and_print(logger, f"Mismatch between pred_probs columns ({pred_probs.shape[1]}) and n_classes ({n_classes}). Skipping ROC.", logging.ERROR)
             print("--- Finished ROC Curve (Shape Mismatch) ---")
             return

        # Define save paths
        plot_save_path = os.path.join(self.viz_dir, f'roc_curves_phase{phase}.png')
        csv_save_path = os.path.join(self.viz_dir, f'roc_data_phase{phase}.csv')
        roc_data_list = []

        try:
            plt.figure(figsize=(10, 8))
            fpr, tpr, roc_auc_dict = dict(), dict(), dict()

            # Setup colors for better distinction if many classes
            # Use matplotlib's colormaps module directly
            try:
                cmap = plt.colormaps.get_cmap('tab10') if n_classes <= 10 else plt.colormaps.get_cmap('tab20')
                colors = [cmap(i) for i in range(n_classes)]
            except AttributeError: # Fallback for older matplotlib
                cmap = plt.cm.get_cmap('tab10') if n_classes <= 10 else plt.cm.get_cmap('tab20')
                colors = [cmap(i) for i in range(n_classes)]


            for i in range(n_classes):
                 y_true_binary = (true_labels == i).astype(int)
                 class_probs = pred_probs[:, i]

                 if len(np.unique(y_true_binary)) < 2:
                      log_and_print(logger, f"Warning: Cannot compute ROC for class '{emotion_names[i]}' - only one class present in true labels.", logging.WARNING)
                      fpr[i], tpr[i], roc_auc_dict[i] = np.array([0, 1]), np.array([0, 1]), 0.0 # Assign default values
                 else:
                      try:
                          fpr[i], tpr[i], _ = roc_curve(y_true_binary, class_probs)
                          roc_auc_dict[i] = auc(fpr[i], tpr[i])
                      except Exception as roc_e:
                           log_and_print(logger, f"Error calculating ROC/AUC for class '{emotion_names[i]}': {roc_e}", logging.WARNING)
                           fpr[i], tpr[i], roc_auc_dict[i] = np.array([0, 1]), np.array([0, 1]), 0.0 # Assign default values

                 # Plotting
                 plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=1.8, label=f'{emotion_names[i]} (AUC = {roc_auc_dict[i]:.3f})')

                 # Store data for CSV
                 for f, t in zip(fpr[i], tpr[i]):
                      roc_data_list.append({'Phase': phase, 'Emotion': emotion_names[i], 'FPR': f, 'TPR': t, 'AUC': roc_auc_dict[i]})

            # Save numerical data
            try:
                roc_df = pd.DataFrame(roc_data_list)
                roc_df.to_csv(csv_save_path, index=False)
                log_and_print(logger, f"ROC curve data saved to {csv_save_path}")
            except Exception as e_save:
                log_and_print(logger, f"Error saving ROC curve data to {csv_save_path}: {e_save}", logging.ERROR)


            # Finalize plot
            plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance Level (AUC = 0.500)')
            plt.xlim([-0.02, 1.0])
            plt.ylim([0.0, 1.02])
            plt.xlabel('False Positive Rate', fontsize=13)
            plt.ylabel('True Positive Rate', fontsize=13)
            plt.title(f'Phase {phase} ROC Curves (One-vs-Rest)', fontsize=16, pad=20)
            plt.legend(loc="lower right", fontsize=9)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt.tight_layout()
            plt.savefig(plot_save_path, dpi=self.dpi, bbox_inches='tight')
            plt.show() # Display plot in notebook
            plt.close()
            log_and_print(logger, f"ROC curves plot saved to {plot_save_path}")
        except Exception as e:
             log_and_print(logger, f"Error generating ROC curves plot/data for Phase {phase}: {e}", logging.ERROR)
             log_and_print(logger, traceback.format_exc(), logging.ERROR)
             plt.close()
        print("--- Finished ROC Curve Plot & Data ---")


    def save_metrics_summary(self, all_phase_metrics: Dict[int, Dict]):
        """Save a text summary of final metrics for all phases."""
        log_and_print(logger, "Attempting to save metrics summary...")
        print("\n--- Saving Metrics Summary ---")
        if not all_phase_metrics:
             log_and_print(logger, "No phase metrics provided, cannot save summary.", logging.WARNING)
             print("--- Finished Saving Metrics Summary (No Data) ---")
             return

        summary_path = os.path.join(self.result_dir, 'metrics_summary.txt')
        try:
            with open(summary_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("          FINAL EVALUATION METRICS SUMMARY\n")
                f.write("="*60 + "\n\n")
                # Add FLOPs/Params at the top if available from phase 3
                first_phase_key = min(all_phase_metrics.keys()) if all_phase_metrics else None # Get first available phase key
                flops = "N/A"
                params = "N/A"
                if first_phase_key and first_phase_key in all_phase_metrics:
                    flops = all_phase_metrics[first_phase_key].get('Estimated FLOPs', 'N/A')
                    params = all_phase_metrics[first_phase_key].get('Parameters', 'N/A')

                f.write(f"Model Complexity:\n")
                f.write(f"- Estimated FLOPs : {flops}\n")
                f.write(f"- Parameters      : {params}\n\n")

                for phase in sorted(all_phase_metrics.keys()):
                    metrics = all_phase_metrics[phase]
                    f.write(f"--- Phase {phase} Best Model Metrics ---\n")
                    # Define order and formatting
                    metric_order = [
                        ('loss', 'Avg Loss', '{:.4f}'),
                        ('accuracy', 'Accuracy', '{:.2f}%'),
                        ('f1_weighted', 'F1 Weighted', '{:.2f}%'),
                        ('f1_macro', 'F1 Macro', '{:.2f}%'),
                        ('f1_micro', 'F1 Micro', '{:.2f}%'),
                        ('precision_weighted', 'Precision Weighted', '{:.2f}%'),
                        ('precision_macro', 'Precision Macro', '{:.2f}%'),
                        ('precision_micro', 'Precision Micro', '{:.2f}%'),
                        ('recall_weighted', 'Recall Weighted', '{:.2f}%'),
                        ('recall_macro', 'Recall Macro', '{:.2f}%'),
                        ('recall_micro', 'Recall Micro', '{:.2f}%'),
                        ('top_2_accuracy', 'Top-2 Accuracy', '{:.2f}%'),
                        ('top_3_accuracy', 'Top-3 Accuracy', '{:.2f}%'),
                    ]
                    for key, display_name, fmt in metric_order:
                        if key in metrics:
                            value = metrics[key]
                            # Handle potential None values before formatting
                            if value is None: value_str = "N/A"
                            elif isinstance(value, str): value_str = value # Already string (like FLOPs/Params)
                            else:
                                try:
                                    value_str = fmt.format(value)
                                except (TypeError, ValueError): # Handle cases where format might fail
                                    value_str = str(value)
                            f.write(f"- {display_name:<20}: {value_str}\n")

                    # Add per-emotion summary (optional, can make file long)
                    # if 'per_emotion' in metrics:
                    #     f.write("\n  Per-Emotion F1 Scores:\n")
                    #     for emotion, e_metrics in metrics['per_emotion'].items():
                    #         f.write(f"    - {emotion:<15}: {e_metrics.get('f1_score', 0.0):.2f}%\n")
                    f.write("\n")

            log_and_print(logger, f"Metrics summary saved to {summary_path}")
        except Exception as e:
            log_and_print(logger, f"Error saving metrics summary to {summary_path}: {e}", logging.ERROR)
        print("--- Finished Saving Metrics Summary ---")

    def save_per_emotion_metrics(self, per_emotion_metrics: Dict[str, Dict[str, float]], phase: int):
        """Saves the detailed per-emotion metrics dictionary to a JSON file."""
        log_and_print(logger, f"Attempting to save per-emotion metrics for Phase {phase}...")
        print(f"\n--- Saving Per-Emotion Metrics for Phase {phase} ---")
        if not per_emotion_metrics:
            log_and_print(logger, f"No per-emotion metrics provided for Phase {phase}. Skipping save.", logging.WARNING)
            print("--- Finished Saving Per-Emotion Metrics (No Data) ---")
            return

        json_save_path = os.path.join(self.viz_dir, f'per_emotion_metrics_phase{phase}.json')
        try:
            # Ensure data is serializable (should be floats/ints already)
            with open(json_save_path, 'w') as f:
                json.dump(per_emotion_metrics, f, indent=4)
            log_and_print(logger, f"Per-emotion metrics saved to {json_save_path}")
        except Exception as e:
            log_and_print(logger, f"Error saving per-emotion metrics to {json_save_path}: {e}", logging.ERROR)
        print("--- Finished Saving Per-Emotion Metrics ---")


    def plot_tsne_features(self, features: np.ndarray, labels: np.ndarray,
                           emotion_names: List[str], title: str, filename_prefix: str):
        """Generates and saves a t-SNE plot of features, also saves numerical data."""
        log_and_print(logger, f"Attempting to generate t-SNE plot: {title}")
        print(f"\n--- Generating t-SNE Plot: {title} ---")

        if TSNE is None:
             log_and_print(logger, "Skipping t-SNE plot: sklearn.manifold.TSNE not available.", logging.WARNING)
             print("--- Finished t-SNE Plot (Skipped) ---")
             return
        if features is None or labels is None or features.shape[0] != labels.shape[0] or features.shape[0] == 0:
             log_and_print(logger, f"Invalid or empty features/labels provided for t-SNE: {title}. Skipping plot.", logging.WARNING)
             print("--- Finished t-SNE Plot (Invalid Data) ---")
             return

        n_samples = features.shape[0]
        # Adjust perplexity: Must be less than n_samples. Common values are 5-50.
        perplexity_value = min(max(5.0, n_samples / 10.0), 50.0, n_samples - 1.0) # Ensure < n_samples
        if n_samples <= 5: perplexity_value = max(1.0, n_samples - 1.0) # Handle very small sample sizes

        # Use config seed if available
        tsne_seed = self.config.SEED if hasattr(self, 'config') and hasattr(self.config, 'SEED') else 42

        log_and_print(logger, f"Running t-SNE with n_samples={n_samples}, perplexity={perplexity_value:.1f}, seed={tsne_seed}")

        # Define save paths
        plot_save_path = os.path.join(self.viz_dir, f"{filename_prefix}.png")
        csv_save_path = os.path.join(self.viz_dir, f"{filename_prefix}_data.csv")

        try:
            # Perform t-SNE
            tsne = TSNE(n_components=2, random_state=tsne_seed,
                        perplexity=perplexity_value, max_iter=300, learning_rate='auto', # Use max_iter instead of n_iter
                        init='pca') # Use PCA init for stability
            tsne_results = tsne.fit_transform(features)
            log_and_print(logger, "t-SNE computation complete.")

            # Create DataFrame for plotting and saving
            df = pd.DataFrame({
                'tsne-1': tsne_results[:, 0],
                'tsne-2': tsne_results[:, 1],
                'label': labels,
                'emotion': [emotion_names[l] if l < len(emotion_names) else 'Unknown' for l in labels]
            })

            # Save numerical data
            try:
                df.to_csv(csv_save_path, index=False)
                log_and_print(logger, f"t-SNE data saved to {csv_save_path}")
            except Exception as e_save:
                log_and_print(logger, f"Error saving t-SNE data to {csv_save_path}: {e_save}", logging.ERROR)

            # Plotting
            plt.figure(figsize=(10, 8))
            n_classes = len(emotion_names)
            # --- FIX: Use a standard Seaborn palette name ---
            palette_name = "tab10" if n_classes <= 10 else "tab20" # Choose appropriate palette

            scatter = sns.scatterplot(
                x="tsne-1", y="tsne-2",
                hue="emotion",
                hue_order=emotion_names, # Ensure consistent color mapping
                palette=palette_name,    # Use the palette name string
                data=df,
                legend="full",
                alpha=0.7,
                s=50 # Adjust marker size
            )

            plt.title(title, fontsize=16, pad=20)
            plt.xlabel("t-SNE Dimension 1", fontsize=13)
            plt.ylabel("t-SNE Dimension 2", fontsize=13)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            # Place legend outside plot
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=9)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

            plt.savefig(plot_save_path, dpi=self.dpi, bbox_inches='tight')
            plt.show() # Display plot in notebook
            plt.close()
            log_and_print(logger, f"t-SNE plot saved to {plot_save_path}")

        except Exception as e:
             log_and_print(logger, f"Error generating t-SNE plot/data for '{title}': {e}", logging.ERROR)
             log_and_print(logger, traceback.format_exc(), logging.ERROR)
             plt.close()
        print(f"--- Finished t-SNE Plot & Data: {title} ---")


# --- Execution Block ---
print("\n--- Starting Cell 3.8 Execution ---")
# This cell defines the VisualizationManager class.
# We instantiate it to ensure initialization works and logs are created.
# The actual plotting methods require data from training/evaluation and will be called later.
if 'config' in locals() and 'logger' in locals():
    try:
        # Ensure config object is passed if needed by methods (like for SEED in t-SNE)
        viz_manager = VisualizationManager(config.RESULT_DIR)
        # Pass config to viz_manager if needed, e.g., viz_manager.config = config
        if hasattr(config, 'SEED'):
             viz_manager.config = config # Make config accessible for SEED
        log_and_print(logger, "VisualizationManager class defined and instance created successfully.")
        print("Note: Plotting methods within VisualizationManager will be called in later cells after training/evaluation.")
    except Exception as e:
        log_and_print(logger, f"Error initializing VisualizationManager: {e}", logging.ERROR)
        log_and_print(logger, traceback.format_exc(), logging.ERROR)
        viz_manager = None
else:
    print("Error: 'config' or 'logger' not defined. Cannot initialize VisualizationManager.")
    viz_manager = None

print("--- Finished Cell 3.8 Execution ---")
print("-" * 50)

# Cell 3.9: Evaluation Functions
# =====================================
# Model Evaluation and Performance Analysis Functions
# =====================================

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import os
import traceback
from typing import Dict, List, Tuple, Optional

# Ensure sklearn metrics are imported
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score # Removed top_k_accuracy_score as we calculate manually
)
# Import visualization manager defined in Cell 3.8
# Assuming VisualizationManager, logger, log_and_print, CombinedLoss are available

# Import FLOPs counter library (install if needed: pip install thop)
try:
    from thop import profile, clever_format
except ImportError:
    print("Warning: 'thop' library not found. FLOPs calculation will be skipped. Install with: pip install thop")
    profile = None # Set profile to None if thop is not available
    clever_format = None

# Assuming initialize_model is available from Cell 3.7 or similar
if 'initialize_model' not in globals():
     # Define a dummy if it's missing, though it should be present
     def initialize_model(*args, **kwargs):
         print("Warning: initialize_model function not found, using dummy.")
         return None
     log_and_print(logger, "Warning: initialize_model function not found.", logging.WARNING)


class ModelEvaluator:
    """Handles comprehensive evaluation of the trained model across phases."""
    def __init__(self, model: Optional[nn.Module], test_loader: DataLoader, device: torch.device,
                 result_dir: str, emotion_names: List[str], config: Config): # Added config, model can be None initially
        print("\n--- Initializing ModelEvaluator ---")
        self.model = model # Store the initial model (might be None or last trained)
        self.test_loader = test_loader # Use validation loader passed as test_loader
        self.device = device
        self.emotion_names = emotion_names
        self.num_classes = len(emotion_names)
        self.result_dir = result_dir
        self.config = config # Store config

        # Instantiate VisualizationManager
        if 'VisualizationManager' in globals() and callable(VisualizationManager):
             self.viz_manager = VisualizationManager(result_dir)
             # Pass config if needed by viz_manager methods
             if hasattr(self.config, 'SEED'):
                 self.viz_manager.config = self.config
             print("- VisualizationManager instantiated.")
        else:
             log_and_print(logger, "Error: VisualizationManager class not found or not callable.", logging.ERROR)
             self.viz_manager = None # Set to None if class not found

        # Instantiate Loss Handler (needed for loss calculation during eval)
        if 'CombinedLoss' in globals() and callable(CombinedLoss):
             # Assuming class_weights are not needed or handled elsewhere for eval loss
             self.loss_handler = CombinedLoss(config=config, num_classes=self.num_classes, class_weights=None)
             print("- CombinedLoss handler instantiated for evaluation.")
        else:
             log_and_print(logger, "Error: CombinedLoss class not found or not callable.", logging.ERROR)
             self.loss_handler = None

        print(f"- Evaluating on device: {self.device}")
        print(f"- Number of classes: {self.num_classes}")
        print(f"--- Finished Initializing ModelEvaluator ---")


    def evaluate_phase(self, phase: int) -> Optional[Dict]:
        """
        Evaluate model performance for a specific phase on the test_loader data.
        Also extracts features for t-SNE.

        Args:
            phase (int): Training phase number (1, 2, or 3).

        Returns:
            Optional[Dict]: Dictionary containing evaluation metrics, or None if evaluation fails.
        """
        log_and_print(logger, f"\n--- Starting Evaluation for Phase {phase} ---")
        if not self.model:
             log_and_print(logger, "Error: Model not loaded in evaluator. Cannot evaluate.", logging.ERROR)
             return None
        if not self.loss_handler:
             log_and_print(logger, "Error: Loss handler not initialized. Cannot calculate loss.", logging.ERROR)
             return None
        if not self.viz_manager:
             log_and_print(logger, "Warning: VisualizationManager not initialized. Plots will not be generated.", logging.WARNING)


        self.model.eval() # Set model to evaluation mode
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        num_samples = 0

        # --- Storage for t-SNE features ---
        # Store features after specific layers
        # Dict structure: features_for_tsne[feature_name] = list_of_batch_features
        features_for_tsne = {
            # BiLSTM Block Outputs (Combined T+A+V)
            'bilstm_block_1': [], 'bilstm_block_2': [], 'bilstm_block_3': [],
            # Standardized Features (Separate first, then combined)
            'std_text': [], 'std_audio': [], 'std_visual': [],
            'std_combined': [] # For combined standardized features
        }
        # ------------------------------------

        pbar = tqdm(self.test_loader, desc=f'Evaluating Phase {phase}', leave=False)

        with torch.no_grad(): # Disable gradient calculations
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move batch to device
                    batch_device = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    targets = batch_device['label']
                    batch_size = targets.size(0)

                    # Forward pass (AMP context optional but good practice)
                    with autocast(enabled=self.config.USE_AMP):
                        outputs = self.model(batch_device, phase=phase)
                        if 'final_logits' not in outputs:
                             log_and_print(logger, f"Warning: 'final_logits' missing in eval output for batch {batch_idx}, phase {phase}. Skipping batch.", logging.WARNING)
                             continue
                        # Calculate loss (using dummy epoch 0 for loss calculation)
                        loss, _ = self.loss_handler(outputs, targets, phase=phase, epoch=0) # Use underscore for loss_dict

                    # Accumulate loss
                    total_loss += loss.item() * batch_size
                    num_samples += batch_size

                    # Get predictions and probabilities
                    logits = outputs['final_logits']
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)

                    # Store results (move to CPU)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(targets.cpu().numpy())
                    all_probs.append(probs.cpu().numpy())

                    # --- Extract and store features for t-SNE (move to CPU) ---
                    # Standardized features (Store separately first)
                    batch_std_features = {}
                    if 'std_features' in outputs:
                        for mod in ['text', 'audio', 'visual']:
                            if mod in outputs['std_features']:
                                mod_feat_cpu = outputs['std_features'][mod].cpu().numpy()
                                features_for_tsne[f'std_{mod}'].append(mod_feat_cpu)
                                batch_std_features[mod] = mod_feat_cpu # Keep for combining

                    # Combine standardized features for this batch
                    if len(batch_std_features) == 3: # Ensure all 3 modalities are present
                         combined_std = np.concatenate(
                             [batch_std_features['text'], batch_std_features['audio'], batch_std_features['visual']],
                             axis=-1
                         )
                         features_for_tsne['std_combined'].append(combined_std)
                    elif batch_std_features: # Log if not all 3 were found
                         log_and_print(logger, f"Warning: Not all std features found in batch {batch_idx} for combining. Keys: {list(batch_std_features.keys())}", logging.WARNING)


                    # BiLSTM block outputs
                    if 'bilstm_block_outputs' in outputs:
                        bilstm_outputs = outputs['bilstm_block_outputs']
                        # Process each block (assuming 3 blocks)
                        for block_idx in range(3):
                            block_features_batch = []
                            # Combine features from all modalities for this block
                            for mod in ['text', 'audio', 'visual']:
                                if mod in bilstm_outputs and bilstm_outputs[mod] is not None and len(bilstm_outputs[mod]) > block_idx:
                                    # Need to handle sequence length: take last time step? mean pool?
                                    # Let's take the last time step for simplicity, similar to final output
                                    mod_block_output = bilstm_outputs[mod][block_idx] # Shape (B, S, D)
                                    if mod_block_output.shape[1] > 0: # Check seq len > 0
                                         block_features_batch.append(mod_block_output[:, -1, :].cpu().numpy()) # Take last time step -> (B, D)
                                    else: # Handle empty sequence case
                                         dummy_feat = np.zeros((batch_size, mod_block_output.shape[2]), dtype=np.float32)
                                         block_features_batch.append(dummy_feat)

                            # Concatenate features from T, A, V for this block if available
                            if len(block_features_batch) == 3: # Ensure all 3 modalities were processed
                                combined_block_features = np.concatenate(block_features_batch, axis=-1) # Shape (B, D_t+D_a+D_v = 1536)
                                features_for_tsne[f'bilstm_block_{block_idx+1}'].append(combined_block_features)
                            elif block_features_batch: # Log if not all 3 were found
                                 log_and_print(logger, f"Warning: Not all BiLSTM block {block_idx+1} features found in batch {batch_idx} for combining.", logging.WARNING)

                    # ---------------------------------------------------------

                    # Update progress bar postfix
                    loss_val = loss.item()
                    pbar.set_postfix({'loss': f"{loss_val:.4f}"})


                except Exception as e:
                    log_and_print(logger, f"Error during evaluation phase {phase}, batch {batch_idx}: {e}", logging.ERROR)
                    log_and_print(logger, traceback.format_exc(), logging.ERROR)
                    continue # Skip to next batch

        if num_samples == 0:
             log_and_print(logger, f"Error: No samples were processed during evaluation for Phase {phase}.", logging.ERROR)
             return None

        # --- Post-Loop Calculations ---
        avg_loss = total_loss / num_samples
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.concatenate(all_probs, axis=0)

        # --- Concatenate t-SNE features ---
        final_tsne_features = {}
        for name, feat_list in features_for_tsne.items():
            if feat_list:
                try:
                    final_tsne_features[name] = np.concatenate(feat_list, axis=0)
                    log_and_print(logger, f"Concatenated features for t-SNE '{name}', shape: {final_tsne_features[name].shape}")
                except ValueError as ve:
                     log_and_print(logger, f"Error concatenating features for t-SNE '{name}': {ve}. Skipping.", logging.ERROR)
                     final_tsne_features[name] = None # Mark as None if concatenation fails
            else:
                final_tsne_features[name] = None
        # ---------------------------------


        # Calculate overall metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0) # Added micro
        precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0) # Added macro
        precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0) # Added micro
        recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0) # Added macro
        recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0) # Added micro

        # Calculate Top-K Accuracy manually
        top_k_values = [k for k in [2, 3] if k < self.num_classes] # Calculate Top-2, Top-3 if possible
        top_k_acc = {}
        for k in top_k_values:
             try:
                 # Get indices of top k probs for each sample
                 top_k_preds_indices = np.argsort(all_probs, axis=1)[:, -k:]
                 # Check if the true label is within the top k predictions for each sample
                 correct_k = np.any(top_k_preds_indices == all_labels[:, np.newaxis], axis=1)
                 top_k_acc[f'top_{k}_accuracy'] = np.mean(correct_k) * 100
             except Exception as topk_e:
                 log_and_print(logger, f"Could not calculate Top-{k} accuracy: {topk_e}", logging.WARNING)
                 top_k_acc[f'top_{k}_accuracy'] = 0.0


        # Generate classification report (text)
        try:
            report_str = classification_report(all_labels, all_preds, target_names=self.emotion_names, zero_division=0, digits=4)
            log_and_print(logger, f"\nClassification Report (Phase {phase}):\n{report_str}")
            # Save report to file
            report_path = os.path.join(self.result_dir, f'classification_report_phase{phase}.txt')
            with open(report_path, 'w') as f: f.write(report_str)
            log_and_print(logger, f"Classification report saved to {report_path}")
        except Exception as report_e:
             log_and_print(logger, f"Could not generate/save classification report: {report_e}", logging.WARNING)


        # Calculate per-emotion metrics (more detailed than classification_report)
        per_emotion_metrics = self.calculate_per_emotion_metrics(all_labels, all_preds, all_probs)

        # --- Generate Visualizations & Save Data ---
        if self.viz_manager:
            # Standard plots + data saving
            self.viz_manager.plot_confusion_matrices(all_labels, all_preds, self.emotion_names, phase)
            self.viz_manager.plot_roc_curves(all_labels, all_probs, self.emotion_names, phase)
            self.viz_manager.save_per_emotion_metrics(per_emotion_metrics, phase) # Save detailed per-class metrics

            # --- Updated t-SNE plotting ---
            # 1. BiLSTM Block Outputs (Combined T+A+V)
            for block_idx in range(1, 4): # Blocks 1, 2, 3
                feature_key = f'bilstm_block_{block_idx}'
                features = final_tsne_features.get(feature_key)
                if features is not None and features.shape[0] == len(all_labels):
                    # Use "Layer" in title as requested
                    title = f"Phase {phase}: BiLSTM Layer {block_idx} Features (T+A+V Combined)"
                    filename_prefix = f"tsne_bilstm_layer{block_idx}_phase{phase}" # Use "layer" in filename too
                    self.viz_manager.plot_tsne_features(features, all_labels, self.emotion_names, title, filename_prefix)
                elif features is not None:
                     log_and_print(logger, f"Skipping t-SNE for '{feature_key}' due to shape mismatch: Features {features.shape[0]}, Labels {len(all_labels)}", logging.WARNING)
                else:
                     log_and_print(logger, f"Skipping t-SNE for '{feature_key}' due to missing features.", logging.WARNING)

            # 2. Combined Standardized Features (Post-Cross-Attention)
            feature_key = 'std_combined'
            features = final_tsne_features.get(feature_key)
            if features is not None and features.shape[0] == len(all_labels):
                # Use "Cross-Attention" in title as requested
                title = f"Phase {phase}: Combined Standardized Features (Post-Cross-Attention)"
                filename_prefix = f"tsne_cross_attention_phase{phase}" # Use "cross_attention" in filename
                self.viz_manager.plot_tsne_features(features, all_labels, self.emotion_names, title, filename_prefix)
            elif features is not None:
                 log_and_print(logger, f"Skipping t-SNE for '{feature_key}' due to shape mismatch: Features {features.shape[0]}, Labels {len(all_labels)}", logging.WARNING)
            else:
                 log_and_print(logger, f"Skipping t-SNE for '{feature_key}' due to missing features.", logging.WARNING)

            # Note: We no longer plot std_text, std_audio, std_visual separately

        # Compile final metrics dictionary
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy * 100,
            'f1_weighted': f1_weighted * 100,
            'f1_macro': f1_macro * 100,
            'f1_micro': f1_micro * 100,
            'precision_weighted': precision_weighted * 100,
            'precision_macro': precision_macro * 100,
            'precision_micro': precision_micro * 100,
            'recall_weighted': recall_weighted * 100,
            'recall_macro': recall_macro * 100,
            'recall_micro': recall_micro * 100,
            **top_k_acc, # Add top-k metrics
            'per_emotion': per_emotion_metrics # Keep detailed per-emotion dict
        }

        log_and_print(logger, f"--- Finished Evaluation for Phase {phase} ---")
        return metrics

    def calculate_per_emotion_metrics(self, true_labels: np.ndarray, predictions: np.ndarray,
                                    probabilities: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate detailed metrics for each emotion class."""
        per_emotion_metrics = {}
        print("\nCalculating Per-Emotion Metrics...")

        for i, emotion in enumerate(self.emotion_names):
            try:
                true_binary = (true_labels == i).astype(int)
                pred_binary = (predictions == i).astype(int)
                prob_binary = probabilities[:, i]

                # Basic metrics
                acc = accuracy_score(true_binary, pred_binary)
                prec = precision_score(true_binary, pred_binary, zero_division=0)
                rec = recall_score(true_binary, pred_binary, zero_division=0)
                f1 = f1_score(true_binary, pred_binary, zero_division=0)

                # Confusion matrix components for binary case
                # Ensure labels=[0, 1] to handle cases where one class might be missing in this binary view
                cm_binary = confusion_matrix(true_binary, pred_binary, labels=[0, 1])
                tn, fp, fn, tp = 0, 0, 0, 0 # Initialize
                if cm_binary.size == 4: # Check if it's a 2x2 matrix
                     tn, fp, fn, tp = cm_binary.ravel()
                else: # Handle cases where confusion matrix is not 2x2 (e.g., only one class predicted/present)
                     log_and_print(logger, f"Warning: Confusion matrix for class '{emotion}' was not 2x2. TN/FP/FN/TP might be inaccurate.", logging.WARNING)
                     # Attempt to infer from shape if possible, otherwise leave as 0
                     if true_binary.sum() == 0: # No positive samples
                         tn = cm_binary[0,0] if cm_binary.shape == (1,1) else 0
                         fp = 0
                         fn = 0
                         tp = 0
                     elif (1-true_binary).sum() == 0: # No negative samples
                         tn = 0
                         fp = 0
                         fn = 0
                         tp = cm_binary[0,0] if cm_binary.shape == (1,1) else 0


                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                # ROC AUC
                roc_auc = 0.0
                if len(np.unique(true_binary)) > 1: # Check if both classes are present
                     try:
                         roc_auc = roc_auc_score(true_binary, prob_binary)
                     except ValueError:
                         roc_auc = 0.0 # Handle cases where roc_auc_score fails

                per_emotion_metrics[emotion] = {
                    'accuracy': acc * 100,
                    'precision': prec * 100,
                    'recall (Sensitivity)': rec * 100,
                    'f1_score': f1 * 100,
                    'specificity': specificity * 100,
                    'roc_auc': roc_auc,
                    'support': int(true_binary.sum()), # Number of true samples for this class
                    'TP': int(tp), # Add TP, TN, FP, FN
                    'TN': int(tn),
                    'FP': int(fp),
                    'FN': int(fn)
                }
            except Exception as per_emotion_e:
                 log_and_print(logger, f"Error calculating metrics for emotion '{emotion}': {per_emotion_e}", logging.ERROR)
                 per_emotion_metrics[emotion] = {metric: 0.0 for metric in ['accuracy', 'precision', 'recall (Sensitivity)', 'f1_score', 'specificity', 'roc_auc']}
                 per_emotion_metrics[emotion]['support'] = int((true_labels == i).sum())
                 per_emotion_metrics[emotion].update({'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0})


        print("Finished Calculating Per-Emotion Metrics.")
        return per_emotion_metrics

    def estimate_flops_params(self) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Estimates FLOPs and parameters using thop."""
        log_and_print(logger, "Estimating FLOPs and Parameters...")
        if not self.model:
             log_and_print(logger, "Model not available for FLOPs estimation.", logging.WARNING)
             return "N/A", "N/A", None
        if profile is None:
             log_and_print(logger, "Cannot estimate FLOPs because 'thop' library is not installed.", logging.WARNING)
             params = sum(p.numel() for p in self.model.parameters())
             params_str = clever_format([params], "%.3f")[0] if clever_format else f"{params:,}"
             return "N/A", params_str, int(params)

        # Need a sample input batch matching the model's forward signature
        # Get one batch from the loader
        try:
            sample_batch = next(iter(self.test_loader))
            # Move only tensors to device, keep others (like utterances list) on CPU
            dummy_batch_for_thop = {}
            # Ensure all required keys are present, even if just with dummy data of correct type/shape
            # Use the actual model input keys if possible
            # Assuming the model forward expects the batch dict directly
            required_keys = ['input_ids', 'attention_mask', 'v1', 'v3', 'v4', 'a2', 'label'] # Label might not be needed by forward
            for key in required_keys:
                 if key in sample_batch:
                     v = sample_batch[key]
                     if isinstance(v, torch.Tensor):
                         # Use only the first item of the batch for FLOPs calculation to reduce memory
                         dummy_batch_for_thop[key] = v[0:1].to(self.device)
                     # else: # thop usually doesn't need non-tensor inputs like utterance lists
                         # Handle non-tensor data (like utterances list) - take first item
                         # dummy_batch_for_thop[key] = v[0:1] if isinstance(v, list) else v
                 else:
                      # Create dummy data if key is missing (this shouldn't happen with dataloader)
                      log_and_print(logger, f"Warning: Key '{key}' missing in sample batch for FLOPs estimation.", logging.WARNING)
                      # Add appropriate dummy data based on expected type/shape if possible
                      if key == 'input_ids': dummy_batch_for_thop[key] = torch.zeros((1, self.config.MAX_LEN), dtype=torch.long, device=self.device)
                      elif key == 'attention_mask': dummy_batch_for_thop[key] = torch.zeros((1, self.config.MAX_LEN), dtype=torch.long, device=self.device)
                      elif key == 'v1': dummy_batch_for_thop[key] = torch.zeros((1, self.config.AUDIO_V1_DIM), dtype=torch.float, device=self.device)
                      elif key == 'v3': dummy_batch_for_thop[key] = torch.zeros((1, self.config.AUDIO_V3_DIM), dtype=torch.float, device=self.device)
                      elif key == 'v4': dummy_batch_for_thop[key] = torch.zeros((1, self.config.AUDIO_V4_DIM), dtype=torch.float, device=self.device)
                      elif key == 'a2': dummy_batch_for_thop[key] = torch.zeros((1, self.config.VISUAL_A2_DIM), dtype=torch.float, device=self.device)
                      elif key == 'label': dummy_batch_for_thop[key] = torch.zeros((1,), dtype=torch.long, device=self.device)
                      else: dummy_batch_for_thop[key] = None # Or handle differently


            # Prepare inputs for model's forward (batch dict, phase int)
            # Use phase 3 as it involves most components
            inputs_for_thop = (dummy_batch_for_thop, 3)

            # Ensure model is on the correct device
            self.model.to(self.device)
            self.model.eval()

            # Use thop.profile
            # Need to handle the tuple input correctly for thop
            macs, params = profile(self.model, inputs=inputs_for_thop, verbose=False)

            # FLOPs are typically ~2 * MACs
            flops = 2 * macs
            flops_str, params_str = clever_format([flops, params], "%.3f")
            log_and_print(logger, f"Estimated FLOPs: {flops_str}")
            log_and_print(logger, f"Total Parameters: {params_str}")
            return flops_str, params_str, int(params)

        except StopIteration:
            log_and_print(logger, "Warning: Test loader is empty. Cannot estimate FLOPs.", logging.WARNING)
            params = sum(p.numel() for p in self.model.parameters())
            params_str = clever_format([params], "%.3f")[0] if clever_format else f"{params:,}"
            return "N/A", params_str, int(params)
        except Exception as e:
            log_and_print(logger, f"Error during FLOPs estimation: {e}", logging.ERROR)
            log_and_print(logger, traceback.format_exc(), logging.ERROR)
            params = sum(p.numel() for p in self.model.parameters())
            params_str = clever_format([params], "%.3f")[0] if clever_format else f"{params:,}"
            log_and_print(logger, f"Total Parameters (manual count): {params_str}")
            return "N/A", params_str, int(params)


    def analyze_prediction_confidence(self, phase: int):
        """Analyzes prediction confidence and calls plotting function."""
        log_and_print(logger, f"--- Analyzing Prediction Confidence for Phase {phase} ---")
        if not self.model:
             log_and_print(logger, "Error: Model not loaded in evaluator. Cannot analyze confidence.", logging.ERROR)
             return
        self.model.eval()
        correct_confidences = []
        incorrect_confidences = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Analyzing Confidence Phase {phase}", leave=False):
                try:
                    batch_device = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    targets = batch_device['label']

                    with autocast(enabled=self.config.USE_AMP):
                        outputs = self.model(batch_device, phase=phase)
                        if 'final_logits' not in outputs: continue

                    probs = F.softmax(outputs['final_logits'], dim=1)
                    confidence, predictions = torch.max(probs, dim=1)
                    correct_mask = (predictions == targets)

                    correct_confidences.extend(confidence[correct_mask].cpu().numpy())
                    incorrect_confidences.extend(confidence[~correct_mask].cpu().numpy())
                except Exception as e:
                     log_and_print(logger, f"Error during confidence analysis batch: {e}", logging.ERROR)
                     continue

        log_and_print(logger, f"Confidence analysis complete for Phase {phase}. Correct: {len(correct_confidences)}, Incorrect: {len(incorrect_confidences)}")

        # Call plotting function if viz_manager exists
        if self.viz_manager:
            self.viz_manager.plot_confidence_distribution(correct_confidences, incorrect_confidences, phase)


    def evaluate_all_phases(self) -> Dict[int, Dict]:
        """Evaluates the best model saved for each phase."""
        log_and_print(logger, "\n" + "="*60)
        log_and_print(logger, "          STARTING OVERALL MODEL EVALUATION ACROSS PHASES")
        log_and_print(logger, "="*60)
        all_phase_metrics = {}

        # Estimate FLOPs and Params using the final phase model structure
        flops_str, params_str, params_int = "N/A", "N/A", None
        model_path_p3 = os.path.join(self.result_dir, f'BEST_model_phase3.pth')
        if os.path.exists(model_path_p3):
             try:
                 log_and_print(logger, f"Loading BEST model from Phase 3 for FLOPs/Param estimation: {model_path_p3}")
                 checkpoint_p3 = torch.load(model_path_p3, map_location=self.device)
                 num_emotions_p3 = checkpoint_p3.get('num_emotions', self.num_classes) # Use num_classes as fallback

                 # Need a valid model instance to estimate FLOPs
                 temp_model_p3 = initialize_model(self.config, num_emotions_p3, self.device)
                 if temp_model_p3:
                     temp_model_p3.load_state_dict(checkpoint_p3['model_state_dict'])
                     self.model = temp_model_p3 # Temporarily set model for estimation
                     flops_str, params_str, params_int = self.estimate_flops_params()
                 else:
                      log_and_print(logger, "Failed to initialize temporary model for FLOPs estimation.", logging.WARNING)

             except Exception as e_flops:
                  log_and_print(logger, f"Could not estimate FLOPs/Params using Phase 3 model: {e_flops}", logging.WARNING)
                  # Fallback: try counting params on the initial model if available
                  if self.model:
                      params_int = sum(p.numel() for p in self.model.parameters())
                      params_str = clever_format([params_int], "%.3f")[0] if clever_format else f"{params_int:,}"
                  else: # If no model at all, cannot count params
                      params_str = "N/A"
                      params_int = None

        else:
             log_and_print(logger, "Phase 3 BEST model not found, cannot estimate FLOPs/Params accurately.", logging.WARNING)
             # Fallback: try counting params on the initial model if available
             if self.model:
                 params_int = sum(p.numel() for p in self.model.parameters())
                 params_str = clever_format([params_int], "%.3f")[0] if clever_format else f"{params_int:,}"


        for phase in range(1, 4): # Evaluate phases 1, 2, 3
            # Load the BEST model checkpoint for the phase
            model_path = os.path.join(self.result_dir, f'BEST_model_phase{phase}.pth')
            if os.path.exists(model_path):
                log_and_print(logger, f"\nLoading BEST model from Phase {phase}: {model_path}")
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    # Re-initialize model to ensure correct architecture before loading state_dict
                    num_emotions_in_ckpt = checkpoint.get('num_emotions', self.num_classes) # Get num_emotions if saved
                    eval_model = initialize_model(self.config, num_emotions_in_ckpt, self.device)
                    if not eval_model:
                         log_and_print(logger, f"Failed to initialize model for Phase {phase} evaluation. Skipping.", logging.ERROR)
                         continue

                    eval_model.load_state_dict(checkpoint['model_state_dict'])
                    self.model = eval_model # Use this loaded model for evaluation
                    log_and_print(logger, f"Successfully loaded BEST model from epoch {checkpoint.get('epoch', 'N/A')}.")

                    # Evaluate the loaded model
                    metrics = self.evaluate_phase(phase)
                    if metrics:
                         # Add FLOPs/Params info (estimated earlier)
                         metrics['Estimated FLOPs'] = flops_str if flops_str else "N/A"
                         metrics['Parameters'] = params_str if params_str else "N/A"
                         all_phase_metrics[phase] = metrics
                         # Analyze confidence for this phase's best model
                         self.analyze_prediction_confidence(phase)

                except Exception as e:
                     log_and_print(logger, f"Error loading or evaluating BEST model for Phase {phase}: {e}", logging.ERROR)
                     log_and_print(logger, traceback.format_exc(), logging.ERROR)
            else:
                log_and_print(logger, f"BEST model checkpoint not found for Phase {phase} at {model_path}. Skipping evaluation.", logging.WARNING)

        # Save summary of metrics across phases
        if self.viz_manager and all_phase_metrics:
            self.viz_manager.save_metrics_summary(all_phase_metrics)

        log_and_print(logger, "\n" + "="*60)
        log_and_print(logger, "          OVERALL MODEL EVALUATION FINISHED")
        log_and_print(logger, "="*60)
        return all_phase_metrics


# --- Wrapper Function ---
def evaluate_model(model: Optional[nn.Module], test_loader: DataLoader, device: torch.device,
                  result_dir: str, emotion_names: List[str], config: Config) -> Dict[int, Dict]:
    """
    Wrapper function to initialize evaluator and run evaluation across all phases.
    """
    log_and_print(logger, "\n--- Initializing Evaluation Process ---")
    final_metrics = {} # Initialize empty dict
    try:
        # Pass the initially loaded model instance (e.g., from end of training or loaded checkpoint)
        # The evaluator will load the *best* checkpoint for each phase internally.
        evaluator = ModelEvaluator(model, test_loader, device, result_dir, emotion_names, config)
        final_metrics = evaluator.evaluate_all_phases()

        # --- Log Final Summary ---
        log_and_print(logger, "\n" + "*"*60)
        log_and_print(logger, "          FINAL EVALUATION SUMMARY (Using BEST model per phase)")
        log_and_print(logger, "*"*60)
        if not final_metrics:
             log_and_print(logger, "No phases were successfully evaluated.")
        else:
             for phase in sorted(final_metrics.keys()):
                 log_and_print(logger, f"\n--- Best Model Performance (Phase {phase}) ---")
                 metrics = final_metrics[phase]
                 # Use the order defined in save_metrics_summary for consistency
                 metric_order = [
                     ('loss', 'Avg Loss', '{:.4f}'),
                     ('accuracy', 'Accuracy', '{:.2f}%'),
                     ('f1_weighted', 'F1 Weighted', '{:.2f}%'),
                     ('f1_macro', 'F1 Macro', '{:.2f}%'),
                     ('f1_micro', 'F1 Micro', '{:.2f}%'),
                     ('precision_weighted', 'Precision Weighted', '{:.2f}%'),
                     ('precision_macro', 'Precision Macro', '{:.2f}%'),
                     ('precision_micro', 'Precision Micro', '{:.2f}%'),
                     ('recall_weighted', 'Recall Weighted', '{:.2f}%'),
                     ('recall_macro', 'Recall Macro', '{:.2f}%'),
                     ('recall_micro', 'Recall Micro', '{:.2f}%'),
                     ('top_2_accuracy', 'Top-2 Accuracy', '{:.2f}%'),
                     ('top_3_accuracy', 'Top-3 Accuracy', '{:.2f}%'),
                     ('Parameters', 'Parameters', '{}'), # Already formatted string
                     ('Estimated FLOPs', 'Est. FLOPs', '{}'), # Already formatted string
                 ]
                 for key, display_name, fmt in metric_order:
                     if key in metrics:
                         value = metrics[key]
                         # Handle potential None values before formatting
                         if value is None: value_str = "N/A"
                         elif isinstance(value, str): value_str = value # Already string (like FLOPs/Params)
                         else:
                             try:
                                 value_str = fmt.format(value)
                             except (TypeError, ValueError): # Handle cases where format might fail
                                 value_str = str(value)
                         log_and_print(logger, f"- {display_name:<20}: {value_str}")


        log_and_print(logger, "*"*60)
        log_and_print(logger, "Evaluation process completed.")
        return final_metrics

    except Exception as e:
        log_and_print(logger, f"Critical error during model evaluation wrapper: {str(e)}", logging.CRITICAL)
        log_and_print(logger, traceback.format_exc(), logging.CRITICAL)
        return final_metrics # Return potentially partially filled dict

# --- Execution Block ---
print("\n--- Starting Cell 3.9 Execution ---")
# This cell defines the ModelEvaluator class and the evaluate_model wrapper.
# The actual evaluation is typically run *after* training is complete.
# We will call evaluate_model in Cell 3.10.
# For now, just print a confirmation that the definitions are ready.

if 'ModelEvaluator' in globals() and 'evaluate_model' in globals():
     print("Cell 3.9 executed successfully: ModelEvaluator class and evaluate_model function defined.")
     print("Evaluation will be triggered in Cell 3.10.")
else:
     print("Error: Failed to define ModelEvaluator or evaluate_model in Cell 3.9.")

print("--- Finished Cell 3.9 Execution ---")
print("-" * 50)

# Cell 3.10: Evaluation Execution
# =====================================
# Run Model Evaluation and Generate Visualizations/Reports
# =====================================

import torch
import os
import traceback

# Assuming config, logger, log_and_print, device, val_loader, label_encoder,
# initialize_model (from Cell 3.7), evaluate_model (from Cell 3.9),
# model_instance (the last trained model instance from Cell 3.7, needed as a fallback if no checkpoint found)
# final_metrics_history (from Cell 3.7 execution block)
# and VisualizationManager, viz_manager (instantiated in Cell 3.8) are available

print("\n--- Starting Cell 3.10: Evaluation Execution ---")

# --- Check Prerequisites ---
# Use a slightly more robust check, ensuring variables are not None where applicable
required_vars = {
    'config': lambda x: x is not None,
    'logger': lambda x: x is not None,
    'log_and_print': lambda x: callable(x),
    'device': lambda x: x is not None,
    'val_loader': lambda x: x is not None,
    'label_encoder': lambda x: x is not None and hasattr(x, 'classes_'), # Check label_encoder has classes_
    'initialize_model': lambda x: callable(x),
    'evaluate_model': lambda x: callable(x),
    # model_instance can be None if training failed, evaluate_model handles loading checkpoints
    'model_instance': lambda x: True, # Allow model_instance to be None
    'VisualizationManager': lambda x: callable(x), # Check if the class is defined
    'viz_manager': lambda x: x is not None, # Check if the instance exists
    'final_metrics_history': lambda x: x is not None and isinstance(x, dict) # Check history exists and is a dict
}
missing_vars_details = []
for var, check in required_vars.items():
    if var not in locals():
        missing_vars_details.append(f"'{var}' is not defined.")
    elif not check(locals()[var]):
         missing_vars_details.append(f"'{var}' is defined but failed check (e.g., is None, not callable, or missing attributes).")

if missing_vars_details:
    print(f"Error: Missing or invalid required variables/functions for evaluation:")
    for detail in missing_vars_details:
        print(f"- {detail}")
    print("Cannot proceed with evaluation.")
    final_evaluation_metrics = None
else:
    try:
        log_and_print(logger, "Preparing for final model evaluation...")

        # 1. Get Emotion Names
        emotion_names = list(label_encoder.classes_)
        num_classes_eval = len(emotion_names)
        log_and_print(logger, f"Using {num_classes_eval} emotion classes for evaluation: {emotion_names}")
        # Verify against config, log warning if mismatch
        if hasattr(config, 'NUM_EMOTIONS') and num_classes_eval != config.NUM_EMOTIONS:
            log_and_print(logger, f"Warning: Number of classes from label encoder ({num_classes_eval}) differs from config ({config.NUM_EMOTIONS}). Evaluation will use {num_classes_eval} classes.", logging.WARNING)
            # Optionally update config if label encoder is source of truth
            # config.NUM_EMOTIONS = num_classes_eval


        # 2. Load the Best Trained Model (Handled internally by evaluate_model)
        # We pass the model_instance trained at the end of Cell 3.7 as the 'model' argument
        # to evaluate_model. The evaluator will then load the *best* checkpoint for each phase.
        log_and_print(logger, "Calling evaluate_model function. It will load the BEST checkpoint for each phase internally.")

        # Retrieve model_instance if it exists, otherwise pass None
        current_model_instance = locals().get('model_instance', None)
        if current_model_instance is None:
            log_and_print(logger, "Warning: 'model_instance' from training not found. Evaluation will rely solely on loading checkpoints.", logging.WARNING)


        # 3. Run Evaluation using the wrapper function
        # Pass the model instance from the end of training (Cell 3.7) if available
        final_evaluation_metrics = evaluate_model(
            model=current_model_instance, # Pass the model potentially trained in Cell 3.7 (can be None)
            test_loader=val_loader, # Using validation set for final evaluation
            device=device,
            result_dir=config.RESULT_DIR,
            emotion_names=emotion_names,
            config=config
        )

        # Optional: Log the returned metrics dictionary structure
        if final_evaluation_metrics:
            log_and_print(logger, f"Evaluation metrics dictionary generated with keys for phases: {list(final_evaluation_metrics.keys())}")
            # Example: Log top-level metrics for the last evaluated phase (if exists)
            last_evaluated_phase = max(final_evaluation_metrics.keys()) if final_evaluation_metrics else None
            if last_evaluated_phase and last_evaluated_phase in final_evaluation_metrics:
                 print(f"\nSummary metrics for best model of Phase {last_evaluated_phase}:")
                 phase_mets = final_evaluation_metrics[last_evaluated_phase]
                 print(f"  Accuracy: {phase_mets.get('accuracy', 'N/A'):.2f}%")
                 print(f"  F1 Weighted: {phase_mets.get('f1_weighted', 'N/A'):.2f}%")
                 print(f"  F1 Macro: {phase_mets.get('f1_macro', 'N/A'):.2f}%")
                 print(f"  Loss: {phase_mets.get('loss', 'N/A'):.4f}")
                 print(f"  Params: {phase_mets.get('Parameters', 'N/A')}")
                 print(f"  FLOPs: {phase_mets.get('Estimated FLOPs', 'N/A')}")
        else:
            log_and_print(logger, "Evaluation process did not return metrics.", logging.WARNING)

    except Exception as e:
        log_and_print(logger, f"An unexpected error occurred during evaluation execution: {e}", logging.CRITICAL)
        log_and_print(logger, traceback.format_exc(), logging.CRITICAL)
        final_evaluation_metrics = None

# Add a final print statement for clarity
if 'final_evaluation_metrics' in locals():
    print(f"\nEvaluation Metrics Object: {'Generated' if final_evaluation_metrics is not None else 'Failed'}")
else:
    print("\nEvaluation Metrics Object: Not generated due to errors.")

# --- Plot Learning Curves --- # <<<--- ADDED THIS SECTION
if 'viz_manager' in locals() and viz_manager is not None and \
   'final_metrics_history' in locals() and final_metrics_history is not None:
    log_and_print(logger, "\n--- Generating Learning Curves ---")
    print("\n--- Generating Learning Curves ---")
    try:
        viz_manager.plot_learning_curves(final_metrics_history)
        log_and_print(logger, "Learning curve generation complete.")
    except Exception as e_lc:
        log_and_print(logger, f"Error generating learning curves: {e_lc}", logging.ERROR)
        log_and_print(logger, traceback.format_exc(), logging.ERROR)
elif 'viz_manager' not in locals() or viz_manager is None:
     log_and_print(logger, "Skipping learning curves: viz_manager not available.", logging.WARNING)
else: # History is missing
     log_and_print(logger, "Skipping learning curves: final_metrics_history not available.", logging.WARNING)
# --------------------------

print("\n--- Finished Cell 3.10 Execution ---")
print("-" * 50)