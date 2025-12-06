
# train_chess_eval_model.py
# ----------------------------------------------
# Train a neural network to evaluate chess positions.
#
# Expected input CSV format:
#   - Column 'FEN': Forsyth-Edwards Notation string for the position
#   - Column 'Evaluation': numeric evaluation (centipawns; + means white is better)
#
# Usage (example):
#   python train_chess_eval_model.py --csv /path/to/chessData.csv --sample 250000
#
# Output:
#   - chess_eval_mlp.joblib (the trained model)
#   - feature_config.json   (meta about the input features)
#
# Notes:
#   * The dataset can be millions of rows; use --sample to train faster on a subset.
#   * We normalize labels with tanh(eval/400) to map to approximately [-1, 1].
#
import argparse
import json
import os
import sys
import math
import numpy as np
import pandas as pd

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle as sk_shuffle
    import joblib
except Exception as e:
    print("This script requires scikit-learn and joblib. Please install them and try again.")
    raise

# ---------- Feature extraction ----------

PIECES = ["P","N","B","R","Q","K","p","n","b","r","q","k"]
PIECE_INDEX = {p:i for i,p in enumerate(PIECES)}

def fen_to_vector(fen: str, include_ep=True):
    r"""Convert a FEN string to a numeric feature vector.

    Representation (length 12*64 + 1 + 4 + 8 = 781 by default):
      - 12 piece planes (6 per side) flattened (binary 0/1)
      - 1 side-to-move bit (1 if white to move else 0)
      - 4 castling rights bits [K, Q, k, q]
      - 8 en-passant file one-hot (a..h) if any, else all zeros

    Parameters
    ----------
    fen : str
        Full position in Forsyth-Edwards Notation.
    include_ep : bool, optional
        Whether to encode the en-passant file as an 8-d one-hot vector.

    Returns
    -------
    np.ndarray
        1D float32 feature vector suitable as input to the MLP.
    """
    parts = fen.split()
    board_part = parts[0]
    stm_part = parts[1] if len(parts) > 1 else "w"
    castling_part = parts[2] if len(parts) > 2 else "-"
    ep_part = parts[3] if len(parts) > 3 else "-"

    ranks = board_part.split('/')
    planes = np.zeros((12, 64), dtype=np.float32)

    # Iterate ranks from 8 to 1
    for r, rank in enumerate(ranks):
        file_idx = 0
        for ch in rank:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                idx = (7 - r) * 8 + file_idx   # a1=0 ... h8=63
                planes[PIECE_INDEX[ch], idx] = 1.0
                file_idx += 1

    # Side to move
    stm = 1.0 if stm_part == 'w' else 0.0

    # Castling rights
    wK = 1.0 if 'K' in castling_part else 0.0
    wQ = 1.0 if 'Q' in castling_part else 0.0
    bK = 1.0 if 'k' in castling_part else 0.0
    bQ = 1.0 if 'q' in castling_part else 0.0

    # En-passant file one-hot (8)
    ep_vec = np.zeros(8, dtype=np.float32)
    if include_ep and ep_part != '-' and len(ep_part) >= 2:
        file_letter = ep_part[0]
        if 'a' <= file_letter <= 'h':
            ep_vec[ord(file_letter) - ord('a')] = 1.0

    vec = np.concatenate([planes.reshape(-1), np.array([stm, wK, wQ, bK, bQ], dtype=np.float32), ep_vec])
    return vec

def normalize_eval_cp_to_tanh(y_cp: np.ndarray, denom: float = 400.0):
    """Normalize centipawn evaluations into the range [-1, 1] using tanh.

    Values are first clipped to a reasonable range to reduce the impact of
    extreme outliers, then mapped with tanh(eval/denom).

    Parameters
    ----------
    y_cp : np.ndarray
        Raw centipawn evaluations (white advantage, + is good for white).
    denom : float, optional
        Scaling factor used inside tanh; larger values make the mapping
        flatter around the origin.

    Returns
    -------
    np.ndarray
        Float32 array of the same shape as ``y_cp`` with values in [-1, 1].
    """
    y_cp = np.clip(y_cp, -4000, 4000)  # avoid extreme outliers
    return np.tanh(y_cp / denom).astype(np.float32)

def denormalize_tanh_to_cp(y_tanh: np.ndarray, denom: float = 400.0):
    """Invert the tanh normalization back to approximate centipawns.

    Parameters
    ----------
    y_tanh : np.ndarray
        Normalized evaluations in [-1, 1].
    denom : float, optional
        Same scaling factor that was used during normalization.

    Returns
    -------
    np.ndarray
        Approximate centipawn evaluations recovered via atanh.
    """
    # inverse of tanh: atanh
    return denom * np.arctanh(np.clip(y_tanh, -0.999999, 0.999999))

def sample_from_csv(csv_path: str, sample: int = 250000, seed: int = 42, chunksize: int = 200000):
    """Sample positions from a large CSV without loading it fully into memory.

    Uses reservoir sampling over the rows to obtain a roughly uniform sample
    of FEN strings and their evaluations.

    Parameters
    ----------
    csv_path : str
        Path to the Kaggle CSV file containing FEN and Evaluation columns.
    sample : int, optional
        Target number of positions to keep in the reservoir.
    seed : int, optional
        Seed for the NumPy random generator used in sampling.
    chunksize : int, optional
        Unused here but kept for compatibility with streaming variants.

    Returns
    -------
    list[str], np.ndarray
        The sampled FEN strings and their centipawn evaluations.
    """
    rng = np.random.default_rng(seed)
    reservoir_fens = []
    reservoir_evals = []
    seen = 0
    k = sample

    def parse_line(line):
        # Split on the last comma to separate FEN and evaluation
        parts = line.rsplit(',', 1)
        if len(parts) != 2:
            return None, None
        fen, eval_str = parts
        fen = fen.strip()
        eval_str = eval_str.strip()
        
        # Clean up the evaluation string (remove any non-numeric prefixes/suffixes)
        eval_str = ''.join(c for c in eval_str if c.isdigit() or c == '-' or c == '.' or c == '+')
        if not eval_str or (eval_str[0] not in '+-' and not eval_str[0].isdigit()):
            return None, None
            
        try:
            eval_val = float(eval_str)
            return fen, eval_val
        except ValueError:
            return None, None

    with open(csv_path, 'r') as f:
        # Skip header
        next(f)
        
        for line in f:
            fen, eval_val = parse_line(line)
            if fen is None or eval_val is None:
                continue
                
            seen += 1
            if len(reservoir_fens) < k:
                reservoir_fens.append(fen)
                reservoir_evals.append(eval_val)
            else:
                j = rng.integers(0, seen)
                if j < k:
                    reservoir_fens[j] = fen
                    reservoir_evals[j] = eval_val

    return reservoir_fens, np.array(reservoir_evals, dtype=np.float32)

def build_Xy(fens, evals_cp, include_ep=True):
    """Build the design matrix X and normalized labels y from raw data.

    Parameters
    ----------
    fens : Sequence[str]
        Iterable of FEN strings.
    evals_cp : np.ndarray or Sequence[float]
        Centipawn evaluations corresponding to each FEN.
    include_ep : bool, optional
        Whether to encode en-passant information in the features.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``X`` is a 2D float32 array of shape (n_positions, n_features),
        ``y`` is a 1D float32 array of tanh-normalized evaluations.
    """
    X = np.stack([fen_to_vector(f, include_ep=include_ep) for f in fens]).astype(np.float32)
    y = normalize_eval_cp_to_tanh(evals_cp)
    return X, y

def main():
    """Entry point for command-line training of the evaluation model.

    This script will:
      1. Sample positions and evaluations from the provided CSV.
      2. Convert FEN strings into numeric feature vectors.
      3. Split the data into train/test sets.
      4. Train an ``MLPRegressor`` to predict normalized evaluations.
      5. Report test MSE in tanh space.
      6. Save the trained model and a feature configuration JSON.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to Kaggle chess evaluations CSV')
    parser.add_argument('--sample', type=int, default=250000, help='How many rows to sample for training')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include_ep', action='store_true', help='Include en-passant file in features')
    parser.add_argument('--hidden', type=str, default='512,256', help='Hidden layer sizes, comma-separated')
    parser.add_argument('--max_iter', type=int, default=30)
    parser.add_argument('--out_model', type=str, default='chess_eval_mlp.joblib')
    parser.add_argument('--out_config', type=str, default='feature_config.json')
    args = parser.parse_args()

    print('Sampling data...')
    fens, evals_cp = sample_from_csv(args.csv, sample=args.sample, seed=args.seed)
    print(f'Got {len(fens)} positions. Building features...')

    X, y = build_Xy(fens, evals_cp, include_ep=args.include_ep)
    print('Feature matrix:', X.shape, 'Label vector:', y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=args.seed)

    hidden = tuple(int(x) for x in args.hidden.split(','))
    print('Training MLPRegressor with hidden sizes:', hidden)

    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation='relu',
        solver='adam',
        learning_rate_init=1e-3,
        alpha=1e-5,
        batch_size=512,
        max_iter=args.max_iter,
        random_state=args.seed,
        verbose=True,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f'Test MSE (tanh-space): {mse:.6f}')

    import joblib
    joblib.dump(model, args.out_model)
    print('Saved model to', args.out_model)

    cfg = {
        "feature_version": 1,
        "pieces": PIECES,
        "vector_length": int(X.shape[1]),
        "label_normalization": "tanh(eval_cp/400)",
        "denom": 400.0,
        "include_ep": bool(args.include_ep)
    }
    with open(args.out_config, 'w') as f:
        json.dump(cfg, f, indent=2)
    print('Saved feature config to', args.out_config)


if __name__ == '__main__':
    main()
