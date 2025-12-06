
# ml_chess_engine.py
# ----------------------------------------------
# Drop-in evaluation and move suggestion using a trained model.
#
# Functions to import into the chess application:
#   - evaluate_board_ml(board, model)
#   - suggest_move_ml(board, model)
#
# Where `board` is a python-chess Board instance.
#
import os
import json
import numpy as np

try:
    import chess
except Exception as e:
    raise RuntimeError("This module requires the 'python-chess' package.") from e

try:
    import joblib
except Exception as e:
    raise RuntimeError("This module requires 'joblib' to load the trained model.") from e

PIECES = ["P","N","B","R","Q","K","p","n","b","r","q","k"]
PIECE_INDEX = {p:i for i,p in enumerate(PIECES)}

def fen_to_vector(fen: str, include_ep=True):
    """Convert a FEN string into the numeric feature vector expected by the model.

    The representation matches the one used during training
    (see ``train_chess_eval_model.py``):
      - 12 piece planes (6 per side) on 64 squares (flattened)
      - 1 side-to-move bit
      - 4 castling rights bits [K, Q, k, q]
      - 8 en-passant file one-hot bits (a..h)

    Parameters
    ----------
    fen : str
        FEN string for the current position.
    include_ep : bool, optional
        Whether to encode the en-passant file in the feature vector.

    Returns
    -------
    np.ndarray
        1D float32 vector of length given by ``CFG['vector_length']``.
    """
    parts = fen.split()
    board_part = parts[0]
    stm_part = parts[1] if len(parts) > 1 else "w"
    castling_part = parts[2] if len(parts) > 2 else "-"
    ep_part = parts[3] if len(parts) > 3 else "-"

    ranks = board_part.split('/')
    planes = np.zeros((12, 64), dtype=np.float32)

    for r, rank in enumerate(ranks):
        file_idx = 0
        for ch in rank:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                idx = (7 - r) * 8 + file_idx
                planes[PIECE_INDEX[ch], idx] = 1.0
                file_idx += 1

    stm = 1.0 if stm_part == 'w' else 0.0

    wK = 1.0 if 'K' in castling_part else 0.0
    wQ = 1.0 if 'Q' in castling_part else 0.0
    bK = 1.0 if 'k' in castling_part else 0.0
    bQ = 1.0 if 'q' in castling_part else 0.0

    ep_vec = np.zeros(8, dtype=np.float32)
    if include_ep and ep_part != '-' and len(ep_part) >= 2:
        file_letter = ep_part[0]
        if 'a' <= file_letter <= 'h':
            ep_vec[ord(file_letter) - ord('a')] = 1.0

    vec = np.concatenate([planes.reshape(-1), np.array([stm, wK, wQ, bK, bQ], dtype=np.float32), ep_vec])
    return vec

def _load_feature_config(cfg_path='feature_config.json'):
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            return json.load(f)
    # sensible defaults if config is missing
    return {
        "feature_version": 1,
        "pieces": PIECES,
        "vector_length": 781,
        "label_normalization": "tanh(eval_cp/400)",
        "denom": 400.0,
        "include_ep": True,
    }

CFG = _load_feature_config()

def _predict_white_advantage(board: 'chess.Board', model) -> float:
    """Return the model's prediction of white's advantage in tanh space.

    The model was trained to output a value in approximately ``[-1, 1]``
    where positive numbers favor white and negative numbers favor black.

    Parameters
    ----------
    board : chess.Board
        Position to evaluate (side to move is encoded in the FEN).
    model : sklearn.base.RegressorMixin
        Trained regression model loaded from ``chess_eval_mlp.joblib``.

    Returns
    -------
    float
        Predicted white advantage in tanh-normalized space.
    """
    vec = fen_to_vector(board.fen(), include_ep=CFG.get("include_ep", True)).reshape(1, -1)
    pred = float(model.predict(vec)[0])
    return pred

def evaluate_board_ml(board: 'chess.Board', model, return_centipawns=False) -> float:
    """Evaluate a position using the ML model.

    This helper wraps ``_predict_white_advantage`` and converts the result
    into the perspective of the side to move. Optionally, the score can be
    mapped back into approximate centipawns using the same scaling as was
    used in training.

    Parameters
    ----------
    board : chess.Board
        Position to evaluate.
    model : sklearn.base.RegressorMixin
        Trained model used to predict normalized evaluations.
    return_centipawns : bool, optional
        If ``True``, return an approximate centipawn score instead of tanh.

    Returns
    -------
    float
        Evaluation from the perspective of the side to move.
    """
    # Terminal checks
    if board.is_game_over():
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            val_white = 0.0
        else:
            val_white = 1e9 if outcome.winner == chess.WHITE else -1e9
        # Convert to current player's perspective
        val = val_white if board.turn == chess.WHITE else -val_white
        return val

    white_tanh = _predict_white_advantage(board, model)
    # Convert to player-to-move perspective
    tanh_for_player = white_tanh if board.turn == chess.WHITE else -white_tanh

    if not return_centipawns:
        return tanh_for_player

    denom = float(CFG.get("denom", 400.0))
    # Invert tanh safely
    tanh_clip = np.clip(tanh_for_player, -0.999999, 0.999999)
    cp = float(denom * np.arctanh(tanh_clip))
    return cp

def suggest_move_ml(board: 'chess.Board', model):
    """Score all legal moves with the ML model and return the best one.

    The function performs a simple 1-ply search: for each legal move, it
    makes the move on the board, queries the model for the resulting
    position, then undoes the move. The move with the highest score from the
    current player's perspective is returned.

    Parameters
    ----------
    board : chess.Board
        Current position whose moves should be evaluated.
    model : sklearn.base.RegressorMixin
        Trained evaluation model.

    Returns
    -------
    Tuple[chess.Move, float]
        The best move and its score (in tanh space, from the side-to-move
        perspective). If there are no legal moves, ``best_move`` will be
        ``None``.
    """
    best_move = None
    best_score = -1e18
    player = board.turn

    for move in board.legal_moves:
        board.push(move)
        score_white = _predict_white_advantage(board, model)  # white perspective
        board.pop()

        score_for_player = score_white if player == chess.WHITE else -score_white
        if score_for_player > best_score:
            best_score = score_for_player
            best_move = move

    return best_move, best_score

def load_model(model_path='chess_eval_mlp.joblib'):
    """Load a trained evaluation model from disk.

    Parameters
    ----------
    model_path : str, optional
        Path to the ``.joblib`` file produced by ``train_chess_eval_model.py``.

    Returns
    -------
    sklearn.neural_network.MLPRegressor
        The deserialized model ready for use in ``evaluate_board_ml`` or
        ``suggest_move_ml``.
    """
    return joblib.load(model_path)

if __name__ == '__main__':
    # Mini demo that prints a suggested move from the initial position if a model is present
    try:
        model = load_model('chess_eval_mlp.joblib')
    except Exception as e:
        print("Could not load model 'chess_eval_mlp.joblib'. Put the trained model in the current directory.")
        raise

    board = chess.Board()
    mv, sc = suggest_move_ml(board, model)
    print("Suggested move from start:", mv, "score(tanh):", sc)
