import random

import chess
import numpy as np
import pytest

from az_chess import ACTION_SIZE, BOARD_FEATURES, decode_action, encode_move, encode_position
from az_mcts import MCTS, choose_move, visit_policy
from az_network import ChessNet
from az_replay import ReplayBuffer, TrainingExample
from az_selfplay import self_play_game
from az_training import train_network


def test_legal_moves_round_trip_through_policy_encoding():
    board = chess.Board()
    for move in board.legal_moves:
        action = encode_move(move, board.turn)
        assert decode_action(action, board) == move


def test_action_encoding_round_trips_for_both_sides_and_promotions():
    positions = [
        chess.Board(),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
        chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
        chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1"),
    ]
    for board in positions:
        for move in board.legal_moves:
            assert decode_action(encode_move(move, board.turn), board) == move


def test_position_encoding_has_expected_planes_and_canonical_turn_view():
    board = chess.Board()
    features = encode_position(board)
    assert features.shape == (BOARD_FEATURES, 8, 8)
    assert features.dtype == np.float32
    assert features[0, 1, 4] == 1  # White pawn on e2 is friendly.

    board.push_uci("e2e4")
    features = encode_position(board)
    assert features[0, 1, 4] == 1  # Black's pawn is friendly in the canonical view.
    assert features[6, 4, 4] == 1  # White pawn is now an enemy pawn.
    assert features[18, 6, 4] == 1  # Previous frame places it back on e2.


def test_mcts_returns_normalized_legal_visit_policy():
    model = ChessNet(channels=16, blocks=1)
    board = chess.Board()
    visits = MCTS(model).search(board, simulations=4, add_root_noise=True,
                                rng=random.Random(3))
    assert set(visits).issubset(set(board.legal_moves))
    assert sum(visits.values()) == 4
    actions, probabilities = visit_policy(board, visits)
    assert len(actions) == len(visits)
    np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-6)
    assert choose_move(visits) in visits
    assert ACTION_SIZE == 4672


def test_self_play_replay_persistence_and_one_training_step(tmp_path):
    model = ChessNet(channels=16, blocks=1)
    examples = self_play_game(
        model, simulations=2, max_plies=4, rng=random.Random(1)
    )
    assert len(examples) == 4
    assert all(example.outcome == 0 for example in examples)

    replay = ReplayBuffer(capacity=10)
    for example in examples:
        replay.add(example)
    replay_path = tmp_path / "replay.npz"
    replay.save(replay_path)
    restored = ReplayBuffer.load(replay_path, capacity=10)
    assert len(restored) == len(replay)
    states, policies, outcomes = restored.batch(2, random.Random(2))
    assert states.shape == (2, BOARD_FEATURES, 8, 8)
    assert policies.shape == (2, ACTION_SIZE)
    np.testing.assert_allclose(policies.sum(axis=1), 1.0)
    assert outcomes.shape == (2,)

    policy_loss, value_loss = train_network(
        model, restored, steps=1, batch_size=2, device="cpu",
        rng=random.Random(4),
    )
    assert np.isfinite(policy_loss) and policy_loss > 0
    assert np.isfinite(value_loss) and value_loss >= 0


def test_replay_rejects_malformed_examples():
    replay = ReplayBuffer()
    with np.testing.assert_raises(ValueError):
        replay.add(
            TrainingExample(
                np.zeros((3, 8, 8)),
                np.asarray([0]),
                np.asarray([1.0]),
                0.0,
            )
        )


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_interrupted_replay_save_preserves_previous_file(tmp_path, monkeypatch, existing, error):
    replay = ReplayBuffer()
    path = tmp_path / "replay.npz"
    if existing:
        replay.save(path)
        original = path.read_bytes()

    def interrupted_save(stream, **arrays):
        stream.write(b"incomplete archive")
        raise error("interrupted replay write")

    monkeypatch.setattr(np, "savez_compressed", interrupted_save)
    with pytest.raises(error, match="interrupted replay write"):
        replay.save(path)

    if existing:
        assert path.read_bytes() == original
        assert len(ReplayBuffer.load(path)) == 0
    else:
        assert not path.exists()
    assert not path.with_suffix(".npz.tmp").exists()


def test_replay_save_replaces_existing_archive(tmp_path):
    replay = ReplayBuffer()
    path = tmp_path / "replay.npz"
    replay.save(path)
    assert len(ReplayBuffer.load(path)) == 0
    replay.add(TrainingExample(
        np.zeros((BOARD_FEATURES, 8, 8)),
        np.asarray([0]),
        np.asarray([1.0]),
        1.0,
    ))
    replay.save(path)
    restored = ReplayBuffer.load(path)
    assert len(restored) == 1
    assert restored.sample(1)[0].outcome == 1.0
    assert not path.with_suffix(".npz.tmp").exists()
