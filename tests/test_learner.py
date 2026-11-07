"""Learner updates and Adam state survive failed promotion and restart."""
import sys

import pytest
import torch

import train
from az_network import ChessNet


def run_training(path, monkeypatch, score=0.5, iterations=1, learning_rate="0.001"):
    monkeypatch.setattr(train, "evaluate_match", lambda *args, **kwargs: score)
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--device", "cpu", "--iterations", str(iterations),
        "--games-per-iteration", "1", "--simulations", "1", "--eval-games", "2",
        "--max-plies", "2", "--train-steps", "1", "--batch-size", "2",
        "--channels", "16", "--blocks", "1", "--output-dir", str(path),
        "--learning-rate", learning_rate,
    ])
    assert train.main() == 0
    return (
        torch.load(path / "champion.pt", weights_only=True),
        torch.load(path / "latest.pt", weights_only=True),
    )


def equal_weights(first, second):
    return all(torch.equal(value, second[key]) for key, value in first.items())


def test_rejected_learner_resumes_and_retains_optimizer(tmp_path, monkeypatch, capsys):
    champion, learner = run_training(tmp_path, monkeypatch)
    assert not equal_weights(champion["state_dict"], learner["state_dict"])
    assert learner["learner_version"] == 1
    assert "scaler_state_dict" in learner
    assert all(state["step"].item() == 1 for state in learner["optimizer_state_dict"]["state"].values())
    champion2, learner2 = run_training(tmp_path, monkeypatch, learning_rate="0.0005")
    assert equal_weights(champion["state_dict"], champion2["state_dict"])
    assert not equal_weights(learner["state_dict"], learner2["state_dict"])
    assert learner2["iteration"] == 2
    assert all(state["step"].item() == 2 for state in learner2["optimizer_state_dict"]["state"].values())
    assert learner2["optimizer_state_dict"]["param_groups"][0]["lr"] == 0.0005
    assert "Resumed learner and optimizer from iteration 1" in capsys.readouterr().out


def test_promoted_learner_becomes_champion(tmp_path, monkeypatch):
    champion, learner = run_training(tmp_path, monkeypatch, score=1.0, iterations=2)
    assert champion["promoted"] and learner["promoted"]
    assert equal_weights(champion["state_dict"], learner["state_dict"])
    assert all(state["step"].item() == 2 for state in learner["optimizer_state_dict"]["state"].values())
    assert "optimizer_state_dict" not in champion


def test_legacy_checkpoint_migrates_from_champion(tmp_path, monkeypatch, capsys):
    model = ChessNet(16, 1)
    train._save_checkpoint(tmp_path / "champion.pt", model, 10, False, 16, 1)
    other = ChessNet(16, 1)
    train._save_checkpoint(tmp_path / "latest.pt", other, 10, False, 16, 1)
    champion, learner = run_training(tmp_path, monkeypatch)
    assert equal_weights(model.state_dict(), champion["state_dict"])
    assert learner["iteration"] == 11
    assert learner["learner_version"] == 1
    assert "Legacy latest.pt" in capsys.readouterr().out


def test_incomplete_learner_checkpoint_fails_explicitly(tmp_path, monkeypatch):
    _, learner = run_training(tmp_path, monkeypatch)
    del learner["optimizer_state_dict"]
    torch.save(learner, tmp_path / "latest.pt")
    with pytest.raises(KeyError, match="optimizer_state_dict"):
        run_training(tmp_path, monkeypatch)


def test_enabled_scaler_restored_across_restart(tmp_path, monkeypatch):
    scaler_class = torch.amp.GradScaler
    # Exercise scaler persistence on CPU without requiring a GPU in the test suite.
    monkeypatch.setattr(
        torch.amp, "GradScaler",
        lambda *args, **kwargs: scaler_class("cpu", enabled=True),
    )
    _, learner = run_training(tmp_path, monkeypatch)
    state = learner["scaler_state_dict"]
    assert state["_growth_tracker"] == 1
    _, resumed = run_training(tmp_path, monkeypatch)
    assert resumed["scaler_state_dict"]["_growth_tracker"] == 2
    assert resumed["scaler_state_dict"]["scale"] == state["scale"]
    monkeypatch.setattr(torch.amp, "GradScaler", scaler_class)
    _, cpu_checkpoint = run_training(tmp_path, monkeypatch)
    assert cpu_checkpoint["scaler_state_dict"] == resumed["scaler_state_dict"]


def test_rejections_keep_champion_fixed_across_iterations(tmp_path, monkeypatch):
    torch.manual_seed(1)
    initial = ChessNet(16, 1)
    champion, learner = run_training(tmp_path, monkeypatch, iterations=2)
    assert equal_weights(initial.state_dict(), champion["state_dict"])
    assert not equal_weights(champion["state_dict"], learner["state_dict"])
    assert all(state["step"].item() == 2 for state in learner["optimizer_state_dict"]["state"].values())


@pytest.mark.parametrize("field,value,message", [
    ("channels", 32, "architecture"),
    ("iteration", 100, "iterations differ"),
    ("learner_version", 2, "Unsupported"),
])
def test_incompatible_checkpoints_fail(tmp_path, monkeypatch, field, value, message):
    _, learner = run_training(tmp_path, monkeypatch)
    learner[field] = value
    torch.save(learner, tmp_path / "latest.pt")
    with pytest.raises(ValueError, match=message):
        run_training(tmp_path, monkeypatch)
