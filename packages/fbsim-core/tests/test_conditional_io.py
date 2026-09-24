"""Conditional export uses the standard question-bank writer."""
from fbsim_core.conditional.io import save_as_question_bank
from fbsim_core.conditional.schema import ConditionalQuestionBank
from fbsim_core.questions.io import load_question_bank


def test_save_as_question_bank_roundtrip(tmp_path):
    bank = ConditionalQuestionBank(
        game_id="synthetic", checkpoint_turn=5, end_turn=10,
        conditions=[], questions=[],
    )
    path = tmp_path / "questions.json"
    save_as_question_bank(bank, path)
    restored = load_question_bank(path)
    assert restored.game_id == "synthetic"
    assert restored.snapshot_turn == 5
    assert restored.game_max_turn == 10
    assert restored.questions == []
