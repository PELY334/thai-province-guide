"""The answer-matching rule used by eval/eval_answers.py."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "eval_answers", Path(__file__).resolve().parents[1] / "eval" / "eval_answers.py")
ea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ea)


def q(answers, all_required=False):
    return {"answers": answers, "all_required": all_required}


def test_alias_and_number_matching():
    assert ea.is_correct("It is the Indian Elm.", q(["indian elm", "holoptelea"]))
    assert ea.is_correct("The difference is 2829 km².", q(["2,829", "2829"]))
    assert ea.is_correct("About 2,829 square kilometres", q(["2829"]))
    assert not ea.is_correct("Chiang Rai", q(["phayao"]))


def test_all_required():
    assert ea.is_correct("Udon Thani, by 2,218 km2", q(["udon thani", "2,218"], True))
    assert not ea.is_correct("Udon Thani is bigger", q(["udon thani", "2,218"], True))


def test_dont_know_detector():
    assert ea.DONT_KNOW.search("I don't know the current governor.")
    assert not ea.DONT_KNOW.search("The provincial tree is the neem.")
