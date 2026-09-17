from pathlib import Path

from lse_v2.mmdit.materialize_librispeech import select_speaker_disjoint


def test_select_speaker_disjoint_fills_quotas_without_leakage():
    paths = [
        Path(f"{speaker}/{speaker}-1-{utterance:04d}.flac")
        for speaker in range(8)
        for utterance in range(5)
    ]
    selected = select_speaker_disjoint(paths, {"train": 15, "validation": 5, "test": 5}, seed=42)
    assert {key: len(value) for key, value in selected.items()} == {
        "train": 15,
        "validation": 5,
        "test": 5,
    }
    speakers = {
        split: {path.stem.split("-", 1)[0] for path in split_paths}
        for split, split_paths in selected.items()
    }
    assert speakers["train"].isdisjoint(speakers["validation"])
    assert speakers["train"].isdisjoint(speakers["test"])
    assert speakers["validation"].isdisjoint(speakers["test"])
