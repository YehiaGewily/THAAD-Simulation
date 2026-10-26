from thaad_sim.__main__ import main, reproduce_command

from conftest import CONFIGS


def test_reproduce_command():
    assert reproduce_command("configs/demo.yaml", 7) == \
        "python -m thaad_sim --config configs/demo.yaml --seed 7"
    assert reproduce_command("my configs/x.yaml", 7, legacy_rng=True) == \
        'python -m thaad_sim --config "my configs/x.yaml" --seed 7 --legacy-rng'


def test_cli_writes_replay_with_reproduce_info(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "demo.yaml").write_text(CONFIGS["demo"].read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    main(["--config", "configs/demo.yaml", "--seed", "5", "--no-browser"])

    replays = list((tmp_path / "replays").glob("replay_*.html"))
    assert len(replays) == 1
    html = replays[0].read_text(encoding="utf-8")
    assert "__REPLAY_DATA__" not in html
    assert '"seed":5,"config":"configs/demo.yaml","legacyRng":false' in html
