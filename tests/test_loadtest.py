import wave
from array import array

from loadtest.run_load import print_table
from loadtest.sim_caller import load_wav, rms


def test_rms_and_wav(tmp_path):
    assert rms(b"") == 0.0
    assert round(rms(array("h", [300, -300] * 10).tobytes())) == 300
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(b"\x00\x00" * 160)
    assert load_wav(path) == (16000, b"\x00\x00" * 160)


def _row(n, ok):
    return {"concurrent_calls": n, "turns": 7 * n, "p50_s": 0.8, "p95_s": 1.0 if ok else 2.0, "max_s": 2.1,
            "timeouts": 0, "failed_calls": 0, "pass": ok}


def test_capacity_message(capsys):
    print_table([_row(10, True), _row(20, True), _row(30, False)], 1.5)
    out = capsys.readouterr().out
    assert "Capacity: 20 calls" in out and "-> 2 box(es)" in out

    print_table([_row(10, True), _row(20, True)], 1.5)
    assert "at least 20" in capsys.readouterr().out
