import pytest

from tests.fakes import FakeLLM


@pytest.fixture(autouse=True, scope="session")
def _no_network_model():
    """Every test gets a fake model unless it passes its own. Real-model checks live in evaluation/, not tests/.

    Session scope, so module-scoped fixtures that build a Pipeline are patched too."""
    import src.pipeline as pipeline

    mp = pytest.MonkeyPatch()
    mp.setattr(pipeline, "LLMClient", lambda settings=None: FakeLLM())
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _isolated_kill_switch(tmp_path, monkeypatch):
    """The kill switch is read from the environment and a flag file; never let a real one leak into a test."""
    monkeypatch.setenv("KILL_SWITCH_FILE", str(tmp_path / "KILL_SWITCH"))
    monkeypatch.delenv("AUTO_RESPOND_DISABLED", raising=False)
