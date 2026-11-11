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
