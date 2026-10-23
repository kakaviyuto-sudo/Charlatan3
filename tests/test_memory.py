from llm.memory import MemoryService


def test_remember_if_important_returns_false_for_short_text():
    memory = MemoryService(storage=None)
    assert memory.remember_if_important("hi") is False


def test_remember_if_important_returns_true_for_substantive_text():
    memory = MemoryService(storage=None)
    text = "I prefer to keep notes about my daily task planning and meeting summaries."
    assert memory.remember_if_important(text) is True
    assert any("daily task planning" in entry["text"] for entry in memory.load())
