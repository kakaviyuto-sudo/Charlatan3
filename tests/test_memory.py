from llm.memory import MemoryService

def test_remember_if_important_returns_false_for_short_text():
    memory = MemoryService(storage=None)
    assert memory.remember_if_important("hi") is False