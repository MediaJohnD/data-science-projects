from src.contextual_triggers.trigger_engine import trigger_event


def test_trigger_event():
    assert trigger_event("start") == "Triggered start"
