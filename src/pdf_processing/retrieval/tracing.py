
import os
from langsmith import Client, traceable, get_current_run_tree

_client = None


def tracing_on():
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def current_run_id():
    # id of the trace we are inside, so the UI can attach feedback to it
    if not tracing_on():
        return None
    run = get_current_run_tree()
    return str(run.id) if run else None


def save_feedback(run_id, score, comment=None):
    global _client
    if not tracing_on():
        return
    if _client is None:
        _client = Client()
    _client.create_feedback(run_id, key="user_score", score=score, comment=comment)


__all__ = ["traceable", "current_run_id", "save_feedback"]
