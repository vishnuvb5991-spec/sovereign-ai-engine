"""
Arize Phoenix Observability Integration.
Tracks LLM requests, prompts, responses, latencies, and token metrics.
View the interactive dashboard at: http://localhost:6006
"""

import os
from contextlib import contextmanager

_phoenix_active = False

def init_observability():
    """Launch local Phoenix UI and instrument clients."""
    global _phoenix_active
    try:
        import phoenix as px
        from openinference.instrumentation.openai import OpenAIInstrumentor
        
        # Launch Phoenix local server on port 6006
        session = px.launch_app(port=6006)
        OpenAIInstrumentor().instrument()
        _phoenix_active = True
        print(f"[+] Arize Phoenix initialized! Dashboard running at: {session.url}")
        return True
    except Exception as e:
        print(f"[!] Arize Phoenix failed to start (or already active): {e}")
        return False


def is_phoenix_active() -> bool:
    return _phoenix_active
