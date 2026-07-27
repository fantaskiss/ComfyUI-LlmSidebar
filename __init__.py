"""
ComfyUI-LlmSidebar
Self-contained LLM sidebar for ComfyUI with Chat + Describe tabs.
Right-click integration: LoadImage → vision, IMAGE outputs → vision, text → copy.
"""
import logging

WEB_DIRECTORY = "./js"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__version__ = "0.1.0"

_log = logging.getLogger("LlmSidebar")


def _register():
    """Hook into ComfyUI's PromptServer to register routes."""
    import server
    try:
        prompt_server = server.PromptServer.instance
    except Exception:
        _log.warning("PromptServer not ready, retrying via timer")
        return

    from . import llm_provider, llm_routes
    llm_routes.register_routes(prompt_server, llm_provider)
    _log.info("LlmSidebar v%s initialized", __version__)


# ComfyUI calls this when loading the node
try:
    _register()
except Exception:
    _log.exception("Registration failed, will retry on first request")
