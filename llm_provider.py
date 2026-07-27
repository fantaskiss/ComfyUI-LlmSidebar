"""
ComfyUI-LlmSidebar - llm_provider.py
Thin wrapper around ComfyUI-llama-cpp_vlm's LLAMA_CPP_STORAGE.
No self-built model loading -- reuses the single process-wide Llama instance.
"""
import os
import sys
import gc
import base64
import logging
from typing import Optional

import folder_paths

_log = logging.getLogger("LlmSidebar")

_SIDEBAR_UID = "sidebar"  # Fixed key in LLAMA_CPP_STORAGE.messages
_DEFAULT_N_CTX = 8192

# ---- LLAMA_CPP_STORAGE import ----
_STORAGE = None
_CHAT_HANDLERS = None
_VLM_MODULE_KEY = None

# ComfyUI loads custom nodes using their folder name as the module key.
# The folder name has a hyphen, so it is NOT a valid Python identifier
# and cannot be imported with a normal 'from nodes import ...'.
# Instead we look it up in sys.modules.
_VLM_FOLDER = "ComfyUI-llama-cpp_vlm"
# ComfyUI 0.28 uses absolute paths as module keys (e.g. N:\\...\\nodes.py).
# Earlier versions use hyphenated folder names.
# We also accept short suffix matches.
_VLM_NODES_KEYS = [
    "ComfyUI-llama-cpp_vlm.nodes",
    "ComfyUI_llama_cpp_vlm.nodes",
    "ComfyUI-llama-cpp-vlm.nodes",
]
# Augment with absolute paths derived from folder_paths
try:
    _vlm_abs = os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-llama-cpp_vlm")
    _VLM_NODES_KEYS.extend([_vlm_abs + ".nodes", _vlm_abs + "\\nodes"])
    _VLM_NODES_KEYS.extend([
        _vlm_abs.replace(os.sep, "/") + ".nodes",
        _vlm_abs.replace(os.sep, "/") + "/nodes",
    ])
except Exception:
    pass




def _get_storage():
    """Get LLAMA_CPP_STORAGE from the already-loaded llama-cpp-vlm module."""
    global _STORAGE, _CHAT_HANDLERS, _VLM_MODULE_KEY
    if _STORAGE is not None:
        return _STORAGE

    # Try each possible module key in sys.modules
    for key in _VLM_NODES_KEYS:
        mod = sys.modules.get(key)
        if mod is not None:
            storage = _safe_getattr(mod, "LLAMA_CPP_STORAGE")
            if storage is not None:
                _STORAGE = storage
                _CHAT_HANDLERS = _safe_getattr(mod, "chat_handlers") or ["None"]
                _VLM_MODULE_KEY = key
                _log.info("LLAMA_CPP_STORAGE found via sys.modules[%s]", key)
                return _STORAGE

    # Fallback: scan all loaded modules
    for name, mod in sys.modules.items():
        if mod is None:
            continue
        storage = _safe_getattr(mod, "LLAMA_CPP_STORAGE")
        if storage is not None:
            _STORAGE = storage
            _CHAT_HANDLERS = _safe_getattr(mod, "chat_handlers") or ["None"]
            _VLM_MODULE_KEY = name
            _log.info("LLAMA_CPP_STORAGE found via scan in sys.modules[%s]", name)
            return _STORAGE

    raise RuntimeError(
        "ComfyUI-llama-cpp_vlm is not loaded. "
        "Ensure it is installed in custom_nodes/ComfyUI-llama-cpp_vlm/ "
        "and ComfyUI has started successfully."
    )



def _safe_getattr(obj, name):
    """Like getattr, but rejects namespace proxy objects from ComfyUI."""
    val = getattr(obj, name, None)
    if val is None:
        return None
    tn = type(val).__name__
    # ComfyUI 0.28 wraps module contents in _OpNamespace, _ClassNamespace, etc.
    # These proxy objects have __getattr__ that returns placeholder objects for
    # any attribute (making hasattr always True). Filter them out.
    if "Namespace" in tn and tn != "SimpleNamespace":
        return None
    return val




# ============================================================
# Model scanning (direct filesystem scan)
# ============================================================

def _get_models_dir() -> str:
    d = os.path.join(folder_paths.models_dir, "LLM")
    os.makedirs(d, exist_ok=True)
    return d

def scan_text_models() -> list[str]:
    """All .gguf files excluding mmproj."""
    try:
        return sorted(
            f for f in os.listdir(_get_models_dir())
            if f.endswith(".gguf") and "mmproj" not in f.lower()
        )
    except OSError:
        return []

def scan_mmproj_files() -> list[str]:
    try:
        return sorted(
            f for f in os.listdir(_get_models_dir())
            if f.endswith(".gguf") and "mmproj" in f.lower()
        )
    except OSError:
        return []

def get_chat_handlers() -> list[str]:
    """Return the chat_handler list from llama-cpp-vlm."""
    try:
        _get_storage()
        return list(_CHAT_HANDLERS) if _CHAT_HANDLERS else ["None"]
    except RuntimeError:
        return ["None"]

def scan_vision_models() -> list[str]:
    """Models that have a matching mmproj (simple heuristic)."""
    try:
        models = scan_text_models()
        mmprojs = scan_mmproj_files()
        if not mmprojs:
            return []
        result = []
        for m in models:
            base = m.lower().replace(".gguf", "")
            for mp in mmprojs:
                mp_lower = mp.lower().replace(".gguf", "").replace("mmproj", "")
                mp_lower = mp_lower.strip("-_")
                if mp_lower and (mp_lower in base or base in mp_lower):
                    result.append(m)
                    break
        return sorted(result)
    except Exception:
        return []


# ============================================================
# Model loading via LLAMA_CPP_STORAGE
# ============================================================

def _build_config(model: str, *, chat_handler: str = "None",
                  mmproj: str = "None", n_ctx: int = _DEFAULT_N_CTX,
                  vram_limit: int = -1, image_min_tokens: int = 0,
                  image_max_tokens: int = 0) -> dict:
    return {
        "model": model,
        "mmproj": mmproj,
        "chat_handler": chat_handler,
        "n_ctx": n_ctx,
        "vram_limit": vram_limit,
        "image_min_tokens": image_min_tokens,
        "image_max_tokens": image_max_tokens,
    }

# Qwen-VL models require at minimum 1024 image tokens (llama.cpp #16842).
# Default image_min/max_tokens are 0 in llama-cpp-vlm's loader node, but
# that causes "Critical Overflow" / "Context Shift" errors at inference time.
# Set sensible defaults so the chat_handler reserves enough context space.
_DEFAULT_IMAGE_MIN_TOKENS = 1024
_DEFAULT_IMAGE_MAX_TOKENS = 4096

def _ensure_model(model: str, *, chat_handler: str = "None",
                  mmproj: str = "None", n_ctx: int = _DEFAULT_N_CTX,
                  vram_limit: int = -1, image_min_tokens: int = _DEFAULT_IMAGE_MIN_TOKENS,
                  image_max_tokens: int = _DEFAULT_IMAGE_MAX_TOKENS):
    """Load model via LLAMA_CPP_STORAGE if not already loaded with same config."""
    storage = _get_storage()
    config = _build_config(model, chat_handler=chat_handler, mmproj=mmproj,
                           n_ctx=n_ctx, vram_limit=vram_limit,
                           image_min_tokens=image_min_tokens,
                           image_max_tokens=image_max_tokens)
    if storage.llm is None or storage.current_config != config:
        # Save sidebar history before reload (load_model cleans all state)
        sidebar_history = storage.messages.pop(_SIDEBAR_UID, [])
        sidebar_sys = storage.sys_prompts.pop(_SIDEBAR_UID, "")
        _log.info("Loading model via LLAMA_CPP_STORAGE: %s (handler=%s, mmproj=%s, ctx=%d, img_min=%d, img_max=%d)",
                  model, chat_handler, mmproj, n_ctx, image_min_tokens, image_max_tokens)
        storage.load_model(config)
        # Restore sidebar history after reload
        if sidebar_history:
            storage.messages[_SIDEBAR_UID] = sidebar_history
            storage.sys_prompts[_SIDEBAR_UID] = sidebar_sys
    return storage.llm

def unload_model():
    """Unload via LLAMA_CPP_STORAGE."""
    storage = _get_storage()
    storage.clean(all=True)
    _log.info("Model unloaded via LLAMA_CPP_STORAGE")

def reset_history():
    """Clear sidebar conversation history from STORAGE."""
    storage = _get_storage()
    storage.clean_state(_SIDEBAR_UID)
    _log.info("Sidebar conversation history cleared")


# ============================================================
# Inference
# ============================================================

def _build_options(opts: Optional[dict]) -> dict:
    if not opts:
        return {}
    result = {}
    for k in ("temperature", "top_p", "top_k", "min_p", "repeat_penalty",
              "frequency_penalty", "presence_penalty", "mirostat_mode",
              "mirostat_eta", "mirostat_tau", "typical_p"):
        if k in opts and opts[k] is not None:
            result[k] = float(opts[k]) if k not in ("top_k", "mirostat_mode") else int(opts[k])
    if "max_tokens" in opts and opts["max_tokens"] is not None:
        result["max_tokens"] = int(opts["max_tokens"])
    if "seed" in opts and opts["seed"] is not None:
        seed = int(opts["seed"])
        if seed >= 0:
            result["seed"] = seed
    return result


def chat(model: str, prompt: str, *,
         system_prompt: str = "",
         chat_handler: str = "None",
         mmproj: str = "None",
         options: Optional[dict] = None,
         stream: bool = False):
    """Generate text. Returns str or generator of {chunk, done, full_response}."""
    opts = dict(options or {})
    n_ctx_val = int(opts.pop("n_ctx", _DEFAULT_N_CTX))
    vram_limit = int(opts.pop("vram_limit", -1))
    image_min = int(opts.pop("image_min_tokens", _DEFAULT_IMAGE_MIN_TOKENS))
    image_max = int(opts.pop("image_max_tokens", _DEFAULT_IMAGE_MAX_TOKENS))

    llm = _ensure_model(model, chat_handler=chat_handler, mmproj=mmproj,
                        n_ctx=n_ctx_val, vram_limit=vram_limit,
                        image_min_tokens=image_min, image_max_tokens=image_max)
    llama_opts = _build_options(opts)

    storage = _get_storage()
    uid = _SIDEBAR_UID

    # Build messages from STORAGE history
    messages = []
    history = storage.messages.get(uid, [])
    # Check if system prompt changed
    last_sys = storage.sys_prompts.get(uid, None)
    if last_sys != system_prompt:
        history = []
        storage.sys_prompts[uid] = system_prompt
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
    else:
        if system_prompt.strip() and not history:
            messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    if not stream:
        resp = llm.create_chat_completion(messages=messages, **llama_opts)
        text = resp["choices"][0]["message"]["content"]
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": text})
        storage.messages[uid] = history
        _trim_history(history)
        return text.strip()

    # Streaming
    gen = llm.create_chat_completion(messages=messages, stream=True, **llama_opts)
    full = ""
    for chunk in gen:
        choices = chunk.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        content = (delta.get("content", "") if isinstance(delta, dict) else "")
        if not content and choice.get("finish_reason"):
            break
        if content:
            full += content
            yield {"chunk": content, "done": False}
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": full.strip()})
    storage.messages[uid] = history
    _trim_history(history)
    
    # Qwen3.5 uses hybrid cache (Mamba+Attention). Clear KV cache after each
    # generation to avoid stale cache interfering with multi-turn conversations.
    cfg = storage.current_config
    if cfg and cfg.get("chat_handler", "") in ("Qwen3.5", "Qwen3.5-Thinking"):
        try:
            storage.llm.n_tokens = 0
            storage.llm._ctx.memory_clear(True)
            if storage.llm.is_hybrid and storage.llm._hybrid_cache_mgr is not None:
                storage.llm._hybrid_cache_mgr.clear()
        except Exception:
            pass
    
    yield {"chunk": "", "done": True, "full_response": full.strip()}


def _trim_history(history: list, max_turns: int = 20):
    limit = max_turns * 2
    if len(history) > limit:
        del history[:len(history) - limit]


def _encode_image_to_uri(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{data}"


def vision(model: str, prompt: str, images, *,
           system_prompt: str = "",
           chat_handler: str = "None",
           mmproj: str = "None",
           options: Optional[dict] = None):
    """Generate description from image(s). Returns str."""
    if chat_handler == "None":
        raise ValueError("Vision requires a chat_handler. Select one in the sidebar.")

    opts = dict(options or {})
    n_ctx_val = int(opts.pop("n_ctx", _DEFAULT_N_CTX))
    vram_limit = int(opts.pop("vram_limit", -1))
    image_min = int(opts.pop("image_min_tokens", _DEFAULT_IMAGE_MIN_TOKENS))
    image_max = int(opts.pop("image_max_tokens", _DEFAULT_IMAGE_MAX_TOKENS))

    llm = _ensure_model(model, chat_handler=chat_handler, mmproj=mmproj,
                        n_ctx=n_ctx_val, vram_limit=vram_limit,
                        image_min_tokens=image_min, image_max_tokens=image_max)
    llama_opts = _build_options(opts)

    encoded = []
    for img in (images if isinstance(images, (list, tuple)) else [images]):
        if isinstance(img, str) and os.path.exists(img):
            encoded.append(_encode_image_to_uri(img))
        elif isinstance(img, str) and img.startswith("data:image"):
            encoded.append(img)
        else:
            _log.warning("Skipping invalid image: %s", type(img))

    if not encoded:
        raise ValueError("No valid images provided")

    content_parts = [{"type": "text", "text": prompt}]
    for uri in encoded:
        content_parts.append({"type": "image_url", "image_url": {"url": uri}})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content_parts})

    resp = llm.create_chat_completion(messages=messages, **llama_opts)
    result = resp["choices"][0]["message"]["content"].strip()

    # Save vision response to sidebar history so chat messages can
    # reference it (e.g. "expand on this description").
    storage = _get_storage()
    storage.messages[_SIDEBAR_UID] = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": result},
    ]
    storage.sys_prompts[_SIDEBAR_UID] = system_prompt

    return result


# ============================================================
# Status
# ============================================================

def get_status() -> dict:
    try:
        storage = _get_storage()
    except RuntimeError as e:
        return {"loaded": False, "error": str(e)}

    text_models = scan_text_models()
    vision_models = scan_vision_models()
    return {
        "loaded": storage.llm is not None,
        "loaded_model": storage.current_config.get("model") if storage.current_config else None,
        "loaded_handler": storage.current_config.get("chat_handler") if storage.current_config else None,
        "text_models": text_models,
        "vision_models": vision_models,
        "mmproj_files": scan_mmproj_files(),
        "chat_handlers": get_chat_handlers(),
        "models_dir": _get_models_dir(),
    }
