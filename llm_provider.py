"""
ComfyUI-LlmSidebar - llm_provider.py
Thin wrapper around ComfyUI-llama-cpp_vlm's LLAMA_CPP_STORAGE.
No self-built model loading -- reuses the single process-wide Llama instance.
"""
import os
import sys
import gc
import base64
import json
import time
import urllib.error
import urllib.request
import logging
from typing import Optional

import folder_paths

_log = logging.getLogger("LlmSidebar")

# 提示词生成系统（阶段 0 程序路由 + 阶段 1 LLM 组装）
try:
    from . import prompt_generator
except ImportError:  # pragma: no cover - 直接文件加载测试
    import prompt_generator

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
                  image_max_tokens: int = 0,
                  n_gpu_layers: int = -1,
                  cache_type_k: str = "default",
                  cache_type_v: str = "default",
                  n_cpu_moe: int = 0,
                  n_seq_max: int = 1) -> dict:
    return {
        "model": model,
        "mmproj": mmproj,
        "chat_handler": chat_handler,
        "n_ctx": n_ctx,
        "vram_limit": vram_limit,
        "image_min_tokens": image_min_tokens,
        "image_max_tokens": image_max_tokens,
        "n_gpu_layers": n_gpu_layers,
        "cache_type_k": cache_type_k,
        "cache_type_v": cache_type_v,
        "n_cpu_moe": n_cpu_moe,
        "n_seq_max": n_seq_max,
    }

# Qwen-VL models require at minimum 1024 image tokens (llama.cpp #16842).
# Default image_min/max_tokens are 0 in llama-cpp-vlm's loader node, but
# that causes "Critical Overflow" / "Context Shift" errors at inference time.
# Set sensible defaults so the chat_handler reserves enough context space.
_DEFAULT_IMAGE_MIN_TOKENS = 1024
_DEFAULT_IMAGE_MAX_TOKENS = 4096

# ============================================================
# Tool registry
# ============================================================

_TOOLS = {}
_TOOLS_ENABLED = False
_WIKI_PATH = ""
_MAX_TOOL_ROUNDS = 10
_TOOL_RESULT_MAX_CHARS = 8000
_LAST_CONTEXT_TOKENS = 0    # estimated tokens in current conversation
_LAST_CONTEXT_LIMIT = 8192  # n_ctx from last model load


def register_tool(name, fn, description, parameters_schema):
    _TOOLS[name] = {"fn": fn, "description": description, "parameters": parameters_schema}


def dispatch_tool(tool_name, tool_params):
    if tool_name not in _TOOLS:
        return f"Error: Tool '{tool_name}' not found. Available: {list(_TOOLS.keys())}"
    try:
        result = _TOOLS[tool_name]["fn"](**tool_params)
        if result is None:
            return "(tool returned no output)"
        s = str(result)
        if len(s) > _TOOL_RESULT_MAX_CHARS:
            s = s[:_TOOL_RESULT_MAX_CHARS] + f"\n...(truncated, total {len(str(result))} chars)"
        return s
    except Exception as e:
        import traceback
        return f"Error: {e}\n{traceback.format_exc()}"


def get_tools_config():
    return {
        "enabled": _TOOLS_ENABLED,
        "wiki_path": _WIKI_PATH,
        "max_rounds": _MAX_TOOL_ROUNDS,
        "result_max_chars": _TOOL_RESULT_MAX_CHARS,
    }


def set_tools_config(enabled=None, wiki_path=None, max_rounds=None, result_max_chars=None):
    global _TOOLS_ENABLED, _WIKI_PATH, _MAX_TOOL_ROUNDS, _TOOL_RESULT_MAX_CHARS
    if enabled is not None:
        _TOOLS_ENABLED = bool(enabled)
    if wiki_path is not None:
        _WIKI_PATH = str(wiki_path)
    if max_rounds is not None:
        _MAX_TOOL_ROUNDS = max(1, int(max_rounds))
    if result_max_chars is not None:
        _TOOL_RESULT_MAX_CHARS = max(100, int(result_max_chars))
    return get_tools_config()


def _ensure_model(model: str, *, chat_handler: str = "None",
                  mmproj: str = "None", n_ctx: int = _DEFAULT_N_CTX,
                  vram_limit: int = -1, image_min_tokens: int = _DEFAULT_IMAGE_MIN_TOKENS,
                  image_max_tokens: int = _DEFAULT_IMAGE_MAX_TOKENS,
                  n_gpu_layers: int = -1,
                  cache_type_k: str = "default",
                  cache_type_v: str = "default",
                  n_cpu_moe: int = 0,
                  n_seq_max: int = 1):
    """Load model via LLAMA_CPP_STORAGE.

    If a model is already loaded (by workflow node or previous sidebar call),
    reuse it AS-IS -- no config comparison, no reload. This prevents the
    infinite reload loop when sidebar and workflow node have different settings.

    Only when no model is loaded, use sidebar's own settings to load.
    """
    storage = _get_storage()

    # Model already loaded -- reuse without config comparison
    if storage.llm is not None:
        return storage.llm

    # Model not loaded -- use sidebar's own settings
    config = _build_config(model, chat_handler=chat_handler, mmproj=mmproj,
                           n_ctx=n_ctx, vram_limit=vram_limit,
                           image_min_tokens=image_min_tokens,
                           image_max_tokens=image_max_tokens,
                           n_gpu_layers=n_gpu_layers,
                           cache_type_k=cache_type_k,
                           cache_type_v=cache_type_v,
                           n_cpu_moe=n_cpu_moe,
                           n_seq_max=n_seq_max)
    _log.info("Loading model via LLAMA_CPP_STORAGE: %s (handler=%s, mmproj=%s, ctx=%d, gpu_layers=%d, kv_k=%s, kv_v=%s)",
              model, chat_handler, mmproj, n_ctx, n_gpu_layers, cache_type_k, cache_type_v)
    storage.load_model(config)
    return storage.llm


def apply_settings(model: str, *, chat_handler: str = "None",
                   mmproj: str = "None", n_ctx: int = _DEFAULT_N_CTX,
                   vram_limit: int = -1, image_min_tokens: int = _DEFAULT_IMAGE_MIN_TOKENS,
                   image_max_tokens: int = _DEFAULT_IMAGE_MAX_TOKENS,
                   n_gpu_layers: int = -1,
                   cache_type_k: str = "default",
                   cache_type_v: str = "default",
                   n_cpu_moe: int = 0,
                   n_seq_max: int = 1,
                   tools_enabled: bool = None,
                   wiki_path: str = None,
                   max_rounds: int = None,
                   result_max_chars: int = None,
                   backend_mode: str = "local",
                   remote_url: str = None):
    """Explicitly unload and reload with new settings. Used by Setup panel.
    Also accepts tool config params forwarded from the frontend Apply button.

    backend_mode='remote': 不操作 LLAMA_CPP_STORAGE，仅校验小主机 llama.cpp 可达。"""
    if backend_mode == "remote":
        set_backend("remote", remote_url)
        if not remote_health_ok():
            raise RuntimeError(
                "小主机 llama.cpp 不可达: {} (/health 非 200)。请先在小主机启动模型。".format(_REMOTE_BASE))
        if tools_enabled is not None or wiki_path is not None:
            set_tools_config(
                enabled=tools_enabled,
                wiki_path=wiki_path,
                max_rounds=max_rounds,
                result_max_chars=result_max_chars,
            )
        _log.info("Remote backend verified at %s", _REMOTE_BASE)
        return True

    set_backend("local", remote_url)
    storage = _get_storage()

    # Save sidebar state before clean
    sidebar_history = storage.messages.pop(_SIDEBAR_UID, [])
    sidebar_sys = storage.sys_prompts.pop(_SIDEBAR_UID, "")

    storage.clean(all=True)

    config = _build_config(model, chat_handler=chat_handler, mmproj=mmproj,
                           n_ctx=n_ctx, vram_limit=vram_limit,
                           image_min_tokens=image_min_tokens,
                           image_max_tokens=image_max_tokens,
                           n_gpu_layers=n_gpu_layers,
                           cache_type_k=cache_type_k,
                           cache_type_v=cache_type_v,
                           n_cpu_moe=n_cpu_moe,
                           n_seq_max=n_seq_max)
    _log.info("Applying settings + reloading: %s (handler=%s, ctx=%d, gpu_layers=%d)",
              model, chat_handler, n_ctx, n_gpu_layers)
    storage.load_model(config)

    # Restore sidebar state
    if sidebar_history:
        storage.messages[_SIDEBAR_UID] = sidebar_history
        storage.sys_prompts[_SIDEBAR_UID] = sidebar_sys

    # Apply tool config if provided (from Apply & Reload button)
    if tools_enabled is not None or wiki_path is not None:
        set_tools_config(
            enabled=tools_enabled,
            wiki_path=wiki_path,
            max_rounds=max_rounds,
            result_max_chars=result_max_chars,
        )

    return True

def unload_model():
    """Unload via LLAMA_CPP_STORAGE (local) or clear history only (remote)."""
    if _BACKEND_MODE == "remote":
        # 远程模型由小主机 llama.cpp 管理（bat 启停），这里只清对话
        reset_history()
        _log.info("Remote backend: sidebar history cleared (server model untouched)")
        return
    storage = _get_storage()
    storage.clean(all=True)
    _log.info("Model unloaded via LLAMA_CPP_STORAGE")

def reset_history():
    """Clear sidebar conversation history from STORAGE."""
    storage = _get_storage()
    storage.clean_state(_SIDEBAR_UID)
    _log.info("Sidebar conversation history cleared")


# ============================================================
# Remote backend (小主机 llama.cpp, OpenAI-compatible HTTP)
# ============================================================

_BACKEND_MODE = "local"          # "local" = LLAMA_CPP_STORAGE, "remote" = 小主机 llama.cpp
_REMOTE_BASE = "http://10.0.0.8:60000"
_REMOTE_SHORT_TIMEOUT = 8
_REMOTE_LONG_TIMEOUT = 600
_REMOTE_CACHE = {"ts": 0.0, "data": None}

# Keys llama.cpp server (OpenAI endpoint) accepts for a single completion.
_REMOTE_OPT_KEYS = ("temperature", "top_p", "top_k", "min_p",
                    "repeat_penalty", "frequency_penalty", "presence_penalty",
                    "max_tokens", "seed", "typical_p")


def set_backend(mode="local", base_url=None):
    """Switch inference backend: 'local' (LLAMA_CPP_STORAGE) or 'remote' (小主机 llama.cpp HTTP)."""
    global _BACKEND_MODE, _REMOTE_BASE
    if mode in ("local", "remote"):
        _BACKEND_MODE = mode
    if base_url:
        url = str(base_url).strip()
        if url:
            _REMOTE_BASE = url.rstrip("/")


def get_backend():
    return {"mode": _BACKEND_MODE, "base_url": _REMOTE_BASE}


def _remote_url(path):
    return _REMOTE_BASE.rstrip("/") + path


def remote_health_ok():
    """True when /health returns 200 (llama-server loads -> 503 until ready)."""
    try:
        with urllib.request.urlopen(_remote_url("/health"), timeout=_REMOTE_SHORT_TIMEOUT) as r:
            r.read()
        return True
    except urllib.error.HTTPError as e:
        return e.code == 200
    except Exception:
        return False


def _remote_get_json(path, timeout=_REMOTE_SHORT_TIMEOUT):
    with urllib.request.urlopen(_remote_url(path), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def remote_info(force=False):
    """Cached server info: model id / n_ctx / model_path / error."""
    now = time.time()
    if not force and _REMOTE_CACHE["data"] and (now - _REMOTE_CACHE["ts"]) < 5:
        return _REMOTE_CACHE["data"]
    info = {"ready": False, "model": None, "n_ctx": None, "path": None, "error": None}
    try:
        models = _remote_get_json("/v1/models")
        ids = [m.get("id") for m in (models.get("data") or []) if m.get("id")]
        if ids:
            info["ready"] = True
            info["model"] = ids[0]
        try:
            props = _remote_get_json("/props")
            info["n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx")
            info["path"] = props.get("model_path")
        except Exception:
            pass
    except Exception as e:
        info["error"] = str(e)
    _REMOTE_CACHE["ts"] = time.time()
    _REMOTE_CACHE["data"] = info
    return info


def _remote_opts(opts):
    """Inference params only (n_ctx / gpu layers etc. are server-side on 小主机)."""
    return {k: v for k, v in (opts or {}).items()
            if k in _REMOTE_OPT_KEYS and v is not None}


def _remote_complete_text(messages, opts):
    """Non-streaming chat completion against 小主机 llama.cpp. Returns content str."""
    info = remote_info()
    payload = {"model": info.get("model") or "local-model",
               "messages": messages, "stream": False}
    payload.update(_remote_opts(opts))
    req = urllib.request.Request(
        _remote_url("/v1/chat/completions"), method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=_REMOTE_LONG_TIMEOUT) as r:
        obj = json.loads(r.read().decode("utf-8", "replace"))
    msg = ((obj.get("choices") or [{}])[0].get("message") or {})
    return msg.get("content") or ""


def _remote_chat_stream(messages, opts):
    """Streaming chat completion. Yields {chunk|done|full_response|error}."""
    info = remote_info()
    payload = {"model": info.get("model") or "local-model",
               "messages": messages, "stream": True}
    payload.update(_remote_opts(opts))
    req = urllib.request.Request(
        _remote_url("/v1/chat/completions"), method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"})
    try:
        with urllib.request.urlopen(req, timeout=_REMOTE_LONG_TIMEOUT) as resp:
            full = ""
            while True:
                line = resp.readline()
                if not line:
                    break
                text = line.decode("utf-8", "replace").strip()
                if not text.startswith("data:"):
                    continue
                data = text[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                c = delta.get("content") if isinstance(delta, dict) else None
                if c:
                    full += c
                    yield {"chunk": c, "done": False}
                if choices[0].get("finish_reason"):
                    break
        yield {"chunk": "", "done": True, "full_response": full}
    except Exception as e:
        yield {"chunk": "", "done": True, "full_response": "", "error": str(e)}


def remote_chat(prompt, system_prompt="", opts=None):
    """Remote chat with the same sidebar-history semantics as local chat()."""
    storage = _get_storage()
    uid = _SIDEBAR_UID
    messages = []
    history = storage.messages.get(uid, [])
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

    full = ""
    for ev in _remote_chat_stream(messages, opts):
        if ev.get("error"):
            yield {"chunk": "", "done": True, "full_response": "",
                   "error": ev["error"]}
            return
        if ev.get("done"):
            full = ev.get("full_response", "")
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": full.strip()})
            storage.messages[uid] = history
            _trim_history(history)
            _update_context_usage(history, storage)
            yield {"chunk": "", "done": True, "full_response": full.strip()}
            return
        c = ev.get("chunk", "")
        if c:
            full += c
            yield {"chunk": c, "done": False}


def remote_vision(prompt, images, system_prompt="", options=None):
    """Vision via remote llama.cpp (model with --mmproj on the server side)."""
    encoded = []
    for img in (images if isinstance(images, (list, tuple)) else [images]):
        if isinstance(img, str) and img.startswith("data:image"):
            encoded.append(img)
        elif isinstance(img, str) and os.path.exists(img):
            encoded.append(_encode_image_to_uri(img))
    if not encoded:
        raise ValueError("No valid images provided")

    content_parts = [{"type": "text", "text": prompt}]
    for uri in encoded:
        content_parts.append({"type": "image_url", "image_url": {"url": uri}})
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content_parts})

    result = _remote_complete_text(messages, options)

    storage = _get_storage()
    storage.messages[_SIDEBAR_UID] = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": result},
    ]
    storage.sys_prompts[_SIDEBAR_UID] = system_prompt
    return result


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
    n_gpu_layers = int(opts.pop("n_gpu_layers", -1))
    cache_type_k = str(opts.pop("cache_type_k", "default"))
    cache_type_v = str(opts.pop("cache_type_v", "default"))
    n_cpu_moe = int(opts.pop("n_cpu_moe", 0))
    n_seq_max = int(opts.pop("n_seq_max", 1))

    if _BACKEND_MODE == "remote":
        # 远程后端：忽略本地 chat_handler / mmproj（模型由小主机 llama.cpp 服务端决定）
        yield from remote_chat(prompt, system_prompt=system_prompt, opts=opts)
        return

    llm = _ensure_model(model, chat_handler=chat_handler, mmproj=mmproj,
                        n_ctx=n_ctx_val, vram_limit=vram_limit,
                        image_min_tokens=image_min, image_max_tokens=image_max,
                        n_gpu_layers=n_gpu_layers,
                        cache_type_k=cache_type_k,
                        cache_type_v=cache_type_v,
                        n_cpu_moe=n_cpu_moe,
                        n_seq_max=n_seq_max)
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
        return _chat_with_tools(
            llm, messages, history, uid, storage,
            prompt, system_prompt, llama_opts, stream=False
        )

    # Streaming (with tool support)
    if _TOOLS_ENABLED:
        result = _chat_with_tools(
            llm, messages, history, uid, storage,
            prompt, system_prompt, llama_opts, stream=True
        )
        if isinstance(result, str):
            yield {"chunk": result, "done": True, "full_response": result}
            return
        for chunk in result:
            yield chunk
        _clear_kv_cache(storage)
        return
    
    gen = llm.create_chat_completion(messages=messages, stream=True, **llama_opts)
    full = ""
    for chunk in gen:
        choices = chunk.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        c = (delta.get("content", "") if isinstance(delta, dict) else "")
        if not c and choice.get("finish_reason"):
            break
        if c:
            full += c
            yield {"chunk": c, "done": False}
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": full.strip()})
    storage.messages[uid] = history
    _trim_history(history)
    _update_context_usage(history, storage)
    _clear_kv_cache(storage)
    yield {"chunk": "", "done": True, "full_response": full.strip()}



def _build_tool_schemas():
    """Build OpenAI-compatible tool schemas for native function calling."""
    if not _TOOLS_ENABLED or not _TOOLS:
        return None
    schemas = []
    for name, info in _TOOLS.items():
        props = {}
        required = []
        for pname, pspec in info["parameters"].items():
            props[pname] = {
                "type": pspec.get("type", "string"),
                "description": pspec.get("description", ""),
            }
            required.append(pname)
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": info["description"],
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        })
    return schemas


def _chat_with_tools(llm, messages, history, uid, storage, prompt, system_prompt, llama_opts, stream):
    """Tool-calling loop using llama-cpp-python native function calling (tools parameter)."""
    
    tool_schemas = _build_tool_schemas()
    _log.info("[TOOLS] enabled=%s, schemas=%s, wiki=%s, max_rounds=%s",
              _TOOLS_ENABLED, len(tool_schemas) if tool_schemas else 0,
              _WIKI_PATH, _MAX_TOOL_ROUNDS)
    if tool_schemas:
        _log.info("[TOOLS-DEBUG] tool names: %s", [t["function"]["name"] for t in tool_schemas])
    else:
        _log.info("[TOOLS-DEBUG] No tool schemas — tools disabled or _TOOLS empty. _TOOLS=%s", list(_TOOLS.keys()))
    tool_msgs = list(messages)
    
    rounds = 0
    max_r = _MAX_TOOL_ROUNDS if _TOOLS_ENABLED and tool_schemas else 1
    history.append({"role": "user", "content": prompt})
    
    while rounds < max_r:
        # Call model with native tools parameter
        if _TOOLS_ENABLED and tool_schemas:
            _log.info("[TOOLS] Round %s: calling with %s tools", rounds, len(tool_schemas))
            resp = llm.create_chat_completion(
                messages=tool_msgs,
                tools=tool_schemas,
                **llama_opts
            )
        else:
            _log.info("[TOOLS] Round %s: calling WITHOUT tools (enabled=%s, schemas=%s)",
                      rounds, _TOOLS_ENABLED, bool(tool_schemas))
            resp = llm.create_chat_completion(messages=tool_msgs, **llama_opts)
        
        msg = resp["choices"][0]["message"]
        tool_calls = msg.get("tool_calls")
        content_text = (msg.get("content") or "").strip()
        
        # Try to parse tool call from either native tool_calls or text content
        tool_name = None
        tool_params = {}
        
        if tool_calls:
            # Native function calling
            tc = tool_calls[0]
            tool_name = tc["function"]["name"]
            try:
                tool_params = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tool_params = {}
            _log.info("Tool call (native): %s(%s)", tool_name, tool_params)
        elif content_text:
            # Text-format tool call: <tool_call><function=name><parameter=key>value...
            import re
            fn_match = re.search(r'<function=([^>]+)>', content_text)
            if fn_match:
                tool_name = fn_match.group(1).strip()
                # Extract parameters: <parameter=key>value (non-greedy, stops at < or end)
                param_matches = re.findall(r'<parameter=([^>]+)>\s*([^<]+)', content_text)
                tool_params = {k.strip(): v.strip() for k, v in param_matches}
                _log.info("Tool call (text): %s(%s)", tool_name, tool_params)
        
        if not tool_name:
            # No tool call — final answer
            history.append({"role": "assistant", "content": content_text or "(empty)"})
            storage.messages[uid] = history
            _trim_history(history)
            _update_context_usage(history, storage)
            if stream:
                return _stream_final(llm, tool_msgs, llama_opts, content_text)
            return content_text
        
        # We have a tool call — execute it
        tool_result = dispatch_tool(tool_name, tool_params)
        
        # Append to conversation
        tool_msgs.append({
            "role": "assistant",
            "content": content_text,
        })
        tool_msgs.append({
            "role": "user",
            "content": f"Tool '{tool_name}' result:\n{tool_result}\n\nBased on this result, continue answering the user's original question."
        })
        history.append({"role": "assistant", "content": f"[tool: {tool_name}]"})
        _update_context_usage(history, storage)
        rounds += 1
    
    # Max rounds — force final
    tool_msgs.append({"role": "user", "content": "Maximum tool calls reached. Provide your final answer now."})
    resp = llm.create_chat_completion(messages=tool_msgs, **llama_opts)
    text = resp["choices"][0]["message"]["content"].strip()
    history.append({"role": "assistant", "content": text or "(empty)"})
    storage.messages[uid] = history
    _trim_history(history)
    _update_context_usage(history, storage)
    return text


def _update_context_usage(history, storage):
    global _LAST_CONTEXT_TOKENS, _LAST_CONTEXT_LIMIT
    # Estimate tokens: total chars / 3 (rough for mixed CN/EN)
    total_chars = sum(len(str(m.get("content", ""))) for m in history)
    _LAST_CONTEXT_TOKENS = max(1, total_chars // 3)
    cfg = storage.current_config
    if cfg:
        _LAST_CONTEXT_LIMIT = cfg.get("n_ctx", 8192)


def _stream_final(llm, messages, llama_opts, fallback_text):
    """Stream the already-generated text. No re-generation needed."""
    text = fallback_text.strip()
    if not text:
        yield {"chunk": "", "done": True, "full_response": ""}
        return
    words = text.split(" ")
    for i, w in enumerate(words):
        chunk = w + (" " if i < len(words) - 1 else "")
        yield {"chunk": chunk, "done": False}
    yield {"chunk": "", "done": True, "full_response": text}


def _clear_kv_cache(storage):
    """尽力清理 KV cache，防止跨调用状态残留（提示词生成系统连续调用时必需）。

    不同 chat_handler 的清理路径不同，全部 try 一遍，失败静默。
    """
    llm = getattr(storage, "llm", None)
    if llm is None:
        return
    # 1. 常见路径：n_tokens 归零 + memory_clear
    try:
        llm.n_tokens = 0
    except Exception:
        pass
    try:
        ctx = getattr(llm, "_ctx", None)
        if ctx is not None and hasattr(ctx, "memory_clear"):
            ctx.memory_clear(True)
    except Exception:
        pass
    # 2. 混合架构缓存
    try:
        if getattr(llm, "is_hybrid", False) and getattr(llm, "_hybrid_cache_mgr", None) is not None:
            llm._hybrid_cache_mgr.clear()
    except Exception:
        pass
    # 3. reset() 兜底（llama-cpp-python 的 Llama.reset() 会清 KV cache）
    try:
        reset = getattr(llm, "reset", None)
        if callable(reset):
            reset()
    except Exception:
        pass


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
    if _BACKEND_MODE == "remote":
        # 远程后端：handler 无关（小主机 llama.cpp 按启动参数决定是否带 mmproj）
        return remote_vision(prompt, images, system_prompt=system_prompt, options=options)

    if chat_handler == "None":
        raise ValueError("Vision requires a chat_handler. Select one in the sidebar.")

    opts = dict(options or {})
    n_ctx_val = int(opts.pop("n_ctx", _DEFAULT_N_CTX))
    vram_limit = int(opts.pop("vram_limit", -1))
    image_min = int(opts.pop("image_min_tokens", _DEFAULT_IMAGE_MIN_TOKENS))
    image_max = int(opts.pop("image_max_tokens", _DEFAULT_IMAGE_MAX_TOKENS))
    n_gpu_layers = int(opts.pop("n_gpu_layers", -1))
    cache_type_k = str(opts.pop("cache_type_k", "default"))
    cache_type_v = str(opts.pop("cache_type_v", "default"))
    n_cpu_moe = int(opts.pop("n_cpu_moe", 0))
    n_seq_max = int(opts.pop("n_seq_max", 1))

    llm = _ensure_model(model, chat_handler=chat_handler, mmproj=mmproj,
                        n_ctx=n_ctx_val, vram_limit=vram_limit,
                        image_min_tokens=image_min, image_max_tokens=image_max,
                        n_gpu_layers=n_gpu_layers,
                        cache_type_k=cache_type_k,
                        cache_type_v=cache_type_v,
                        n_cpu_moe=n_cpu_moe,
                        n_seq_max=n_seq_max)
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


# ============================================================
# 提示词生成系统入口（阶段 0 程序路由 + 阶段 1 LLM 组装）
# ============================================================

def generate_prompt(model: str, intent: str, wiki_path: str, *,
                    system_prompt: str = "",
                    chat_handler: str = "None",
                    mmproj: str = "None",
                    options: Optional[dict] = None,
                    fallback: bool = True,
                    max_rounds: int = 2) -> dict:
    """用 wiki 生成绘画提示词：程序路由 → LLM 一次组装 → 降级兜底。

    与 chat() 不同，本函数不走工具循环（程序直接读候选文件），
    是"一次调用"路径。结果写入 sidebar 历史（保留生成历史）。
    返回 dict（含 prompt / mode / route 等），由路由层转 JSON。
    """
    opts = dict(options or {})
    n_ctx_val = int(opts.pop("n_ctx", _DEFAULT_N_CTX))
    vram_limit = int(opts.pop("vram_limit", -1))
    image_min = int(opts.pop("image_min_tokens", _DEFAULT_IMAGE_MIN_TOKENS))
    image_max = int(opts.pop("image_max_tokens", _DEFAULT_IMAGE_MAX_TOKENS))
    n_gpu_layers = int(opts.pop("n_gpu_layers", -1))
    cache_type_k = str(opts.pop("cache_type_k", "default"))
    cache_type_v = str(opts.pop("cache_type_v", "default"))
    n_cpu_moe = int(opts.pop("n_cpu_moe", 0))
    n_seq_max = int(opts.pop("n_seq_max", 1))

    if _BACKEND_MODE == "remote":
        def _chat_fn(messages):
            return _remote_complete_text(messages, opts)
    else:
        llm = _ensure_model(model, chat_handler=chat_handler, mmproj=mmproj,
                            n_ctx=n_ctx_val, vram_limit=vram_limit,
                            image_min_tokens=image_min, image_max_tokens=image_max,
                            n_gpu_layers=n_gpu_layers,
                            cache_type_k=cache_type_k,
                            cache_type_v=cache_type_v,
                            n_cpu_moe=n_cpu_moe,
                            n_seq_max=n_seq_max)
        llama_opts = _build_options(opts)

        def _chat_fn(messages):
            resp = llm.create_chat_completion(messages=messages, **llama_opts)
            return resp["choices"][0]["message"].get("content") or ""

    result = prompt_generator.generate_prompt(
        intent, wiki_path, _chat_fn,
        # system_prompt 是 Chat 聊天用的提示词，绝不能覆盖组装输出协议
        # 组装协议永远用 DEFAULT_OUTPUT_PROTOCOL（含"禁止思考/只输出正文"硬规则）
        output_protocol=prompt_generator.DEFAULT_OUTPUT_PROTOCOL,
        max_rounds=max_rounds,
        fallback=fallback,
        max_ctx_tokens=(remote_info().get("n_ctx") if _BACKEND_MODE == "remote" else n_ctx_val),
    )

    # 保留生成历史：意图 + 结果写入 sidebar 对话（工具中间产物不进历史）
    if result.get("ok"):
        storage = _get_storage()
        history = storage.messages.get(_SIDEBAR_UID, [])
        history.append({"role": "user", "content": f"[生成提示词] {intent}"})
        history.append({"role": "assistant", "content": result.get("prompt", "")})
        storage.messages[_SIDEBAR_UID] = history
        _trim_history(history)
        _update_context_usage(history, storage)

    # 生成任务无状态：清 KV cache，防止连续生成时状态残留污染下一次
    try:
        storage = _get_storage()
        _clear_kv_cache(storage)
    except Exception:
        pass

    return result


# ============================================================
# File reading tool (file_read)
# ============================================================

import json as _json

_PROGRAMMING_EXTENSIONS = [".py", ".js", ".java", ".c", ".cpp", ".html", ".css", ".sql", ".r", ".swift"]


def _read_one(path):
    text = ""
    lp = path.lower()
    if lp.endswith(".docx"):
        try:
            import docx2txt
            text = docx2txt.process(path) or ""
        except ImportError:
            text = "[Error: docx2txt not installed]"
    elif lp.endswith(".md") or lp.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    elif lp.endswith(".pdf"):
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text += t + "\n"
        except ImportError:
            text = "[Error: pdfplumber not installed]"
    elif lp.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            text = _json.dumps(_json.load(f), ensure_ascii=False, indent=2)
    elif lp.endswith(".xlsx"):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(path, data_only=True)
            for sn in wb.sheetnames:
                ws = wb[sn]
                text += f"## {sn}\n"
                for row in ws.iter_rows(values_only=True):
                    text += " | ".join([str(c) if c is not None else "" for c in row]) + "\n"
        except ImportError:
            text = "[Error: openpyxl not installed]"
    elif lp.endswith(".csv"):
        try:
            import pandas as pd
            df = pd.read_csv(path)
            text = df.to_string()
        except ImportError:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
    elif any(lp.endswith(ext) for ext in _PROGRAMMING_EXTENSIONS):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as f:
                text = f.read()
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, IsADirectoryError):
            text = f"[Cannot read: {path}]"
    return text


def _get_file_tree(folder_path):
    tree = {}
    try:
        for item in sorted(os.listdir(folder_path)):
            full = os.path.join(folder_path, item)
            if os.path.isdir(full):
                tree[item + "/"] = _get_file_tree(full)
            else:
                tree[item] = None
    except PermissionError:
        tree["(permission denied)"] = None
    return tree


def tool_file_read(file_path):
    if not _WIKI_PATH:
        return "Error: Wiki path not configured. Set it in the Setup tab."
    wiki = os.path.abspath(_WIKI_PATH)
    target = os.path.abspath(file_path) if os.path.isabs(file_path) else os.path.abspath(os.path.join(wiki, file_path))
    wiki_sep = wiki.rstrip(os.sep) + os.sep
    if not (target.startswith(wiki_sep) or target == wiki):
        tree = _get_file_tree(wiki)
        return f"Error: Access denied. Path outside wiki.\nWiki: {wiki}\nContents:\n{_json.dumps(tree, ensure_ascii=False, indent=2)}"
    if os.path.isfile(target):
        content = _read_one(target)
        rel = os.path.relpath(target, wiki)
        return f"File: {rel}\n\n{content}"
    elif os.path.isdir(target):
        tree = _get_file_tree(target)
        rel = os.path.relpath(target, wiki) if target != wiki else "."
        return f"Folder: {rel}\n\n{_json.dumps(tree, ensure_ascii=False, indent=2)}"
    else:
        tree = _get_file_tree(wiki)
        return f"Path not found: '{file_path}'.\nWiki contents:\n{_json.dumps(tree, ensure_ascii=False, indent=2)}"

def get_status() -> dict:
    backend = get_backend()
    base = {
        "text_models": scan_text_models(),
        "vision_models": scan_vision_models(),
        "mmproj_files": scan_mmproj_files(),
        "chat_handlers": get_chat_handlers(),
        "models_dir": _get_models_dir(),
    }

    if backend["mode"] == "remote":
        info = remote_info()
        base["backend"] = backend
        base["remote_info"] = info
        base["context_tokens"] = _LAST_CONTEXT_TOKENS
        base["context_limit"] = info.get("n_ctx") or _LAST_CONTEXT_LIMIT
        if info.get("ready"):
            base["loaded"] = True
            base["loaded_model"] = info.get("model")
            base["loaded_handler"] = "remote"
            base["current_config"] = {
                "model": info.get("model"),
                "chat_handler": "remote",
                "mmproj": "remote",
                "n_ctx": info.get("n_ctx") or 0,
                "n_gpu_layers": -1,
                "vram_limit": -1,
                "cache_type_k": "default",
                "cache_type_v": "default",
                "n_cpu_moe": 0,
                "n_seq_max": 1,
                "image_min_tokens": 0,
                "image_max_tokens": 0,
            }
        else:
            base["loaded"] = False
            base["loaded_model"] = None
            base["loaded_handler"] = None
            base["current_config"] = None
        return base

    try:
        storage = _get_storage()
    except RuntimeError as e:
        base["backend"] = backend
        base["loaded"] = False
        base["error"] = str(e)
        base["context_tokens"] = _LAST_CONTEXT_TOKENS
        base["context_limit"] = _LAST_CONTEXT_LIMIT
        return base

    cfg = storage.current_config
    base["backend"] = backend
    base["loaded"] = storage.llm is not None
    base["context_tokens"] = _LAST_CONTEXT_TOKENS
    base["context_limit"] = _LAST_CONTEXT_LIMIT
    base["loaded_model"] = cfg.get("model") if cfg else None
    base["loaded_handler"] = cfg.get("chat_handler") if cfg else None
    base["current_config"] = dict(cfg) if cfg else None
    return base

# ---- Register built-in tools ----
register_tool(
    "file_read",
    tool_file_read,
    "Read a file or list a folder within the configured wiki directory. "
    "Use this to look up information you do not know. "
    "Pass a relative path like 'INDEX.md' or an absolute path within the wiki.",
    {
        "file_path": {
            "type": "string",
            "description": "Path to file/folder. Relative paths resolve against the wiki root."
        }
    }
)

