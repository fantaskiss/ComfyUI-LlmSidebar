"""
ComfyUI-LlmSidebar - HTTP routes.
Registered with ComfyUI's PromptServer at setup time.
Depends on ComfyUI-llama-cpp_vlm's LLAMA_CPP_STORAGE for model loading.
"""
import json
import os
import base64
import logging
from aiohttp import web

_log = logging.getLogger("LlmSidebar.routes")

_llm = None


def register_routes(prompt_server, llm_module):
    global _llm
    _llm = llm_module

    async def _sse_stream(request, generator):
        resp = web.StreamResponse(
            status=200, reason="OK",
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)
        try:
            for chunk in generator:
                line = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await resp.write(line.encode("utf-8"))
        except Exception as e:
            err = json.dumps({"error": str(e), "done": True})
            await resp.write(f"data: {err}\n\n".encode("utf-8"))
        return resp

    # ---- GET /llm-sidebar/status ----
    @prompt_server.routes.get("/llm-sidebar/status")
    async def status_handler(request):
        try:
            return web.json_response({"success": True, "data": _llm.get_status()})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # ---- GET /llm-sidebar/models ----
    @prompt_server.routes.get("/llm-sidebar/models")
    async def models_handler(request):
        try:
            return web.json_response({
                "success": True,
                "data": {
                    "text": _llm.scan_text_models(),
                    "vision": _llm.scan_vision_models(),
                    "mmproj": _llm.scan_mmproj_files(),
                    "chat_handlers": _llm.get_chat_handlers(),
                },
            })
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # ---- POST /llm-sidebar/chat/stream ----
    @prompt_server.routes.post("/llm-sidebar/chat/stream")
    async def chat_stream_handler(request):
        try:
            data = await request.json()
            model = data.get("model", "")
            prompt = data.get("prompt", "")
            if not model or not prompt:
                return web.json_response(
                    {"success": False, "error": "model and prompt required"}, status=400)

            system_prompt = data.get("system_prompt", "")
            chat_handler = data.get("chat_handler", "None")
            mmproj = data.get("mmproj_file", "None")
            options = data.get("options", None)

            # Apply tools config from request
            tools_enabled = data.get("tools_enabled", False)
            wiki_path = data.get("wiki_path", "")
            max_rounds = data.get("max_tool_rounds", 10)
            result_max_chars = data.get("tool_result_max_chars", 8000)
            _llm.set_tools_config(
                enabled=tools_enabled,
                wiki_path=wiki_path,
                max_rounds=max_rounds,
                result_max_chars=result_max_chars,
            )

            generator = _llm.chat(
                model, prompt,
                system_prompt=system_prompt,
                chat_handler=chat_handler,
                mmproj=mmproj,
                options=options,
                stream=True,
            )
            return await _sse_stream(request, generator)

        except Exception as e:
            _log.exception("chat/stream error")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # ---- POST /llm-sidebar/generate-prompt ----
    @prompt_server.routes.post("/llm-sidebar/generate-prompt")
    async def generate_prompt_handler(request):
        """提示词生成系统：阶段 0 程序路由 + 阶段 1 LLM 一次组装。

        body: {
          "model": "...",
          "intent": "用户意图",
          "wiki_path": "N:\\...\\wiki目录",
          "system_prompt": "...",        # 可选，覆盖默认输出协议
          "chat_handler": "None",
          "mmproj_file": "None",
          "options": {...},               # 推理参数
          "fallback": true,               # 可选，LLM 失败时程序直出
        }
        返回 JSON（含 prompt / mode / route）。
        """
        try:
            data = await request.json()
            model = data.get("model", "")
            intent = data.get("intent", "")
            wiki_path = data.get("wiki_path", "")
            if not model or not intent or not wiki_path:
                return web.json_response(
                    {"success": False,
                     "error": "model, intent, wiki_path are required"},
                    status=400)

            system_prompt = data.get("system_prompt", "")
            chat_handler = data.get("chat_handler", "None")
            mmproj = data.get("mmproj_file", "None")
            options = data.get("options", None)
            fallback = data.get("fallback", True)

            result = _llm.generate_prompt(
                model, intent, wiki_path,
                system_prompt=system_prompt,
                chat_handler=chat_handler,
                mmproj=mmproj,
                options=options,
                fallback=bool(fallback),
            )
            return web.json_response({"success": True, "data": result})
        except Exception as e:
            _log.exception("generate-prompt error")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # ---- POST /llm-sidebar/vision ----
    @prompt_server.routes.post("/llm-sidebar/vision")
    async def vision_handler(request):
        try:
            data = await request.json()
            model = data.get("model", "")
            images = data.get("images", [])
            prompt = data.get("prompt", "Describe this image in detail.")
            chat_handler = data.get("chat_handler", "None")
            mmproj = data.get("mmproj_file", "None")

            if not model or not images:
                return web.json_response(
                    {"success": False, "error": "model and images required"}, status=400)

            if chat_handler == "None":
                return web.json_response(
                    {"success": False, "error": "Vision requires a chat_handler. Select one in the sidebar."},
                    status=400)

            # Resolve image paths
            resolved = []
            import tempfile
            for img in images:
                if isinstance(img, str) and img.startswith("data:image"):
                    header, b64 = img.split(",", 1)
                    raw = base64.b64decode(b64)
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    tmp.write(raw)
                    tmp.close()
                    resolved.append(tmp.name)
                elif isinstance(img, str) and not img.startswith("data:") and not os.path.isabs(img):
                    import folder_paths
                    full_path = folder_paths.get_annotated_filepath(img)
                    if os.path.exists(full_path):
                        resolved.append(full_path)
                    else:
                        _log.warning("Image not found: %s (resolved: %s)", img, full_path)
                elif isinstance(img, str):
                    resolved.append(img)

            system_prompt = data.get("system_prompt", "")
            options = data.get("options", None)

            result = _llm.vision(
                model, prompt, resolved,
                system_prompt=system_prompt,
                chat_handler=chat_handler,
                mmproj=mmproj,
                options=options,
            )

            # Clean up temp files
            for p in resolved:
                if p.startswith(tempfile.gettempdir()):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

            return web.json_response({"success": True, "data": {"response": result}})

        except Exception as e:
            _log.exception("vision error")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # ---- POST /llm-sidebar/unload ----
    @prompt_server.routes.post("/llm-sidebar/unload")
    async def unload_handler(request):
        try:
            _llm.unload_model()
            return web.json_response({"success": True, "data": {"loaded": False}})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # ---- POST /llm-sidebar/reset ----
    @prompt_server.routes.post("/llm-sidebar/reset")
    async def reset_handler(request):
        try:
            _llm.reset_history()
            return web.json_response({"success": True, "data": {"history_cleared": True}})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # ---- POST /llm-sidebar/apply-settings ----
    @prompt_server.routes.post("/llm-sidebar/apply-settings")
    async def apply_settings_handler(request):
        """Explicitly unload and reload with new settings from Setup tab."""
        try:
            data = await request.json()
            model = data.get("model", "")
            if not model:
                return web.json_response(
                    {"success": False, "error": "model required"}, status=400)

            _llm.apply_settings(
                model=model,
                chat_handler=data.get("chat_handler", "None"),
                mmproj=data.get("mmproj", "None"),
                n_ctx=int(data.get("n_ctx", 8192)),
                vram_limit=int(data.get("vram_limit", -1)),
                image_min_tokens=int(data.get("image_min_tokens", 1024)),
                image_max_tokens=int(data.get("image_max_tokens", 4096)),
                n_gpu_layers=int(data.get("n_gpu_layers", -1)),
                cache_type_k=str(data.get("cache_type_k", "default")),
                cache_type_v=str(data.get("cache_type_v", "default")),
                n_cpu_moe=int(data.get("n_cpu_moe", 0)),
                n_seq_max=int(data.get("n_seq_max", 1)),
                tools_enabled=data.get("tools_enabled"),
                wiki_path=data.get("wiki_path"),
                max_rounds=data.get("max_tool_rounds"),
                result_max_chars=data.get("tool_result_max_chars"),
            )
            return web.json_response({"success": True, "data": {"loaded": True}})
        except Exception as e:
            _log.exception("apply-settings error")
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # ---- GET /llm-sidebar/diag ----
    @prompt_server.routes.get("/llm-sidebar/diag")
    async def diag_handler(request):
        import sys
        result = {"sys_modules_keys": [], "storage_info": None}
        
        # Find all modules with llama/vlm in name
        for name, mod in sorted(sys.modules.items()):
            if mod is None:
                continue
            nl = name.lower()
            if "llama" in nl or "vlm" in nl:
                attrs = [a for a in dir(mod) if not a.startswith("_")][:20]
                result["sys_modules_keys"].append({
                    "key": name,
                    "file": getattr(mod, "__file__", None),
                    "has_STORAGE": hasattr(mod, "LLAMA_CPP_STORAGE"),
                    "has_handlers": hasattr(mod, "chat_handlers"),
                    "attrs_sample": attrs,
                })
        
        # Try to get STORAGE and inspect it
        try:
            from . import llm_provider
            storage = llm_provider._get_storage()
            result["storage_info"] = {
                "type": str(type(storage)),
                "type_name": type(storage).__name__,
                "module": type(storage).__module__,
                "has_llm": hasattr(storage, "llm"),
                "has_clean": hasattr(storage, "clean"),
                "has_load_model": hasattr(storage, "load_model"),
                "attrs": [a for a in dir(storage) if not a.startswith("_")],
            }
        except Exception as e:
            result["storage_info"] = {"error": str(e)}
        
        return web.json_response({"success": True, "data": result})

    # ---- GET /llm-sidebar/tools/config ----
    @prompt_server.routes.get("/llm-sidebar/tools/config")
    async def tools_config_get(request):
        try:
            return web.json_response({"success": True, "data": _llm.get_tools_config()})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=500)

    # ---- POST /llm-sidebar/tools/config ----
    @prompt_server.routes.post("/llm-sidebar/tools/config")
    async def tools_config_set(request):
        try:
            data = await request.json()
            result = _llm.set_tools_config(
                enabled=data.get("enabled"),
                wiki_path=data.get("wiki_path"),
                max_rounds=data.get("max_rounds"),
                result_max_chars=data.get("result_max_chars"),
            )
            return web.json_response({"success": True, "data": result})
        except Exception as e:
            return web.json_response({"success": False, "error": str(e)}, status=400)

    _log.info("LlmSidebar routes registered")
