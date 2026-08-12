# -*- coding: utf-8 -*-
"""
prompt_generator.py — 提示词生成系统：阶段 1 LLM 组装 + 降级档
==============================================================
阶段 0（prompt_router.route）→ 阶段 1（本模块）：
  1. 读候选文件内容 + 常驻规则文件（程序直接读，不走 LLM 工具循环）
  2. 构建 system prompt（候选内容 + 冲突规则 + 六轴 + 输出协议）
  3. 一次 LLM 调用完成选词+组装+冲突自检
  4. 失败 → 降级档（fallback_assemble 程序直出）
"""
from __future__ import annotations

import logging
import os
import re
from typing import Callable, Optional

try:
    from . import prompt_router
except ImportError:  # pragma: no cover - 直接文件加载测试
    import prompt_router

_log = logging.getLogger("LlmSidebar.prompt_gen")

# 输出协议（可被 wiki 的 output-protocol 覆盖，这里给默认）
DEFAULT_OUTPUT_PROTOCOL = """\
1. 只输出最终提示词正文，不输出标题、解释、分析、Markdown 代码块。
2. 语言跟随输入：中文输入输出中文，英文输入输出英文。
3. 输出为一段自然完整的提示词正文，不要分条，不要换行。
4. 保留主体、服装、动作、场景、构图、镜头、光影、画质等关键锚点。
5. 禁止输出思考过程（thinking/chain-of-thought/分析步骤）。不要以
   "Here's a thinking process"、"好的，我来"、"首先" 等开头，直接给结果。
6. 禁止输出自检过程（如"自检：""检查："等），组装完成后直接结束。"""


def _read_candidates(wiki_path: str, route_result: dict,
                     max_chars_per_file: int = 3000,
                     total_budget_chars: int = 20000) -> dict[str, str]:
    """读取候选文件内容（含常驻规则文件）。返回 {rel_path: content}。

    total_budget_chars: 总注入预算（字符）。常驻规则文件优先分配，
    剩余预算按候选文件数均分——防止多文件注入撑爆 context。
    """
    contents: dict[str, str] = {}
    always = list(route_result.get("always_include", []))
    candidates = route_result.get("candidates", [])

    # 硬约束：index.md 必须读（LLM 全局地图）。不依赖 conflicts.json 配置——
    # 即使 always_include 没配 index，也强制注入。大小写容错（INDEX.md/index.md）。
    # 2026-08-07 入口统一约定：index.md = LLM 入口，README.md = 人类入口。
    if not any(rel.lower() == "index.md" for rel in always):
        always.insert(0, "index.md")

    # 常驻规则文件优先（每个最多 2500，共最多 len(always)*2500）
    for rel in always:
        contents[rel] = prompt_router.read_wiki_file(wiki_path, rel, 2500)

    # 剩余预算给候选文件均分
    used = sum(len(v) for v in contents.values())
    remaining = total_budget_chars - used
    n_cand = len(candidates)
    if n_cand > 0:
        per_file = max(500, remaining // n_cand)
        per_file = min(per_file, max_chars_per_file)
        for c in candidates:
            rel = c.get("path", "")
            if rel not in contents:
                contents[rel] = prompt_router.read_wiki_file(wiki_path, rel, per_file)
    return contents


def _extract_need_files(text: str) -> list[str]:
    """解析 LLM 输出中的 NEED: 补充请求。"""
    m = re.search(r"(?im)^NEED:\s*(.+?)\s*$", text)
    if not m:
        return []
    return [p.strip() for p in m.group(1).split(",") if p.strip()]


def _clean_output(text: str) -> str:
    """去掉 LLM 可能输出的包装（代码块、NEED 行、thinking 段、多余空行）。"""
    t = re.sub(r"```(?:json|markdown|text)?", "", text)
    t = re.sub(r"(?im)^NEED:.*$", "", t)
    # 剥掉思考段：从 thinking 标记开头，一直吞到出现"最终提示词/正文开始"特征
    # 匹配模式：thinking 标题 → 任意内容（非贪婪）→ 正文起始（中文描述词/英文主题词）
    m = re.search(
        r"(?is)(?:here's\s+a\s+thinking\s+process|思考过程|分析步骤|好的，我来|首先，让我).*?"
        r"(?=(?:最终提示词[:：]?|一位|一名|一个|身穿|画面|构图|镜头|subject|a\s+\w+\s+(?:girl|woman|man))|\Z)",
        t,
    )
    if m and m.end() < len(t):
        t = t[m.end():]
    return t.strip()


def generate_prompt(intent: str,
                    wiki_path: str,
                    chat_fn: Callable,
                    output_protocol: str = DEFAULT_OUTPUT_PROTOCOL,
                    max_rounds: int = 2,
                    fallback: bool = True,
                    max_ctx_tokens: Optional[int] = None) -> dict:
    """阶段 0 + 阶段 1 完整流程。

    chat_fn(messages) → str：注入的 LLM 调用函数（由 llm_provider 提供）。
    max_ctx_tokens: 模型 context 上限（token）。用于动态计算注入预算，
        默认 None = 用 20000 字符固定预算。n_ctx 越小预算越紧。
    返回：
    {
      "ok": true,
      "route": {...},          # 阶段 0 结果
      "prompt": "...",         # 最终提示词
      "mode": "llm" | "fallback",
      "need_files": [...],     # LLM 请求补充但未提供的文件（诊断用）
    }
    """
    # 阶段 0：路由
    route_result = prompt_router.route(intent, wiki_path)
    if not route_result.get("ok"):
        return {"ok": False, "error": route_result.get("error", "路由失败"), "route": route_result}

    # 注入预算：留 60% 给输入，40% 给输出/协议/历史
    if max_ctx_tokens and max_ctx_tokens > 0:
        budget = int(max_ctx_tokens * 0.6 * 3)  # chars ≈ tokens * 3
    else:
        budget = 20000

    # 阶段 1：LLM 组装（最多 max_rounds 次，通常 1 次）
    wiki = os.path.abspath(wiki_path)
    round_count = 0
    need_files = []
    last_error = None
    while round_count < max_rounds:
        round_count += 1
        contents = _read_candidates(wiki, route_result,
                                    total_budget_chars=budget)
        system_prompt = prompt_router.build_assemble_prompt(
            route_result, wiki, contents, output_protocol)
        try:
            reply = chat_fn([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"用户意图：{intent}\n请组装提示词。"},
            ])
        except Exception as e:
            _log.warning("LLM 组装失败 (round %s): %s", round_count, e)
            last_error = f"{type(e).__name__}: {e}"
            break

        need = _extract_need_files(reply or "")
        if need and round_count < max_rounds:
            # 补充请求：把 NEED 文件加入候选，再试一轮
            need_files = need
            existing = {c.get("path") for c in route_result.get("candidates", [])}
            for rel in need:
                if rel not in existing:
                    route_result.setdefault("candidates", []).append(
                        {"path": rel, "score": 0, "summary": ""})
            continue

        prompt = _clean_output(reply or "")
        if prompt:
            return {
                "ok": True,
                "route": route_result,
                "prompt": prompt,
                "mode": "llm",
                "rounds": round_count,
                "need_files": need_files,
            }
        break

    # 降级档
    if fallback:
        fallback_text = prompt_router.fallback_assemble(route_result, wiki)
        if fallback_text:
            return {
                "ok": True,
                "route": route_result,
                "prompt": fallback_text,
                "mode": "fallback",
                "rounds": round_count,
                "need_files": need_files,
                "llm_error": last_error,
            }

    return {
        "ok": False,
        "error": "LLM 组装失败且降级档无输出",
        "llm_error": last_error,
        "route": route_result,
        "mode": "none",
        "need_files": need_files,
    }
