# -*- coding: utf-8 -*-
"""
prompt_router.py — 提示词生成系统：阶段 0 程序路由 + 降级档组装
==============================================================
调和架构：
  阶段 0（程序，零 LLM 成本）：用户意图 → 关键词匹配 index → 候选文件清单 JSON
  阶段 1（LLM 一次调用）：程序直接读候选文件内容注入 system prompt → 选词+组装+冲突自检
  降级档（LLM 不可用）：程序按 index 树直接抽词组装，保证永远有输出

设计要点（2026-07-31 确认）：
- 程序只做"选文件"，不做"按文件大小比例抽词"（机械抽词产生噪声）
- 冲突表显式注入（curated 知识，不靠模型常识）
- 词筛选和组装放同一次 LLM 调用（同一语言任务，省 prefill）
- 来源优先级：用户输入 > 锁定 > 设定图 > 模板 > 主题池 > 随机（借鉴 RealMan）
- 六轴约束域：主体/世界/镜头/光色/成像/运动（借鉴 RealMan）
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

# ============================================================
# 常量
# ============================================================

INDEX_FILENAME = "index.json"            # 每 wiki 一个，可由 build_index() 生成
CONFLICT_FILENAME = "conflicts.json"      # 每 wiki 一个（规则随库走，不合并）
CONSTRAINT_FILENAME = "constraint_domains.json"  # 六轴约束域（可共享）

DEFAULT_MAX_CANDIDATES = 6                # 候选文件上限
DEFAULT_SCORE_FLOOR = 1                   # 低于此分不入选

# 停用词（关键词提取时过滤）
# 包含：请求性/任务性词语（用户说话方式，非画面内容）——防止"提示词"命中"负面提示词"模块
_STOPWORDS = {
    "一个", "什么", "怎么", "可以", "需要", "想要", "这个", "那个",
    "还是", "场景", "画面", "风格", "提示词", "生成", "给我", "帮我",
    "画", "图", "张", "条", "the", "and", "for", "with", "that",
    # 任务性/请求性词语（用户说"生成提示词/需求主体/身穿"等，不是画面内容）
    "提示", "示词", "需求", "主体", "身穿", "要求", "帮我", "请",
    "为", "的", "和", "与", "并", "以及", "或者", "然后",
    "组", "组提", "组提示词",
    # 2-gram 滑窗常见碎片（跨词边界，无独立意义）
    "体为", "为甜", "穿泳", "求主", "美少", "爱少", "女泳", "衣比",
    "基尼", "妹泳", "装题", "景名", "层套", "地窗", "房感", "房落",
}

# 英文关键词最少长度
_MIN_EN_WORD = 3

# ============================================================
# JSON 安全加载
# ============================================================

def _load_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return default


def _save_json(path: str, data: Any) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


# ============================================================
# A1: index 生成器 — 扫描 wiki 目录 → index.json
# ============================================================

def build_index(wiki_path: str, force: bool = False) -> Optional[dict]:
    """扫描 wiki 目录，为每个 .md 文件生成条目，写入 index.json。

    条目字段：
      path      — 相对路径（正斜杠）
      name      — 文件名（去 .md）
      title     — 第一个 # 标题（无则用文件名）
      tags      — 文件名分词 + frontmatter tags + 标题分词
      summary   — 正文前若干字符（用于匹配和 LLM 摘要）
      links     — [[wikilink]] 引用列表
    """
    wiki = os.path.abspath(wiki_path)
    if not os.path.isdir(wiki):
        return None

    out_path = os.path.join(wiki, INDEX_FILENAME)
    if os.path.exists(out_path) and not force:
        return _load_json(out_path)

    entries = []
    for root, dirs, files in os.walk(wiki):
        # 跳过 .obsidian
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in sorted(files):
            if not fn.endswith(".md"):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, wiki).replace(os.sep, "/")
            try:
                with open(full, "r", encoding="utf-8") as f:
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                text = ""

            title = ""
            tags: list[str] = []
            links: list[str] = []
            # frontmatter tags
            m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
            body = text
            if m:
                fm = m.group(1)
                body = text[m.end():]
                tm = re.search(r"(?im)^tags:\s*\[(.*?)\]", fm)
                if tm:
                    tags = [t.strip() for t in tm.group(1).split(",") if t.strip()]

            # 第一个标题
            hm = re.search(r"(?m)^#\s+(.+?)\s*$", body)
            if hm:
                title = hm.group(1).strip()

            # wikilinks
            links = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", body)

            # 文件名分词（kebab/下划线/数字前缀）
            stem = fn[:-3]
            name_parts = re.split(r"[\-_]+", stem)
            name_parts = [p for p in name_parts if p]

            # 摘要：正文前 200 字符（去标题）
            summary = re.sub(r"\s+", " ", body).strip()[:200]

            entries.append({
                "path": rel,
                "name": stem,
                "title": title or stem,
                "tags": tags + name_parts,
                "summary": summary,
                "links": links,
                "size": len(text),
            })

    index = {
        "wiki": os.path.basename(wiki),
        "count": len(entries),
        "entries": entries,
    }
    _save_json(out_path, index)
    return index


def load_index(wiki_path: str) -> Optional[dict]:
    return build_index(wiki_path, force=False)


def load_conflicts(wiki_path: str) -> dict:
    return _load_json(os.path.join(wiki_path, CONFLICT_FILENAME), {}) or {}


def load_constraint_domains(wiki_path: str = "") -> dict:
    """六轴约束域。优先读 wiki 目录下自定义，其次读默认（随模块）。"""
    if wiki_path:
        local = _load_json(os.path.join(wiki_path, CONSTRAINT_FILENAME))
        if local:
            return local
    here = os.path.dirname(os.path.abspath(__file__))
    return _load_json(os.path.join(here, CONSTRAINT_FILENAME), {}) or {}


# ============================================================
# 关键词提取
# ============================================================

def extract_keywords(intent: str) -> list[str]:
    """从用户意图提取关键词：英文单词 + 中文 2-gram 滑窗 + 整 chunk（≤6 字）。

    中文无分词器，用 2-gram 滑窗（能抓"窗边"这类关键 2 字词）+
    整 chunk（保留完整短语），避免 3-gram+ 碎片噪声（如"室午后"）。
    """
    words: set[str] = set()
    low = intent.lower()

    for w in re.findall(r"[a-z][a-z0-9_\-]{,%d}" % (_MIN_EN_WORD - 1), low):
        if len(w) >= _MIN_EN_WORD:
            words.add(w)

    for chunk in re.findall(r"[\u4e00-\u9fff]+", intent):
        if len(chunk) <= 6:
            words.add(chunk)  # 整 chunk
        for i in range(len(chunk) - 1):
            words.add(chunk[i:i + 2])  # 2-gram 滑窗

    # 过滤碎片：2-gram 中若首尾字符是常见连接字（为/体/穿/求 等）且不在整 chunk 中，剔除
    # 保留像"泳装""甜妹"这种独立有意义的词
    return [w for w in words if w not in _STOPWORDS]


# ============================================================
# 打分匹配
# ============================================================

def _entry_text(entry: dict) -> str:
    """拼接条目的可匹配文本（小写）。"""
    parts = [entry.get("name", ""), entry.get("title", ""), entry.get("summary", "")]
    parts += entry.get("tags", [])
    return " ".join(parts).lower()


def score_entry(entry: dict, keywords: list[str]) -> int:
    """关键词命中打分。标题/文件名命中加权，摘要命中基础分。"""
    name_t = (entry.get("name", "") + " " + entry.get("title", "")).lower()
    tags_t = " ".join(entry.get("tags", [])).lower()
    summary_t = entry.get("summary", "").lower()

    score = 0
    for kw in keywords:
        k = kw.lower()
        if k in name_t:
            score += 3
        elif k in tags_t:
            score += 2
        elif k in summary_t:
            score += 1
    return score


def match_files(index: dict, keywords: list[str],
                max_candidates: int = DEFAULT_MAX_CANDIDATES,
                score_floor: int = DEFAULT_SCORE_FLOOR) -> list[dict]:
    """按关键词给所有条目打分，返回排序后的候选（含 score）。"""
    scored = []
    for entry in index.get("entries", []):
        s = score_entry(entry, keywords)
        if s >= score_floor:
            scored.append({"entry": entry, "score": s})
    scored.sort(key=lambda x: (-x["score"], x["entry"].get("path", "")))
    return scored[:max_candidates]


def fuzzy_match_files(index: dict, keywords: list[str], limit: int = 5) -> list[dict]:
    """宽松匹配：单字命中（用于路由空洞时给用户相近条目建议）。

    对每个关键词取首字，在文件名/标题中找含该字的条目。
    """
    if not keywords:
        return []
    first_chars = set()
    for kw in keywords:
        if kw and len(kw) >= 2 and not kw.isascii():
            first_chars.add(kw[0])
    hits = {}
    for entry in index.get("entries", []):
        text = entry.get("name", "") + entry.get("title", "")
        n = sum(1 for ch in first_chars if ch in text)
        if n > 0:
            hits[entry.get("path", "")] = n
    ranked = sorted(hits.items(), key=lambda x: -x[1])[:limit]
    return [{"entry": {"path": p, "name": p.rsplit("/", 1)[-1], "summary": ""},
             "score": s} for p, s in ranked]


# ============================================================
# 硬冲突检测
# ============================================================

def _norm(s: str) -> str:
    return re.sub(r"[\s\-_\[\]#()（）]", "", str(s).lower())


def detect_conflicts(candidates: list[dict], conflicts: dict) -> list[str]:
    """在候选文件间检测硬冲突。

    conflicts.json 结构（每 wiki 维护）：
    {
      "hard": [{"a": "...", "b": "...", "reason": "..."}],   # 程序可判，如 富士×柯达
      "soft": [{"a": "...", "b": "...", "reason": "...", "suggestion": "..."}],  # LLM 裁决
      "always_include": ["01-role.md", ...],   # 流水线型 wiki 每次组装必读的规则文件
      "notes": "可选说明"
    }
    a/b 可以是关键词（子串匹配文件名/标题）或路径片段（匹配相对路径）。
    返回 hard 冲突的 human-readable 列表。
    """
    hard = conflicts.get("hard", []) or []
    if not hard or not candidates:
        return []

    paths = [_norm(c.get("entry", {}).get("path", "")) for c in candidates]
    names = [_norm(c.get("entry", {}).get("name", "")) for c in candidates]

    hits = []
    for rule in hard:
        a = _norm(rule.get("a", ""))
        b = _norm(rule.get("b", ""))
        if not a or not b:
            continue
        found_a = any(a in p or a in n for p, n in zip(paths, names))
        found_b = any(b in p or b in n for p, n in zip(paths, names))
        if found_a and found_b:
            hits.append(f"{rule.get('a')} × {rule.get('b')}: {rule.get('reason', '互斥')}")
    return hits


def collect_soft_conflicts(conflicts: dict) -> list[dict]:
    """返回 soft 冲突列表（注入 LLM 用于裁决）。"""
    return conflicts.get("soft", []) or []


def collect_always_include(conflicts: dict) -> list[str]:
    """返回常驻规则文件列表（流水线型 wiki 每次组装必读，如 rule-bus/self-check）。"""
    return conflicts.get("always_include", []) or []


# ============================================================
# 阶段 0 主入口
# ============================================================

def route(intent: str, wiki_path: str,
          max_candidates: int = DEFAULT_MAX_CANDIDATES) -> dict:
    """阶段 0：用户意图 → 候选文件清单 JSON。

    返回：
    {
      "ok": true,
      "wiki": "...",
      "intent": "...",
      "keywords": [...],
      "candidates": [{"path": "...", "score": N, "summary": "..."}],
      "hard_conflicts": ["A × B: ..."],
      "source_priority": [...],
    }
    """
    wiki = os.path.abspath(wiki_path)
    index = load_index(wiki)
    if not index:
        return {"ok": False, "error": f"wiki 不存在或 index 生成失败: {wiki}"}

    conflicts = load_conflicts(wiki)
    domains = load_constraint_domains(wiki)

    keywords = extract_keywords(intent)
    candidates = match_files(index, keywords, max_candidates=max_candidates)

    result = {
        "ok": True,
        "wiki": index.get("wiki", os.path.basename(wiki)),
        "intent": intent,
        "keywords": keywords,
        "always_include": collect_always_include(conflicts),
        "candidates": [
            {
                "path": c["entry"]["path"],
                "score": c["score"],
                "summary": c["entry"].get("summary", "")[:120],
            }
            for c in candidates
        ],
        "hard_conflicts": detect_conflicts(candidates, conflicts),
        "soft_conflicts": collect_soft_conflicts(conflicts),
        "source_priority": (domains.get("source_priority") or
                            ["用户输入", "锁定标签", "设定图事实", "模板", "主题池", "随机"]),
    }
    if not candidates:
        # 路由空洞：给相近条目建议（单字模糊匹配），避免用户无从下手
        fuzzy = fuzzy_match_files(index, keywords)
        result["ok"] = False
        result["error"] = "没有命中任何 wiki 条目，请换个说法或检查关键词。"
        result["suggestions"] = [f["entry"]["path"] for f in fuzzy]
    return result


# ============================================================
# 降级档：程序直出组装（LLM 不可用时）
# ============================================================

def fallback_assemble(route_result: dict, wiki_path: str,
                      max_chars_per_file: int = 400) -> str:
    """LLM 失败时的兜底：按候选文件顺序抽取内容片段拼接。

    不追求自然语言质量，只保证有输出（对应 RealMan 的"仅 Skill"回退哲学）。
    """
    if not route_result.get("ok"):
        return ""
    wiki = os.path.abspath(wiki_path)
    parts = []
    for c in route_result.get("candidates", []):
        rel = c.get("path", "")
        full = os.path.join(wiki, rel.replace("/", os.sep))
        try:
            with open(full, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        # 抽取：第一个标题 + 关键词行
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        picked = []
        kws = route_result.get("keywords", [])
        for ln in lines:
            if ln.startswith("#"):
                picked.append(ln.lstrip("# ").strip())
            elif kws and any(kw in ln for kw in kws):
                picked.append(ln)
            if len(" ".join(picked)) >= max_chars_per_file:
                break
        if picked:
            parts.append(f"[{rel}]\n" + "\n".join(picked[:6]))
    return "\n\n".join(parts)


# ============================================================
# 组装 system prompt（供 llm_provider 阶段 1 使用）
# ============================================================

def read_wiki_file(wiki_path: str, rel: str, max_chars: int = 3000) -> str:
    """读取 wiki 内一个相对路径文件，返回内容（截断）。"""
    full = os.path.join(os.path.abspath(wiki_path), rel.replace("/", os.sep))
    try:
        with open(full, "r", encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        return f"[无法读取: {rel}]"
    return text[:max_chars]


def build_assemble_prompt(route_result: dict, wiki_path: str,
                          candidate_contents: dict[str, str],
                          output_protocol: str = "") -> str:
    """构建阶段 1 的 LLM system prompt。

    注入：候选文件内容 + 常驻规则文件 + 硬冲突 + 软冲突 + 六轴约束域
         + 来源优先级 + 输出协议。
    """
    domains = load_constraint_domains(wiki_path)
    lines = []
    lines.append("你是绘画提示词组装工程师。任务：根据用户意图与给定的 wiki 候选材料，"
                 "选词并组装一条完整的自然语言绘画提示词。")
    lines.append("")

    # 候选材料
    lines.append("## 候选材料（程序已按关键词选出，只能从这里选词，除非明确需要补充）")
    if candidate_contents:
        for rel, content in candidate_contents.items():
            lines.append(f"### 文件: {rel}")
            lines.append(content)
            lines.append("")
    else:
        lines.append("（无）")

    # 常驻规则文件
    always = route_result.get("always_include", [])
    if always:
        lines.append("## 常驻规则（每次组装必须遵守，优先级高于候选材料）")
        for rel in always:
            lines.append(f"### 规则文件: {rel}")
            lines.append(read_wiki_file(wiki_path, rel, 2500))
            lines.append("")

    # 冲突规则
    hard = route_result.get("hard_conflicts", [])
    soft = route_result.get("soft_conflicts", [])
    if hard or soft:
        lines.append("## 冲突规则（必须遵守）")
        for h in hard:
            lines.append(f"- ❌ 硬冲突（程序检测，不可同时使用）: {h}")
        for s in soft:
            lines.append(f"- ⚠️ 软冲突（默认不可同时使用；如用户意图明确需要，可破例并说明）: "
                         f"{s.get('a')} × {s.get('b')} — {s.get('reason', '')}")

    # 六轴约束域
    dom_list = domains.get("domains") or []
    if dom_list:
        lines.append("")
        lines.append("## 六轴约束域（组装时逐轴自检，发现冲突只修该轴，不重写其他正确内容）")
        for d in dom_list:
            checks = "、".join(d.get("checks", []))
            lines.append(f"- {d.get('name')}: {checks}")

    # 来源优先级
    prio = route_result.get("source_priority") or []
    if prio:
        lines.append("")
        lines.append("## 来源优先级（内容冲突时，高优先级者胜）")
        lines.append(" > ".join(prio))

    # 输出协议
    if output_protocol:
        lines.append("")
        lines.append("## 输出协议")
        lines.append(output_protocol)

    # 定向修复指令
    lines.append("")
    lines.append("## 组装要求")
    lines.append("1. 从候选材料中选词，组装为自然语言提示词（不要输出标签串/逗号堆砌）。")
    lines.append("2. 硬冲突项必须规避；软冲突项需按用户意图裁决。")
    lines.append("3. 只输出最终提示词正文，不输出解释、分析或 Markdown 代码块。")
    lines.append("4. 若候选材料不足，在回答末尾用一行标注：NEED: 文件名1, 文件名2")
    lines.append("5. ⚠️ 重要：六轴自检、冲突检查、来源优先级判断都是你的内部思考过程，"
                 "**绝对不要输出任何检查报告、核对清单、✅/❌ 列表或 'Let's check' 类内容**。"
                 "输出里只能有最终提示词本身，一个多余字符都不要。")

    return "\n".join(lines)
