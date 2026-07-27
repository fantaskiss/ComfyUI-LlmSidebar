/**
 * ComfyUI-LlmSidebar - sidebar.js
 * Self-contained floating panel with Chat + Describe tabs.
 * Reuses LLAMA_CPP_STORAGE from ComfyUI-llama-cpp_vlm for model loading.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// ---- state ----
let panel = null;
let visible = false;
let activeTab = "chat";
let selectedModel = "";
let selectedMmproj = "";  // "None" = no mmproj, "" = auto (not used in path A)
let selectedHandler = "None";
let chatHistory = [];
let describeText = "";
let systemPrompt = "";
let systemPromptEnabled = true;
// Inference params
let nCtx = 8192;
let maxTokens = 300;
let temperature = "";
let topP = "";
let topK = "";
let repeatPenalty = "";
let allModels = [];
let allMmproj = [];
let allHandlers = ["None"];

const STORAGE_KEY = "llm_sidebar";
const VISION_PROMPT =
    "Describe this image in detail. Focus on: subject, composition, lighting, " +
    "color palette, style, mood, and any notable visual elements. " +
    "Write as a prompt for AI image generation. Be concise and direct.";

// ---- persistence ----
function loadState() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {
            const s = JSON.parse(raw);
            selectedModel = s.selectedModel || "";
            selectedMmproj = s.selectedMmproj || "";
            selectedHandler = s.selectedHandler || "None";
            chatHistory = s.chatHistory || [];
            describeText = s.describeText || "";
            systemPrompt = s.systemPrompt || "";
            systemPromptEnabled = s.systemPromptEnabled !== false;
            nCtx = s.nCtx || 8192;
            maxTokens = s.maxTokens || 300;
            temperature = s.temperature || "";
            topP = s.topP || "";
            topK = s.topK || "";
            repeatPenalty = s.repeatPenalty || "";
        }
    } catch (e) {}
}

function saveState() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
            selectedModel,
            selectedMmproj,
            selectedHandler,
            chatHistory: chatHistory.slice(-100),
            describeText,
            systemPrompt,
            systemPromptEnabled,
            nCtx, maxTokens,
            temperature, topP, topK, repeatPenalty,
        }));
    } catch (e) {}
}

// ---- DOM ----
function createPanel() {
    if (panel) return;

    panel = document.createElement("div");
    panel.id = "llm-sidebar-panel";
    Object.assign(panel.style, {
        position: "fixed",
        width: "380px",
        minWidth: "300px",
        height: "420px",
        minHeight: "200px",
        maxHeight: "90vh",
        zIndex: "9998",
        background: "var(--bg-color, #1a1a2e)",
        border: "1px solid var(--border-color, #333)",
        borderRadius: "8px",
        display: "none",
        flexDirection: "column",
        fontFamily: "monospace",
        fontSize: "13px",
        color: "var(--fg-color, #ddd)",
        boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
        resize: "both",
        overflow: "hidden",
    });

    panel.innerHTML = `
    <div id="llm-tabs" style="display:flex;border-bottom:1px solid var(--border-color,#333);flex-shrink:0">
      <button id="llm-tab-chat" class="llm-tab active" style="flex:1;padding:10px;border:none;background:transparent;color:inherit;cursor:pointer;font-size:14px">💬 Chat</button>
      <button id="llm-tab-desc" class="llm-tab" style="flex:1;padding:10px;border:none;background:transparent;color:inherit;cursor:pointer;font-size:14px">📝 Describe</button>
      <button id="llm-tab-sys" class="llm-tab" style="flex:1;padding:10px;border:none;background:transparent;color:inherit;cursor:pointer;font-size:14px">⚙️ SysPrompt</button>
      <button id="llm-close" style="padding:10px 14px;border:none;background:transparent;color:inherit;cursor:pointer;font-size:16px">✕</button>
    </div>

    <!-- Chat tab -->
    <div id="llm-chat-panel" style="display:flex;flex-direction:column;flex:1;min-height:0">
      <div style="display:flex;align-items:center;padding:8px;gap:6px;border-bottom:1px solid var(--border-color,#333);flex-shrink:0;flex-wrap:wrap">
        <select id="llm-model-select" style="flex:1;min-width:120px;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:4px;font-size:12px;border-radius:4px"></select>
        <select id="llm-handler-select" style="max-width:110px;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:4px;font-size:11px;border-radius:4px" title="chat_handler (required for vision)">
          <option value="None">handler: None</option>
        </select>
        <select id="llm-mmproj-select" style="max-width:100px;background:var(--bg-color,#222);color:var(--fg-color,#888);border:1px solid var(--border-color,#444);padding:4px;font-size:11px;border-radius:4px" title="mmproj for vision (None = no vision)">
          <option value="None">mmproj: None</option>
        </select>
        <button id="llm-unload" style="padding:4px 8px;background:#633;color:#faa;border:1px solid #844;border-radius:4px;cursor:pointer;font-size:11px">Unload</button>
        <button id="llm-clear-chat" style="padding:4px 8px;background:#333;color:#ccc;border:1px solid #555;border-radius:4px;cursor:pointer;font-size:11px">Clear</button>
      </div>
      <div id="llm-messages" style="flex:1;overflow-y:auto;padding:8px;min-height:0"></div>
      <div style="display:flex;padding:8px;gap:6px;border-top:1px solid var(--border-color,#333);flex-shrink:0">
        <textarea id="llm-input" rows="2" placeholder="Ask something..." style="flex:1;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:6px;resize:none;font-size:12px;font-family:inherit"></textarea>
        <button id="llm-send" style="padding:6px 14px;background:var(--primary,#4a6);color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:13px;font-weight:bold">Send</button>
        <button id="llm-new" style="padding:6px 10px;background:#333;color:#ccc;border:1px solid #555;border-radius:4px;cursor:pointer;font-size:12px">New</button>
        <button id="llm-unload-bottom" style="padding:6px 10px;background:#633;color:#faa;border:1px solid #844;border-radius:4px;cursor:pointer;font-size:12px">Unload</button>
      </div>
    </div>

    <!-- Describe tab -->
    <div id="llm-desc-panel" style="display:none;flex-direction:column;flex:1;min-height:0">
      <textarea id="llm-desc-textarea" placeholder="Paste and compose your prompt here.&#10;Use [-> Describe] buttons in Chat to append messages." style="flex:1;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:none;padding:10px;resize:none;font-size:12px;font-family:inherit;min-height:0"></textarea>
      <div style="display:flex;padding:8px;gap:6px;border-top:1px solid var(--border-color,#333);flex-shrink:0">
        <button id="llm-desc-send" style="flex:1;padding:8px;background:var(--primary,#4a6);color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:13px">Send to Chat</button>
        <button id="llm-desc-clear" style="padding:8px 14px;background:#333;color:#ccc;border:1px solid #555;border-radius:4px;cursor:pointer">Clear</button>
      </div>
    </div>

    <!-- System Prompt tab -->
    <div id="llm-sys-panel" style="display:none;flex-direction:column;flex:1;min-height:0">
      <div style="display:flex;align-items:center;padding:6px 10px;gap:8px;border-bottom:1px solid var(--border-color,#333);flex-shrink:0">
        <label style="display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer">
          <input type="checkbox" id="llm-sys-enabled" checked style="accent-color:var(--primary,#4a6)">
          <span style="color:var(--fg-color,#ddd)">Enable</span>
        </label>
        <span style="flex:1"></span>
        <button id="llm-sys-reset" style="padding:2px 8px;background:#333;color:#ccc;border:1px solid #555;border-radius:3px;cursor:pointer;font-size:11px">Reset</button>
      </div>
      <textarea id="llm-sys-textarea" placeholder="System prompt (sent at start of each conversation)&#10;Example: You are a helpful creative assistant specialized in image prompt engineering." style="flex:1;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:none;padding:10px;resize:none;font-size:12px;font-family:inherit;min-height:0"></textarea>
      <div id="llm-sys-params" style="padding:6px 10px;border-top:1px solid var(--border-color,#333);flex-shrink:0;display:grid;grid-template-columns:1fr 1fr;gap:4px 10px;font-size:11px">
        <div><label style="color:#888">context</label><input id="llm-param-ctx" type="number" min="512" max="32768" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px"></div>
        <div><label style="color:#888">max_tokens</label><input id="llm-param-tokens" type="number" min="1" max="4096" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px"></div>
        <div><label style="color:#888">temperature</label><input id="llm-param-temp" type="number" step="0.01" min="0" max="2" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" placeholder="model default"></div>
        <div><label style="color:#888">top_p</label><input id="llm-param-topp" type="number" step="0.01" min="0" max="1" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" placeholder="model default"></div>
        <div><label style="color:#888">top_k</label><input id="llm-param-topk" type="number" min="0" max="200" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" placeholder="model default"></div>
        <div><label style="color:#888">repeat_penalty</label><input id="llm-param-repp" type="number" step="0.01" min="1" max="2" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" placeholder="model default"></div>
      </div>
    </div>
    `;

    document.body.appendChild(panel);
    bindEvents();
}

function bindEvents() {
    // Tabs
    panel.querySelector("#llm-tab-chat").onclick = () => switchTab("chat");
    panel.querySelector("#llm-tab-desc").onclick = () => switchTab("desc");
    panel.querySelector("#llm-tab-sys").onclick = () => switchTab("sys");
    panel.querySelector("#llm-close").onclick = togglePanel;

    // Chat
    panel.querySelector("#llm-send").onclick = sendMessage;
    panel.querySelector("#llm-input").onkeydown = (e) => {
        if (e.key === "Enter" && !e.ctrlKey && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };
    panel.querySelector("#llm-unload").onclick = unloadModel;
    panel.querySelector("#llm-unload-bottom").onclick = unloadModel;
    panel.querySelector("#llm-clear-chat").onclick = clearChat;
    panel.querySelector("#llm-new").onclick = newChat;

    // Model selector
    const modelSel = panel.querySelector("#llm-model-select");
    modelSel.onchange = () => {
        selectedModel = modelSel.value;
        saveState();
    };

    // Handler selector
    const handlerSel = panel.querySelector("#llm-handler-select");
    handlerSel.onchange = () => {
        selectedHandler = handlerSel.value;
        saveState();
    };

    // mmproj selector
    const mmprojSel = panel.querySelector("#llm-mmproj-select");
    mmprojSel.onchange = () => {
        selectedMmproj = mmprojSel.value;
        saveState();
    };

    // Describe
    panel.querySelector("#llm-desc-send").onclick = describeToChat;
    panel.querySelector("#llm-desc-clear").onclick = () => {
        describeText = "";
        updateDescribeUI();
        saveState();
    };

    const ta = panel.querySelector("#llm-desc-textarea");
    ta.oninput = () => {
        describeText = ta.value;
        saveState();
    };

    // System Prompt
    const sysEnabled = panel.querySelector("#llm-sys-enabled");
    sysEnabled.checked = systemPromptEnabled;
    sysEnabled.onchange = () => {
        systemPromptEnabled = sysEnabled.checked;
        saveState();
    };
    const sysTa = panel.querySelector("#llm-sys-textarea");
    sysTa.value = systemPrompt;
    sysTa.oninput = () => {
        systemPrompt = sysTa.value;
        saveState();
    };
    panel.querySelector("#llm-sys-reset").onclick = () => {
        systemPrompt = "";
        sysTa.value = "";
        saveState();
    };

    // Inference params
    const paramInputs = {
        "llm-param-ctx": { get: () => nCtx, set: (v) => { nCtx = parseInt(v) || 8192; } },
        "llm-param-tokens": { get: () => maxTokens, set: (v) => { maxTokens = parseInt(v) || 300; } },
        "llm-param-temp": { get: () => temperature, set: (v) => { temperature = v; } },
        "llm-param-topp": { get: () => topP, set: (v) => { topP = v; } },
        "llm-param-topk": { get: () => topK, set: (v) => { topK = v; } },
        "llm-param-repp": { get: () => repeatPenalty, set: (v) => { repeatPenalty = v; } },
    };
    for (const [id, p] of Object.entries(paramInputs)) {
        const el = panel.querySelector("#" + id);
        el.value = p.get();
        el.oninput = () => { p.set(el.value); saveState(); };
    }
}

// ---- Tab switching ----
function switchTab(tab) {
    activeTab = tab;
    panel.querySelector("#llm-tab-chat").classList.toggle("active", tab === "chat");
    panel.querySelector("#llm-tab-desc").classList.toggle("active", tab === "desc");
    panel.querySelector("#llm-tab-sys").classList.toggle("active", tab === "sys");
    panel.querySelector("#llm-chat-panel").style.display = tab === "chat" ? "flex" : "none";
    panel.querySelector("#llm-desc-panel").style.display = tab === "desc" ? "flex" : "none";
    panel.querySelector("#llm-sys-panel").style.display = tab === "sys" ? "flex" : "none";
    if (tab === "chat") updateChatUI();
    if (tab === "desc") updateDescribeUI();
}

// ---- Toggle ----
function togglePanel() {
    visible = !visible;
    if (visible) {
        const btn = document.getElementById("llm-sidebar-btn");
        if (btn) {
            const br = btn.getBoundingClientRect();
            let left = br.left;
            let top = br.bottom + 6;
            if (left + panel.offsetWidth > window.innerWidth) left = window.innerWidth - panel.offsetWidth - 10;
            if (top + panel.offsetHeight > window.innerHeight) top = br.top - panel.offsetHeight - 6;
            if (left < 8) left = 8;
            if (top < 8) top = 8;
            panel.style.left = left + "px";
            panel.style.top = top + "px";
            panel.style.right = "";
            panel.style.bottom = "";
        }
        panel.style.display = "flex";
        if (activeTab === "chat") {
            refreshModels();
            updateChatUI();
        } else {
            updateDescribeUI();
        }
    } else {
        panel.style.display = "none";
    }
}

// ---- Model loading ----
async function refreshModels() {
    try {
        const resp = await api.fetchApi("/llm-sidebar/models");
        const data = await resp.json();
        allModels = data?.data?.text || [];
        allMmproj = data?.data?.mmproj || [];
        allHandlers = data?.data?.chat_handlers || ["None"];

        // Update model dropdown
        const sel = panel.querySelector("#llm-model-select");
        sel.innerHTML = "";
        for (const m of allModels) {
            const opt = document.createElement("option");
            opt.value = m;
            opt.textContent = m;
            if (m === selectedModel) opt.selected = true;
            sel.appendChild(opt);
        }
        if (!selectedModel && allModels.length > 0) {
            selectedModel = allModels[0];
            sel.value = selectedModel;
            saveState();
        }

        // Update handler dropdown
        const handlerSel = panel.querySelector("#llm-handler-select");
        handlerSel.innerHTML = "";
        for (const h of allHandlers) {
            const opt = document.createElement("option");
            opt.value = h;
            opt.textContent = h;
            if (h === selectedHandler) opt.selected = true;
            handlerSel.appendChild(opt);
        }
        if (!allHandlers.includes(selectedHandler)) {
            selectedHandler = "None";
            handlerSel.value = "None";
            saveState();
        }

        // Update mmproj dropdown
        const mmprojSel = panel.querySelector("#llm-mmproj-select");
        mmprojSel.innerHTML = '<option value="None">mmproj: None</option>';
        for (const m of allMmproj) {
            const opt = document.createElement("option");
            opt.value = m;
            opt.textContent = m.replace(".gguf", "").replace(/mmproj/i, "mmp").substring(0, 28);
            if (m === selectedMmproj) opt.selected = true;
            mmprojSel.appendChild(opt);
        }
        if (selectedMmproj && selectedMmproj !== "None") mmprojSel.value = selectedMmproj;
    } catch (e) {
        console.error("LlmSidebar: failed to load models", e);
    }
}

function buildOptions() {
    const opts = {};
    if (temperature !== "") opts.temperature = parseFloat(temperature);
    if (topP !== "") opts.top_p = parseFloat(topP);
    if (topK !== "") opts.top_k = parseInt(topK);
    if (repeatPenalty !== "") opts.repeat_penalty = parseFloat(repeatPenalty);
    if (maxTokens) opts.max_tokens = parseInt(maxTokens);
    if (nCtx) opts.n_ctx = parseInt(nCtx);
    return opts;
}

// ---- Chat ----
function updateChatUI() {
    const container = panel.querySelector("#llm-messages");
    container.innerHTML = "";
    for (const msg of chatHistory) {
        appendMessageDOM(container, msg.role, msg.content);
    }
    container.scrollTop = container.scrollHeight;
}

function appendMessageDOM(container, role, content) {
    const div = document.createElement("div");
    const isUser = role === "user";
    Object.assign(div.style, {
        marginBottom: "10px",
        padding: "6px 10px",
        borderRadius: "6px",
        background: isUser ? "var(--bg-color, #222)" : "#1a2a1a",
        borderLeft: isUser ? "3px solid var(--primary, #4a6)" : "3px solid #666",
        whiteSpace: "pre-wrap",
        wordBreak: "break-word",
        fontSize: "12px",
        lineHeight: "1.4",
    });
    const label = document.createElement("div");
    label.style.cssText = "font-size:10px;color:#888;margin-bottom:4px";
    label.textContent = isUser ? "👤 You" : "🤖 LLM";
    div.appendChild(label);
    const text = document.createElement("div");
    text.textContent = content;
    div.appendChild(text);
    const btn = document.createElement("button");
    btn.textContent = "-> Describe";
    Object.assign(btn.style, {
        marginTop: "4px", padding: "2px 8px",
        background: "#333", color: "#aaa",
        border: "1px solid #555", borderRadius: "3px",
        cursor: "pointer", fontSize: "10px",
    });
    btn.onclick = () => appendToDescribe(content);
    div.appendChild(btn);
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

async function sendMessage() {
    const input = panel.querySelector("#llm-input");
    const text = input.value.trim();
    if (!text || !selectedModel) return;
    input.value = "";
    input.disabled = true;
    panel.querySelector("#llm-send").disabled = true;

    chatHistory.push({ role: "user", content: text });
    updateChatUI();

    chatHistory.push({ role: "assistant", content: "..." });
    updateChatUI();
    const container = panel.querySelector("#llm-messages");
    const lastMsg = container.lastElementChild;
    const lastText = lastMsg?.querySelector("div:last-child");

    try {
        const resp = await api.fetchApi("/llm-sidebar/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: selectedModel,
                prompt: text,
                system_prompt: systemPromptEnabled ? systemPrompt : "",
                chat_handler: selectedHandler,
                mmproj_file: selectedMmproj || "None",
                options: buildOptions(),
            }),
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let full = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split("\n");
            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        if (data.chunk) full += data.chunk;
                        if (data.done && data.full_response) full = data.full_response;
                        if (lastText) lastText.textContent = full || "...";
                    } catch (e) {}
                }
            }
        }

        chatHistory[chatHistory.length - 1] = { role: "assistant", content: full };
        updateChatUI();
    } catch (e) {
        chatHistory[chatHistory.length - 1] = {
            role: "assistant",
            content: `Error: ${e.message}`,
        };
        updateChatUI();
    }

    saveState();
    input.disabled = false;
    panel.querySelector("#llm-send").disabled = false;
    input.focus();
}

async function unloadModel() {
    try {
        await api.fetchApi("/llm-sidebar/unload", { method: "POST" });
        chatHistory.push({ role: "system", content: "🔄 Model unloaded. Select a model and send a message to reload." });
        updateChatUI();
        saveState();
    } catch (e) {
        console.error("LlmSidebar: unload failed", e);
    }
}

function clearChat() {
    chatHistory = [];
    updateChatUI();
    saveState();
}

async function newChat() {
    chatHistory = [];
    updateChatUI();
    saveState();
    try {
        await api.fetchApi("/llm-sidebar/reset", { method: "POST" });
    } catch (e) {
        console.warn("LlmSidebar: reset failed", e);
    }
}

// ---- Describe ----
function updateDescribeUI() {
    const ta = panel.querySelector("#llm-desc-textarea");
    if (ta) ta.value = describeText;
}

function appendToDescribe(text) {
    if (describeText) describeText += "\n\n";
    describeText += text;
    updateDescribeUI();
    saveState();
    switchTab("desc");
}

function describeToChat() {
    const ta = panel.querySelector("#llm-desc-textarea");
    const text = ta.value.trim();
    if (!text) return;
    switchTab("chat");
    const input = panel.querySelector("#llm-input");
    input.value = text;
    input.focus();
}

// ---- Right-click integration entry point (called from rightClick.js) ----
function onRightClickDescribe(text) {
    appendToDescribe(text);
}

function onRightClickVision(images, filename) {
    if (!selectedModel) {
        alert("Select a model in the LLM sidebar first.");
        return;
    }
    if (selectedHandler === "None") {
        alert("Select a chat_handler in the LLM sidebar for vision.\nOptions: Qwen3.5, Qwen3-VL, Qwen2.5-VL, etc.");
        togglePanel();
        return;
    }
    const prompt = systemPromptEnabled && systemPrompt
        ? (filename ? `Describe this image.\nFilename: ${filename}` : "Describe this image.")
        : (filename ? `${VISION_PROMPT}\n\nFilename: ${filename}` : VISION_PROMPT);
    visionDescribe(images, prompt);
}

async function visionDescribe(images, prompt) {
    try {
        chatHistory.push({ role: "system", content: "🔍 Analyzing image..." });
        updateChatUI();

        const resp = await api.fetchApi("/llm-sidebar/vision", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: selectedModel,
                images: images,
                prompt: prompt,
                chat_handler: selectedHandler,
                mmproj_file: selectedMmproj || "None",
                system_prompt: systemPromptEnabled ? systemPrompt : "",
                options: buildOptions(),
            }),
        });
        const data = await resp.json();

        if (data?.success) {
            const desc = data.data.response;
            chatHistory.pop();
            chatHistory.push({ role: "assistant", content: desc });
        } else {
            chatHistory.pop();
            chatHistory.push({ role: "system", content: `Error: ${data?.error || "Vision failed"}` });
        }
        updateChatUI();
        saveState();
        switchTab("chat");
    } catch (e) {
        chatHistory.pop();
        chatHistory.push({ role: "system", content: `Error: ${e.message}` });
        updateChatUI();
        saveState();
    }
}

// ---- Init ----
function createToggleButton() {
    const btn = document.createElement("button");
    btn.id = "llm-sidebar-btn";
    btn.textContent = "💬 LLM";
    btn.title = "Toggle LLM Sidebar (drag to reposition, click to open)";

    let pos = { x: -1, y: -1 };
    try {
        const saved = localStorage.getItem(STORAGE_KEY + "_btnpos");
        if (saved) pos = JSON.parse(saved);
    } catch (e) {}

    Object.assign(btn.style, {
        position: "fixed",
        zIndex: "9999",
        padding: "5px 12px",
        background: "var(--primary, #4a6)",
        color: "#fff",
        border: "none",
        borderRadius: "6px",
        cursor: "grab",
        fontSize: "13px",
        fontWeight: "bold",
        boxShadow: "0 2px 8px rgba(0,0,0,0.3)",
        userSelect: "none",
        transition: "box-shadow 0.15s",
    });

    function placeBtn() {
        if (pos.x >= 0 && pos.y >= 0) {
            btn.style.left = pos.x + "px";
            btn.style.top = pos.y + "px";
            btn.style.right = "";
        } else {
            btn.style.left = "50%";
            btn.style.top = "";
            btn.style.bottom = "60px";
            btn.style.transform = "translateX(-50%)";
        }
    }
    placeBtn();

    let dragging = false, startX, startY, origLeft, origTop, moved = false;

    btn.addEventListener("mousedown", (e) => {
        if (e.button !== 0) return;
        dragging = true;
        moved = false;
        startX = e.clientX;
        startY = e.clientY;
        const rect = btn.getBoundingClientRect();
        origLeft = rect.left;
        origTop = rect.top;
        btn.style.cursor = "grabbing";
        btn.style.transition = "none";
        btn.style.transform = "";
        btn.style.right = "";
        btn.style.bottom = "";
        btn.style.left = origLeft + "px";
        btn.style.top = origTop + "px";
        e.preventDefault();
    });

    document.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        if (Math.abs(dx) > 3 || Math.abs(dy) > 3) moved = true;
        btn.style.left = (origLeft + dx) + "px";
        btn.style.top = (origTop + dy) + "px";
    });

    document.addEventListener("mouseup", () => {
        if (!dragging) return;
        dragging = false;
        btn.style.cursor = "grab";
        btn.style.transition = "box-shadow 0.15s";
        const rect = btn.getBoundingClientRect();
        pos = { x: rect.left, y: rect.top };
        try { localStorage.setItem(STORAGE_KEY + "_btnpos", JSON.stringify(pos)); } catch (e) {}
    });

    btn.addEventListener("click", (e) => {
        if (moved) {
            e.preventDefault();
            e.stopPropagation();
            return;
        }
        togglePanel();
    });

    document.body.appendChild(btn);
}

// ---- Extension registration ----
app.registerExtension({
    name: "ComfyUI.LlmSidebar",

    async setup() {
        loadState();
        createPanel();
        createToggleButton();
        refreshModels();

        // Expose for rightClick.js
        window.LlmSidebar = {
            appendToDescribe,
            onRightClickVision,
            togglePanel,
            get selectedModel() { return selectedModel; },
            get selectedHandler() { return selectedHandler; },
        };
    },
});
