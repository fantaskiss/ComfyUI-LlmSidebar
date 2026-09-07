/**
 * ComfyUI-LlmSidebar - sidebar.js
 * Self-contained floating panel with Chat + Describe + SysPrompt + Setup tabs.
 * Reuses LLAMA_CPP_STORAGE from ComfyUI-llama-cpp_vlm for model loading.
 *
 * Key behavior:
 * - If model is ALREADY loaded (by workflow node or previously), sidebar
 *   reuses it without config comparison. No accidental reloads.
 * - Setup tab sets the load params used when NO model is loaded yet.
 * - "Apply & Reload" button in Setup tab force-reloads with new params.
 */


// ---- state ----
let panel = null;
let visible = false;
let activeTab = "chat";
let selectedModel = "";
let selectedMmproj = "";  // "None" = no mmproj, "" = auto
let selectedHandler = "None";
let chatHistory = [];
let describeText = "";
let systemPromptEnabled = true;
// 预设槽 v3.0：chatPRESET = Chat 系统提示词 3 套；descPRESET = 右键反推指令 3 套
// 每槽 {name, text}；激活槽 = 实际生效的预设；name 空时显示 "预设 N"
let chatPresets = [
    { name: "", text: "" },
    { name: "", text: "" },
    { name: "", text: "" },
];
let chatPresetIdx = 0;
let descPresets = [
    { name: "", text: "" },
    { name: "", text: "" },
    { name: "", text: "" },
];
let descPresetIdx = 0;
// Inference params (SysPrompt tab)
let maxTokens = 300;
let temperature = "";
let topP = "";
let topK = "";
let repeatPenalty = "";
// Load params (Setup tab)
let nCtx = 8192;
let nGpuLayers = -1;
let vramLimit = -1;
let cacheTypeK = "default";
let cacheTypeV = "default";
let nCpuMoe = 0;
let nSeqMax = 1;
let imageMinTokens = 1024;
let imageMaxTokens = 4096;

// Tool params (Setup tab)
let toolsEnabled = false;
let wikiPath = "";
let maxToolRounds = 10;
let toolResultMaxChars = 8000;

// Remote backend (Setup tab): 小主机 llama.cpp
let backendMode = "local";          // "local" | "remote"
let remoteBaseUrl = "http://10.0.0.8:60000";

// Gen tab (提示词生成系统)
let genIntent = "";
let genWikiPath = "";
let genResult = null;
let genBusy = false;

// Context usage (Chat panel)
let contextTokens = 0;
let contextLimit = 8192;

let allModels = [];
let allMmproj = [];
let allHandlers = ["None"];

const STORAGE_KEY = "llm_sidebar";
// 右键 "Describe with LLM" 的默认指令：英文生图向描述 + 【中文对照】
const DESCRIBE_PROMPT =
    "Describe this image in English, written as a prompt for AI image generation. " +
    "Cover subject, composition, lighting, color palette, style, mood, and notable " +
    "visual details. Be concise and direct. After the English description, output a " +
    "complete Chinese translation prefixed with 【中文对照】.";

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
            systemPromptEnabled = s.systemPromptEnabled !== false;
            // 预设槽：v3.0 结构；旧版单条 systemPrompt 迁移到 chatPresets[0]
            if (Array.isArray(s.chatPresets) && s.chatPresets.length === 3) {
                chatPresets = s.chatPresets.map(p => ({ name: (p && p.name) || "", text: (p && p.text) || "" }));
                chatPresetIdx = (s.chatPresetIdx >= 0 && s.chatPresetIdx < 3) ? s.chatPresetIdx : 0;
            } else {
                chatPresets = [
                    { name: "", text: s.systemPrompt || "" },
                    { name: "", text: "" },
                    { name: "", text: "" },
                ];
                chatPresetIdx = 0;
            }
            if (Array.isArray(s.descPresets) && s.descPresets.length === 3) {
                descPresets = s.descPresets.map(p => ({ name: (p && p.name) || "", text: (p && p.text) || "" }));
                descPresetIdx = (s.descPresetIdx >= 0 && s.descPresetIdx < 3) ? s.descPresetIdx : 0;
            } else {
                descPresets = [
                    { name: "", text: DESCRIBE_PROMPT },
                    { name: "", text: "" },
                    { name: "", text: "" },
                ];
                descPresetIdx = 0;
            }
            maxTokens = s.maxTokens || 300;
            temperature = s.temperature || "";
            topP = s.topP || "";
            topK = s.topK || "";
            repeatPenalty = s.repeatPenalty || "";
            // Load params
            nCtx = s.nCtx || 8192;
            nGpuLayers = s.nGpuLayers != null ? s.nGpuLayers : -1;
            vramLimit = s.vramLimit != null ? s.vramLimit : -1;
            cacheTypeK = s.cacheTypeK || "default";
            cacheTypeV = s.cacheTypeV || "default";
            nCpuMoe = s.nCpuMoe != null ? s.nCpuMoe : 0;
            nSeqMax = s.nSeqMax || 1;
            imageMinTokens = s.imageMinTokens || 1024;
            imageMaxTokens = s.imageMaxTokens || 4096;
            backendMode = s.backendMode === "remote" ? "remote" : "local";
            remoteBaseUrl = s.remoteBaseUrl || "http://10.0.0.8:60000";
            toolsEnabled = s.toolsEnabled === true;
            wikiPath = s.wikiPath || "";
            maxToolRounds = s.maxToolRounds || 10;
            toolResultMaxChars = s.toolResultMaxChars || 8000;
            genIntent = s.genIntent || "";
            genWikiPath = s.genWikiPath || s.wikiPath || "";
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
            systemPromptEnabled,
            chatPresets,
            chatPresetIdx,
            descPresets,
            descPresetIdx,
            maxTokens,
            temperature, topP, topK, repeatPenalty,
            nCtx, nGpuLayers, vramLimit,
            cacheTypeK, cacheTypeV,
            nCpuMoe, nSeqMax,
            imageMinTokens, imageMaxTokens,
            backendMode, remoteBaseUrl,
            toolsEnabled, wikiPath, maxToolRounds, toolResultMaxChars,
            genIntent, genWikiPath,
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
      <button id="llm-tab-chat" class="llm-tab active" style="flex:1;padding:10px 6px;border:none;background:transparent;color:inherit;cursor:pointer;font-size:13px">💬 Chat</button>
      <button id="llm-tab-desc" class="llm-tab" style="flex:1;padding:10px 6px;border:none;background:transparent;color:inherit;cursor:pointer;font-size:13px">📝 Desc</button>
      <button id="llm-tab-sys" class="llm-tab" style="flex:1;padding:10px 6px;border:none;background:transparent;color:inherit;cursor:pointer;font-size:13px">⚙️ Prompt</button>
      <button id="llm-tab-gen" class="llm-tab" style="flex:1;padding:10px 6px;border:none;background:transparent;color:inherit;cursor:pointer;font-size:13px">🎨 Gen</button>
      <button id="llm-tab-setup" class="llm-tab" style="flex:1;padding:10px 6px;border:none;background:transparent;color:inherit;cursor:pointer;font-size:13px">🔧 Setup</button>
      <button id="llm-close" style="padding:10px 14px;border:none;background:transparent;color:inherit;cursor:pointer;font-size:16px">✕</button>
    </div>

    <!-- Chat tab -->
    <div id="llm-chat-panel" style="display:flex;flex-direction:column;flex:1;min-height:0">
      <div style="display:flex;align-items:center;padding:8px;gap:6px;border-bottom:1px solid var(--border-color,#333);flex-shrink:0;flex-wrap:wrap">
        <span id="llm-current-model" style="flex:1;min-width:0;font-size:12px;color:#888;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="当前载入模型。选择/加载在 Setup 标签">未加载</span>
        <button id="llm-unload" style="padding:4px 8px;background:#633;color:#faa;border:1px solid #844;border-radius:4px;cursor:pointer;font-size:11px">Unload</button>
      </div>
      <div id="llm-messages" style="flex:1;overflow-y:auto;padding:8px;min-height:0"></div>
      <div id="llm-ctx-bar" style="padding:2px 10px;font-size:10px;color:#666;text-align:right;flex-shrink:0;border-top:1px solid var(--border-color,#333)">ctx: --/--</div>
      <div style="display:flex;padding:8px;gap:6px;border-top:none;flex:0 0 33.33%;min-height:0">
        <textarea id="llm-input" rows="2" placeholder="Ask something..." style="flex:1;height:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:6px;resize:none;font-size:16px;font-family:inherit"></textarea>
        <button id="llm-send" style="align-self:flex-end;padding:6px 14px;background:var(--primary,#4a6);color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:13px;font-weight:bold">Send</button>
        <button id="llm-new" style="align-self:flex-end;padding:6px 10px;background:#333;color:#ccc;border:1px solid #555;border-radius:4px;cursor:pointer;font-size:12px">New</button>
      </div>
    </div>

    <!-- Describe tab -->
    <div id="llm-desc-panel" style="display:none;flex-direction:column;flex:1;min-height:0">
      <textarea id="llm-desc-textarea" placeholder="Paste and compose your prompt here.&#10;Use [-> Describe] buttons in Chat to append messages." style="flex:1;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:none;padding:10px;resize:none;font-size:16px;font-family:inherit;min-height:0"></textarea>
      <div style="display:flex;padding:8px;gap:6px;border-top:1px solid var(--border-color,#333);flex-shrink:0">
        <button id="llm-desc-send" style="flex:1;padding:8px;background:var(--primary,#4a6);color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:13px">Send to Chat</button>
        <button id="llm-desc-clear" style="padding:8px 14px;background:#333;color:#ccc;border:1px solid #555;border-radius:4px;cursor:pointer">Clear</button>
      </div>
    </div>

    <!-- Prompt tab (v3.0: chatPRESET + descPRESET preset manager) -->
    <div id="llm-sys-panel" style="display:none;flex-direction:column;flex:1;min-height:0;overflow-y:auto">
      <div style="display:flex;flex-direction:column;flex:1;min-height:0">
        <div style="display:flex;align-items:center;padding:6px 10px;gap:8px;border-bottom:1px solid var(--border-color,#333);flex-wrap:wrap;flex-shrink:0">
          <span style="font-size:12px;font-weight:bold;color:var(--fg-color,#ddd)">💬 Chat 指令</span>
          <span style="font-size:10px;color:#888">chatPRESET · 激活槽随每条消息发送</span>
          <span style="flex:1"></span>
          <label style="display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer">
            <input type="checkbox" id="llm-sys-enabled" style="accent-color:var(--primary,#4a6)">
            <span style="color:var(--fg-color,#ddd)">✅ Enable</span>
          </label>
        </div>
        <div id="llm-chat-preset-tabs" style="display:flex;gap:4px;padding:5px 10px 0;flex-shrink:0"></div>
        <div style="padding:4px 10px 0;flex-shrink:0">
          <input id="llm-chat-preset-name" type="text" placeholder="槽位名字（如：翻译助手），留空显示 预设 N" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 6px;font-size:11px;border-radius:3px">
        </div>
        <div style="padding:4px 10px 6px;flex:1;min-height:0;display:flex">
          <textarea id="llm-chat-preset-textarea" placeholder="System prompt：新对话开场指令。例如 You are a helpful creative assistant specialized in image prompt engineering." style="flex:1;width:100%;box-sizing:border-box;min-height:40px;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:6px;resize:none;font-size:12px;font-family:inherit;border-radius:3px"></textarea>
        </div>
      </div>
      <div style="display:flex;flex-direction:column;flex:1;min-height:0;border-top:1px solid var(--border-color,#333)">
        <div style="display:flex;align-items:center;padding:6px 10px;gap:8px;border-bottom:1px solid var(--border-color,#333);flex-shrink:0">
          <span style="font-size:12px;font-weight:bold;color:var(--fg-color,#ddd)">🖼️ 反推指令</span>
          <span style="font-size:10px;color:#888">descPRESET · 右键 Describe with LLM 使用激活槽</span>
          <span style="flex:1"></span>
        </div>
        <div id="llm-desc-preset-tabs" style="display:flex;gap:4px;padding:5px 10px 0;flex-shrink:0"></div>
        <div style="padding:4px 10px 0;flex-shrink:0">
          <input id="llm-desc-preset-name" type="text" placeholder="槽位名字（如：英文生图），留空显示 预设 N" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 6px;font-size:11px;border-radius:3px">
        </div>
        <div style="padding:4px 10px 8px;flex:1;min-height:0;display:flex">
          <textarea id="llm-desc-preset-textarea" placeholder="反推指令：右键图片时发给模型的指令。槽1内置英文生图向描述（含【中文对照】）" style="flex:1;width:100%;box-sizing:border-box;min-height:40px;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:6px;resize:none;font-size:12px;font-family:inherit;border-radius:3px"></textarea>
        </div>
      </div>
    </div>

    <!-- Setup tab (model loading + tool parameters) -->
    <div id="llm-setup-panel" style="display:none;flex-direction:column;flex:1;min-height:0;overflow-y:auto">
      <div style="padding:8px 10px;font-size:11px;color:#888;border-bottom:1px solid var(--border-color,#333);flex-shrink:0">
        Model loading parameters. Used when no model is loaded yet.<br>
        After a model is loaded by a workflow node, sidebar reuses it as-is.
      </div>
      <div style="display:flex;align-items:center;padding:6px 10px;gap:8px;border-bottom:1px solid var(--border-color,#333);flex-shrink:0;flex-wrap:wrap">
        <label style="display:flex;align-items:center;gap:6px;color:#4a6;cursor:pointer;flex-shrink:0;font-size:12px" title="勾选后 Chat / Gen / 右键描述全部改走小主机 llama.cpp（OpenAI 兼容 HTTP），本机不再加载 GGUF 模型；模型与上下文由服务端决定">
          <input id="llm-setup-remote" type="checkbox" style="accent-color:var(--primary,#4a6);transform:scale(1.2)">
          🖥️ 使用小主机 llama.cpp
        </label>
        <label style="color:#888;flex-shrink:0">URL</label>
        <input id="llm-setup-remote-url" type="text" placeholder="http://10.0.0.8:60000" style="flex:1;min-width:0;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" title="小主机 llama.cpp 的 OpenAI 兼容服务地址">
      </div>
      <div id="llm-setup-model-box" style="display:flex;align-items:center;padding:6px 10px;gap:6px;border-bottom:1px solid var(--border-color,#333);flex-shrink:0;flex-wrap:wrap">
        <select id="llm-setup-model-select" style="flex:2;min-width:140px;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:4px;font-size:12px;border-radius:4px" title="模型（全局唯一加载入口）"></select>
        <select id="llm-setup-handler-select" style="flex:1;min-width:90px;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:4px;font-size:11px;border-radius:4px" title="chat_handler（视觉模型需要，纯文本用 None）">
          <option value="None">handler: None</option>
        </select>
        <select id="llm-setup-mmproj-select" style="flex:1;min-width:90px;background:var(--bg-color,#222);color:var(--fg-color,#888);border:1px solid var(--border-color,#444);padding:4px;font-size:11px;border-radius:4px" title="mmproj（视觉投影器，None=纯文本）">
          <option value="None">mmproj: None</option>
        </select>
      </div>
      <div id="llm-setup-params" style="padding:6px 10px;display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px 10px;font-size:11px;flex-shrink:0">
        <!-- Col 1: load params -->
        <div><label style="color:#888">context (n_ctx)</label><input id="llm-setup-ctx" type="number" min="512" max="327680" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px"></div>
        <div><label style="color:#888">n_gpu_layers</label><input id="llm-setup-gpu" type="number" min="-1" max="1024" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" title="-1 = auto/all layers. Overrides vram_limit calculation."></div>
        <div><label style="color:#888">vram_limit (GB)</label><input id="llm-setup-vram" type="number" min="-1" max="1024" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" title="-1 = no limit"></div>
        <div><label style="color:#888">n_cpu_moe</label><input id="llm-setup-cpu-moe" type="number" min="0" max="512" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" title="Keep MoE expert weights on CPU. For Qwen3.6-35B try 34."></div>
        <!-- Col 2: load params -->
        <div><label style="color:#888">cache_type_k</label><select id="llm-setup-ctk" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px">
          <option value="default">default</option>
          <option value="f16">f16</option>
          <option value="q8_0">q8_0</option>
          <option value="q4_0">q4_0</option>
          <option value="q5_0">q5_0</option>
        </select></div>
        <div><label style="color:#888">cache_type_v</label><select id="llm-setup-ctv" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px">
          <option value="default">default</option>
          <option value="f16">f16</option>
          <option value="q8_0">q8_0</option>
          <option value="q4_0">q4_0</option>
          <option value="q5_0">q5_0</option>
        </select></div>
        <div><label style="color:#888">n_seq_max</label><input id="llm-setup-seq" type="number" min="1" max="32" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" title="Max parallel sequences for batched VLM."></div>
        <div><label style="color:#888">image_min_tokens</label><input id="llm-setup-img-min" type="number" min="0" max="4096" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" title="Minimum image tokens. 1024 recommended for Qwen-VL."></div>
        <div><label style="color:#888">image_max_tokens</label><input id="llm-setup-img-max" type="number" min="0" max="4096" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" title="Maximum image tokens. 4096 recommended for Qwen-VL."></div>
        <!-- 工具参数已移至下方 Harness 单行（Enable / Wiki path / Max rounds） -->
      </div>
      <div style="display:flex;align-items:center;padding:6px 10px;gap:8px;border-top:1px solid var(--border-color,#333);flex-shrink:0;font-size:11px">
        <label style="display:flex;align-items:center;gap:6px;color:#4a6;cursor:pointer;flex-shrink:0;font-size:11px" title="Harness 工具层开关：控制下方 Wiki path / Max rounds 是否生效">
          <input id="llm-setup-tools-enable" type="checkbox" style="accent-color:var(--primary,#4a6);transform:scale(1.2)">
          ✅ Enable
        </label>
        <label style="color:#888;flex-shrink:0">Wiki path</label>
        <input id="llm-setup-wiki-path" type="text" style="flex:1;min-width:0;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" placeholder="e.g. F:/wiki">
        <label style="color:#888;flex-shrink:0">Max rounds</label>
        <input id="llm-setup-max-rounds" type="number" min="1" max="50" style="width:64px;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" title="How many tool call cycles before forced final answer.">
      </div>
      <div id="llm-setup-status" style="padding:8px 10px;font-size:11px;border-top:1px solid var(--border-color,#333);flex-shrink:0">
        <span id="llm-setup-loaded" style="color:#888">No model loaded</span>
      </div>
      <div style="display:flex;padding:8px 10px;gap:6px;flex-shrink:0">
        <button id="llm-setup-apply" style="flex:1;padding:8px;background:var(--primary,#4a6);color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:13px;font-weight:bold">Apply &amp; Reload</button>
        <button id="llm-setup-unload" style="padding:8px 14px;background:#633;color:#faa;border:1px solid #844;border-radius:4px;cursor:pointer;font-size:13px">Unload</button>
      </div>
      <div style="padding:6px 10px;border-top:1px solid var(--border-color,#333);flex-shrink:0">
        <div style="font-size:11px;font-weight:bold;color:#4a6;margin-bottom:4px">⚡ 推理参数（即时生效，无需 Apply · Chat / Gen / 反推共用）</div>
        <div id="llm-setup-inf-params" style="display:grid;grid-template-columns:1fr 1fr;gap:4px 10px;font-size:11px">
          <div><label style="color:#888">max_tokens</label><input id="llm-param-tokens" type="number" min="1" max="4096" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px"></div>
          <div><label style="color:#888">temperature</label><input id="llm-param-temp" type="number" step="0.01" min="0" max="2" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" placeholder="model default"></div>
          <div><label style="color:#888">top_p</label><input id="llm-param-topp" type="number" step="0.01" min="0" max="1" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" placeholder="model default"></div>
          <div><label style="color:#888">top_k</label><input id="llm-param-topk" type="number" min="0" max="200" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" placeholder="model default"></div>
          <div><label style="color:#888">repeat_penalty</label><input id="llm-param-repp" type="number" step="0.01" min="1" max="2" style="width:100%;box-sizing:border-box;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:2px 4px;font-size:11px;border-radius:3px" placeholder="model default"></div>
        </div>
      </div>
    </div>

    <!-- Gen tab (提示词生成系统：阶段0程序路由 + 阶段1 LLM组装) -->
    <div id="llm-gen-panel" style="display:none;flex-direction:column;flex:1;min-height:0">
      <div style="padding:6px 10px;border-bottom:1px solid var(--border-color,#333);flex-shrink:0;font-size:11px;color:#888">
        Wiki 路径：使用 Setup 标签的 Wiki path（harness）
      </div>
      <div style="padding:8px 10px;flex-shrink:0;font-size:11px;color:#888">
        描述一个画面（一次一个，别写数量）。避免用"提示词"字眼，直接说画面内容。
      </div>
      <textarea id="llm-gen-intent" placeholder="例如：校园教室窗边午后，女学生坐在课桌上看窗外&#10;例如：夜景名媛，顶层套房落地窗前，冷色调" style="flex:0 0 33.33%;min-height:70px;background:var(--bg-color,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#444);padding:8px;resize:none;font-size:16px;font-family:inherit;margin:0 10px 8px;border-radius:4px"></textarea>
      <div style="display:flex;padding:0 10px 8px;gap:6px;flex-shrink:0">
        <button id="llm-gen-run" style="flex:1;padding:8px;background:var(--primary,#4a6);color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:13px;font-weight:bold">🎨 生成提示词</button>
        <button id="llm-gen-copy" style="padding:8px 12px;background:#333;color:#ccc;border:1px solid #555;border-radius:4px;cursor:pointer;font-size:12px">Copy</button>
        <button id="llm-gen-desc" style="padding:8px 12px;background:#333;color:#ccc;border:1px solid #555;border-radius:4px;cursor:pointer;font-size:12px" title="将结果追加到 Describe">→ Desc</button>
        <button id="llm-gen-clear" style="padding:8px 12px;background:#633;color:#faa;border:1px solid #844;border-radius:4px;cursor:pointer;font-size:12px" title="清空结果区（不清历史）">Clear</button>
      </div>
      <div id="llm-gen-meta" style="padding:2px 10px;font-size:10px;color:#888;flex-shrink:0;min-height:14px"></div>
      <div id="llm-gen-result" style="flex:1;overflow-y:auto;padding:8px 10px;min-height:0;background:var(--bg-color,#1a2a1a);border-top:1px solid var(--border-color,#333);white-space:pre-wrap;word-break:break-word;font-size:16px;line-height:1.5"></div>
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
    panel.querySelector("#llm-tab-gen").onclick = () => switchTab("gen");
    panel.querySelector("#llm-tab-setup").onclick = () => switchTab("setup");
    panel.querySelector("#llm-gen-run").onclick = generatePrompt;
    panel.querySelector("#llm-gen-copy").onclick = copyGenResult;
    panel.querySelector("#llm-gen-desc").onclick = genToDescribe;
    panel.querySelector("#llm-gen-clear").onclick = clearGenResult;
    panel.querySelector("#llm-gen-intent").onkeydown = (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            generatePrompt();
        }
    };
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
    panel.querySelector("#llm-new").onclick = newChat;

    // Setup model selector (全局唯一加载入口)
    const modelSel = panel.querySelector("#llm-setup-model-select");
    if (modelSel) {
        modelSel.onchange = () => {
            selectedModel = modelSel.value;
            saveState();
        };
    }

    // Setup handler selector
    const handlerSel = panel.querySelector("#llm-setup-handler-select");
    if (handlerSel) {
        handlerSel.onchange = () => {
            selectedHandler = handlerSel.value;
            saveState();
        };
    }

    // Setup mmproj selector
    const mmprojSel = panel.querySelector("#llm-setup-mmproj-select");
    if (mmprojSel) {
        mmprojSel.onchange = () => {
            selectedMmproj = mmprojSel.value;
            saveState();
        };
    }

    // Remote backend (小主机 llama.cpp)
    const remoteEl = panel.querySelector("#llm-setup-remote");
    if (remoteEl) {
        remoteEl.checked = backendMode === "remote";
        remoteEl.onchange = () => {
            backendMode = remoteEl.checked ? "remote" : "local";
            saveState();
            updateBackendUI();
            updateContextDisplay();
        };
    }
    const remoteUrlEl = panel.querySelector("#llm-setup-remote-url");
    if (remoteUrlEl) {
        remoteUrlEl.value = remoteBaseUrl;
        remoteUrlEl.oninput = () => {
            remoteBaseUrl = remoteUrlEl.value.trim() || "http://10.0.0.8:60000";
            saveState();
        };
    }

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

    // ---- Prompt tab: preset groups (chatPRESET + descPRESET) ----
    const sysEnabled = panel.querySelector("#llm-sys-enabled");
    sysEnabled.checked = systemPromptEnabled;
    sysEnabled.onchange = () => {
        systemPromptEnabled = sysEnabled.checked;
        saveState();
    };

    function bindPresetGroup(group) {
        const isChat = group === "chat";
        const wrap = panel.querySelector(isChat ? "#llm-chat-preset-tabs" : "#llm-desc-preset-tabs");
        const nameEl = panel.querySelector(isChat ? "#llm-chat-preset-name" : "#llm-desc-preset-name");
        const ta = panel.querySelector(isChat ? "#llm-chat-preset-textarea" : "#llm-desc-preset-textarea");
        const presets = isChat ? chatPresets : descPresets;
        const getIdx = () => (isChat ? chatPresetIdx : descPresetIdx);
        const setIdx = (i) => { if (isChat) chatPresetIdx = i; else descPresetIdx = i; };

        const buttons = [];
        for (let i = 0; i < 3; i++) {
            const b = document.createElement("button");
            b.type = "button";
            b.style.cssText = "flex:1;padding:3px 0;border:1px solid #555;border-radius:3px;cursor:pointer;font-size:11px;background:#333;color:#ddd;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
            b.onclick = () => { setIdx(i); refresh(); saveState(); };
            wrap.appendChild(b);
            buttons.push(b);
        }
        function refresh() {
            const idx = getIdx();
            buttons.forEach((b, bi) => {
                const nm = (presets[bi] && presets[bi].name) || "预设 " + (bi + 1);
                b.textContent = nm;
                b.title = nm;
                b.style.background = bi === idx ? "var(--primary,#4a6)" : "#333";
                b.style.color = bi === idx ? "#fff" : "#ddd";
            });
            const cur = presets[idx] || { name: "", text: "" };
            ta.value = cur.text;
            nameEl.value = cur.name;
        }
        ta.oninput = () => { const cur = presets[getIdx()]; if (cur) { cur.text = ta.value; saveState(); } };
        nameEl.oninput = () => { const cur = presets[getIdx()]; if (cur) { cur.name = nameEl.value; refresh(); saveState(); } };
        refresh();
    }
    bindPresetGroup("chat");
    bindPresetGroup("desc");

    // Inference params (shared; input elements now live in Setup tab below Apply & Reload)
    const paramInputs = {
        "llm-param-tokens": { get: () => maxTokens, set: (v) => { maxTokens = parseInt(v) || 300; } },
        "llm-param-temp": { get: () => temperature, set: (v) => { temperature = v; } },
        "llm-param-topp": { get: () => topP, set: (v) => { topP = v; } },
        "llm-param-topk": { get: () => topK, set: (v) => { topK = v; } },
        "llm-param-repp": { get: () => repeatPenalty, set: (v) => { repeatPenalty = v; } },
    };
    for (const [id, p] of Object.entries(paramInputs)) {
        const el = panel.querySelector("#" + id);
        if (el) {
            el.value = p.get();
            el.oninput = () => { p.set(el.value); saveState(); };
        }
    }

    // Setup params (Setup tab)
    const setupInputs = {
        "llm-setup-ctx": { get: () => nCtx, set: (v) => { nCtx = parseInt(v) || 8192; } },
        "llm-setup-gpu": { get: () => nGpuLayers, set: (v) => { nGpuLayers = parseInt(v); if (isNaN(nGpuLayers)) nGpuLayers = -1; } },
        "llm-setup-vram": { get: () => vramLimit, set: (v) => { vramLimit = parseInt(v); if (isNaN(vramLimit)) vramLimit = -1; } },
        "llm-setup-cpu-moe": { get: () => nCpuMoe, set: (v) => { nCpuMoe = parseInt(v) || 0; } },
        "llm-setup-seq": { get: () => nSeqMax, set: (v) => { nSeqMax = parseInt(v) || 1; } },
        "llm-setup-img-min": { get: () => imageMinTokens, set: (v) => { imageMinTokens = parseInt(v) || 0; } },
        "llm-setup-img-max": { get: () => imageMaxTokens, set: (v) => { imageMaxTokens = parseInt(v) || 0; } },
    };
    for (const [id, p] of Object.entries(setupInputs)) {
        const el = panel.querySelector("#" + id);
        if (el) {
            el.value = p.get();
            el.oninput = () => { p.set(el.value); saveState(); };
        }
    }

    // Setup selects
    const ctkEl = panel.querySelector("#llm-setup-ctk");
    if (ctkEl) {
        ctkEl.value = cacheTypeK;
        ctkEl.onchange = () => { cacheTypeK = ctkEl.value; saveState(); };
    }
    const ctvEl = panel.querySelector("#llm-setup-ctv");
    if (ctvEl) {
        ctvEl.value = cacheTypeV;
        ctvEl.onchange = () => { cacheTypeV = ctvEl.value; saveState(); };
    }

    // Tool params (Setup tab)
    const toolsEnableEl = panel.querySelector("#llm-setup-tools-enable");
    if (toolsEnableEl) {
        toolsEnableEl.checked = toolsEnabled;
        toolsEnableEl.onchange = () => { toolsEnabled = toolsEnableEl.checked; saveState(); };
    }
    const wikiPathEl = panel.querySelector("#llm-setup-wiki-path");
    if (wikiPathEl) {
        wikiPathEl.value = wikiPath;
        wikiPathEl.oninput = () => { wikiPath = wikiPathEl.value; saveState(); };
    }
    const maxRoundsEl = panel.querySelector("#llm-setup-max-rounds");
    if (maxRoundsEl) {
        maxRoundsEl.value = maxToolRounds;
        maxRoundsEl.oninput = () => { maxToolRounds = parseInt(maxRoundsEl.value) || 10; saveState(); };
    }

    // Apply & Reload + Unload
    panel.querySelector("#llm-setup-apply").onclick = applySettings;
    panel.querySelector("#llm-setup-unload").onclick = unloadModel;
}

// ---- Tab switching ----
function switchTab(tab) {
    activeTab = tab;
    panel.querySelector("#llm-tab-chat").classList.toggle("active", tab === "chat");
    panel.querySelector("#llm-tab-desc").classList.toggle("active", tab === "desc");
    panel.querySelector("#llm-tab-sys").classList.toggle("active", tab === "sys");
    panel.querySelector("#llm-tab-gen").classList.toggle("active", tab === "gen");
    panel.querySelector("#llm-tab-setup").classList.toggle("active", tab === "setup");
    panel.querySelector("#llm-chat-panel").style.display = tab === "chat" ? "flex" : "none";
    panel.querySelector("#llm-desc-panel").style.display = tab === "desc" ? "flex" : "none";
    panel.querySelector("#llm-sys-panel").style.display = tab === "sys" ? "flex" : "none";
    panel.querySelector("#llm-gen-panel").style.display = tab === "gen" ? "flex" : "none";
    panel.querySelector("#llm-setup-panel").style.display = tab === "setup" ? "flex" : "none";
    if (tab === "chat") { refreshModels(); updateChatUI(); updateContextDisplay(); updateCurrentModelDisplay(); }
    if (tab === "desc") updateDescribeUI();
    if (tab === "gen") updateGenUI();
    if (tab === "setup") updateSetupUI();
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
            updateCurrentModelDisplay();
        } else if (activeTab === "desc") {
            updateDescribeUI();
        } else if (activeTab === "gen") {
            updateGenUI();
        } else if (activeTab === "setup") {
            updateSetupUI();
        }
    } else {
        panel.style.display = "none";
    }
}

// ---- Model loading ----
async function refreshModels() {
    try {
        const resp = await fetch("/llm-sidebar/models");
        const data = await resp.json();
        allModels = data?.data?.text || [];
        allMmproj = data?.data?.mmproj || [];
        allHandlers = data?.data?.chat_handlers || ["None"];

        // Update Setup model dropdown
        const sel = panel.querySelector("#llm-setup-model-select");
        if (sel) {
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
        }

        // Update Setup handler dropdown
        const handlerSel = panel.querySelector("#llm-setup-handler-select");
        if (handlerSel) {
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
        }

        // Update Setup mmproj dropdown
        const mmprojSel = panel.querySelector("#llm-setup-mmproj-select");
        if (mmprojSel) {
            mmprojSel.innerHTML = '<option value="None">mmproj: None</option>';
            for (const m of allMmproj) {
                const opt = document.createElement("option");
                opt.value = m;
                opt.textContent = m.replace(".gguf", "").replace(/mmproj/i, "mmp").substring(0, 28);
                if (m === selectedMmproj) opt.selected = true;
                mmprojSel.appendChild(opt);
            }
            if (selectedMmproj && selectedMmproj !== "None") mmprojSel.value = selectedMmproj;
        }
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
    // Load params for first-load (ignored if model already loaded)
    if (nCtx) opts.n_ctx = parseInt(nCtx);
    opts.n_gpu_layers = nGpuLayers;
    opts.vram_limit = vramLimit;
    opts.cache_type_k = cacheTypeK;
    opts.cache_type_v = cacheTypeV;
    opts.n_cpu_moe = nCpuMoe;
    opts.n_seq_max = nSeqMax;
    opts.image_min_tokens = imageMinTokens;
    opts.image_max_tokens = imageMaxTokens;
    return opts;
}

// ---- Remote backend UI (小主机 llama.cpp) ----
function updateBackendUI() {
    const modelBox = panel.querySelector("#llm-setup-model-box");
    const paramsGrid = panel.querySelector("#llm-setup-params");
    const remote = backendMode === "remote";
    if (modelBox) modelBox.style.display = remote ? "none" : "flex";
    if (paramsGrid) paramsGrid.style.display = remote ? "none" : "grid";
    const remoteEl = panel.querySelector("#llm-setup-remote");
    if (remoteEl) remoteEl.checked = remote;
}

function statusUrl() {
    const q = new URLSearchParams();
    q.set("backend", backendMode);
    if (backendMode === "remote") q.set("url", remoteBaseUrl);
    return "/llm-sidebar/status?" + q.toString();
}

// ---- Setup tab ----
async function updateSetupUI() {
    updateBackendUI();
    // Fetch current status to show loaded config
    try {
        const resp = await fetch(statusUrl());
        const data = await resp.json();
        if (data?.success) {
            const d = data.data;
            const loadedEl = panel.querySelector("#llm-setup-loaded");
            if (d.backend && d.backend.mode === "remote") {
                const rinfo = d.remote_info || {};
                if (d.loaded && rinfo.ready) {
                    loadedEl.innerHTML = '<span style="color:#4a6">🖥️ Remote connected</span> ' +
                        '<span style="color:#aaa">' + (rinfo.model || "?") + '</span>' +
                        (rinfo.n_ctx ? ' <span style="color:#888">· ctx ' + rinfo.n_ctx + '</span>' : '');
                } else {
                    loadedEl.innerHTML = '<span style="color:#f55">🖥️ Remote unreachable</span>';
                    if (rinfo.error) loadedEl.innerHTML += ' <span style="color:#888">' + rinfo.error + '</span>';
                }
                // 远程模式：不覆盖本地加载参数
                return;
            }
            if (d.loaded && d.current_config) {
                const cfg = d.current_config;
                loadedEl.innerHTML = '<span style="color:#4a6">Model loaded</span> ' +
                    '<span style="color:#aaa">' + (d.loaded_model || "?") + '</span>';
                // Fill fields with current loaded values
                const ctkEl = panel.querySelector("#llm-setup-ctk");
                const ctvEl = panel.querySelector("#llm-setup-ctv");
                panel.querySelector("#llm-setup-ctx").value = cfg.n_ctx || 8192;
                panel.querySelector("#llm-setup-gpu").value = cfg.n_gpu_layers != null ? cfg.n_gpu_layers : -1;
                panel.querySelector("#llm-setup-vram").value = cfg.vram_limit || -1;
                panel.querySelector("#llm-setup-cpu-moe").value = cfg.n_cpu_moe || 0;
                panel.querySelector("#llm-setup-seq").value = cfg.n_seq_max || 1;
                panel.querySelector("#llm-setup-img-min").value = cfg.image_min_tokens || 0;
                panel.querySelector("#llm-setup-img-max").value = cfg.image_max_tokens || 0;
                if (ctkEl) ctkEl.value = cfg.cache_type_k || "default";
                if (ctvEl) ctvEl.value = cfg.cache_type_v || "default";
            } else {
                loadedEl.innerHTML = '<span style="color:#888">No model loaded</span>';
                // Restore our saved values
                panel.querySelector("#llm-setup-ctx").value = nCtx;
                panel.querySelector("#llm-setup-gpu").value = nGpuLayers;
                panel.querySelector("#llm-setup-vram").value = vramLimit;
                panel.querySelector("#llm-setup-cpu-moe").value = nCpuMoe;
                panel.querySelector("#llm-setup-seq").value = nSeqMax;
                panel.querySelector("#llm-setup-img-min").value = imageMinTokens;
                panel.querySelector("#llm-setup-img-max").value = imageMaxTokens;
                const ctkEl = panel.querySelector("#llm-setup-ctk");
                const ctvEl = panel.querySelector("#llm-setup-ctv");
                if (ctkEl) ctkEl.value = cacheTypeK;
                if (ctvEl) ctvEl.value = cacheTypeV;
                // Restore tool fields
                const teEl = panel.querySelector("#llm-setup-tools-enable");
                if (teEl) teEl.checked = toolsEnabled;
                const wpEl = panel.querySelector("#llm-setup-wiki-path");
                if (wpEl) wpEl.value = wikiPath;
                const mrEl = panel.querySelector("#llm-setup-max-rounds");
                if (mrEl) mrEl.value = maxToolRounds;
            }
        }
    } catch (e) {
        console.warn("LlmSidebar: setup status fetch failed", e);
    }
}

async function applySettings() {
    const remote = backendMode === "remote";
    if (!remote && !selectedModel) {
        alert("请先在 Setup 标签选择模型");
        return;
    }
    if (remote && !remoteBaseUrl) {
        alert("请先填写小主机 llama.cpp 的 URL");
        return;
    }
    const btn = panel.querySelector("#llm-setup-apply");
    btn.disabled = true;
    btn.textContent = "Reloading...";

    try {
        const resp = await fetch("/llm-sidebar/apply-settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                backend_mode: backendMode,
                remote_url: remote ? remoteBaseUrl : "",
                model: selectedModel,
                chat_handler: selectedHandler,
                mmproj: selectedMmproj || "None",
                n_ctx: nCtx,
                n_gpu_layers: nGpuLayers,
                vram_limit: vramLimit,
                cache_type_k: cacheTypeK,
                cache_type_v: cacheTypeV,
                n_cpu_moe: nCpuMoe,
                n_seq_max: nSeqMax,
                image_min_tokens: imageMinTokens,
                image_max_tokens: imageMaxTokens,
                tools_enabled: toolsEnabled,
                wiki_path: wikiPath,
                max_tool_rounds: maxToolRounds,
                tool_result_max_chars: toolResultMaxChars,
            }),
        });
        const data = await resp.json();
        if (data?.success) {
            btn.textContent = remote ? "Connected" : "Reloaded";
            btn.style.background = "#484";
            setTimeout(() => {
                btn.textContent = "Apply & Reload";
                btn.style.background = "";
                btn.disabled = false;
            }, 2000);
            updateSetupUI();
            updateCurrentModelDisplay();
        } else {
            btn.textContent = "Apply & Reload";
            btn.disabled = false;
            alert("Reload failed: " + (data?.error || "unknown error"));
        }
    } catch (e) {
        btn.textContent = "Apply & Reload";
        btn.disabled = false;
        alert("Reload failed: " + e.message);
    }
}

// ---- Chat ----
async function updateCurrentModelDisplay() {
    try {
        const resp = await fetch(statusUrl());
        const data = await resp.json();
        const el = panel.querySelector("#llm-current-model");
        if (!el) return;
        if (!data?.success) return;
        const d = data.data;
        if (d.backend && d.backend.mode === "remote") {
            const rinfo = d.remote_info || {};
            if (d.loaded && rinfo.ready) {
                el.textContent = "🖥️ " + (d.loaded_model || "?") + " · 小主机";
                el.style.color = "#4a6";
                el.title = "小主机 llama.cpp 远程模型（Setup 勾选远程后端）";
            } else {
                el.textContent = "🖥️ 远程不可达";
                el.style.color = "#f55";
                el.title = rinfo.error || "检查小主机 llama.cpp 是否已启动（桌面 bat）";
            }
            return;
        }
        if (d.loaded && d.current_config) {
            const h = d.current_config.chat_handler;
            el.textContent = (h && h !== "None") ? (d.loaded_model + " · " + h) : d.loaded_model;
            el.style.color = "#4a6";
            el.title = "当前载入模型（Setup 标签选择与加载）";
        } else {
            el.textContent = "未加载";
            el.style.color = "#888";
            el.title = "模型未加载。在 Setup 选择模型后 Apply，或直接发送消息自动重载";
        }
    } catch (e) {}
}

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
        fontSize: "16px",
        lineHeight: "1.4",
    });
    const label = document.createElement("div");
    label.style.cssText = "font-size:10px;color:#888;margin-bottom:4px";
    label.textContent = isUser ? ">> You" : ">> LLM";
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

// ---- Gen tab (提示词生成系统) ----
function updateGenUI() {
    const intentEl = panel.querySelector("#llm-gen-intent");
    if (intentEl && genIntent && intentEl.value !== genIntent) intentEl.value = genIntent;
    renderGenResult();
}

function renderGenResult() {
    const box = panel.querySelector("#llm-gen-result");
    const meta = panel.querySelector("#llm-gen-meta");
    if (!box || !meta) return;
    if (!genResult) {
        box.textContent = "（结果区：输入意图后点击「生成提示词」）";
        meta.textContent = "";
        return;
    }
    if (genResult.error) {
        box.textContent = "❌ " + genResult.error;
        // 路由空洞时显示相近条目建议
        if (genResult.suggestions && genResult.suggestions.length) {
            box.textContent += "\n\n相近条目（可参考这些词重新描述）：\n" +
                genResult.suggestions.map(s => "  · " + s).join("\n");
        }
        meta.textContent = "error";
        return;
    }
    const r = genResult;
    let metaText = "mode: " + (r.mode || "?");
    if (r.rounds) metaText += " | rounds: " + r.rounds;
    if (r.hard_conflicts && r.hard_conflicts.length) metaText += " | ⚠️ 冲突: " + r.hard_conflicts.join("; ");
    if (r.candidates && r.candidates.length) metaText += " | 候选: " + r.candidates.length + " 文件";
    meta.textContent = metaText;
    box.textContent = r.prompt || "(空输出)";
}

async function generatePrompt() {
    if (genBusy) return;
    const intentEl = panel.querySelector("#llm-gen-intent");
    genIntent = (intentEl.value || "").trim();
    if (!genIntent) { alert("请输入用户意图"); return; }
    if (!wikiPath) { alert("请在 Setup 标签设置 Wiki path（harness）"); return; }
    if (backendMode !== "remote" && !selectedModel) { alert("请在 Setup 标签选择模型"); return; }
    saveState();

    genBusy = true;
    const box = panel.querySelector("#llm-gen-result");
    const runBtn = panel.querySelector("#llm-gen-run");
    box.textContent = "⏳ 阶段0 路由 wiki ...";
    runBtn.disabled = true;

    try {
        const resp = await fetch("/llm-sidebar/generate-prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                backend_mode: backendMode,
                remote_url: backendMode === "remote" ? remoteBaseUrl : "",
                model: selectedModel,
                intent: genIntent,
                wiki_path: wikiPath,
                system_prompt: "", // Gen 独立：不走 chat 预设，协议由 prompt_generator/DEFAULT_OUTPUT_PROTOCOL 控制
                chat_handler: selectedHandler,
                mmproj_file: selectedMmproj || "None",
                options: buildOptions(),
                fallback: true,
            }),
        });
        const data = await resp.json();
        genResult = data?.data || { error: (data?.error || "unknown error") };
        renderGenResult();
    } catch (e) {
        genResult = { error: e.message };
        renderGenResult();
    }
    genBusy = false;
    runBtn.disabled = false;
}

function clearGenResult() {
    genResult = null;
    renderGenResult();
}

function copyGenResult() {
    if (!genResult || !genResult.prompt) return;
    navigator.clipboard?.writeText(genResult.prompt).then(() => {
        const btn = panel.querySelector("#llm-gen-copy");
        if (btn) { btn.textContent = "Copied"; setTimeout(() => btn.textContent = "Copy", 1200); }
    }).catch(() => {});
}

function genToDescribe() {
    if (!genResult || !genResult.prompt) return;
    if (describeText) describeText += "\n\n";
    describeText += genResult.prompt;
    updateDescribeUI();
    saveState();
    switchTab("desc");
}

async function sendMessage() {
    const input = panel.querySelector("#llm-input");
    const text = input.value.trim();
    if (!text) return;
    if (backendMode !== "remote" && !selectedModel) return;
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
        const resp = await fetch("/llm-sidebar/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                backend_mode: backendMode,
                remote_url: backendMode === "remote" ? remoteBaseUrl : "",
                model: selectedModel,
                prompt: text,
                system_prompt: systemPromptEnabled ? chatPresets[chatPresetIdx].text : "",
                chat_handler: selectedHandler,
                mmproj_file: selectedMmproj || "None",
                options: buildOptions(),
                tools_enabled: toolsEnabled,
                wiki_path: wikiPath,
                max_tool_rounds: maxToolRounds,
                tool_result_max_chars: toolResultMaxChars,
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
                        if (data.error) throw new Error(data.error);
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
    updateContextDisplay();
    updateCurrentModelDisplay();
    input.disabled = false;
    panel.querySelector("#llm-send").disabled = false;
    input.focus();
}

async function unloadModel() {
    try {
        await fetch("/llm-sidebar/unload", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                backend_mode: backendMode,
                remote_url: backendMode === "remote" ? remoteBaseUrl : "",
            }),
        });
        chatHistory.push({ role: "system", content: backendMode === "remote"
            ? "🧹 已清除对话。远程模型由小主机 llama.cpp 管理（桌面 bat 启停），不会卸载。"
            : "🧹 Model unloaded. 在 Setup 标签选择模型后 Apply，或直接发送消息自动重载。" });
        updateChatUI();
        updateCurrentModelDisplay();
        saveState();
    } catch (e) {
        console.error("LlmSidebar: unload failed", e);
    }
}

async function newChat() {
    chatHistory = [];
    updateChatUI();
    saveState();
    try {
        await fetch("/llm-sidebar/reset", { method: "POST" });
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

// ---- Context display ----
async function updateContextDisplay() {
    try {
        const resp = await fetch(statusUrl());
        const data = await resp.json();
        if (data?.success) {
            const d = data.data;
            contextTokens = d.context_tokens || 0;
            contextLimit = d.context_limit || 8192;
            const el = document.getElementById("llm-ctx-bar");
            if (el) {
                const pct = contextLimit > 0 ? (contextTokens / contextLimit * 100) : 0;
                const txt = "ctx: " + contextTokens + "/" + contextLimit + " tok (" + pct.toFixed(1) + "%)";
                el.textContent = txt;
                if (pct > 80) {
                    el.style.color = "#f55";
                } else if (pct > 60) {
                    el.style.color = "#fa5";
                } else {
                    el.style.color = "#888";
                }
            }
        }
    } catch (e) {}
}

// ---- Right-click integration entry point (called from rightClick.js) ----
function onRightClickDescribe(text) {
    appendToDescribe(text);
}

function onRightClickVision(images, filename) {
    const remote = backendMode === "remote";
    if (!remote) {
        if (!selectedModel) {
            alert("请先在 Setup 标签选择模型");
            return;
        }
        if (selectedHandler === "None") {
            alert("请先在 Setup 标签选择 chat_handler（视觉需要非 None handler，如 Qwen3.5 / Qwen3-VL）。");
            togglePanel();
            return;
        }
    } else if (!remoteBaseUrl) {
        alert("请先在 Setup 填写小主机 llama.cpp 的 URL");
        return;
    }
    // 反推指令 = 激活的 desc 预设；槽空/越界回退内置 DESCRIBE_PROMPT 保底
    const activeDesc = (descPresetIdx >= 0 && descPresetIdx < descPresets.length)
        ? descPresets[descPresetIdx] : null;
    const descBase = (activeDesc && activeDesc.text) ? activeDesc.text : DESCRIBE_PROMPT;
    const prompt = filename
        ? `${descBase}\n\nFilename: ${filename}`
        : descBase;
    visionDescribe(images, prompt);
}

async function visionDescribe(images, prompt) {
    try {
        const resp = await fetch("/llm-sidebar/vision", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                backend_mode: backendMode,
                remote_url: backendMode === "remote" ? remoteBaseUrl : "",
                model: selectedModel,
                images: images,
                prompt: prompt,
                chat_handler: selectedHandler,
                mmproj_file: selectedMmproj || "None",
                system_prompt: '', // 反推只吃 desc 预设；Sys/chat 提示词不叠加（v3.0）
                options: buildOptions(),
            }),
        });
        const data = await resp.json();

        if (data?.success) {
            // 反推是无状态工具：结果进 Desc 文本框（appendToDescribe 自动切到 Desc 标签），不写 Chat 池
            appendToDescribe(data.data.response);
        } else {
            alert(`描述失败：${data?.error || "Vision failed"}`);
        }
    } catch (e) {
        alert(`描述失败：${e.message}`);
    }
}

// ---- Init ----
function createToggleButton() {
    const btn = document.createElement("button");
    btn.id = "llm-sidebar-btn";
    btn.textContent = "📝 LLM";
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
// Register with the Vite frontend's extension system
(async function() {
    // Poll for window.app to be ready
    for (let i = 0; i < 100; i++) {
        const a = window.app;
        if (a && typeof a.registerExtension === 'function') {
            a.registerExtension({
                name: "ComfyUI.LlmSidebar",
                async setup() {
                    loadState();
                    createPanel();
                    createToggleButton();
                    refreshModels();
                    updateBackendUI();

                    window.LlmSidebar = {
                        appendToDescribe,
                        onRightClickVision,
                        togglePanel,
                        get selectedModel() { return selectedModel; },
                        get selectedHandler() { return selectedHandler; },
                    };
                },
            });
            return;
        }
        await new Promise(r => setTimeout(r, 200));
    }
    console.warn("[LlmSidebar] app.registerExtension not available after 20s");
})();
