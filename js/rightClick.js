/**
 * ComfyUI-LlmSidebar  rightClick.js
 * Hooks into node right-click menus:
 *   LoadImage       😯 Describe with LLM" (file path 鈫?vision)
 *   IMAGE output    😯 Describe with LLM" (cached output 鈫?vision)
 *   Text nodes      😯 Copy to Describe"
 */

const { app } = window.comfyAPI.app;

const IMAGE_NODE_TYPES = new Set([
    "LoadImage",
    "LoadImageMask",
    "PreviewImage",
    "SaveImage",
]);

const TEXT_NODE_TYPES = new Set([
    "CLIPTextEncode",
    "CLIPTextEncodeSDXL",
    "CLIPTextEncodeSD3",
    "CLIPTextEncodeFlux",
]);

app.registerExtension({
    name: "ComfyUI.LlmSidebar.RightClick",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const isImageNode = IMAGE_NODE_TYPES.has(nodeData.name);
        const isTextNode = TEXT_NODE_TYPES.has(nodeData.name);
        const hasImageOutput = nodeData.output_node !== false &&
            (nodeData.output_types || []).some(t =>
                t === "IMAGE" || (typeof t === "string" && t.toUpperCase() === "IMAGE")
            );

        if (!isImageNode && !isTextNode && !hasImageOutput) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const origGetExtraMenuOptions = this.getExtraMenuOptions;
            this.getExtraMenuOptions = function (canvas) {
                const options =
                    origGetExtraMenuOptions?.apply(this, arguments) || [];

                if (isImageNode) {
                    options.push({
                        content: "😯 Describe with LLM",
                        callback: () => describeImageNode(this),
                    });
                } else if (hasImageOutput) {
                    options.push({
                        content: "😯 Describe with LLM",
                        callback: () => describeOutputImage(this),
                    });
                }

                if (isTextNode) {
                    options.push({
                        content: "😯 Copy to Describe",
                        callback: () => copyTextToDescribe(this),
                    });
                }

                return options;
            };

            return result;
        };
    },
});

// ---- Image nodes (LoadImage, PreviewImage, etc.) ----
async function describeImageNode(node) {
    if (!ensureSidebar()) return;

    const sidebar = window.LlmSidebar;

    // Try to get image from widget (LoadImage)
    const imageWidget = node.widgets?.find(
        (w) => w.name === "image" || w.type === "image"
    );
    if (imageWidget && imageWidget.value) {
        const filename = String(imageWidget.value).trim();
        if (filename && filename !== "none" && filename !== "Choose file to upload") {
            // LoadImage: use file path. The backend resolves folder_paths.get_annotated_filepath.
            sidebar.onRightClickVision([filename], filename);
            return;
        }
    }

    // Fallback: get from node.imgs (works for PreviewImage, VAE Decode, etc.)
    const imageData = await getNodePreviewImage(node);
    if (imageData) {
        sidebar.onRightClickVision([imageData], null);
    } else {
        alert("Could not get image data from this node.\nMake sure the node has been executed.");
    }
}

// ---- IMAGE output nodes (VAE Decode, etc.) ----
async function describeOutputImage(node) {
    if (!ensureSidebar()) return;

    const sidebar = window.LlmSidebar;

    const imageData = await getNodePreviewImage(node);
    if (imageData) {
        sidebar.onRightClickVision([imageData], null);
    } else {
        alert("No image output available.\nExecute the workflow first.");
    }
}

// ---- Text nodes ----
function copyTextToDescribe(node) {
    if (!ensureSidebar()) return;

    // Find the text widget
    const textWidget = node.widgets?.find(
        (w) => w.name === "text" || w.type === "text" || w.type === "customtext"
    );
    if (textWidget && textWidget.value) {
        window.LlmSidebar.appendToDescribe(String(textWidget.value).trim());
    } else {
        alert("No text found in this node.");
    }
}

// ---- Helpers ----
function ensureSidebar() {
    if (!window.LlmSidebar) {
        alert("LLM sidebar not loaded. Try refreshing the page.");
        return false;
    }
    return true;
}

/**
 * Try to get the preview image from a node's rendered output.
 * After execution, ComfyUI stores <img> DOM elements in node.imgs.
 */
async function getNodePreviewImage(node) {
    // node.imgs 鈥?<img> DOM elements populated by ComfyUI after execution
    if (node.imgs && node.imgs.length > 0) {
        const img = node.imgs[0];

        // It's a DOM <img> element 鈥?use its src directly
        if (img instanceof HTMLImageElement && img.currentSrc) {
            return fetchNodeImage(img.currentSrc);
        }
        if (img instanceof HTMLImageElement && img.src) {
            return fetchNodeImage(img.src);
        }

        // Fallback for object format {filename, subfolder, type}
        if (img && typeof img === "object" && img.filename && img.type) {
            const params = new URLSearchParams({
                filename: img.filename,
                type: img.type,
                subfolder: img.subfolder || "",
            });
            return fetchNodeImage(`/api/view?${params}`);
        }

        // String data URI
        if (typeof img === "string" && img.startsWith("data:image")) {
            return img;
        }
    }

    return null;
}

/**
 * Fetch an image from a URL and return it as a base64 data URI.
 */
async function fetchNodeImage(url) {
    try {
        const resp = await fetch(url);
        if (!resp.ok) return null;
        const blob = await resp.blob();
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(blob);
        });
    } catch (e) {
        console.warn("LlmSidebar: failed to fetch image", url, e);
        return null;
    }
}

