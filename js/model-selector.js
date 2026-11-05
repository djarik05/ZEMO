/**
 * ZEMO TARGET AI MODEL SELECTOR
 * Manages provider card selection, dynamic configuration panel updates,
 * and backend synchronization via /api/v1/models/*.
 */

import { showToast } from "./interactions.js";
import {
  getModelProviders,
  saveModelConfig,
  assessReadiness,
  getWorkflowStatus,
  getAppState,
  emitAppEvent,
} from "./api.js";

const FALLBACK_PROVIDERS = [
  {
    id: "openai",
    name: "OpenAI",
    defaultModel: "gpt-4o",
    models: ["gpt-4o", "gpt-4o-mini", "o1-preview", "o3-mini"],
    envDefault: "Cloud Isolated",
  },
  {
    id: "anthropic",
    name: "Anthropic",
    defaultModel: "claude-3-5-sonnet-20241022",
    models: ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
    envDefault: "Cloud Isolated",
  },
  {
    id: "gemini",
    name: "Google Gemini",
    defaultModel: "gemini-1.5-pro",
    models: ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-thinking"],
    envDefault: "Cloud Isolated",
  },
  {
    id: "meta",
    name: "Meta Llama",
    defaultModel: "llama-3.3-70b-instruct",
    models: ["llama-3.3-70b-instruct", "llama-3.1-8b-instruct", "llama-guard-3-8b"],
    envDefault: "Hybrid Enclave",
  },
  {
    id: "mistral",
    name: "Mistral AI",
    defaultModel: "mistral-large-2411",
    models: ["mistral-large-2411", "pixtral-large-2411", "mistral-small-2409"],
    envDefault: "Cloud Isolated",
  },
  {
    id: "ollama",
    name: "Ollama / Local",
    defaultModel: "llama3.2:latest",
    models: ["llama3.2:latest", "qwen2.5-coder:32b", "deepseek-r1:14b", "phi4:latest"],
    envDefault: "Local Air-Gapped",
  },
];

let providersCatalog = [...FALLBACK_PROVIDERS];
let selectedProvider = "openai";

export function initModelSelector() {
  const cards = document.querySelectorAll(".model-card");
  const providerTitle = document.getElementById("config-provider-name");
  const modelSelect = document.getElementById("config-model-select");
  const envSelect = document.getElementById("config-env-select");
  const taskSelect = document.getElementById("config-task-select");
  const privacySelect = document.getElementById("config-privacy-select");
  const saveBtn = document.getElementById("save-config-btn");

  if (!cards.length) return;

  // 1. Fetch Provider Catalog & Saved Config from FastAPI Backend
  (async () => {
    try {
      const data = await getModelProviders();
      if (Array.isArray(data.providers) && data.providers.length > 0) {
        providersCatalog = data.providers;
      }
      if (data.active_config) {
        selectedProvider = data.active_config.provider || "openai";
        cards.forEach(c => {
          const isMatch = c.getAttribute("data-provider") === selectedProvider;
          c.classList.toggle("selected", isMatch);
          c.setAttribute("aria-pressed", String(isMatch));
        });
        updateConfigurationPanel(selectedProvider, data.active_config);
      } else {
        updateConfigurationPanel(selectedProvider);
      }
    } catch (err) {
      console.warn("[ZEMO] Using fallback model catalog:", err);
      updateConfigurationPanel("openai");
    }
  })();

  // 2. Provider Card Click Selection
  cards.forEach(card => {
    card.addEventListener("click", () => {
      const providerId = card.getAttribute("data-provider");
      if (!providerId) return;

      cards.forEach(c => {
        c.classList.remove("selected");
        c.setAttribute("aria-pressed", "false");
      });
      card.classList.add("selected");
      card.setAttribute("aria-pressed", "true");

      selectedProvider = providerId;
      updateConfigurationPanel(providerId);

      const providerData = providersCatalog.find(p => p.id === providerId);
      if (providerData) {
        showToast(`Target provider selected: ${providerData.name}`, "info");
      }
    });
  });

  // 3. Save Target Configuration to Backend
  saveBtn?.addEventListener("click", async (e) => {
    e.preventDefault();
    const providerData = providersCatalog.find(p => p.id === selectedProvider);
    const chosenModel = modelSelect?.value || providerData?.defaultModel || "gpt-4o";
    const chosenEnv = envSelect?.value || "Cloud Isolated";
    const chosenTask = taskSelect?.value || "Supervised Fine-Tuning";
    const chosenPrivacy = privacySelect?.value || "Strict";

    const origText = saveBtn.textContent;
    saveBtn.disabled = true;
    saveBtn.textContent = "Saving Configuration...";

    try {
      const res = await saveModelConfig({
        provider: selectedProvider,
        model: chosenModel,
        environment: chosenEnv,
        task: chosenTask,
        privacy_level: chosenPrivacy,
      });

      // If a dataset is loaded, re-assess readiness for the updated intended operation
      const state = getAppState();
      const activeDatasetId = state.sanitizedDatasetId || state.currentDatasetId;
      if (activeDatasetId) {
        await assessReadiness(activeDatasetId, res.config?.intended_operation);
        const wf = await getWorkflowStatus(activeDatasetId);
        emitAppEvent("zemo:workflow-updated", wf);
      }

      showToast(
        res.message || `Configuration saved: ${providerData?.name} (${chosenModel}) in ${chosenEnv}.`,
        "success"
      );
    } catch (err) {
      console.error("[ZEMO] Failed to save model config:", err);
      showToast(`Failed to save configuration: ${err.message}`, "warning");
    } finally {
      saveBtn.disabled = false;
      saveBtn.textContent = origText;
    }
  });

  function updateConfigurationPanel(providerId, savedConfig = null) {
    const data = providersCatalog.find(p => p.id === providerId);
    if (!data) return;

    if (providerTitle) {
      providerTitle.textContent = `${data.name} Configuration`;
    }

    if (modelSelect) {
      modelSelect.innerHTML = "";
      const targetModel = savedConfig?.model && data.models.includes(savedConfig.model)
        ? savedConfig.model
        : data.defaultModel;

      data.models.forEach(modelName => {
        const opt = document.createElement("option");
        opt.value = modelName;
        opt.textContent = modelName;
        if (modelName === targetModel) opt.selected = true;
        modelSelect.appendChild(opt);
      });
    }

    const targetEnv = savedConfig?.environment || data.envDefault;
    if (envSelect && targetEnv) {
      for (let i = 0; i < envSelect.options.length; i++) {
        if (envSelect.options[i].text.includes(targetEnv) || envSelect.options[i].value === targetEnv) {
          envSelect.selectedIndex = i;
          break;
        }
      }
    }

    if (taskSelect && savedConfig?.task) {
      for (let i = 0; i < taskSelect.options.length; i++) {
        if (taskSelect.options[i].value === savedConfig.task) {
          taskSelect.selectedIndex = i;
          break;
        }
      }
    }

    if (privacySelect && savedConfig?.privacy_level) {
      for (let i = 0; i < privacySelect.options.length; i++) {
        if (privacySelect.options[i].value === savedConfig.privacy_level) {
          privacySelect.selectedIndex = i;
          break;
        }
      }
    }
  }
}
