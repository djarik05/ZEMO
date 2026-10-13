/**
 * ZEMO APPLICATION ENTRY POINT
 * Initializes all modular controllers and verifies FastAPI backend connectivity.
 */

import { initNavigation } from "./navigation.js";
import { initOrb } from "./orb.js";
import { initUpload } from "./upload.js";
import { initModelSelector } from "./model-selector.js";
import { initSampleAnalysis } from "./sample-analysis.js";
import { initWorkflow } from "./workflow.js";
import { initInteractions, showToast } from "./interactions.js";
import * as api from "./api.js";

document.addEventListener("DOMContentLoaded", async () => {
  console.log("%c[ZEMO]%c AI Data Safety Platform initialized.", "color: #4D5837; font-weight: bold;", "color: #9EAC95;");

  window.zemo = window.zemo || {};
  window.zemo.api = api;

  // Initialize UI Subsystems
  initInteractions();
  initNavigation();
  initOrb();
  initUpload();
  initModelSelector();
  initSampleAnalysis();
  initWorkflow();

  // Verify FastAPI Backend Connection
  const statusTextEl = document.querySelector(".system-status span:last-child");
  try {
    const health = await api.checkHealth();
    if (statusTextEl) {
      statusTextEl.textContent = `BACKEND ONLINE (v${health.version || "0.1.0"})`;
    }
    console.log("[ZEMO] Connected to FastAPI backend:", health);
  } catch (err) {
    if (statusTextEl) {
      statusTextEl.textContent = "BACKEND OFFLINE";
    }
    console.warn("[ZEMO] FastAPI backend not reachable:", err);
    showToast("FastAPI backend is not reachable. Start uvicorn on port 8000.", "warning");
  }
});
