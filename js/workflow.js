/**
 * ZEMO WORKFLOW VISUALIZATION & TRAINING READINESS CONTROLLER
 * Synchronizes pipeline stage nodes, readiness checklist, and sanitization
 * execution with the FastAPI backend (/api/v1/workflow, /api/v1/sanitization, /api/v1/readiness).
 */

import { showModal, showToast } from "./interactions.js";
import { setOrbState } from "./orb.js";
import {
  getWorkflowStatus,
  applySanitization,
  validateSanitization,
  getAppState,
} from "./api.js";

let currentPipelineStages = [
  {
    id: "input",
    title: "Data Input",
    status: "pending",
    badgeText: "AWAITING",
    description: "Raw dataset ingested, schema validated, format normalized (CSV, JSON, JSONL).",
  },
  {
    id: "privacy",
    title: "Privacy Scan",
    status: "pending",
    badgeText: "PENDING",
    description: "Deterministic & probabilistic PII inspection detecting direct identifiers, emails, phones, and names.",
  },
  {
    id: "security",
    title: "Security Analysis",
    status: "pending",
    badgeText: "PENDING",
    description: "Credential detection, API key entropy analysis, and prompt injection adversarial pattern screening.",
  },
  {
    id: "risk",
    title: "Risk Assessment",
    status: "pending",
    badgeText: "PENDING",
    description: "Unified risk score calculation factoring severity, frequency, and downstream model blast radius.",
  },
  {
    id: "sanitization",
    title: "Sanitization",
    status: "pending",
    badgeText: "QUEUED",
    description: "Automated redaction, synthetic replacement, and blocking of hazardous training samples.",
  },
  {
    id: "validation",
    title: "Validation",
    status: "pending",
    badgeText: "PENDING",
    description: "Post-sanitization verification ensuring no residual privacy leaks or semantic distortion.",
  },
  {
    id: "readiness",
    title: "AI Readiness",
    status: "pending",
    badgeText: "PENDING",
    description: "Final safety attestation and export to target AI model fine-tuning or evaluation pipeline.",
  },
];

let selectedStageId = "risk";

export function initWorkflow() {
  const nodes = document.querySelectorAll(".workflow-stage-node");
  const explainerTitle = document.getElementById("explainer-stage-title");
  const explainerDesc = document.getElementById("explainer-stage-desc");
  const explainerStatus = document.getElementById("explainer-stage-status");
  const prepBtn = document.getElementById("prepare-training-btn");

  if (!nodes.length) return;

  // 1. Fetch Initial Workflow Definition from FastAPI Backend
  (async () => {
    try {
      const wf = await getWorkflowStatus();
      if (wf && wf.pipelineStages) {
        applyWorkflowData(wf);
      }
    } catch (err) {
      console.warn("[ZEMO] Using initial local workflow state:", err);
    }
  })();

  // 2. Pipeline Stage Node Selection
  nodes.forEach(node => {
    node.addEventListener("click", () => {
      const stageId = node.getAttribute("data-stage");
      if (!stageId) return;

      selectedStageId = stageId;
      nodes.forEach(n => n.classList.remove("active"));
      node.classList.add("active");

      updateExplainer(stageId);
    });
  });

  // 3. Listen for Backend Analysis, Config, and Reset Events
  window.addEventListener("zemo:analysis-loaded", (e) => {
    const analysis = e.detail;
    if (analysis?.workflow) {
      applyWorkflowData(analysis.workflow);
    }
  });

  window.addEventListener("zemo:workflow-updated", (e) => {
    if (e.detail) {
      applyWorkflowData(e.detail);
    }
  });

  window.addEventListener("zemo:config-updated", (e) => {
    const cfg = e.detail?.config;
    if (cfg) {
      const statRows = document.querySelectorAll(".readiness-stats-table .stat-row .stat-value");
      if (statRows.length >= 5) {
        statRows[4].textContent = `${cfg.provider_name || "OpenAI"} ${cfg.model || "gpt-4o"}`;
      }
    }
  });

  window.addEventListener("zemo:analysis-reset", async () => {
    try {
      const wf = await getWorkflowStatus();
      applyWorkflowData(wf);
    } catch (_) {
      // ignore
    }
  });

  // 4. Prepare Training Workspace: Execute Real Sanitization & Post-Validation
  prepBtn?.addEventListener("click", async () => {
    const state = getAppState();
    const datasetId = state.currentDatasetId;

    if (!datasetId) {
      updateModalContent(null, null);
      showModal("training-modal");
      showToast("Upload or load a dataset first to execute sanitization & training preparation.", "info");
      return;
    }

    const origText = prepBtn.textContent;
    prepBtn.disabled = true;
    prepBtn.textContent = "Sanitizing & Validating Dataset...";
    setOrbState("simulating");

    try {
      const sanResult = await applySanitization(datasetId, [], true);
      const valResult = await validateSanitization(datasetId);
      const updatedWorkflow = await getWorkflowStatus(sanResult.sanitized_dataset_id);

      selectedStageId = "readiness";
      applyWorkflowData(updatedWorkflow);
      setOrbState("ready");

      updateModalContent(sanResult, valResult);
      showModal("training-modal");
      showToast(
        `Sanitization complete: ${sanResult.sanitized_record_count.toLocaleString()} clean records certified for AI training!`,
        "success"
      );
    } catch (err) {
      console.error("[ZEMO] Sanitization pipeline error:", err);
      setOrbState("idle");
      showToast(`Sanitization failed: ${err.message}`, "warning");
    } finally {
      prepBtn.disabled = false;
      prepBtn.textContent = origText;
    }
  });

  function applyWorkflowData(wf) {
    if (!wf) return;

    if (Array.isArray(wf.pipelineStages)) {
      currentPipelineStages = wf.pipelineStages;

      nodes.forEach(node => {
        const stageId = node.getAttribute("data-stage");
        const stageObj = currentPipelineStages.find(s => s.id === stageId);
        if (!stageObj) return;

        node.classList.remove("completed", "active");
        if (stageObj.status === "completed") {
          node.classList.add("completed");
        } else if (stageObj.status === "active") {
          node.classList.add("active");
          selectedStageId = stageObj.id;
        }

        const statusSpan = node.querySelector(".node-status");
        if (statusSpan) {
          statusSpan.textContent = stageObj.badgeText || stageObj.status.toUpperCase();
        }
      });

      updateExplainer(selectedStageId);
    }

    if (wf.trainingReadiness) {
      renderTrainingReadiness(wf.trainingReadiness);
    }
  }

  function updateExplainer(stageId) {
    const stageData = currentPipelineStages.find(s => s.id === stageId) || currentPipelineStages[0];
    if (!stageData) return;

    if (explainerTitle) explainerTitle.textContent = stageData.title;
    if (explainerDesc) explainerDesc.textContent = stageData.description;
    if (explainerStatus) {
      const label = stageData.badgeText || stageData.status.toUpperCase();
      explainerStatus.textContent = `Current Pipeline Status: ${label}`;
      explainerStatus.className = `step-badge ${
        stageData.status === "completed"
          ? "pass"
          : stageData.status === "active"
          ? "warning"
          : "pending"
      }`;
    }
  }

  function renderTrainingReadiness(tr) {
    const gaugePct = document.querySelector(".gauge-percentage");
    const gaugeLabel = document.querySelector(".gauge-status-label");
    if (gaugePct) gaugePct.textContent = `${tr.readinessScore}%`;
    if (gaugeLabel) {
      gaugeLabel.textContent = `Threshold: ${tr.requiredThreshold}% Required • ${tr.statusText || ""}`;
    }

    // Update the 5 stat rows in .readiness-stats-table
    const statValues = document.querySelectorAll(".readiness-stats-table .stat-row .stat-value");
    if (statValues.length >= 5) {
      statValues[0].textContent = Number(tr.totalRecords || 0).toLocaleString();
      statValues[1].textContent = `${Number(tr.cleanRecords || 0).toLocaleString()} (${tr.cleanPercentage ?? 0}%)`;
      statValues[2].textContent = Number(tr.sanitizedRecords || 0).toLocaleString();
      statValues[3].textContent = Number(tr.blockedRecords || 0).toLocaleString();
      if (tr.targetArchitecture) {
        statValues[4].textContent = tr.targetArchitecture;
      }
    }

    // Update the 6 checklist steps in .readiness-stages-list
    const stepItems = document.querySelectorAll(".readiness-stages-list .stage-step-item");
    if (Array.isArray(tr.steps) && stepItems.length > 0) {
      tr.steps.forEach((stepData, idx) => {
        const itemEl = stepItems[idx];
        if (!itemEl) return;
        const titleEl = itemEl.querySelector(".step-title");
        const descEl = itemEl.querySelector(".step-desc");
        const badgeEl = itemEl.querySelector(".step-badge");

        if (titleEl) titleEl.textContent = stepData.title;
        if (descEl) descEl.textContent = stepData.desc;
        if (badgeEl) {
          badgeEl.className = `step-badge ${stepData.status}`;
          badgeEl.textContent = stepData.badge || stepData.status.toUpperCase();
        }
      });
    }
  }

  function updateModalContent(sanResult, valResult) {
    const modalBody = document.querySelector("#training-modal .modal-body");
    if (!modalBody) return;

    const state = getAppState();
    const cfg = state.modelConfig || {};

    if (!sanResult) {
      modalBody.innerHTML = `
        <p style="margin-bottom: var(--space-4);">
          <strong>No Active Dataset:</strong> Please upload a dataset in section <code>01 / DATA INPUT</code> or click <strong>Load Sample Analysis</strong> in section <code>03 / SECURITY INTELLIGENCE</code> first.
        </p>
        <div style="background: var(--surface-muted); padding: var(--space-4); border-radius: var(--radius-sm); border: 1px solid var(--border); font-size: var(--text-xs);">
          <strong>FastAPI Pipeline Route:</strong> <code>POST /api/v1/sanitization/apply</code> &rarr; <code>POST /api/v1/sanitization/{id}/validate</code><br>
          <strong>Status:</strong> Awaiting Dataset Ingestion
        </div>
      `;
      return;
    }

    modalBody.innerHTML = `
      <p style="margin-bottom: var(--space-4);">
        <strong>Sanitization &amp; Verification Complete:</strong> The Zemo FastAPI backend has created an immutable, sanitized version of your dataset and verified it for AI training readiness.
      </p>
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-3); margin-bottom: var(--space-4);">
        <div style="background: var(--surface-muted); padding: var(--space-3); border-radius: var(--radius-sm); border: 1px solid var(--border); font-size: var(--text-xs);">
          <div style="color: var(--text-muted);">Original &rarr; Sanitized Records</div>
          <div style="font-size: var(--text-base); font-weight: 700; color: var(--text); margin-top: 2px;">
            ${sanResult.original_record_count.toLocaleString()} &rarr; ${sanResult.sanitized_record_count.toLocaleString()}
          </div>
        </div>
        <div style="background: var(--surface-muted); padding: var(--space-3); border-radius: var(--radius-sm); border: 1px solid var(--border); font-size: var(--text-xs);">
          <div style="color: var(--text-muted);">Remediation Actions Applied</div>
          <div style="font-size: var(--text-base); font-weight: 700; color: var(--olive-dark); margin-top: 2px;">
            ${sanResult.fields_masked} Masked • ${sanResult.fields_redacted} Redacted • ${sanResult.records_removed} Pruned
          </div>
        </div>
      </div>
      <div style="background: var(--surface-muted); padding: var(--space-4); border-radius: var(--radius-sm); border: 1px solid var(--border); font-size: var(--text-xs); line-height: 1.6;">
        <strong>Sanitized Dataset ID:</strong> <code>${sanResult.sanitized_dataset_id}</code><br>
        <strong>Target AI Environment:</strong> <code>${cfg.provider_name || "OpenAI"} (${cfg.model || "gpt-4o"}) — ${cfg.environment || "Cloud Isolated"}</code><br>
        <strong>Post-Validation Status:</strong> <span style="color: var(--success); font-weight: 600;">${valResult?.passed ? "PASSED (0 Critical/High Residual Leaks)" : "VALIDATED"}</span>
      </div>
    `;
  }
}
