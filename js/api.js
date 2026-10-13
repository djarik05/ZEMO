/**
 * ZEMO BACKEND API CLIENT & SHARED STATE ORCHESTRATOR
 * Connects the frontend to the FastAPI backend (/api/v1/...).
 * Supports both unified serving (FastAPI StaticFiles) and cross-origin dev servers.
 */

const DEFAULT_BACKEND_PORT = "8000";
const SEPARATE_DEV_PORTS = new Set(["3000", "4173", "4200", "5173", "5500", "8080", "8081"]);

/**
 * Compute the backend origin dynamically from the browser's current hostname
 * so both localhost and LAN IP addresses (e.g., 192.168.x.x, 172.16.x.x, 10.x.x.x)
 * resolve to the host server rather than the client device's loopback interface.
 */
function getDynamicBackendOrigin() {
  if (typeof window === "undefined") return `http://127.0.0.1:${DEFAULT_BACKEND_PORT}`;
  const { protocol, hostname } = window.location;
  const backendPort = String(window.ZEMO_BACKEND_PORT || DEFAULT_BACKEND_PORT);
  if (protocol === "file:" || !hostname) {
    return `http://127.0.0.1:${backendPort}`;
  }
  const scheme = protocol === "https:" ? "https:" : "http:";
  return `${scheme}//${hostname}:${backendPort}`;
}

export function resolveApiBase() {
  if (typeof window === "undefined") return "";
  // 1. Explicit runtime override
  if (window.ZEMO_API_BASE !== undefined && window.ZEMO_API_BASE !== null) {
    return String(window.ZEMO_API_BASE).replace(/\/+$/, "");
  }
  // 2. URL query parameter (?api_base=...) or localStorage override
  try {
    const params = new URLSearchParams(window.location.search);
    const queryApiBase = params.get("api_base");
    if (queryApiBase) {
      return queryApiBase.replace(/\/+$/, "");
    }
    const storedApiBase = window.localStorage?.getItem("ZEMO_API_BASE");
    if (storedApiBase) {
      return storedApiBase.replace(/\/+$/, "");
    }
  } catch (_) {
    // Ignore storage/URL parsing errors in restricted browser contexts
  }
  // 3. HTML <meta name="zemo-api-base" content="..."> override
  if (typeof document !== "undefined") {
    const metaEl = document.querySelector('meta[name="zemo-api-base"]');
    const metaContent = metaEl?.getAttribute("content")?.trim();
    if (metaContent) {
      return metaContent.replace(/\/+$/, "");
    }
  }
  const { protocol, port } = window.location;
  if (protocol === "file:") return getDynamicBackendOrigin();

  const backendPort = String(window.ZEMO_BACKEND_PORT || DEFAULT_BACKEND_PORT);
  // 4. When served from a separate frontend dev server (e.g. 3000, 5500, 8080),
  // target the FastAPI backend on the same host IP/hostname.
  if (port && port !== backendPort && SEPARATE_DEV_PORTS.has(port)) {
    return getDynamicBackendOrigin();
  }
  // 5. When served directly by FastAPI (unified server on port 8000 or custom port), use relative URLs.
  return "";
}

/**
 * Resolve WebSocket base URL matching the active server origin and protocol (ws:// or wss://).
 */
export function resolveWsBase() {
  if (typeof window === "undefined") return `ws://127.0.0.1:${DEFAULT_BACKEND_PORT}`;
  const base = resolveApiBase() || window.location.origin;
  return base.replace(/^http/, "ws");
}

let apiBase = resolveApiBase();

/**
 * Shared reactive application state across frontend modules.
 */
const appState = {
  backendOnline: false,
  currentDatasetId: null,
  sanitizedDatasetId: null,
  datasetMeta: null,
  analysisData: null,
  workflowData: null,
  sanitizationResult: null,
  validationResult: null,
  modelConfig: {
    provider: "openai",
    provider_name: "OpenAI",
    model: "gpt-4o",
    environment: "Cloud Isolated",
    task: "Supervised Fine-Tuning",
    intended_operation: "fine_tuning",
    privacy_level: "Strict",
  },
  providers: [],
};

export function getAppState() {
  return appState;
}

export function emitAppEvent(eventName, detail = {}) {
  window.dispatchEvent(new CustomEvent(eventName, { detail }));
}

/**
 * Core HTTP helper with automatic LAN-aware fallback if running on a separate frontend port.
 */
async function apiFetch(path, options = {}) {
  const url = `${apiBase}${path}`;
  const headers = new Headers(options.headers || {});
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const fallbackOrigin = getDynamicBackendOrigin();
  let response;
  try {
    response = await fetch(url, { ...options, headers });
  } catch (networkErr) {
    // If relative fetch failed while on a separate port, try the host's backend origin once
    if (apiBase === "" && typeof window !== "undefined" && window.location.origin !== fallbackOrigin) {
      apiBase = fallbackOrigin;
      response = await fetch(`${apiBase}${path}`, { ...options, headers });
    } else {
      throw networkErr;
    }
  }

  // If served on a static server (e.g. python -m http.server) where API routes return non-JSON 404/405/501
  if (
    (response.status === 404 || response.status === 405 || response.status === 501) &&
    apiBase === "" &&
    typeof window !== "undefined" &&
    window.location.origin !== fallbackOrigin
  ) {
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      apiBase = fallbackOrigin;
      response = await fetch(`${apiBase}${path}`, { ...options, headers });
    }
  }

  const isJson = (response.headers.get("content-type") || "").includes("application/json");
  const payload = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    const message =
      (payload && typeof payload === "object" && (payload.message || payload.detail)) ||
      `API request failed with status ${response.status}`;
    const error = new Error(typeof message === "string" ? message : JSON.stringify(message));
    error.status = response.status;
    error.payload = payload;
    throw error;
  }

  return payload;
}

// ── 1. System Health ─────────────────────────────────────────────

export async function checkHealth() {
  const data = await apiFetch("/health", { method: "GET" });
  appState.backendOnline = data?.status === "healthy";
  return data;
}

// ── 2. Dataset Endpoints ─────────────────────────────────────────

export async function uploadDataset(file) {
  const formData = new FormData();
  formData.append("file", file);

  const data = await apiFetch("/api/v1/datasets/upload", {
    method: "POST",
    body: formData,
  });
  appState.currentDatasetId = data.dataset_id;
  appState.sanitizedDatasetId = null;
  appState.datasetMeta = data;
  return data;
}

export async function createSampleDataset() {
  const data = await apiFetch("/api/v1/datasets/sample", {
    method: "POST",
  });
  appState.currentDatasetId = data.dataset_id;
  appState.sanitizedDatasetId = null;
  appState.datasetMeta = data;
  return data;
}

export async function listDatasets() {
  return apiFetch("/api/v1/datasets", { method: "GET" });
}

export async function getDataset(datasetId) {
  return apiFetch(`/api/v1/datasets/${encodeURIComponent(datasetId)}`, { method: "GET" });
}

export async function deleteDataset(datasetId) {
  const res = await apiFetch(`/api/v1/datasets/${encodeURIComponent(datasetId)}`, {
    method: "DELETE",
  });
  if (appState.currentDatasetId === datasetId) {
    appState.currentDatasetId = null;
    appState.sanitizedDatasetId = null;
    appState.datasetMeta = null;
    appState.analysisData = null;
    appState.sanitizationResult = null;
    appState.validationResult = null;
  }
  return res;
}

export async function getDatasetProfile(datasetId) {
  return apiFetch(`/api/v1/datasets/${encodeURIComponent(datasetId)}/profile`, { method: "GET" });
}

export async function getDatasetFindings(datasetId) {
  return apiFetch(`/api/v1/datasets/${encodeURIComponent(datasetId)}/findings`, { method: "GET" });
}

export async function getDatasetRisk(datasetId) {
  return apiFetch(`/api/v1/datasets/${encodeURIComponent(datasetId)}/risk`, { method: "GET" });
}

// ── 3. Scan Endpoints ────────────────────────────────────────────

export async function runScan(datasetId, options = {}) {
  const body = {
    dataset_id: datasetId,
    enable_privacy: options.enable_privacy ?? true,
    enable_secrets: options.enable_secrets ?? true,
    enable_security: options.enable_security ?? true,
    enable_quality: options.enable_quality ?? true,
    enable_ml: options.enable_ml ?? false,
  };
  return apiFetch("/api/v1/scans", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function getScan(datasetId) {
  return apiFetch(`/api/v1/scans/${encodeURIComponent(datasetId)}`, { method: "GET" });
}

// ── 4. Sanitization Endpoints ────────────────────────────────────

export async function applySanitization(datasetId, actions = [], autoApplyPolicy = true) {
  const data = await apiFetch("/api/v1/sanitization/apply", {
    method: "POST",
    body: JSON.stringify({
      dataset_id: datasetId,
      actions,
      auto_apply_policy: autoApplyPolicy,
    }),
  });
  appState.sanitizedDatasetId = data.sanitized_dataset_id;
  appState.sanitizationResult = data;
  return data;
}

export async function validateSanitization(datasetId) {
  const data = await apiFetch(`/api/v1/sanitization/${encodeURIComponent(datasetId)}/validate`, {
    method: "POST",
  });
  appState.validationResult = data;
  return data;
}

// ── 5. Readiness & Workflow Endpoints ────────────────────────────

export async function assessReadiness(datasetId, intendedOperation = null) {
  const op = intendedOperation || appState.modelConfig?.intended_operation || "training";
  return apiFetch("/api/v1/readiness/assess", {
    method: "POST",
    body: JSON.stringify({
      dataset_id: datasetId,
      intended_operation: op,
    }),
  });
}

export async function getReadiness(datasetId) {
  return apiFetch(`/api/v1/readiness/${encodeURIComponent(datasetId)}`, { method: "GET" });
}

export async function getWorkflowStatus(datasetId = null) {
  const path = datasetId
    ? `/api/v1/workflow/${encodeURIComponent(datasetId)}`
    : "/api/v1/workflow";
  const data = await apiFetch(path, { method: "GET" });
  appState.workflowData = data;
  return data;
}

// ── 6. Model Provider & Config Endpoints ─────────────────────────

export async function getModelProviders() {
  const data = await apiFetch("/api/v1/models/providers", { method: "GET" });
  if (data.providers) appState.providers = data.providers;
  if (data.active_config) appState.modelConfig = data.active_config;
  return data;
}

export async function getModelConfig() {
  const data = await apiFetch("/api/v1/models/config", { method: "GET" });
  appState.modelConfig = data;
  return data;
}

export async function saveModelConfig(config) {
  const res = await apiFetch("/api/v1/models/config", {
    method: "POST",
    body: JSON.stringify(config),
  });
  if (res.config) {
    appState.modelConfig = res.config;
  }
  emitAppEvent("zemo:config-updated", { config: appState.modelConfig });
  return res;
}

// ── 7. Full Pipeline Orchestrator & Finding Adapter ──────────────

const CATEGORY_METADATA = {
  email: {
    title: "Personal Email Address Detected",
    displayCategory: "Personal Information",
    group: "privacy",
    impact: "Direct violation of privacy regulations (GDPR, CCPA) if model memorizes and regurgitates user email addresses.",
    remediation: "Apply irreversible email masking (e.g. `a***@example.com`) or synthetic token replacement.",
  },
  phone: {
    title: "Customer Telephone Number in Record",
    displayCategory: "Personal Information",
    group: "privacy",
    impact: "Risk of personal contact number exposure in downstream model completions.",
    remediation: "Mask telephone digits preserving only the trailing 4 digits (`***-***-1234`).",
  },
  address: {
    title: "Physical Street Address Identifier",
    displayCategory: "Personal Information",
    group: "privacy",
    impact: "Location PII can enable re-identification of individuals in training corpora.",
    remediation: "Mask or generalize street address fields prior to model training.",
  },
  personal_id: {
    title: "National / Social Security Identifier (SSN)",
    displayCategory: "Critical PII",
    group: "privacy",
    impact: "Severe compliance and identity theft risk if national IDs enter model weights.",
    remediation: "Mandatory masking or full redaction of government identification numbers.",
  },
  credit_card: {
    title: "Payment Card Number Pattern (PCI)",
    displayCategory: "Financial PII",
    group: "privacy",
    impact: "Critical PCI-DSS violation and financial exposure risk if ingested into AI training.",
    remediation: "Mask card digits immediately (`****-****-****-1234`) or redact field.",
  },
  financial: {
    title: "Sensitive Financial Data Pattern",
    displayCategory: "Financial PII",
    group: "privacy",
    impact: "Confidential financial data could be surfaced in model outputs.",
    remediation: "Mask or generalize financial identifiers.",
  },
  api_key: {
    title: "Exposed API Key / Cloud Secret",
    displayCategory: "Exposed Credentials",
    group: "security",
    impact: "Critical risk of credential leakage and unauthorized cloud resource access if memorized.",
    remediation: "Block or completely redact secret (`[REDACTED]`) and rotate the compromised key.",
  },
  password: {
    title: "Plaintext Password or Credential Assignment",
    displayCategory: "Exposed Credentials",
    group: "security",
    impact: "Cleartext authentication credentials in training data pose immediate security breach risk.",
    remediation: "Redact password value (`[REDACTED]`) and remove credential columns.",
  },
  access_token: {
    title: "Bearer / Authentication Access Token",
    displayCategory: "Exposed Credentials",
    group: "security",
    impact: "Session or bearer tokens can allow unauthorized API impersonation.",
    remediation: "Redact token value (`[REDACTED]`) before dataset export.",
  },
  private_key: {
    title: "Cryptographic Private Key Header (PEM)",
    displayCategory: "Exposed Credentials",
    group: "security",
    impact: "Model memorization could reconstruct private cryptographic key material.",
    remediation: "Redact PEM private key material immediately and revoke affected certificates.",
  },
  db_credential: {
    title: "Database Connection URI with Credentials",
    displayCategory: "Exposed Credentials",
    group: "security",
    impact: "Exposes internal database hostnames and authentication secrets.",
    remediation: "Redact connection string (`[REDACTED]`) and restrict network access.",
  },
  prompt_injection: {
    title: "Adversarial Prompt Injection / Jailbreak Pattern",
    displayCategory: "Prompt Injection",
    group: "security",
    impact: "Training on unfiltered adversarial overrides weakens model alignment and safety guardrails.",
    remediation: "Remove affected record completely from fine-tuning dataset.",
  },
  suspicious_content: {
    title: "Malicious Exploit Payload (SQLi / XSS / Shell)",
    displayCategory: "Exploit Payload",
    group: "security",
    impact: "Infects training corpus with injection attack vectors and unsafe code patterns.",
    remediation: "Remove affected record or sanitize untrusted payload strings.",
  },
  malformed_record: {
    title: "Malformed or Binary Record Payload",
    displayCategory: "Security Anomaly",
    group: "security",
    impact: "Oversized or binary payloads can trigger tokenizer anomalies or context overflow.",
    remediation: "Prune corrupted or oversized records prior to training.",
  },
  missing_value: {
    title: "High Null Ratio on Schema Column",
    displayCategory: "Data Quality",
    group: "quality",
    impact: "High missing value rates can skew loss convergence and degrade fine-tuning quality.",
    remediation: "Impute missing values or drop incomplete records.",
  },
  duplicate_record: {
    title: "Exact Duplicate Records Detected",
    displayCategory: "Data Quality",
    group: "quality",
    impact: "Duplicate rows over-weight specific examples and increase memorization risk.",
    remediation: "Deduplicate dataset records before training.",
  },
  empty_record: {
    title: "Blank / Empty Dataset Rows",
    displayCategory: "Data Quality",
    group: "quality",
    impact: "Empty rows waste training compute and introduce noise.",
    remediation: "Remove blank rows during preprocessing.",
  },
  schema_inconsistency: {
    title: "Mixed Data Types in Column Schema",
    displayCategory: "Data Quality",
    group: "quality",
    impact: "Inconsistent field types may cause parsing or serialization failures.",
    remediation: "Normalize column values to a consistent schema type.",
  },
};

function formatLocation(rawFinding) {
  if (rawFinding.record_index !== null && rawFinding.record_index !== undefined && rawFinding.field_name) {
    return `Record #${Number(rawFinding.record_index) + 1} (field: \`${rawFinding.field_name}\`)`;
  }
  if (rawFinding.field_name) {
    return `Column \`${rawFinding.field_name}\``;
  }
  return rawFinding.location || "Dataset";
}

export function normalizeFinding(f, idx) {
  const rawCat = (f.category || "unknown").toLowerCase();
  const meta = CATEGORY_METADATA[rawCat] || {
    title: `Detected ${rawCat.replace(/_/g, " ")} Issue`,
    displayCategory: rawCat.replace(/_/g, " "),
    group: "quality",
    impact: "May affect downstream AI model safety or data quality.",
    remediation: "Review and remediate finding before training.",
  };

  const shortId = f.finding_id
    ? `FND-${String(idx + 1).padStart(3, "0")} (${f.finding_id.slice(0, 6)})`
    : `FND-${String(idx + 1).padStart(3, "0")}`;

  return {
    id: shortId,
    finding_id: f.finding_id,
    title: meta.title,
    category: meta.displayCategory,
    rawCategory: rawCat,
    group: meta.group,
    severity: (f.severity || "low").toLowerCase(),
    location: formatLocation(f),
    action: (f.recommended_action || "review").toLowerCase(),
    description: f.explanation || meta.title,
    evidence: f.evidence || "Redacted sample",
    impact: meta.impact,
    remediation: meta.remediation,
  };
}

/**
 * Runs the complete backend analysis pipeline on an already-ingested dataset_id:
 * 1. POST /api/v1/scans
 * 2. GET /api/v1/datasets/{id}/findings
 * 3. GET /api/v1/datasets/{id}/risk
 * 4. POST /api/v1/readiness/assess
 * 5. GET /api/v1/workflow/{id}
 */
export async function runFullDatasetAnalysis(datasetId, onStageProgress = null) {
  if (onStageProgress) onStageProgress(35, "Running privacy, secret & adversarial security scans...");
  const scanResult = await runScan(datasetId);

  if (onStageProgress) onStageProgress(65, "Fetching findings & composite risk assessment...");
  const [datasetMeta, findingsResp, riskResp] = await Promise.all([
    getDataset(datasetId),
    getDatasetFindings(datasetId),
    getDatasetRisk(datasetId),
  ]);

  if (onStageProgress) onStageProgress(85, "Evaluating AI training readiness & pipeline state...");
  const readinessReport = await assessReadiness(datasetId);
  const workflowStatus = await getWorkflowStatus(datasetId);

  const normalizedFindings = (findingsResp.findings || []).map((f, idx) => normalizeFinding(f, idx));

  const sensitiveCount = normalizedFindings.filter(f => f.group === "privacy").length;
  const securityCount = normalizedFindings.filter(f => f.group === "security").length;
  const qualityCount = normalizedFindings.filter(f => f.group === "quality").length;

  const summary = findingsResp.summary || {
    total: normalizedFindings.length,
    critical: 0,
    high: 0,
    medium: 0,
    low: 0,
  };

  const overallRiskScore = Math.round(riskResp?.risk?.overall_score ?? 0);
  const riskLevel = String(riskResp?.risk?.risk_level || "LOW").toUpperCase();

  const analysisPayload = {
    metadata: {
      datasetId,
      filename: datasetMeta.original_filename || datasetMeta.filename || "dataset",
      format: String(datasetMeta.format || "csv").toUpperCase(),
      fileSizeBytes: datasetMeta.file_size_bytes || 0,
      totalRecords: datasetMeta.record_count || 0,
      columnCount: datasetMeta.column_count || 0,
      analyzedAt: new Date().toLocaleTimeString(),
      isDemo: false,
    },
    metrics: {
      overallRiskScore,
      riskLevel,
      sensitiveDataFindings: sensitiveCount,
      securityFindings: securityCount,
      dataQualitySignals: qualityCount,
    },
    riskDistribution: {
      critical: summary.critical || 0,
      high: summary.high || 0,
      medium: summary.medium || 0,
      low: (summary.low || 0) + (summary.informational || 0),
      total: summary.total || normalizedFindings.length,
    },
    findings: normalizedFindings,
    rawScan: scanResult,
    rawRisk: riskResp,
    rawReadiness: readinessReport,
    workflow: workflowStatus,
  };

  appState.currentDatasetId = datasetId;
  appState.datasetMeta = datasetMeta;
  appState.analysisData = analysisPayload;
  appState.workflowData = workflowStatus;

  if (onStageProgress) onStageProgress(100, "Inspection complete!");
  emitAppEvent("zemo:analysis-loaded", analysisPayload);

  return analysisPayload;
}
