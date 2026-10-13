/**
 * ZEMO SECURITY ANALYSIS DASHBOARD CONTROLLER
 * Renders live backend scan metrics, risk distribution charts,
 * findings filtering & search, and detail drawer presentation.
 */

import { setOrbState } from "./orb.js";
import { showToast } from "./interactions.js";
import {
  createSampleDataset,
  runFullDatasetAnalysis,
  deleteDataset,
  getAppState,
  emitAppEvent,
} from "./api.js";

let allFindings = [];
let currentFindings = [];
let activeSeverityFilter = "all";
let activeCategoryFilter = "all";
let searchQuery = "";

export function initSampleAnalysis() {
  const loadBtn = document.getElementById("load-sample-btn");
  const resetBtn = document.getElementById("reset-analysis-btn");
  const emptyState = document.getElementById("analysis-empty-state");
  const workspace = document.getElementById("analysis-workspace");
  const tableBody = document.getElementById("findings-table-body");
  const searchInput = document.getElementById("findings-search-input");
  const filterBtns = document.querySelectorAll(".filter-btn[data-severity]");

  // Drawer Elements
  const drawer = document.getElementById("finding-drawer");
  const closeDrawerBtn = document.getElementById("close-drawer-btn");
  const drawerTitle = document.getElementById("drawer-title");
  const drawerCategory = document.getElementById("drawer-category");
  const drawerSeverityBadge = document.getElementById("drawer-severity-badge");
  const drawerLocation = document.getElementById("drawer-location");
  const drawerDescription = document.getElementById("drawer-description");
  const drawerEvidence = document.getElementById("drawer-evidence");
  const drawerImpact = document.getElementById("drawer-impact");
  const drawerRemediation = document.getElementById("drawer-remediation");
  const drawerActionBadge = document.getElementById("drawer-action-badge");

  if (!loadBtn || !workspace) return;

  // 1. Load Sample Analysis Click -> Real Backend Ingestion & Scan
  loadBtn.addEventListener("click", async () => {
    const origHtml = loadBtn.innerHTML;
    loadBtn.disabled = true;
    loadBtn.textContent = "Running Backend Scan...";
    setOrbState("inspecting");

    try {
      const sampleMeta = await createSampleDataset();
      await runFullDatasetAnalysis(sampleMeta.dataset_id);
      setOrbState("idle");
      showToast(
        `Sample dataset "${sampleMeta.filename}" ingested and scanned via FastAPI backend.`,
        "success"
      );
    } catch (err) {
      console.error("[ZEMO] Failed to run sample analysis on backend:", err);
      setOrbState("idle");
      showToast(`Backend Error: ${err.message || "Ensure FastAPI server is running."}`, "warning");
    } finally {
      loadBtn.disabled = false;
      loadBtn.innerHTML = origHtml;
    }
  });

  // 2. Listen for Live Backend Analysis Payload (from upload.js or sample button)
  window.addEventListener("zemo:analysis-loaded", (e) => {
    const data = e.detail;
    if (!data) return;

    allFindings = Array.isArray(data.findings) ? [...data.findings] : [];
    activeSeverityFilter = "all";
    activeCategoryFilter = "all";
    searchQuery = "";
    if (searchInput) searchInput.value = "";

    filterBtns.forEach(b => {
      b.classList.toggle("active", b.getAttribute("data-severity") === "all");
    });

    emptyState.style.display = "none";
    workspace.style.display = "flex";

    updateBanner(data.metadata);
    animateMetrics(data.metrics);
    renderRiskBar(data.riskDistribution);
    updateFilterButtonCounts(data.riskDistribution, allFindings.length);
    applyFilters();
  });

  // 3. Listen for Workspace Reset
  window.addEventListener("zemo:analysis-reset", () => {
    workspace.style.display = "none";
    emptyState.style.display = "block";
    allFindings = [];
    currentFindings = [];
    closeDrawer();
  });

  // 4. Reset Analysis Click
  resetBtn?.addEventListener("click", async () => {
    const state = getAppState();
    if (state.currentDatasetId) {
      try {
        await deleteDataset(state.currentDatasetId);
      } catch (err) {
        console.warn("[ZEMO] Error deleting dataset during reset:", err);
      }
    }
    emitAppEvent("zemo:analysis-reset");
    setOrbState("idle");
    showToast("Analysis workspace reset.", "info");
  });

  // 5. Severity Filter Tabs
  filterBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      filterBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      activeSeverityFilter = btn.getAttribute("data-severity") || "all";
      applyFilters();
    });
  });

  // 6. Live Search Input
  searchInput?.addEventListener("input", (e) => {
    searchQuery = e.target.value.toLowerCase().trim();
    applyFilters();
  });

  // 7. Drawer Dismiss Handlers
  closeDrawerBtn?.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && drawer?.classList.contains("open")) {
      closeDrawer();
    }
  });

  function updateBanner(meta) {
    if (!meta) return;
    const bannerDiv = workspace.querySelector(".demo-notice div");
    if (bannerDiv) {
      bannerDiv.innerHTML = `
        <strong>LIVE BACKEND ANALYSIS:</strong> Showing real-time detection findings for <code>${meta.filename}</code> (${Number(meta.totalRecords || 0).toLocaleString()} records • ID: <code>${meta.datasetId.slice(0, 8)}</code>). All sensitive evidence is automatically masked by Zemo.
      `;
    }
  }

  function updateFilterButtonCounts(dist, totalCount) {
    if (!dist) return;
    filterBtns.forEach(btn => {
      const sev = btn.getAttribute("data-severity");
      if (sev === "all") btn.textContent = `All Findings (${totalCount})`;
      else if (sev === "critical") btn.textContent = `Critical (${dist.critical || 0})`;
      else if (sev === "high") btn.textContent = `High (${dist.high || 0})`;
      else if (sev === "medium") btn.textContent = `Medium (${dist.medium || 0})`;
      else if (sev === "low") btn.textContent = `Low (${dist.low || 0})`;
    });
  }

  function applyFilters() {
    currentFindings = allFindings.filter(f => {
      const matchSeverity = activeSeverityFilter === "all" || f.severity === activeSeverityFilter;
      const matchCategory = activeCategoryFilter === "all" || f.category === activeCategoryFilter;
      const matchSearch =
        searchQuery === "" ||
        f.title.toLowerCase().includes(searchQuery) ||
        f.category.toLowerCase().includes(searchQuery) ||
        f.location.toLowerCase().includes(searchQuery) ||
        f.description.toLowerCase().includes(searchQuery) ||
        f.id.toLowerCase().includes(searchQuery);

      return matchSeverity && matchCategory && matchSearch;
    });

    renderFindingsTable();
  }

  function renderFindingsTable() {
    if (!tableBody) return;
    tableBody.innerHTML = "";

    if (currentFindings.length === 0) {
      tableBody.innerHTML = `
        <tr>
          <td colspan="5" style="text-align: center; padding: 2.5rem; color: var(--text-muted);">
            No findings match the current filter criteria.
          </td>
        </tr>
      `;
      return;
    }

    currentFindings.forEach(finding => {
      const tr = document.createElement("tr");
      tr.setAttribute("data-id", finding.id);

      tr.innerHTML = `
        <td style="font-weight: 500;">
          <div style="color: var(--text);">${finding.title}</div>
          <div style="font-size: 0.75rem; color: var(--text-muted); font-family: var(--font-mono); margin-top: 2px;">${finding.id}</div>
        </td>
        <td style="color: var(--text-muted); font-size: 0.813rem;">${finding.category}</td>
        <td>
          <span class="severity-badge ${finding.severity}">${finding.severity}</span>
        </td>
        <td style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted);">${finding.location}</td>
        <td>
          <span class="action-badge ${finding.action}">${finding.action.toUpperCase()}</span>
        </td>
      `;

      tr.addEventListener("click", () => {
        openFindingDrawer(finding);
      });

      tableBody.appendChild(tr);
    });
  }

  function openFindingDrawer(finding) {
    if (!drawer) return;

    if (drawerTitle) drawerTitle.textContent = finding.title;
    if (drawerCategory) drawerCategory.textContent = finding.category;
    if (drawerLocation) drawerLocation.textContent = finding.location;
    if (drawerDescription) drawerDescription.textContent = finding.description;
    if (drawerEvidence) drawerEvidence.textContent = finding.evidence;
    if (drawerImpact) drawerImpact.textContent = finding.impact;
    if (drawerRemediation) drawerRemediation.textContent = finding.remediation;

    if (drawerSeverityBadge) {
      drawerSeverityBadge.className = `severity-badge ${finding.severity}`;
      drawerSeverityBadge.textContent = finding.severity.toUpperCase();
    }

    if (drawerActionBadge) {
      drawerActionBadge.className = `action-badge ${finding.action}`;
      drawerActionBadge.textContent = finding.action.toUpperCase();
    }

    drawer.classList.add("open");
  }

  function closeDrawer() {
    drawer?.classList.remove("open");
  }

  function animateMetrics(metrics) {
    if (!metrics) return;
    animateCounter("metric-risk-score", metrics.overallRiskScore || 0, 750);
    animateCounter("metric-sensitive-count", metrics.sensitiveDataFindings || 0, 700);
    animateCounter("metric-security-count", metrics.securityFindings || 0, 650);
    animateCounter("metric-quality-count", metrics.dataQualitySignals || 0, 600);

    // Update overall risk level badge on first metric card
    const firstCardBadge = workspace.querySelector(".metrics-row .metric-card:first-child .severity-badge");
    if (firstCardBadge && metrics.riskLevel) {
      const lvl = metrics.riskLevel.toLowerCase();
      firstCardBadge.className = `severity-badge ${lvl}`;
      firstCardBadge.textContent = `${metrics.riskLevel.toUpperCase()} RISK`;
    }
  }

  function animateCounter(elementId, targetVal, durationMs) {
    const el = document.getElementById(elementId);
    if (!el) return;

    if (targetVal === 0) {
      el.textContent = "0";
      return;
    }

    let start = 0;
    const stepTime = 16;
    const totalSteps = Math.max(1, durationMs / stepTime);
    const increment = targetVal / totalSteps;

    const timer = setInterval(() => {
      start += increment;
      if (start >= targetVal) {
        clearInterval(timer);
        el.textContent = String(targetVal);
      } else {
        el.textContent = String(Math.floor(start));
      }
    }, stepTime);
  }

  function renderRiskBar(dist) {
    if (!dist) return;
    const total = Math.max(1, dist.total || 0);
    const segCritical = document.getElementById("risk-seg-critical");
    const segHigh = document.getElementById("risk-seg-high");
    const segMedium = document.getElementById("risk-seg-medium");
    const segLow = document.getElementById("risk-seg-low");

    if (segCritical) {
      segCritical.style.width = `${(dist.critical / total) * 100}%`;
      segCritical.title = `Critical: ${dist.critical}`;
    }
    if (segHigh) {
      segHigh.style.width = `${(dist.high / total) * 100}%`;
      segHigh.title = `High: ${dist.high}`;
    }
    if (segMedium) {
      segMedium.style.width = `${(dist.medium / total) * 100}%`;
      segMedium.title = `Medium: ${dist.medium}`;
    }
    if (segLow) {
      segLow.style.width = `${(dist.low / total) * 100}%`;
      segLow.title = `Low: ${dist.low}`;
    }

    // Update total count header & legend items
    const totalHeaderSpans = workspace.querySelectorAll(".risk-distrib-header span");
    if (totalHeaderSpans.length >= 2) {
      totalHeaderSpans[1].textContent = `${dist.total || 0} Total Detected Incidents`;
    }

    const legendItems = workspace.querySelectorAll(".risk-legend .legend-item");
    if (legendItems.length >= 4) {
      legendItems[0].innerHTML = `<span class="legend-dot critical"></span> Critical (${dist.critical || 0})`;
      legendItems[1].innerHTML = `<span class="legend-dot high"></span> High (${dist.high || 0})`;
      legendItems[2].innerHTML = `<span class="legend-dot medium"></span> Medium (${dist.medium || 0})`;
      legendItems[3].innerHTML = `<span class="legend-dot low"></span> Low (${dist.low || 0})`;
    }
  }
}
