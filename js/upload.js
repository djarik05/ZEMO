/**
 * ZEMO DATASET UPLOAD CONTROLLER
 * Client-side file selection, format validation, drag-and-drop feedback,
 * and real FastAPI backend dataset ingestion & multi-stage safety inspection.
 */

import { setOrbState } from "./orb.js";
import { showToast } from "./interactions.js";
import {
  uploadDataset,
  runFullDatasetAnalysis,
  deleteDataset,
  getAppState,
  emitAppEvent,
} from "./api.js";

const VALID_EXTENSIONS = [".csv", ".json", ".jsonl"];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

let currentFile = null;

export function initUpload() {
  const dropzone = document.getElementById("upload-dropzone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("browse-file-btn");
  const selectedBox = document.getElementById("selected-file-box");
  const fileNameEl = document.getElementById("file-name-text");
  const fileMetaEl = document.getElementById("file-meta-text");
  const removeFileBtn = document.getElementById("remove-file-btn");
  const inspectBtn = document.getElementById("inspect-dataset-btn");
  const progressWrapper = document.getElementById("upload-progress-wrapper");
  const progressFill = document.getElementById("progress-fill");
  const progressPercent = document.getElementById("progress-percent");
  const progressStage = document.getElementById("progress-stage");

  if (!dropzone || !fileInput) return;

  // 1. Browse Button & Dropzone Click
  browseBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener("click", () => {
    if (!currentFile) {
      fileInput.click();
    }
  });

  // 2. Drag & Drop Event Handlers
  ["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("drag-active");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("drag-active");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  // 3. Remove File Handler (Deletes from Backend if uploaded)
  removeFileBtn?.addEventListener("click", async (e) => {
    e.stopPropagation();
    const state = getAppState();
    if (state.currentDatasetId) {
      try {
        await deleteDataset(state.currentDatasetId);
      } catch (err) {
        console.warn("[ZEMO] Failed to delete dataset on backend:", err);
      }
    }
    resetUploadState();
    emitAppEvent("zemo:analysis-reset");
  });

  // 4. File Validation & Display
  function handleFileSelected(file) {
    const ext = "." + file.name.split(".").pop().toLowerCase();

    if (!VALID_EXTENSIONS.includes(ext)) {
      showToast(`Unsupported format (${ext}). Please provide a CSV, JSON, or JSONL dataset.`, "warning");
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      showToast("File size exceeds 50MB limit.", "warning");
      return;
    }

    currentFile = file;
    dropzone.classList.add("has-file");

    if (fileNameEl) fileNameEl.textContent = file.name;
    if (fileMetaEl) {
      fileMetaEl.textContent = `${formatBytes(file.size)} • ${ext.toUpperCase().replace(".", "")} Format • Ready to Upload`;
    }

    if (selectedBox) selectedBox.style.display = "flex";
    if (inspectBtn) {
      inspectBtn.style.display = "inline-flex";
      inspectBtn.disabled = false;
      inspectBtn.textContent = "Inspect Dataset";
    }
    if (progressWrapper) {
      progressWrapper.style.display = "none";
      if (progressFill) progressFill.style.width = "0%";
      if (progressPercent) progressPercent.textContent = "0%";
    }

    setOrbState("inspecting");
    setTimeout(() => {
      if (currentFile && !progressWrapper?.style.display.includes("block")) {
        setOrbState("idle");
      }
    }, 1000);

    showToast(`Dataset "${file.name}" selected. Click "Inspect Dataset" to upload & scan.`, "info");
  }

  // 5. Real FastAPI Dataset Upload & Multi-Stage Scan
  inspectBtn?.addEventListener("click", async (e) => {
    e.stopPropagation();
    if (!currentFile) return;

    inspectBtn.disabled = true;
    inspectBtn.textContent = "Uploading & Scanning...";
    if (progressWrapper) progressWrapper.style.display = "block";

    setOrbState("inspecting");

    function updateProgress(pct, stageText) {
      if (progressFill) progressFill.style.width = `${pct}%`;
      if (progressPercent) progressPercent.textContent = `${pct}%`;
      if (progressStage && stageText) progressStage.textContent = stageText;
    }

    try {
      updateProgress(15, "Uploading & ingesting dataset into FastAPI backend...");
      const uploadedMeta = await uploadDataset(currentFile);

      if (fileMetaEl) {
        fileMetaEl.textContent = `${formatBytes(uploadedMeta.file_size_bytes)} • ${String(uploadedMeta.format).toUpperCase()} • ${uploadedMeta.record_count.toLocaleString()} Records (${uploadedMeta.column_count} Cols)`;
      }

      const analysis = await runFullDatasetAnalysis(uploadedMeta.dataset_id, updateProgress);

      setOrbState("idle");
      inspectBtn.textContent = "Inspection Complete";

      const totalFindings = analysis.riskDistribution.total;
      showToast(
        `Dataset "${uploadedMeta.filename}" scanned: ${uploadedMeta.record_count.toLocaleString()} records, ${totalFindings} findings detected.`,
        totalFindings > 0 ? "warning" : "success"
      );

      // Scroll smoothly to Security Intelligence dashboard to show live findings
      const securitySec = document.getElementById("security");
      if (securitySec) {
        setTimeout(() => {
          securitySec.scrollIntoView({ behavior: "smooth" });
        }, 500);
      }
    } catch (err) {
      console.error("[ZEMO] Dataset upload/scan failed:", err);
      setOrbState("idle");
      inspectBtn.disabled = false;
      inspectBtn.textContent = "Retry Inspection";
      updateProgress(0, "Upload failed.");
      if (progressWrapper) progressWrapper.style.display = "none";
      showToast(`Backend Error: ${err.message || "Could not connect to FastAPI server."}`, "warning");
    }
  });

  // Sync upload card if sample dataset is loaded from the Security section
  window.addEventListener("zemo:analysis-loaded", (e) => {
    const data = e.detail;
    if (!data || !data.metadata) return;
    if (!currentFile && fileNameEl && fileMetaEl && selectedBox) {
      dropzone.classList.add("has-file");
      fileNameEl.textContent = data.metadata.filename;
      fileMetaEl.textContent = `${formatBytes(data.metadata.fileSizeBytes)} • ${data.metadata.format} • ${data.metadata.totalRecords.toLocaleString()} Records (Backend Active)`;
      selectedBox.style.display = "flex";
      if (inspectBtn) {
        inspectBtn.style.display = "inline-flex";
        inspectBtn.disabled = true;
        inspectBtn.textContent = "Inspection Complete";
      }
    }
  });

  function resetUploadState() {
    currentFile = null;
    fileInput.value = "";
    dropzone.classList.remove("has-file");
    if (selectedBox) selectedBox.style.display = "none";
    if (inspectBtn) {
      inspectBtn.style.display = "none";
      inspectBtn.disabled = false;
      inspectBtn.textContent = "Inspect Dataset";
    }
    if (progressWrapper) {
      progressWrapper.style.display = "none";
      if (progressFill) progressFill.style.width = "0%";
      if (progressPercent) progressPercent.textContent = "0%";
    }
    setOrbState("idle");
    showToast("Dataset removed from workspace.", "info");
  }

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }
}
