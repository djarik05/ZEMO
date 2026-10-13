# ZEMO — AI Data Privacy, Security & Training Safety Platform

> **Frontend Architecture & Integration Guide**  
> Premium AI Data Safety Infrastructure designed for secure dataset inspection, sensitive data anonymization, adversarial security evaluation, and AI training readiness.

---

## 1. Product Overview

Zemo is an enterprise-grade AI data safety platform built to inspect datasets, detect personal information (PII), uncover exposed secrets and prompt injections, apply verifiable sanitization, and certify datasets before they enter model fine-tuning or evaluation pipelines.

```
RAW DATASET (CSV, JSON, JSONL)
       │
       ▼
[01 / PRIVACY SCAN] ──────► Detects PII, Names, Contact Details
       │
       ▼
[02 / SECURITY SCAN] ─────► Identifies API Keys, Credentials, Injections
       │
       ▼
[03 / RISK ASSESSMENT] ───► Calculates Unified Risk Score (0-100)
       │
       ▼
[04 / SANITIZATION] ──────► Masks, Redacts, or Blocks Hazardous Records
       │
       ▼
[05 / REVALIDATION] ──────► Confirms Zero Residual Leaks
       │
       ▼
[06 / TRAINING READINESS] ─► Certified Export to OpenAI, Anthropic, Gemini, Llama, Mistral, Ollama
```

---

## 2. Brand Identity & Design System

The visual identity is inspired by natural Scandinavian materials, warm paper tones, and organic structures:

| Design Token | Color Hex / Value | Semantic Role |
| :--- | :--- | :--- |
| `--bg` | `#F7F6F2` | Warm white primary canvas & background |
| `--surface` | `#FFFFFF` | Elevated cards, panels & modals |
| `--surface-muted` | `#EFEEE7` | Subtle stone grey for headers, tables & chips |
| `--sage` | `#9EAC95` | Muted sage green for nodes, accents & status |
| `--sage-light` | `#DCE2D6` | Soft sage wash for selections & highlights |
| `--olive` | `#4D5837` | Deep olive green for primary CTA buttons & active states |
| `--olive-dark` | `#353E29` | Rich dark olive for hover states & typography |
| `--text` | `#292D25` | Deep charcoal-olive for headings & high-contrast text |
| `--text-muted` | `#73776C` | Muted olive-grey for descriptions & metadata |
| `--border` | `rgba(77, 88, 55, 0.16)` | Understated warm 1px boundaries |
| `--danger` | `#B85C50` | Reserved exclusively for critical security & risk alerts |
| `--warning` | `#B38B4E` | Reserved for medium/high risk warnings |
| `--success` | `#627B52` | Reserved for verified safety & pass states |

---

## 3. Directory Structure

```
zemo/
│
├── index.html                   # Single-page application entry point
├── README.md                    # Technical documentation & backend integration guide
│
├── css/
│   ├── variables.css            # Design tokens, color palette, typography scale, radii, shadows
│   ├── base.css                 # CSS reset, @font-face, focus states, custom scrollbar
│   ├── layout.css               # Container widths, responsive grid system, section spacing
│   ├── components.css           # Navigation, buttons, upload dropzone, model cards, dashboard, drawers
│   ├── animations.css           # Keyframes, pulsing dots, scroll reveals, prefers-reduced-motion
│   └── responsive.css           # Breakpoints for desktop, tablet, and mobile views
│
└── js/
    ├── app.js                   # Application bootstrap orchestrator
    ├── orb.js                   # High-performance Canvas 2D organic AI Intelligence Orb
    ├── navigation.js            # Sticky navbar, smooth anchor navigation, scrollspy, mobile menu
    ├── upload.js                # Client-side file validation, dropzone handlers, simulated inspection
    ├── model-selector.js        # AI provider card selection & dynamic configuration panel
    ├── sample-analysis.js       # Security findings dashboard, metrics animation, search & detail drawer
    ├── workflow.js              # Pipeline node map navigation & training readiness workspace
    ├── interactions.js          # Toast notification system, modal controllers, scroll reveals
    └── mock-service.js          # Typed API contracts & demonstration data fixtures
```

---

## 4. Key Functional Features

1. **Zemo Intelligence Orb (`js/orb.js`)**:
   - Built on HTML5 Canvas 2D with retina display pixel ratio scaling (`devicePixelRatio`).
   - Layered volumetric sphere with ambient aura, rotating dashed orbital rings with planetary markers, and a dynamic 28-node constellation network that draws proximity connections.
   - Responds to mouse movements with gentle 3D parallax tilt on desktop, and autonomous breathing drift on mobile devices.
   - Fully supports `prefers-reduced-motion` with an elegant static render.
   - States: `AWAITING DATASET`, `INSPECTING DATASET`, `ANALYSIS IN PROGRESS`, `VERIFIED / READY`.

2. **Dataset Upload (`js/upload.js`)**:
   - Drag-and-drop file target with custom SVG iconography.
   - Client-side format checks (`.csv`, `.json`, `.jsonl`) and size limits (<50MB).
   - Real-time file metadata presentation (name, formatted size, type).
   - Simulated inspection progress bar with staged parsing feedback.
   - Strictly client-side: no server transmission occurs in demo mode.

3. **Target AI Model Selection (`js/model-selector.js`)**:
   - 6 supported provider categories: OpenAI, Anthropic, Google Gemini, Meta Llama, Mistral AI, Ollama / Local Models.
   - Live synchronization with the configuration panel (model architecture, processing environment, intended operation, privacy enforcement level).

4. **Security Intelligence Dashboard (`js/sample-analysis.js`)**:
   - Clean empty state with "Load Sample Analysis" button.
   - Summary metric counters with animated numerical count-up.
   - Risk distribution segmented bar (Critical, High, Medium, Low).
   - Findings table with real-time text search and severity filtering.
   - Interactive sliding drawer with technical explanations, harmless fictional evidence, and recommended remediation.

5. **Data Safety Pipeline & Training Readiness (`js/workflow.js`)**:
   - 7-node interactive pipeline map explaining the lifecycle from Data Input to AI Readiness.
   - Readiness gauge (78%) with a 6-step compliance checklist.
   - "Prepare Training Workspace" action dialog detailing backend integration boundaries.

---

## 5. Future Python FastAPI Backend Integration

The frontend was specifically engineered to decouple presentation and client state from backend logic. The `js/mock-service.js` file documents the exact endpoints and request/response payloads:

### Expected FastAPI Endpoints:

#### 1. Dataset Ingestion
- **Route:** `POST /api/v1/datasets/upload`
- **Payload:** `multipart/form-data` containing `file: UploadFile`
- **Response:**
  ```json
  {
    "dataset_id": "ds_78201",
    "filename": "customer_dialogues_v2.jsonl",
    "format": "JSONL",
    "record_count": 18420,
    "file_size_bytes": 4210540,
    "status": "validated"
  }
  ```

#### 2. Run Privacy & Security Scan
- **Route:** `POST /api/v1/analysis/scan`
- **Payload:**
  ```json
  {
    "dataset_id": "ds_78201",
    "target_provider": "openai",
    "privacy_level": "strict"
  }
  ```
- **Response:**
  ```json
  {
    "overall_risk_score": 68,
    "risk_level": "HIGH",
    "metrics": {
      "sensitive_data_findings": 24,
      "security_findings": 9,
      "data_quality_signals": 7
    },
    "risk_distribution": {
      "critical": 3,
      "high": 8,
      "medium": 15,
      "low": 14
    },
    "findings": [ ... ]
  }
  ```

#### 3. Execute Sanitization & Masking
- **Route:** `POST /api/v1/sanitization/apply`
- **Payload:**
  ```json
  {
    "dataset_id": "ds_78201",
    "actions": {
      "FND-001": "block",
      "FND-004": "sanitize"
    }
  }
  ```

#### 4. Prepare Training Export
- **Route:** `POST /api/v1/training/prepare`
- **Payload:**
  ```json
  {
    "dataset_id": "ds_78201",
    "target_model": "gpt-4o",
    "export_format": "sft_jsonl"
  }
  ```

---

## 6. How to Run Locally & Over LAN (Same Wi-Fi Network)

Start the unified Zemo server (serves both the frontend SPA and FastAPI backend on port `8000` across `localhost` and your local network):

```bash
# Windows (from project root):
.\start_lan.bat

# macOS / Linux (from project root):
./start_lan.sh

# Or directly via Uvicorn (from zemo_backend/):
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- **On the Host Computer:** Open `http://localhost:8000`
- **From Other Devices on the Same Wi-Fi/LAN:** Open `http://<HOST_LAN_IP>:8000` (displayed automatically by `start_lan.bat`)
- See [`LAN_SETUP.md`](LAN_SETUP.md) for firewall configuration, multi-port setups, and LAN troubleshooting.

---

## 7. License & Compliance
© 2026 Zemo AI Safety Infrastructure. All rights reserved.
