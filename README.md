# Zemo — AI/ML Model Training & Experimentation Platform

A modern, cinematic, dark editorial frontend for **Zemo**, an AI/ML model training and experimentation platform.

### Design Philosophy & Brand Identity

- **Calm Organic & Dark Editorial Identity**:
  - **Calm Organic Palette**: Background `#F7F6F2`, Sage `#9EAC95`, Deep Olive `#4D5837` for a calm, editorial, organic machine learning experience.
  - **Dark Editorial Palette**: Deep charcoal backgrounds (`#171716`, `#20201E`, `#292824`) paired with warm orange accents (`#FF603D`) and warm off-white typography (`#F1EEE7`).
  - **1-Click Theme Switcher**: Header toggle enabling instantaneous switching between **Calm Organic** and **Dark Editorial** themes with state persistence.
- **Sophisticated Typography**: Pairing of **Instrument Serif** for expressive editorial headlines with **Inter** and **JetBrains Mono** for clean UI controls and technical metrics.
- **Scroll-Driven Motion**: Smooth scroll navigation, active section indicators, intersection-observed reveals, and deliberate horizontal micro-interactions.
- **Frontend-Only Prototype with Full Honesty**: Complete browser-side interactivity with client-side parsing, dynamic dataset previews, restriction controls, interactive CSS-based visualizations, dynamic hyperparameter forms, and demonstration training queues clearly marked `[DEMO]`.

---

## Website Structure & Core Components

The application features five core product sections along with **Zemo Core Intelligence** and the **Training Assistant**:

1. **Section 01 — Dataset Upload (`#upload`)**
   - Drag-and-drop file upload zone supporting `CSV`, `XLSX`, `JSON`, `JSONL`, and `Parquet`.
   - Client-side parser generating instant row/column count, file size, and data type inference (`numeric`, `categorical`, `boolean`, `text`).
   - Sticky header preview table with scrollable data and data-type badges.
   - Status indicator for **Zemo Dataset Analysis Engine Readiness**.
   - 1-Click sample dataset loaders (`Housing Sample`, `Customer Churn`) for instant demonstration.

2. **Section 02 — Data Governance / Restrictions (`#restrictions`)**
   - **Column Restrictions**: Tri-state toggles (`Include`, `Exclude`, `Review`) for every detected dataset column.
   - **Sensitive Data Detection**: Configurable cards for Email, Phone, Personal Names, Personal IDs, Financial Data, and Custom Fields.
   - **Data Quality Rules**: Toggles for Missing Values, Duplicates, Invalid Values, Outliers, and High Cardinality.
   - **Training Constraints**: Target variable selection, feature exclusion preview, missing value handling strategy, categorical encoding preferences (`Label`, `One-Hot`, `Automatic`), and train/test splits (`70/30`, `80/20`, `80/10/10`).
   - Real-time **Configuration Summary** tracking all active rules.

3. **Section 03 — Output Preview & Zemo Core Intelligence (`#output`)**
   - **Zemo Core Intelligence**:
     - Dedicated interface element representing Zemo's own **Deep Learning Meta-Model (v3.4)**.
     - Autonomous planning intelligence that interprets natural language experiment requirements and synthesizes training blueprints for downstream candidate models.
     - Visual orchestration architecture graph illustrating the relationship between Dataset/Restrictions ➔ Zemo Core Intelligence ➔ Synthesized Blueprint ➔ Downstream User Models.
     - Requirement input area with quick preset triggers ("High Precision Classification", "Outlier-Resilient Regression", "Low Latency Edge Model", etc.).
     - Live **Synthesized Training Plan Preview** with a 1-click "Apply Blueprint to Model Lab" action.
     - Clear architectural separation disclaimer clarifying that Core Intelligence is the planning meta-model, not the user model being trained.
   - **Dataset Overview Metrics**: Real-time cards calculating Total Rows, Total Columns, Numeric Columns, Categorical Columns, and Missing Values from the loaded preview with clear distinction badges (`Local Preview Data` vs `Illustrative Placeholder`).
   - **Interactive Visualizations**:
     - *Distribution*: Dynamic CSS bar chart displaying histogram bins for numeric columns.
     - *Categories*: Horizontal bar chart displaying top frequency categories.
     - *Missing Values*: Per-column data completeness indicators.
     - *Correlations*: Neutral labeled illustrative placeholder.
   - **Feature Pipeline Blueprint & Table**: Explicit specification for feature extraction, threshold filtering, encoding, and train/test allocation.
   - **Downstream Model Selection Matrix**: Multi-task filtering by task and architecture family across traditional and deep learning models.

4. **Section 04 — Model Lab / Training Waiting Room (`#training`)**
   - **Experiment Configuration**: Experiment name, dataset binding, target column, selected model, encoding method, split ratio, random seed, and adaptive evaluation metrics.
   - **Dynamic Hyperparameter Panel**: Automatically updates visible sliders and controls based on model type.
   - **Multi-Status Training Queue**: Live queue interface with demonstration cards (`Draft`, `Ready`, `Queued`, `Training`, `Completed`, `Failed`).
   - **Training Progress Preview**: Epoch progress bar, duration counters, and demonstration log ticker marked with `[DEMO]`.
   - **Experiment Evaluation & Result Placeholder**: Benchmarks overview with confusion matrix and residual error distribution placeholders.

5. **Section 05 — AI Training Studio & Training Assistant (`#ai-training`)**
   - **Mode A: Pre-trained Model Adaptation**:
     - Supported model registry: DistilBERT-Base (66.4M), RoBERTa-Base (125M), BERT-Mini (11.3M), T5-Small (60M), MobileNet-V3-Small (2.5M), ResNet-50-v2 (25.6M).
     - Strict honesty: Unsupported models (Llama-3-70B, GPT-4) are blocked with explicit educational alerts explaining why fine-tuning is unsupported.
     - Supported adaptation methods: Classification-Head Training (freeze backbone, low VRAM), Full Fine-Tuning, and LoRA / PEFT.
     - Hyperparameter controls: Epochs, Learning Rate, Batch Size, Validation Split, and VRAM/compute estimation.
   - **Mode B: Train from Scratch (Cold Start)**:
     - Strict isolation: Random weight initialization ($W_0 \sim \text{Glorot Uniform}$, He Normal, LeCun, Orthogonal) with zero pre-trained weights or checkpoints loaded.
     - Supported neural architectures: Deep MLP with LayerNorm, 1D-ConvNet, BiLSTM, Mini-Transformer (2-layer self-attention), and TabResNet.
     - Model capacity sizing: Micro (~45K), Small (~220K), Medium (~1.1M), and Large (~4.8M).
     - Task validation: Automatically checks dataset compatibility against Binary Classification, Multi-Class, Regression, and Anomaly Detection.
     - Clear architectural scope callout: Explains the distinction between training a small task-specific model from scratch vs. training a multi-billion parameter foundation LLM from scratch.
   - **Training Assistant Chatbox**:
     - Conversational assistant embedded in the AI Training Studio.
     - Suggested prompt chips:
       1. *"Train a sentiment classifier using my uploaded dataset."*
       2. *"Help me fine-tune a supported pre-trained model."*
       3. *"Build a new text classification model from scratch using only my dataset."*
       4. *"Analyze my dataset and recommend a suitable training approach."*
       5. *"Explain my model's evaluation metrics."*
     - Interactive configuration cards in chat with 1-click "Apply Configuration" to update UI forms.
     - Session message history with "Clear History" capability.
   - **Zemo Governance & Policy Scan Gateway**:
     - Real-time privacy verification linked directly to Section 02 Restrictions rules.
     - Flags or blocks datasets with sensitive unmasked PII (Email, Phone, Names, Financial, Personal IDs).
   - **Pre-Flight Review & Explicit Confirmation**:
     - Structured modal review step before any execution.
     - Requires explicit user confirmation button click (`Confirm & Dispatch Job [DEMO]`).
   - **Live Training Monitor [DEMO]**:
     - Animated progress bar, decaying train & validation loss, accuracy/F1 progression.
     - Live terminal logs stream with timestamps and `[DEMO]` tags.
     - Job cancellation support (`Cancel Job`) immediately halts simulation.
     - Generates final evaluation metrics and saved checkpoints table.
   - **Backend Integration Specification**:
     - Complete documentation of REST (`POST /api/v1/training/pretrained`, `POST /api/v1/training/scratch`) and WebSocket (`WSS /api/v1/training/jobs/{job_id}/stream`) data contracts.

---

## How to Run

Because Zemo is built with vanilla HTML5, CSS3, and modern JavaScript, no build step or node installation is required!

### Option 1: Development Server
```bash
node serve.js
```
Then visit `http://localhost:3000` in your browser.

### Option 2: Direct File Open
Simply double-click or open `index.html` in any modern web browser.

---

## Phase 8 — Zemo AI Assistant (Gemini-Powered Chatbot)

Zemo includes an intelligent, beginner-friendly AI assistant powered by Google Gemini (`gemini-2.5-flash`), integrated directly into the platform through a floating liquid-glass widget at the bottom-right corner of the screen.

### Capabilities:
- **Dataset-Aware Intelligence**: Retrieves compact factual summaries of active datasets (rows, columns, numeric/categorical types, missing values, target candidates, recent model metrics) and grounds recommendations in real data without hallucinating statistics.
- **Beginner-Friendly ML Tutor**: Explains machine learning concepts (One-Hot Encoding, StandardScaler vs MinMaxScaler, imputation, overfitting, ROC-AUC, F1-score) using everyday analogies and plain English.
- **Workflow Mentorship**: Guides users through dataset inspection, preprocessing selection, model training, evaluation analysis, and model export.
- **Multi-Turn Conversational Memory**: Preserves context across follow-up queries (e.g. *"Which scaling method should I use?"* followed by *"Why not MinMaxScaler?"*).
- **Security & Privacy**: The Gemini API key remains strictly on the backend and is never exposed in frontend code, client requests, responses, or logs.

### Chatbot Endpoints:
- `POST /api/v1/chatbot/message` — Send user message with optional conversation ID and dataset ID.
- `POST /api/v1/chatbot/clear` — Clear conversation history for a given conversation ID.
- `GET /api/v1/chatbot/status` — Health check verifying Gemini client configuration and active model.


