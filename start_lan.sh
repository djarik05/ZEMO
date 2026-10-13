#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# ZEMO — LAN / Wi-Fi Server Startup Script (macOS / Linux)
# Starts the unified Zemo Frontend + FastAPI Backend server bound to 0.0.0.0
# Optional: pass --dual-port to also start a separate frontend server on 3000
# ============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${ROOT_DIR}/zemo_backend"

if [[ ! -f "${BACKEND_DIR}/app/main.py" ]]; then
  echo "[ERROR] Could not locate Zemo backend at ${BACKEND_DIR}/app/main.py" >&2
  exit 1
fi

# 1. Locate Python executable
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_CMD="${ROOT_DIR}/.venv/bin/python"
elif [[ -x "${BACKEND_DIR}/.venv/bin/python" ]]; then
  PYTHON_CMD="${BACKEND_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "[ERROR] Python 3.11+ was not found on your PATH." >&2
  exit 1
fi

export ZEMO_HOST="${ZEMO_HOST:-0.0.0.0}"
export ZEMO_PORT="${ZEMO_PORT:-8000}"
export ZEMO_FRONTEND_PORT="${ZEMO_FRONTEND_PORT:-3000}"
DUAL_PORT=0
if [[ "${1:-}" == "--dual-port" ]]; then
  DUAL_PORT=1
fi

cd "${BACKEND_DIR}"

# 2. Verify dependencies
if ! "${PYTHON_CMD}" -c "import fastapi, uvicorn, pydantic_settings" >/dev/null 2>&1; then
  echo "[ERROR] Required Python packages are missing." >&2
  echo "Run: ${PYTHON_CMD} -m pip install -r ${BACKEND_DIR}/requirements.txt" >&2
  exit 1
fi

# 3. Dynamically detect LAN IPv4 address
LAN_IP="$("${PYTHON_CMD}" -c "from app.core.config import get_primary_lan_ip; print(get_primary_lan_ip())" 2>/dev/null || echo "127.0.0.1")"

# 4. Prevent duplicate server processes on ZEMO_PORT
if ! "${PYTHON_CMD}" -c "import socket, sys; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(0.5); res = s.connect_ex(('127.0.0.1', int('${ZEMO_PORT}'))); s.close(); sys.exit(1 if res == 0 else 0)"; then
  echo "============================================================================"
  echo " [NOTICE] Port ${ZEMO_PORT} is already in use on this computer."
  echo " Avoiding duplicate server startup."
  echo "============================================================================"
  echo " Host Computer Local URL : http://localhost:${ZEMO_PORT}"
  echo " LAN Client URL          : http://${LAN_IP}:${ZEMO_PORT}"
  echo " Backend API URL         : http://${LAN_IP}:${ZEMO_PORT}/api/v1"
  echo "============================================================================"
  exit 0
fi

echo "============================================================================"
echo " ZEMO — AI DATA PRIVACY, SECURITY & TRAINING SAFETY PLATFORM (LAN MODE)"
echo "============================================================================"
echo " Host Bind Interface     : ${ZEMO_HOST} (All Network Interfaces)"
echo " Host LAN IPv4 Address   : ${LAN_IP}"
if [[ "${DUAL_PORT}" -eq 1 ]]; then
  echo " Architecture Mode       : Dual-Port (Frontend: ${ZEMO_FRONTEND_PORT}, Backend: ${ZEMO_PORT})"
  echo " Host Local Frontend URL : http://localhost:${ZEMO_FRONTEND_PORT}"
  echo " LAN Client Frontend URL : http://${LAN_IP}:${ZEMO_FRONTEND_PORT}"
  echo " LAN Backend API URL     : http://${LAN_IP}:${ZEMO_PORT}/api/v1"
else
  echo " Architecture Mode       : Unified Single-Port (Frontend + Backend on ${ZEMO_PORT})"
  echo " Host Local URL          : http://localhost:${ZEMO_PORT}"
  echo " LAN Client URL (Open)   : http://${LAN_IP}:${ZEMO_PORT}"
  echo " LAN Backend API URL     : http://${LAN_IP}:${ZEMO_PORT}/api/v1"
fi
echo " LAN Health Check        : http://${LAN_IP}:${ZEMO_PORT}/health"
echo " LAN Swagger Docs        : http://${LAN_IP}:${ZEMO_PORT}/docs"
echo "============================================================================"
echo " Press Ctrl+C to stop the server."
echo "============================================================================"
echo ""

FRONTEND_PID=""
cleanup() {
  if [[ -n "${FRONTEND_PID}" ]]; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${DUAL_PORT}" -eq 1 ]]; then
  (cd "${ROOT_DIR}" && "${PYTHON_CMD}" -m http.server "${ZEMO_FRONTEND_PORT}" --bind "${ZEMO_HOST}") &
  FRONTEND_PID=$!
fi

exec "${PYTHON_CMD}" -m uvicorn app.main:app --host "${ZEMO_HOST}" --port "${ZEMO_PORT}"
