# Zemo — Local Area Network (LAN / Wi-Fi) Setup & Deployment Guide

This guide explains how to run **Zemo** on a central **Host Computer / Main Server** so that multiple client devices (laptops, desktops, tablets, and smartphones) connected to the same Wi-Fi network or LAN can access the application simultaneously through their web browsers.

---

## 1. Architecture Overview

Zemo uses a **Centralized Local-Network Architecture**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HOST COMPUTER / MAIN SERVER                          │
│                   (Binds to 0.0.0.0 : Port 8000)                        │
│                                                                         │
│  • FastAPI + Uvicorn Server (Serves Frontend SPA + REST APIs)           │
│  • Multi-Engine Detection (Privacy, Secrets, Adversarial, Quality, ML)  │
│  • Centralized Storage (zemo_backend/storage/ — Parquet & JSON)         │
└───────────────────▲─────────────────────────────────▲───────────────────┘
                    │ HTTP (/ & /api/v1/*)            │ HTTP (/ & /api/v1/*)
                    │ Same Wi-Fi / LAN                │ Same Wi-Fi / LAN
        ┌───────────┴───────────┐         ┌───────────┴───────────┐
        │   Client Device A     │         │   Client Device B     │
        │  (Laptop / Desktop)   │         │ (Smartphone / Tablet) │
        │ http://<HOST_IP>:8000 │         │ http://<HOST_IP>:8000 │
        └───────────────────────┘         └───────────────────────┘
```

- **Unified Single-Port Serving (Default, Port `8000`):** FastAPI serves the HTML/CSS/JS/WOFF2 frontend assets at `/` and the backend REST API at `/api/v1/*` and `/health` over a single port (`8000`).
- **Dual-Port Mode (Optional, Ports `3000` + `8000`):** If you run a separate frontend development server on port `3000` (`5500`, `8080`, etc.), the frontend API client (`js/api.js`) dynamically resolves the backend URL using `window.location.hostname` (`http://<HOST_IP>:8000`) instead of `127.0.0.1`.
- **Zero Database Exposure:** Client devices never connect directly to the filesystem or database storage; all operations go through the FastAPI backend.

---

## 2. Prerequisites & Dependencies

On the **Host Computer** only (client devices only need a modern web browser):

1. **Python 3.11+** installed and added to `PATH`.
2. Install backend dependencies (run once from the project root):
   ```powershell
   py -m pip install -r zemo_backend\requirements.txt
   ```

---

## 3. How to Start the Application on the Host Computer

### Option A: Windows One-Command Startup (Recommended)

From the project root directory (`zemo final`), run:

```cmd
.\start_lan.bat
```

Or double-click `start_lan.bat` in File Explorer.

What `start_lan.bat` does automatically:
1. Verifies Python and required packages (`fastapi`, `uvicorn`, `pandas`, `scikit-learn`, `pydantic-settings`).
2. Dynamically detects the host computer's active LAN IPv4 address (prioritizing physical Wi-Fi/Ethernet adapters over VPN tunnels like Cloudflare WARP).
3. Checks whether port `8000` is already listening to prevent duplicate server processes.
4. Prints the exact URLs for the host computer and for other LAN devices.
5. Starts Uvicorn bound to `0.0.0.0:8000`.

### Option B: macOS / Linux Startup

```bash
chmod +x start_lan.sh
./start_lan.sh
```

### Option C: Direct Uvicorn Command

```powershell
cd zemo_backend
py -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 4. Finding the Host Computer's Local IPv4 Address

`start_lan.bat` automatically detects and prints your LAN IPv4 address at startup. You can also check it manually at any time:

### Windows (PowerShell)
```powershell
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -match 'Wi-Fi|Ethernet' -and $_.PrefixOrigin -ne 'WellKnown' } | Select-Object InterfaceAlias, IPAddress
```

### Windows (Command Prompt)
```cmd
ipconfig
```
Look under **Wireless LAN adapter Wi-Fi** (or **Ethernet adapter Ethernet**) for **IPv4 Address** (for example, `172.16.9.195`, `192.168.1.10`, or `10.0.0.15`).

> **Important Note on VPN / Cloudflare WARP:**  
> If you have **Cloudflare WARP** or another VPN active, `ipconfig` will also show a virtual adapter (such as `CloudflareWARP` with IP `172.16.0.2`). **Do not** use the `CloudflareWARP` IP address on other devices — always use the IPv4 address under your **Wi-Fi** or **Ethernet** adapter (e.g. `172.16.9.195`). `start_lan.bat` automatically filters out `172.16.0.2` and selects your real LAN IP.

---

## 5. Accessing Zemo from Another Laptop, Tablet, or Smartphone

1. Connect the client device (another laptop, phone, or tablet) to the **exact same Wi-Fi network / LAN** as the host computer.
2. Open any modern browser (Chrome, Edge, Safari, Firefox) on the client device.
3. Navigate to the **LAN Client URL** printed by `start_lan.bat`:
   - **Application Workspace:** `http://<HOST_LAN_IP>:8000` *(e.g. `http://172.16.9.195:8000`)*
   - **Interactive API Docs (Swagger):** `http://<HOST_LAN_IP>:8000/docs`
   - **Health Check Endpoint:** `http://<HOST_LAN_IP>:8000/health`
4. Meanwhile, on the host computer itself, you can continue to use either:
   - `http://localhost:8000`
   - `http://127.0.0.1:8000`
   - `http://<HOST_LAN_IP>:8000`

---

## 6. Windows Defender Firewall Configuration

By default, **Windows Defender Firewall** may block incoming TCP connections on port `8000` from other devices on your network (especially if Windows classifies your Wi-Fi network profile as **Public**).

### Step 1: Allow Inbound TCP Port `8000` (Run in Administrator PowerShell)

Open PowerShell **as Administrator** on the host computer and run:

```powershell
New-NetFirewallRule -DisplayName "Zemo LAN Server (TCP 8000)" `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort 8000 `
    -Profile Private,Domain
```

*(If your Wi-Fi network is set to the `Public` network profile in Windows Settings and you are on a trusted home/office router, either switch the network profile to **Private** in **Settings → Network & internet → Wi-Fi → Properties → Network profile type → Private**, or include `-Profile Any` in the rule above.)*

### Step 2: Or Configure via Windows Firewall GUI
1. Press `Win + R`, type `wf.msc`, and press **Enter** to open **Windows Defender Firewall with Advanced Security**.
2. Click **Inbound Rules** in the left pane → **New Rule...** in the right pane.
3. Select **Port** → **Next**.
4. Select **TCP**, choose **Specific local ports**, and enter `8000` (or `8000, 3000` if using dual-port mode) → **Next**.
5. Select **Allow the connection** → **Next**.
6. Check **Private** (and **Domain**) → **Next**.
7. Name the rule `Zemo LAN Server (TCP 8000)` and click **Finish**.

---

## 7. Verifying Same-Network Connectivity

To confirm that a client device and the host computer are on the same network and can reach each other:

1. **Check Subnet Match:**
   - Compare the host computer's IP (e.g. `172.16.9.195` with `/19` subnet mask `255.255.224.0`, or `192.168.1.10` with `/24` mask `255.255.255.0`) with the client device's Wi-Fi IP address. Both must belong to the same subnet.
2. **Ping Test (from another laptop):**
   ```bash
   ping <HOST_LAN_IP>
   ```
3. **Port / Health Check Test (from another laptop's terminal or browser):**
   ```bash
   curl http://<HOST_LAN_IP>:8000/health
   ```
   Expected response:
   ```json
   {"status":"healthy","version":"0.1.0","service":"Zemo","storage_path":"..."}
   ```

---

## 8. Configuring Different Ports for Frontend and Backend (Dual-Port Mode)

While Zemo serves both the frontend and backend from port `8000` by default, you can also run the frontend on a separate port (such as `3000`, `5500`, or `8080`):

1. **Automatic Dual-Port Startup:**
   ```cmd
   .\start_lan.bat --dual-port
   ```
   This starts:
   - Frontend static server on `0.0.0.0:3000` (`http://<HOST_LAN_IP>:3000`)
   - FastAPI backend server on `0.0.0.0:8000` (`http://<HOST_LAN_IP>:8000`)
2. **How Frontend-Backend Resolution Works (`js/api.js`):**
   - When accessed at `http://<HOST_LAN_IP>:3000`, `js/api.js` reads `window.location.hostname` (`<HOST_LAN_IP>`) and automatically sends all API requests to `http://<HOST_LAN_IP>:8000`.
3. **Custom Backend Port or Custom API Base URL:**
   - To change the backend port, set `ZEMO_PORT=8001` in `zemo_backend/.env`.
   - If using a separate frontend server with a non-default backend port, you can pass `?api_base=http://<HOST_LAN_IP>:8001` in the browser URL, set `localStorage.setItem("ZEMO_API_BASE", "http://<HOST_LAN_IP>:8001")`, or define `window.ZEMO_API_BASE = "http://<HOST_LAN_IP>:8001"`.

---

## 9. Handling a Changing Local IP Address (DHCP)

Because Wi-Fi routers assign IPv4 addresses dynamically via DHCP, the host computer's LAN IP may change if it disconnects and reconnects to Wi-Fi or restarts:

1. **Re-run `start_lan.bat`:** The startup script detects the current IPv4 address dynamically every time it runs. Also, backend CORS uses an RFC 1918 private-network regex (`ZEMO_ALLOWED_ORIGIN_REGEX`), so you **never** need to edit `.env` when your local IP changes.
2. **Use Hostname / mDNS (Optional):** On networks that support mDNS/Bonjour or local DNS, client devices can often access `http://<COMPUTER_NAME>.local:8000` (allowed automatically by Zemo's CORS regex).
3. **Set a DHCP Reservation on Your Router (Recommended for Permanent Setup):** In your Wi-Fi router's admin settings (under **LAN / DHCP Static Lease / Address Reservation**), bind the host computer's Wi-Fi MAC address to a fixed local IP.

---

## 10. Troubleshooting Common LAN Issues

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **`ERR_CONNECTION_TIMED_OUT` on client device** | Windows Defender Firewall is blocking port `8000`, or Wi-Fi is set to **Public** network profile. | Add the inbound TCP port `8000` rule in Windows Firewall (Section 6) and set your Wi-Fi network profile to **Private**. |
| **`ERR_CONNECTION_TIMED_OUT` on public/campus/hotel/guest Wi-Fi** | **AP / Client Isolation** is enabled on the wireless router/access point, preventing peer-to-peer traffic between Wi-Fi clients. | Connect both devices to a standard private Wi-Fi SSID (not a Guest network), use a mobile hotspot, or disable "AP Isolation / Client Isolation" in router settings. |
| **VPN / Cloudflare WARP interfering with LAN traffic** | VPN software routes traffic through a virtual tunnel (`172.16.0.2`) or blocks local LAN inbound packets. | Ensure you use the **Wi-Fi** adapter IP (e.g., `172.16.9.195`), not `172.16.0.2`. In Cloudflare WARP settings, ensure **Exclude local networks** is enabled, or pause WARP while hosting on LAN. |
| **`ERR_CONNECTION_REFUSED` on client device** | Uvicorn was started with `--host 127.0.0.1` instead of `0.0.0.0`, or the server process was stopped. | Start the server using `.\start_lan.bat` or ensure `--host 0.0.0.0` is used. |
| **Client disconnects after a few minutes of inactivity** | Host computer enters Sleep / Modern Standby, turning off the Wi-Fi adapter. | In Windows **Settings → System → Power & battery → Screen and sleep**, set **When plugged in, put my device to sleep after** to **Never** while hosting. |
| **Different Wi-Fi bands / VLANs cannot see each other** | Some mesh systems or enterprise networks place 2.4 GHz, 5 GHz, or Ethernet on isolated VLAN subnets. | Verify both devices share the same subnet prefix (Section 7). |
