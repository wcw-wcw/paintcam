import { invoke } from "@tauri-apps/api/core";
import "./styles.css";

type EngineStatus = {
  running: boolean;
  pid: number | null;
  cameraIndex: number | null;
  handsDetected: number;
  activeGesture: string;
  gestureConfidence: number;
  gestureDebugText: string;
  gestureConflicts: string[];
  selectedColor: string;
  brushSize: number;
  zoom: number;
  virtualCameraStatus: string;
  lastError: string | null;
  recentLogLines: string[];
};

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Missing app root");

app.innerHTML = `
  <main class="shell">
    <header class="topbar">
      <div><h1>PaintCam</h1><p>Gesture paint and zoom engine monitor.</p></div>
      <span id="status" class="status">Stopped</span>
    </header>

    <section class="panel controls" aria-label="Engine controls">
      <label>Camera index<input id="camera-index" type="number" min="0" step="1" value="0"></label>
      <label class="check"><input id="preview" type="checkbox" checked> Preview enabled</label>
      <label class="check"><input id="virtual-camera" type="checkbox" checked> Virtual camera enabled</label>
      <label class="check"><input id="draw-landmarks" type="checkbox"> Draw landmarks</label>
      <label class="check"><input id="debug-overlay" type="checkbox"> Debug overlay</label>
      <label>Brush size (px)<input id="brush-size" type="number" min="1" max="100" step="1" value="16"></label>
      <div class="actions">
        <button id="start" class="primary" type="button">Start engine</button>
        <button id="stop" type="button">Stop</button>
      </div>
    </section>

    <section class="panel">
      <h2>Engine state</h2>
      <dl class="metrics">
        <div><dt>Process</dt><dd id="pid">—</dd></div>
        <div><dt>Camera</dt><dd id="camera">—</dd></div>
        <div><dt>Hands detected</dt><dd id="hands">0</dd></div>
        <div class="gesture-metric"><dt>Active gesture</dt><dd><strong id="gesture">none</strong><span id="confidence" class="confidence">0%</span></dd></div>
        <div><dt>Gesture detail</dt><dd id="gesture-detail">—</dd></div>
        <div><dt>Selected color</dt><dd><i id="color-chip"></i><span id="color">—</span></dd></div>
        <div><dt>Brush size</dt><dd id="brush">16px</dd></div>
        <div><dt>Zoom</dt><dd id="zoom">1.00×</dd></div>
        <div><dt>Virtual camera</dt><dd id="virtual-status">—</dd></div>
      </dl>
      <div id="conflict-wrap" class="notice" hidden><strong>Gesture conflict/cooldown</strong><span id="conflicts"></span></div>
      <div id="error-wrap" class="error" hidden><strong>Last error</strong><span id="error"></span></div>
    </section>

    <section class="panel logs-panel">
      <h2>Recent engine logs</h2>
      <pre id="logs" aria-live="polite">No engine events yet.</pre>
    </section>
  </main>
`;

const byId = <T extends HTMLElement>(id: string) =>
  document.querySelector<T>(`#${id}`);
const startButton = byId<HTMLButtonElement>("start");
const stopButton = byId<HTMLButtonElement>("stop");
const statusBadge = byId<HTMLSpanElement>("status");

function text(id: string, value: string) {
  const element = byId(id);
  if (element) element.textContent = value;
}

function setStatus(status: EngineStatus) {
  if (statusBadge) {
    statusBadge.textContent = status.running ? "Running" : "Stopped";
    statusBadge.dataset.running = String(status.running);
  }
  text("pid", status.pid == null ? "—" : `#${status.pid}`);
  text("camera", status.cameraIndex == null ? "—" : String(status.cameraIndex));
  text("hands", String(status.handsDetected));
  text("gesture", status.activeGesture || "none");
  text("confidence", `${Math.round((status.gestureConfidence || 0) * 100)}%`);
  text("gesture-detail", status.gestureDebugText || "—");
  text("color", status.selectedColor || "—");
  text("brush", `${status.brushSize || 16}px`);
  text("zoom", `${(status.zoom || 1).toFixed(2)}×`);
  text("virtual-status", status.virtualCameraStatus || "—");
  const chip = byId<HTMLElement>("color-chip");
  if (chip) chip.style.backgroundColor = status.selectedColor || "transparent";
  const errorWrap = byId<HTMLElement>("error-wrap");
  if (errorWrap) errorWrap.hidden = !status.lastError;
  text("error", status.lastError || "");
  const conflictWrap = byId<HTMLElement>("conflict-wrap");
  if (conflictWrap) conflictWrap.hidden = !status.gestureConflicts.length;
  text("conflicts", status.gestureConflicts.join(", "));
  text("logs", status.recentLogLines.length
    ? status.recentLogLines.join("\n")
    : "No engine events yet.");
  if (startButton) startButton.disabled = status.running;
  if (stopButton) stopButton.disabled = !status.running;
}

async function refreshStatus() {
  setStatus(await invoke<EngineStatus>("engine_status"));
}

startButton?.addEventListener("click", async () => {
  const cameraIndex = Number(byId<HTMLInputElement>("camera-index")?.value ?? 0);
  try {
    await invoke("start_engine", {
      cameraIndex,
      previewEnabled: byId<HTMLInputElement>("preview")?.checked ?? true,
      virtualCameraEnabled: byId<HTMLInputElement>("virtual-camera")?.checked ?? true,
      drawLandmarks: byId<HTMLInputElement>("draw-landmarks")?.checked ?? false,
      debugOverlay: byId<HTMLInputElement>("debug-overlay")?.checked ?? false,
      brushSize: Number(byId<HTMLInputElement>("brush-size")?.value ?? 16),
    });
    await refreshStatus();
  } catch (error) {
    console.error(error);
    text("error", String(error));
    const wrap = byId<HTMLElement>("error-wrap");
    if (wrap) wrap.hidden = false;
  }
});

stopButton?.addEventListener("click", async () => {
  await invoke("stop_engine");
  await refreshStatus();
});

refreshStatus().catch(console.error);
window.setInterval(() => refreshStatus().catch(console.error), 1000);
