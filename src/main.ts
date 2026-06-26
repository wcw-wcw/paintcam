import { invoke } from "@tauri-apps/api/core";
import "./styles.css";

type EngineStatus = {
  running: boolean;
  pid?: number;
};

const app = document.querySelector<HTMLDivElement>("#app");

if (!app) {
  throw new Error("Missing app root");
}

app.innerHTML = `
  <section class="shell">
    <header class="topbar">
      <div>
        <h1>PaintCam</h1>
        <p>Gesture paint and zoom for a virtual camera feed.</p>
      </div>
      <span id="status" class="status">Stopped</span>
    </header>

    <section class="controls" aria-label="Engine controls">
      <button id="start" class="primary" type="button">Start Camera Engine</button>
      <button id="stop" type="button">Stop</button>
    </section>

    <section class="panel">
      <h2>Current Gestures</h2>
      <dl>
        <div>
          <dt>Choose color</dt>
          <dd>Point at a palette swatch along the bottom of the video.</dd>
        </div>
        <div>
          <dt>Draw</dt>
          <dd>Pinch thumb and index finger above the palette.</dd>
        </div>
        <div>
          <dt>Zoom</dt>
          <dd>Use two visible hands; spreading them zooms in, moving them closer zooms out.</dd>
        </div>
      </dl>
    </section>
  </section>
`;

const startButton = document.querySelector<HTMLButtonElement>("#start");
const stopButton = document.querySelector<HTMLButtonElement>("#stop");
const statusBadge = document.querySelector<HTMLSpanElement>("#status");

function setStatus(status: EngineStatus) {
  if (!statusBadge) return;
  statusBadge.textContent = status.running
    ? `Running${status.pid ? ` #${status.pid}` : ""}`
    : "Stopped";
  statusBadge.dataset.running = String(status.running);
}

async function refreshStatus() {
  const status = await invoke<EngineStatus>("engine_status");
  setStatus(status);
}

startButton?.addEventListener("click", async () => {
  await invoke("start_engine");
  await refreshStatus();
});

stopButton?.addEventListener("click", async () => {
  await invoke("stop_engine");
  await refreshStatus();
});

refreshStatus().catch((error) => {
  console.error(error);
});
