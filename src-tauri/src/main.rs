use serde::Serialize;
use serde_json::Value;
use std::{
    io::{BufRead, BufReader},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
};
use tauri::State;

const MAX_LOG_LINES: usize = 100;

#[derive(Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct EngineStatus {
    running: bool,
    pid: Option<u32>,
    camera_index: Option<i64>,
    hands_detected: usize,
    active_gesture: String,
    gesture_confidence: f64,
    gesture_debug_text: String,
    gesture_conflicts: Vec<String>,
    selected_color: String,
    brush_size: u64,
    zoom: f64,
    virtual_camera_status: String,
    last_error: Option<String>,
    recent_log_lines: Vec<String>,
}

#[derive(Default)]
struct EngineProcess {
    child: Mutex<Option<Child>>,
    status: Arc<Mutex<EngineStatus>>,
}

fn project_root() -> Result<PathBuf, String> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(PathBuf::from)
        .ok_or_else(|| "Could not resolve project root".to_string())
}

fn append_log(status: &mut EngineStatus, line: String) {
    status.recent_log_lines.push(line);
    if status.recent_log_lines.len() > MAX_LOG_LINES {
        let excess = status.recent_log_lines.len() - MAX_LOG_LINES;
        status.recent_log_lines.drain(0..excess);
    }
}

fn update_from_event(status: &mut EngineStatus, value: &Value, raw: &str) {
    append_log(status, raw.to_string());
    if let Some(index) = value.get("camera_index").and_then(Value::as_i64) {
        status.camera_index = Some(index);
    }
    if let Some(count) = value.get("hands_detected").and_then(Value::as_u64) {
        status.hands_detected = count as usize;
    }
    if let Some(gesture) = value.get("active_gesture").and_then(Value::as_str) {
        status.active_gesture = gesture.to_string();
    }
    if let Some(confidence) = value.get("confidence").and_then(Value::as_f64) {
        status.gesture_confidence = confidence;
    }
    if let Some(debug_text) = value.get("debug_text").and_then(Value::as_str) {
        status.gesture_debug_text = debug_text.to_string();
    }
    if let Some(conflicts) = value.get("conflicts").and_then(Value::as_array) {
        status.gesture_conflicts = conflicts
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
    }
    if let Some(color) = value.get("selected_color").and_then(Value::as_str) {
        status.selected_color = color.to_string();
    }
    if let Some(brush_size) = value.get("brush_size").and_then(Value::as_u64) {
        status.brush_size = brush_size;
    }
    if let Some(zoom) = value.get("zoom").and_then(Value::as_f64) {
        status.zoom = zoom;
    }
    if let Some(error) = value
        .get("last_error")
        .or_else(|| value.get("message"))
        .and_then(Value::as_str)
    {
        if value.get("event").and_then(Value::as_str) == Some("error")
            || value.get("last_error").is_some()
        {
            status.last_error = Some(error.to_string());
        }
    }
    match value.get("event").and_then(Value::as_str) {
        Some("engine_started") => status.running = true,
        Some("engine_stopped") => {
            status.running = false;
            status.pid = None;
        }
        Some("virtual_camera_status") => {
            if let Some(value) = value.get("status").and_then(Value::as_str) {
                status.virtual_camera_status = value.to_string();
            }
        }
        _ => {}
    }
}

fn spawn_stdout_reader(stdout: std::process::ChildStdout, status: Arc<Mutex<EngineStatus>>) {
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(mut snapshot) = status.lock() {
                match serde_json::from_str::<Value>(&line) {
                    Ok(value) => update_from_event(&mut snapshot, &value, &line),
                    Err(_) => append_log(&mut snapshot, format!("stdout: {line}")),
                }
            }
        }
    });
}

fn spawn_stderr_reader(stderr: std::process::ChildStderr, status: Arc<Mutex<EngineStatus>>) {
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if let Ok(mut snapshot) = status.lock() {
                append_log(&mut snapshot, format!("stderr: {line}"));
            }
        }
    });
}

#[tauri::command]
fn start_engine(
    state: State<EngineProcess>,
    camera_index: i64,
    preview_enabled: bool,
    virtual_camera_enabled: bool,
    draw_landmarks: bool,
    debug_overlay: bool,
    brush_size: u64,
) -> Result<(), String> {
    let mut child_slot = state.child.lock().map_err(|_| "Engine lock poisoned")?;
    if let Some(child) = child_slot.as_mut() {
        if child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_none()
        {
            return Ok(());
        }
    }
    *child_slot = None;

    let script = project_root()?.join("engine").join("paintcam_engine.py");
    let mut command = Command::new("python3");
    command
        .arg(script)
        .arg("--camera-index")
        .arg(camera_index.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if !preview_enabled {
        command.arg("--no-preview");
    }
    if !virtual_camera_enabled {
        command.arg("--no-virtual-camera");
    }
    if draw_landmarks {
        command.arg("--draw-landmarks");
    }
    if debug_overlay {
        command.arg("--debug-overlay");
    }
    command.arg("--brush-size").arg(brush_size.to_string());
    let mut child = command
        .spawn()
        .map_err(|error| format!("Failed to start Python engine: {error}"))?;
    let pid = child.id();
    let stdout = child
        .stdout
        .take()
        .ok_or("Could not capture engine stdout")?;
    let stderr = child
        .stderr
        .take()
        .ok_or("Could not capture engine stderr")?;

    {
        let mut snapshot = state.status.lock().map_err(|_| "Status lock poisoned")?;
        *snapshot = EngineStatus {
            running: true,
            pid: Some(pid),
            camera_index: Some(camera_index),
            active_gesture: "none".to_string(),
            selected_color: "#f67834".to_string(),
            brush_size,
            zoom: 1.0,
            virtual_camera_status: if virtual_camera_enabled {
                "starting"
            } else {
                "disabled"
            }
            .to_string(),
            ..EngineStatus::default()
        };
    }
    spawn_stdout_reader(stdout, Arc::clone(&state.status));
    spawn_stderr_reader(stderr, Arc::clone(&state.status));
    *child_slot = Some(child);
    Ok(())
}

#[tauri::command]
fn stop_engine(state: State<EngineProcess>) -> Result<(), String> {
    let mut child_slot = state.child.lock().map_err(|_| "Engine lock poisoned")?;
    if let Some(child) = child_slot.as_mut() {
        if child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_none()
        {
            child.kill().map_err(|error| error.to_string())?;
        }
        let _ = child.wait();
    }
    *child_slot = None;
    let mut snapshot = state.status.lock().map_err(|_| "Status lock poisoned")?;
    snapshot.running = false;
    snapshot.pid = None;
    snapshot.hands_detected = 0;
    if snapshot.virtual_camera_status == "active" {
        snapshot.virtual_camera_status = "stopped".to_string();
    }
    append_log(&mut snapshot, "Engine stopped by user".to_string());
    Ok(())
}

#[tauri::command]
fn engine_status(state: State<EngineProcess>) -> Result<EngineStatus, String> {
    let mut child_slot = state.child.lock().map_err(|_| "Engine lock poisoned")?;
    if let Some(child) = child_slot.as_mut() {
        if let Some(exit) = child.try_wait().map_err(|error| error.to_string())? {
            *child_slot = None;
            let mut snapshot = state.status.lock().map_err(|_| "Status lock poisoned")?;
            snapshot.running = false;
            snapshot.pid = None;
            append_log(&mut snapshot, format!("Engine process exited with {exit}"));
        }
    }
    state
        .status
        .lock()
        .map(|snapshot| snapshot.clone())
        .map_err(|_| "Status lock poisoned".to_string())
}

fn main() {
    tauri::Builder::default()
        .manage(EngineProcess::default())
        .invoke_handler(tauri::generate_handler![
            start_engine,
            stop_engine,
            engine_status
        ])
        .run(tauri::generate_context!())
        .expect("error while running PaintCam");
}
