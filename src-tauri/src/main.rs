use serde::Serialize;
use serde_json::Value;
use std::{
    env,
    ffi::OsStr,
    fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
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
    resolved_python: Option<String>,
    camera_index: Option<i64>,
    camera_open: bool,
    measured_fps: f64,
    frame_count: u64,
    last_frame_time: Option<f64>,
    hands_detected: usize,
    active_gesture: String,
    gesture_confidence: f64,
    gesture_debug_text: String,
    gesture_conflicts: Vec<String>,
    selected_color: String,
    brush_size: u64,
    zoom: f64,
    drawing_enabled: bool,
    canvas_dirty: bool,
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

fn usable_python(path: &Path) -> bool {
    fs::metadata(path)
        .map(|metadata| metadata.is_file())
        .unwrap_or(false)
}

fn find_on_path(program: &OsStr, path: Option<&OsStr>) -> Option<PathBuf> {
    env::split_paths(path?)
        .map(|directory| directory.join(program))
        .find(|candidate| usable_python(candidate))
}

fn resolve_python_from(
    explicit: Option<&str>,
    root: &Path,
    path: Option<&OsStr>,
    fallbacks: &[&Path],
) -> Result<PathBuf, String> {
    if let Some(value) = explicit.map(str::trim).filter(|value| !value.is_empty()) {
        let candidate = PathBuf::from(value);
        let resolved = if candidate.components().count() == 1 {
            find_on_path(candidate.as_os_str(), path).unwrap_or(candidate)
        } else {
            candidate
        };
        if usable_python(&resolved) {
            return Ok(resolved);
        }
        return Err(format!(
            "Configured Python executable is not a usable file: {}",
            resolved.display()
        ));
    }
    let venv_python = root.join(".venv").join("bin").join("python");
    if usable_python(&venv_python) {
        return Ok(venv_python);
    }
    if let Some(python) = find_on_path(OsStr::new("python3"), path) {
        return Ok(python);
    }
    if let Some(python) = fallbacks.iter().find(|candidate| usable_python(candidate)) {
        return Ok((*python).to_path_buf());
    }
    Err(
        "No usable Python interpreter found. Create .venv, put python3 on PATH, or configure an explicit Python path."
            .to_string(),
    )
}

fn resolve_python(explicit: Option<&str>) -> Result<PathBuf, String> {
    let root = project_root()?;
    let fallbacks = [
        Path::new("/usr/bin/python3"),
        Path::new("/usr/local/bin/python3"),
    ];
    resolve_python_from(explicit, &root, env::var_os("PATH").as_deref(), &fallbacks)
}

fn display_path(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
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
    if let Some(open) = value.get("camera_open").and_then(Value::as_bool) {
        status.camera_open = open;
    }
    if let Some(fps) = value.get("measured_fps").and_then(Value::as_f64) {
        status.measured_fps = fps;
    }
    if let Some(frame) = value.get("frame_index").and_then(Value::as_u64) {
        status.frame_count = frame;
    }
    if let Some(timestamp) = value.get("last_frame_time").and_then(Value::as_f64) {
        status.last_frame_time = Some(timestamp);
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
    if let Some(enabled) = value.get("drawing_enabled").and_then(Value::as_bool) {
        status.drawing_enabled = enabled;
    }
    if let Some(dirty) = value.get("canvas_dirty").and_then(Value::as_bool) {
        status.canvas_dirty = dirty;
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
            status.camera_open = false;
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
    python_path: Option<String>,
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

    let root = project_root()?;
    let python = resolve_python(python_path.as_deref())?;
    let model = root.join("engine/models/hand_landmarker.task");
    if !model.is_file() {
        let message = format!(
            "Hand Landmarker model is missing at {}. Run: python3 scripts/download-mediapipe-models.py",
            model.display()
        );
        let mut snapshot = state.status.lock().map_err(|_| "Status lock poisoned")?;
        snapshot.resolved_python = Some(display_path(&python));
        snapshot.last_error = Some(message.clone());
        append_log(&mut snapshot, message.clone());
        return Err(message);
    }
    let script = root.join("engine").join("paintcam_engine.py");
    let mut command = Command::new(&python);
    command
        .arg(script)
        .arg("--camera-index")
        .arg(camera_index.to_string())
        .stdin(Stdio::piped())
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
            resolved_python: Some(display_path(&python)),
            camera_index: Some(camera_index),
            active_gesture: "none".to_string(),
            selected_color: "#f67834".to_string(),
            brush_size,
            zoom: 1.0,
            drawing_enabled: true,
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

fn validate_command(value: &Value) -> Result<(), String> {
    let name = value
        .get("command")
        .and_then(Value::as_str)
        .ok_or("Engine command requires a command string")?;
    match name {
        "clear_canvas" | "reset_zoom" => Ok(()),
        "set_drawing_enabled" if value.get("enabled").and_then(Value::as_bool).is_some() => Ok(()),
        "set_brush_size"
            if value
                .get("brush_size")
                .and_then(Value::as_u64)
                .is_some_and(|size| (1..=100).contains(&size)) =>
        {
            Ok(())
        }
        "set_drawing_enabled" => Err("set_drawing_enabled requires boolean enabled".to_string()),
        "set_brush_size" => Err("set_brush_size requires an integer from 1 to 100".to_string()),
        _ => Err(format!("Unsupported engine command: {name}")),
    }
}

#[tauri::command]
fn send_engine_command(state: State<EngineProcess>, command: Value) -> Result<(), String> {
    validate_command(&command)?;
    let mut child_slot = state.child.lock().map_err(|_| "Engine lock poisoned")?;
    let child = child_slot
        .as_mut()
        .ok_or("Engine is not running; start it before sending canvas controls.")?;
    if let Some(exit) = child.try_wait().map_err(|error| error.to_string())? {
        *child_slot = None;
        return Err(format!(
            "Engine is not running; process exited with {exit}."
        ));
    }
    let stdin = child
        .stdin
        .as_mut()
        .ok_or("Engine command channel is unavailable.")?;
    serde_json::to_writer(&mut *stdin, &command).map_err(|error| error.to_string())?;
    stdin.write_all(b"\n").map_err(|error| error.to_string())?;
    stdin.flush().map_err(|error| error.to_string())
}

#[tauri::command]
fn resolve_python_path(python_path: Option<String>) -> Result<String, String> {
    resolve_python(python_path.as_deref()).map(|path| display_path(&path))
}

#[tauri::command]
fn run_engine_doctor(python_path: Option<String>) -> Result<Value, String> {
    run_engine_utility(python_path.as_deref(), "--doctor")
}

#[tauri::command]
fn list_cameras(python_path: Option<String>) -> Result<Value, String> {
    run_engine_utility(python_path.as_deref(), "--list-cameras")
}

fn run_engine_utility(python_path: Option<&str>, argument: &str) -> Result<Value, String> {
    let python = resolve_python(python_path)?;
    let script = project_root()?.join("engine/paintcam_engine.py");
    let output = Command::new(&python)
        .arg(script)
        .arg(argument)
        .output()
        .map_err(|error| {
            format!(
                "Failed to run {} with {}: {error}",
                argument,
                python.display()
            )
        })?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let event = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|value| {
            value.get("event").and_then(Value::as_str)
                == Some(if argument == "--doctor" {
                    "doctor"
                } else {
                    "camera_probe"
                })
        });
    event.ok_or_else(|| {
        let stderr = String::from_utf8_lossy(&output.stderr);
        format!(
            "{} failed using {} (exit {}). {}{}",
            argument,
            display_path(&python),
            output.status,
            stdout,
            stderr
        )
    })
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
    snapshot.camera_open = false;
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
            snapshot.camera_open = false;
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
            engine_status,
            send_engine_command,
            resolve_python_path,
            run_engine_doctor,
            list_cameras
        ])
        .run(tauri::generate_context!())
        .expect("error while running PaintCam");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;

    fn touch(path: &Path) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        File::create(path).unwrap();
    }

    #[test]
    fn resolution_prefers_explicit_then_venv_then_path_then_fallback() {
        let temp = env::temp_dir().join(format!("paintcam-python-test-{}", std::process::id()));
        let root = temp.join("repo");
        let explicit = temp.join("explicit-python");
        let venv = root.join(".venv/bin/python");
        let bin = temp.join("bin");
        let path_python = bin.join("python3");
        let fallback = temp.join("fallback-python");
        for path in [&explicit, &venv, &path_python, &fallback] {
            touch(path);
        }
        assert_eq!(
            resolve_python_from(
                explicit.to_str(),
                &root,
                Some(bin.as_os_str()),
                &[&fallback]
            )
            .unwrap(),
            explicit
        );
        assert_eq!(
            resolve_python_from(None, &root, Some(bin.as_os_str()), &[&fallback]).unwrap(),
            venv
        );
        fs::remove_file(&venv).unwrap();
        assert_eq!(
            resolve_python_from(None, &root, Some(bin.as_os_str()), &[&fallback]).unwrap(),
            path_python
        );
        fs::remove_file(&path_python).unwrap();
        assert_eq!(
            resolve_python_from(None, &root, Some(bin.as_os_str()), &[&fallback]).unwrap(),
            fallback
        );
        fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn invalid_explicit_python_is_an_error_without_falling_through() {
        let root = env::temp_dir();
        assert!(resolve_python_from(
            Some("/definitely/missing/paintcam-python"),
            &root,
            None,
            &[]
        )
        .unwrap_err()
        .contains("Configured Python"));
    }

    #[test]
    fn engine_commands_are_strictly_bounded() {
        assert!(validate_command(&serde_json::json!({"command": "clear_canvas"})).is_ok());
        assert!(validate_command(
            &serde_json::json!({"command": "set_drawing_enabled", "enabled": false})
        )
        .is_ok());
        assert!(validate_command(
            &serde_json::json!({"command": "set_brush_size", "brush_size": 101})
        )
        .is_err());
        assert!(validate_command(&serde_json::json!({"command": "launch_missiles"})).is_err());
    }
}
