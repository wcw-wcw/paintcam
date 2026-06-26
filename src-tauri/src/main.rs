use serde::Serialize;
use std::{
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::Mutex,
};
use tauri::State;

#[derive(Default)]
struct EngineProcess(Mutex<Option<Child>>);

#[derive(Serialize)]
struct EngineStatus {
    running: bool,
    pid: Option<u32>,
}

fn project_root() -> Result<PathBuf, String> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .map(PathBuf::from)
        .ok_or_else(|| "Could not resolve project root".to_string())
}

#[tauri::command]
fn start_engine(state: State<EngineProcess>) -> Result<(), String> {
    let mut slot = state
        .0
        .lock()
        .map_err(|_| "Engine lock poisoned".to_string())?;

    if let Some(child) = slot.as_mut() {
        if child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_none()
        {
            return Ok(());
        }
    }

    let script = project_root()?.join("engine").join("paintcam_engine.py");
    let child = Command::new("python3")
        .arg(script)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("Failed to start Python engine: {error}"))?;

    *slot = Some(child);
    Ok(())
}

#[tauri::command]
fn stop_engine(state: State<EngineProcess>) -> Result<(), String> {
    let mut slot = state
        .0
        .lock()
        .map_err(|_| "Engine lock poisoned".to_string())?;

    if let Some(child) = slot.as_mut() {
        child.kill().map_err(|error| error.to_string())?;
        let _ = child.wait();
    }

    *slot = None;
    Ok(())
}

#[tauri::command]
fn engine_status(state: State<EngineProcess>) -> Result<EngineStatus, String> {
    let mut slot = state
        .0
        .lock()
        .map_err(|_| "Engine lock poisoned".to_string())?;

    if let Some(child) = slot.as_mut() {
        if child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_none()
        {
            return Ok(EngineStatus {
                running: true,
                pid: Some(child.id()),
            });
        }
    }

    *slot = None;
    Ok(EngineStatus {
        running: false,
        pid: None,
    })
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
