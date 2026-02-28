// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use btleplug::api::Peripheral;
use btleplug::api::{Central, CentralEvent, Manager as _, ScanFilter};
use btleplug::platform::{Adapter, Manager};
use futures::stream::StreamExt;
use serialport::{available_ports, DataBits, FlowControl, Parity, StopBits};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;
use std::sync::mpsc::{self, Sender, SyncSender, TryRecvError};
use std::sync::Mutex;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant, SystemTime};
use tauri::{Emitter, Listener, Manager as _};

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct BleDevice {
    pub address: String,
    pub local_name: String,
    pub rssi: i16,
    pub manufacturer_data: HashMap<u16, Vec<u8>>,
    pub services: Vec<String>,
    pub service_data: HashMap<String, Vec<u8>>,
    pub adv: Vec<u8>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct FileInfo {
    name: String,
    is_dir: bool,
    is_file: bool,
    len: u64,
    create_time: u64,
}

struct SerialSession {
    command_tx: SyncSender<SerialCommand>,
    stop_tx: Sender<()>,
    closing: Arc<AtomicBool>,
    reader_handle: Option<thread::JoinHandle<()>>,
    writer_handle: Option<thread::JoinHandle<()>>,
}

#[derive(Default)]
struct SerialAssistantState {
    sessions: Mutex<HashMap<String, SerialSession>>,
}

#[derive(serde::Serialize, Clone)]
struct SerialAssistantEvent {
    kind: String,
    text: String,
    hex: String,
}

enum SerialCommand {
    Send {
        data: Vec<u8>,
        response_tx: Sender<Result<usize, String>>,
    },
    SetSignals {
        rts: bool,
        dtr: bool,
        response_tx: Sender<Result<(), String>>,
    },
    Shutdown,
}

impl SerialAssistantEvent {
    fn status(text: impl Into<String>) -> Self {
        Self {
            kind: "status".to_string(),
            text: text.into(),
            hex: String::new(),
        }
    }

    fn error(text: impl Into<String>) -> Self {
        Self {
            kind: "error".to_string(),
            text: text.into(),
            hex: String::new(),
        }
    }

    fn data(bytes: &[u8]) -> Self {
        let hex = bytes
            .iter()
            .map(|b| format!("{b:02X}"))
            .collect::<Vec<_>>()
            .join(" ");
        let text = String::from_utf8_lossy(bytes).to_string();
        Self {
            kind: "data".to_string(),
            text,
            hex,
        }
    }
}

fn parse_data_bits(bits: u8) -> Result<DataBits, String> {
    match bits {
        5 => Ok(DataBits::Five),
        6 => Ok(DataBits::Six),
        7 => Ok(DataBits::Seven),
        8 => Ok(DataBits::Eight),
        _ => Err("data_bits must be one of: 5, 6, 7, 8".to_string()),
    }
}

fn parse_stop_bits(bits: u8) -> Result<StopBits, String> {
    match bits {
        1 => Ok(StopBits::One),
        2 => Ok(StopBits::Two),
        _ => Err("stop_bits must be one of: 1, 2".to_string()),
    }
}

fn parse_parity(parity: &str) -> Result<Parity, String> {
    match parity.to_ascii_lowercase().as_str() {
        "none" => Ok(Parity::None),
        "odd" => Ok(Parity::Odd),
        "even" => Ok(Parity::Even),
        _ => Err("parity must be one of: none, odd, even".to_string()),
    }
}

fn parse_flow_control(flow_control: &str) -> Result<FlowControl, String> {
    match flow_control.to_ascii_lowercase().as_str() {
        "none" => Ok(FlowControl::None),
        "software" => Ok(FlowControl::Software),
        "hardware" => Ok(FlowControl::Hardware),
        _ => Err("flow_control must be one of: none, software, hardware".to_string()),
    }
}

fn close_serial_session(state: &SerialAssistantState, label: &str) -> Result<(), String> {
    let session = {
        let mut sessions = state
            .sessions
            .lock()
            .map_err(|_| "failed to lock serial sessions".to_string())?;
        sessions.remove(label)
    };

    if let Some(session) = session {
        shutdown_serial_session(session, true);
    }

    Ok(())
}

fn shutdown_serial_session(mut session: SerialSession, wait_for_threads: bool) {
    session.closing.store(true, Ordering::SeqCst);
    let _ = session.command_tx.send(SerialCommand::Shutdown);
    let _ = session.stop_tx.send(());

    let writer_handle = session.writer_handle.take();
    let reader_handle = session.reader_handle.take();

    if wait_for_threads {
        if let Some(handle) = writer_handle {
            let _ = handle.join();
        }
        if let Some(handle) = reader_handle {
            let _ = handle.join();
        }
        return;
    }

    thread::spawn(move || {
        if let Some(handle) = writer_handle {
            let _ = handle.join();
        }
        if let Some(handle) = reader_handle {
            let _ = handle.join();
        }
    });
}

fn wait_for_output_drain(
    writer_port: &mut Box<dyn serialport::SerialPort>,
    closing: &AtomicBool,
) -> Result<(), String> {
    let start = Instant::now();
    let max_wait = Duration::from_millis(300);

    loop {
        if closing.load(Ordering::SeqCst) {
            return Err("serial is closing".to_string());
        }

        match writer_port.bytes_to_write() {
            Ok(0) => return Ok(()),
            Ok(_) => {}
            Err(_) => return Ok(()),
        }

        if start.elapsed() >= max_wait {
            return Ok(());
        }

        thread::sleep(Duration::from_millis(2));
    }
}

async fn get_central(manager: &Manager) -> Result<Adapter, String> {
    let adapters = manager
        .adapters()
        .await
        .map_err(|e| format!("failed to get bluetooth adapters: {e}"))?;
    adapters
        .into_iter()
        .next()
        .ok_or_else(|| "no bluetooth adapter found".to_string())
}

#[tauri::command]
async fn start_ble_advertisement_scan(window: tauri::Window) -> Result<(), String> {
    let (tx, rx) = mpsc::channel();

    let manager = Manager::new()
        .await
        .map_err(|e| format!("failed to create bluetooth manager: {e}"))?;

    // get the first bluetooth adapter
    // connect to the adapter
    let central = get_central(&manager).await?;

    // Each adapter has an event stream, we fetch via events(),
    // simplifying the type, this will return what is essentially a
    // Future<Result<Stream<Item=CentralEvent>>>.
    let mut events = central
        .events()
        .await
        .map_err(|e| format!("failed to subscribe bluetooth events: {e}"))?;

    // start scanning for devices
    central
        .start_scan(ScanFilter::default())
        .await
        .map_err(|e| format!("failed to start bluetooth scan: {e}"))?;

    let listen = window.listen("stop_ble_advertisement_scan", move |_event| {
        let _ = tx.send(());
    });
    let mut listener_active = true;

    // Print based on whatever the event receiver outputs. Note that the event
    // receiver blocks, so in a real program, this should be run in its own
    // thread (not task, as this library does not yet use async channels).
    while let Some(event) = events.next().await {
        match rx.try_recv() {
            Ok(_) | Err(TryRecvError::Disconnected) => {
                window.unlisten(listen);
                listener_active = false;
                break;
            }
            Err(TryRecvError::Empty) => {}
        }
        match event {
            CentralEvent::DeviceConnected(_id) => {}
            CentralEvent::DeviceDisconnected(_id) => {}
            CentralEvent::DeviceDiscovered(id)
            | CentralEvent::ManufacturerDataAdvertisement { id, .. }
            | CentralEvent::ServiceDataAdvertisement { id, .. }
            | CentralEvent::ServicesAdvertisement { id, .. } => {
                let peripheral = central.peripheral(&id).await;
                match peripheral {
                    Ok(peripheral) => {
                        let device = match peripheral.properties().await {
                            Ok(Some(device)) => device,
                            _ => continue,
                        };
                        let addr = id.to_string();
                        let mr = BleDevice {
                            address: addr,
                            local_name: device.local_name.unwrap_or(String::from("")),
                            rssi: device.rssi.unwrap_or(0),
                            manufacturer_data: device.manufacturer_data,
                            services: device.services.iter().map(|x| x.to_string()).collect(),
                            service_data: device
                                .service_data
                                .iter()
                                .map(|(x, y)| (x.to_string(), y.clone()))
                                .collect(),
                            adv: device
                                .service_data
                                .iter()
                                .flat_map(|x| x.1.clone())
                                .collect(),
                        };

                        if let Ok(payload) = serde_json::to_string(&mr) {
                            let _ = window.emit("ble_advertisement_scan_event", payload);
                        }
                    }
                    Err(_) => {}
                }
            }
            _ => {}
        }

        // thread::sleep(Duration::from_millis(100));
    }

    if listener_active {
        window.unlisten(listen);
    }
    Ok(())
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
fn get_serial_port_list() -> Vec<String> {
    match available_ports() {
        Ok(port_info_list) => port_info_list
            .iter()
            .map(|x| x.port_name.to_string())
            .collect(),
        Err(_) => Vec::new(),
    }
}

#[tauri::command]
fn serial_assistant_open(
    window: tauri::Window,
    state: tauri::State<SerialAssistantState>,
    port: String,
    baud_rate: u32,
    data_bits: u8,
    stop_bits: u8,
    parity: String,
    flow_control: String,
) -> Result<(), String> {
    if port.is_empty() {
        return Err("port is required".to_string());
    }

    close_serial_session(&state, window.label())?;

    let data_bits = parse_data_bits(data_bits)?;
    let stop_bits = parse_stop_bits(stop_bits)?;
    let parity = parse_parity(&parity)?;
    let flow_control = parse_flow_control(&flow_control)?;

    let serial = serialport::new(&port, baud_rate)
        .data_bits(data_bits)
        .stop_bits(stop_bits)
        .parity(parity)
        .flow_control(flow_control)
        .timeout(Duration::from_millis(100))
        .open()
        .map_err(|e| format!("failed to open serial port: {e}"))?;

    let mut reader_port = serial
        .try_clone()
        .map_err(|e| format!("failed to clone serial port: {e}"))?;
    let mut writer_port = serial;
    let (command_tx, command_rx) = mpsc::sync_channel::<SerialCommand>(1);
    let (stop_tx, stop_rx) = mpsc::channel();
    let closing = Arc::new(AtomicBool::new(false));
    let writer_closing = Arc::clone(&closing);
    let cloned_window = window.clone();
    let port_for_reader = port.clone();

    let reader_handle = thread::spawn(move || {
        let mut buffer = [0u8; 4096];
        loop {
            match stop_rx.try_recv() {
                Ok(_) | Err(TryRecvError::Disconnected) => break,
                Err(TryRecvError::Empty) => {}
            }

            match reader_port.read(&mut buffer) {
                Ok(size) if size > 0 => {
                    let payload = SerialAssistantEvent::data(&buffer[..size]);
                    let _ = cloned_window.emit("serial_assistant_event", payload);
                }
                Ok(_) => {}
                Err(err) if err.kind() == std::io::ErrorKind::TimedOut => {}
                Err(err) => {
                    let payload = SerialAssistantEvent::error(format!(
                        "serial read failed ({port_for_reader}): {err}"
                    ));
                    let _ = cloned_window.emit("serial_assistant_event", payload);
                    break;
                }
            }
        }
    });

    let writer_handle = thread::spawn(move || {
        while let Ok(command) = command_rx.recv() {
            match command {
                SerialCommand::Send { data, response_tx } => {
                    let result = writer_port
                        .write_all(&data)
                        .map_err(|e| format!("failed to write serial data: {e}"))
                        .and_then(|_| wait_for_output_drain(&mut writer_port, writer_closing.as_ref()))
                        .map(|_| data.len())
                        .map_err(|e| e.to_string());
                    let _ = response_tx.send(result);
                }
                SerialCommand::SetSignals {
                    rts,
                    dtr,
                    response_tx,
                } => {
                    let result = writer_port
                        .write_request_to_send(rts)
                        .map_err(|e| format!("failed to set RTS: {e}"))
                        .and_then(|_| {
                            writer_port
                                .write_data_terminal_ready(dtr)
                                .map_err(|e| format!("failed to set DTR: {e}"))
                        });
                    let _ = response_tx.send(result);
                }
                SerialCommand::Shutdown => break,
            }
        }
    });

    {
        let mut sessions = state
            .sessions
            .lock()
            .map_err(|_| "failed to lock serial sessions".to_string())?;
        sessions.insert(
            window.label().to_string(),
            SerialSession {
                command_tx,
                stop_tx,
                closing,
                reader_handle: Some(reader_handle),
                writer_handle: Some(writer_handle),
            },
        );
    }

    let _ = window.emit(
        "serial_assistant_event",
        SerialAssistantEvent::status(format!("serial connected: {port} @ {baud_rate}")),
    );
    Ok(())
}

#[tauri::command]
fn serial_assistant_set_signals(
    window: tauri::Window,
    state: tauri::State<SerialAssistantState>,
    rts: bool,
    dtr: bool,
) -> Result<(), String> {
    let command_tx = {
        let sessions = state
            .sessions
            .lock()
            .map_err(|_| "failed to lock serial sessions".to_string())?;
        let session = sessions
            .get(window.label())
            .ok_or_else(|| "serial is not connected".to_string())?;
        session.command_tx.clone()
    };

    let (response_tx, response_rx) = mpsc::channel();
    command_tx
        .send(SerialCommand::SetSignals {
            rts,
            dtr,
            response_tx,
        })
        .map_err(|_| "serial writer is unavailable".to_string())?;

    response_rx
        .recv()
        .map_err(|_| "serial writer did not respond".to_string())?
}

#[tauri::command]
fn serial_assistant_send(
    window: tauri::Window,
    state: tauri::State<SerialAssistantState>,
    data: Vec<u8>,
) -> Result<usize, String> {
    if data.is_empty() {
        return Ok(0);
    }

    let command_tx = {
        let sessions = state
            .sessions
            .lock()
            .map_err(|_| "failed to lock serial sessions".to_string())?;
        let session = sessions
            .get(window.label())
            .ok_or_else(|| "serial is not connected".to_string())?;
        session.command_tx.clone()
    };

    let (response_tx, response_rx) = mpsc::channel();
    command_tx
        .send(SerialCommand::Send { data, response_tx })
        .map_err(|_| "serial writer is unavailable".to_string())?;

    response_rx
        .recv()
        .map_err(|_| "serial writer did not respond".to_string())?
}

#[tauri::command]
fn serial_assistant_close(
    window: tauri::Window,
    state: tauri::State<SerialAssistantState>,
) -> Result<(), String> {
    let session = {
        let mut sessions = state
            .sessions
            .lock()
            .map_err(|_| "failed to lock serial sessions".to_string())?;
        sessions.remove(window.label())
    };
    if let Some(session) = session {
        shutdown_serial_session(session, false);
    }
    let _ = window.emit(
        "serial_assistant_event",
        SerialAssistantEvent::status("serial disconnected"),
    );
    Ok(())
}

#[tauri::command]
fn serial_assistant_is_open(
    window: tauri::Window,
    state: tauri::State<SerialAssistantState>,
) -> Result<bool, String> {
    let sessions = state
        .sessions
        .lock()
        .map_err(|_| "failed to lock serial sessions".to_string())?;
    Ok(sessions.contains_key(window.label()))
}

#[tauri::command]
fn get_current_dir(app_handle: tauri::AppHandle) -> String {
    let path = app_handle.path().app_data_dir().unwrap_or_else(|_| {
        let mut fallback = env::temp_dir();
        fallback.push("wheat-esp-tools");
        fallback
    });
    path.display().to_string()
}

#[tauri::command]
fn open_file_in_explorer(path: &str) {
    #[cfg(target_os = "windows")]
    {
        let file_path = format!(r#"{}"#, path);
        let _ = Command::new("explorer")
            .arg("/select,")
            .arg(file_path)
            .status();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg("-R").arg(path).status();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("xdg-open")
            .arg(Path::new(path).parent().unwrap_or(Path::new(path)))
            .status();
    }
}

#[tauri::command]
fn open_directory_in_explorer(path: &str) {
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("explorer").arg(path).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = Command::new("open").arg(path).spawn();
    }
    #[cfg(target_os = "linux")]
    {
        let _ = Command::new("xdg-open").arg(path).spawn();
    }
}

#[tauri::command]
fn get_file_info(path: &str) -> FileInfo {
    let name = Path::new(path)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(path)
        .to_string();

    match fs::metadata(path) {
        Ok(metadata) => {
            let create_time = metadata
                .created()
                .or_else(|_| metadata.modified())
                .ok()
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());

            FileInfo {
                name,
                is_dir: metadata.is_dir(),
                is_file: metadata.is_file(),
                len: metadata.len(),
                create_time,
            }
        }
        Err(_) => FileInfo {
            name,
            is_dir: false,
            is_file: false,
            len: 0,
            create_time: 0,
        },
    }
}

fn main() {
    tauri::Builder::default()
        .manage(SerialAssistantState::default())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_notification::init())
        .setup(|app| {
            let _ = app.get_webview_window("main");
            let app_dir = app.path().app_data_dir().unwrap_or_else(|_| {
                let mut fallback = env::temp_dir();
                fallback.push("wheat-esp-tools");
                fallback
            });
            if !app_dir.exists() {
                let _ = fs::create_dir_all(&app_dir);
            }

            for item in ["firmware", "partitions"].iter() {
                let dir = app_dir.join(item);
                if !dir.exists() {
                    let _ = fs::create_dir_all(&dir);
                }
            }

            let chip_list = app_dir.join("chip.list.json");
            if !chip_list.exists() {
                let data = "[\"ESP32\",\"ESP32C2\",\"ESP32C3\",\"ESP32C6\",\"ESP32S2\",\"ESP32S3\",\"ESP32H2\",\"ESP8266\"]";
                let _ = fs::write(&chip_list, data);
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            get_serial_port_list,
            get_current_dir,
            open_file_in_explorer,
            open_directory_in_explorer,
            get_file_info,
            start_ble_advertisement_scan,
            serial_assistant_open,
            serial_assistant_send,
            serial_assistant_close,
            serial_assistant_is_open,
            serial_assistant_set_signals
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
