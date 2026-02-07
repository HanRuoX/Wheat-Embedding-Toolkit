// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use btleplug::api::Peripheral;
use btleplug::api::{Central, CentralEvent, Manager as _, ScanFilter};
use btleplug::platform::{Adapter, Manager};
use futures::stream::StreamExt;
use serialport::available_ports;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::mpsc::{self, TryRecvError};
use std::time::SystemTime;
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
            start_ble_advertisement_scan
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
