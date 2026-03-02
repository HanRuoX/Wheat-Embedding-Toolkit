use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn is_lfs_pointer_file(path: &Path) -> bool {
    let Ok(content) = fs::read(path) else {
        return false;
    };

    let prefix_len = content.len().min(128);
    let prefix = &content[..prefix_len];
    std::str::from_utf8(prefix)
        .map(|text| text.starts_with("version https://git-lfs.github.com/spec/v1"))
        .unwrap_or(false)
}

fn copy_file_if_exists(source: &Path, target: &Path) {
    if !source.exists() {
        return;
    }

    if is_lfs_pointer_file(source) {
        panic!(
            "source binary is still a Git LFS pointer: {}. Run `git lfs pull` before building.",
            source.display()
        );
    }

    if let Some(parent) = target.parent() {
        let _ = fs::create_dir_all(parent);
    }

    fs::copy(source, target).unwrap_or_else(|error| {
        panic!(
            "failed to copy {} to {}: {error}",
            source.display(),
            target.display()
        )
    });
}

fn sidecar_file_name(binary_name: &str, target_triple: &str, target_os: &str) -> String {
    if target_os == "windows" {
        format!("{binary_name}-{target_triple}.exe")
    } else {
        format!("{binary_name}-{target_triple}")
    }
}

fn stage_macos_binaries(
    manifest_dir: &Path,
    target_arch: &str,
    target_triple: &str,
    staged_bin_dir: &Path,
) {
    let source_dir = match target_arch {
        "aarch64" => manifest_dir.join("bin").join("macos-arm64"),
        "x86_64" => manifest_dir.join("bin").join("macos-x64"),
        _ => return,
    };

    println!("cargo:rerun-if-changed={}", source_dir.display());

    copy_file_if_exists(
        &source_dir.join("ffmpeg"),
        &staged_bin_dir.join(sidecar_file_name("ffmpeg", target_triple, "macos")),
    );
    copy_file_if_exists(
        &source_dir.join("ffprobe"),
        &staged_bin_dir.join(sidecar_file_name("ffprobe", target_triple, "macos")),
    );
}

fn stage_windows_binaries(manifest_dir: &Path, target_triple: &str, staged_bin_dir: &Path) {
    let source_dir = manifest_dir.join("bin");
    println!("cargo:rerun-if-changed={}", source_dir.display());

    if let Ok(entries) = fs::read_dir(&source_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let Some(file_name) = path.file_name() else {
                continue;
            };
            let file_name = file_name.to_string_lossy();
            let lower_name = file_name.to_ascii_lowercase();

            if lower_name.ends_with(".dll") {
                copy_file_if_exists(&path, &staged_bin_dir.join(file_name.as_ref()));
                continue;
            }

            if lower_name == "ffmpeg.exe" {
                copy_file_if_exists(
                    &path,
                    &staged_bin_dir.join(sidecar_file_name("ffmpeg", target_triple, "windows")),
                );
                continue;
            }

            if lower_name == "ffprobe.exe" {
                copy_file_if_exists(
                    &path,
                    &staged_bin_dir.join(sidecar_file_name("ffprobe", target_triple, "windows")),
                );
            }
        }
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_triple = env::var("TARGET").unwrap_or_default();
    let staged_bin_dir = manifest_dir.join("binaries");

    if staged_bin_dir.exists() {
        fs::remove_dir_all(&staged_bin_dir).unwrap_or_else(|error| {
            panic!(
                "failed to reset staged bin directory {}: {error}",
                staged_bin_dir.display()
            )
        });
    }
    fs::create_dir_all(&staged_bin_dir).unwrap_or_else(|error| {
        panic!(
            "failed to create staged bin directory {}: {error}",
            staged_bin_dir.display()
        )
    });

    match target_os.as_str() {
        "macos" => stage_macos_binaries(&manifest_dir, &target_arch, &target_triple, &staged_bin_dir),
        "windows" => stage_windows_binaries(&manifest_dir, &target_triple, &staged_bin_dir),
        _ => {}
    }

    println!("cargo:rerun-if-changed={}", manifest_dir.join("bin").display());
    tauri_build::build()
}
