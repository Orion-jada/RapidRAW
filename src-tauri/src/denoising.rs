use crate::app_settings::{AppSettings, load_settings};
use crate::app_state::AppState;
use crate::file_management::parse_virtual_path;
use crate::formats::is_raw_file;
use crate::image_loader::{load_base_image_from_bytes, raw_preprocessing_amounts};
use crate::image_processing::{apply_cpu_default_raw_processing, remove_raw_artifacts_and_enhance};
use base64::{Engine as _, engine::general_purpose};
use image::{DynamicImage, GenericImageView, ImageFormat, Rgb32FImage};
use rayon::prelude::*;
use std::fs;
use std::io::Cursor;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering as AtomicOrdering;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};

#[tauri::command]
pub async fn apply_denoising(
    path: String,
    intensity: f32,
    method: String,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let (source_path, _) = parse_virtual_path(&path);
    let path_str = source_path.to_string_lossy().to_string();

    // A run started after this one (say, after the modal was closed mid-run and opened again)
    // owns the result; this one must not replace it or the previews on screen.
    let generation = state
        .denoise_generation
        .fetch_add(1, AtomicOrdering::SeqCst)
        + 1;
    let latest_generation = state.denoise_generation.clone();

    let mut ai_session = None;
    let mut bayer_session = None;
    if method == "ai_bayer" && is_raw_file(&path_str) {
        let session = crate::ai_processing::get_or_init_raw_bayer_denoise_model(
            &app_handle,
            &state.ai_state,
            &state.ai_init_lock,
        )
        .await
        .map_err(|e| e.to_string())?;
        bayer_session = Some(session);
    } else if method == "ai" || method == "ai_bayer" {
        let session = if is_raw_file(&path_str) {
            crate::ai_processing::get_or_init_raw_denoise_model(
                &app_handle,
                &state.ai_state,
                &state.ai_init_lock,
            )
            .await
        } else {
            crate::ai_processing::get_or_init_denoise_model(
                &app_handle,
                &state.ai_state,
                &state.ai_init_lock,
            )
            .await
        }
        .map_err(|e| e.to_string())?;
        ai_session = Some(session);
    }

    let denoise_result_handle = state.denoise_result.clone();

    tokio::task::spawn_blocking(move || {
        let result = denoise_image(
            path_str.clone(),
            intensity,
            method,
            app_handle.clone(),
            ai_session,
            bayer_session,
            true,
        );
        // Stored before the previews are sent, so Save always gets the image they show.
        let mut stored = denoise_result_handle.lock().unwrap();
        if latest_generation.load(AtomicOrdering::SeqCst) != generation {
            log::info!(
                "Dropping the denoise of '{}': a newer run replaced it",
                path_str
            );
            return;
        }
        match result {
            Ok((image, previews)) => {
                *stored = Some(image);
                drop(stored);
                if let Err(e) = app_handle.emit("denoise-complete", &previews) {
                    log::error!("Couldn't send the denoised previews: {}", e);
                }
            }
            Err(e) => {
                *stored = None;
                drop(stored);
                log::error!("Denoising failed: {}", e);
                let _ = app_handle.emit("denoise-error", e);
            }
        }
    })
    .await
    .map_err(|e| format!("Denoising task failed: {}", e))
}

#[tauri::command]
pub async fn batch_denoise_images(
    paths: Vec<String>,
    intensity: f32,
    method: String,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<Vec<String>, String> {
    let is_raw_path = |p: &String| is_raw_file(parse_virtual_path(p).0);
    let mut raw_session = None;
    let mut bayer_session = None;
    let mut rgb_session = None;
    if method == "ai_bayer" && paths.iter().any(is_raw_path) {
        let session = crate::ai_processing::get_or_init_raw_bayer_denoise_model(
            &app_handle,
            &state.ai_state,
            &state.ai_init_lock,
        )
        .await
        .map_err(|e| e.to_string())?;
        bayer_session = Some(session);
    }
    if method == "ai" || method == "ai_bayer" {
        if method == "ai" && paths.iter().any(is_raw_path) {
            let session = crate::ai_processing::get_or_init_raw_denoise_model(
                &app_handle,
                &state.ai_state,
                &state.ai_init_lock,
            )
            .await
            .map_err(|e| e.to_string())?;
            raw_session = Some(session);
        }
        if !paths.iter().all(is_raw_path) {
            let session = crate::ai_processing::get_or_init_denoise_model(
                &app_handle,
                &state.ai_state,
                &state.ai_init_lock,
            )
            .await
            .map_err(|e| e.to_string())?;
            rgb_session = Some(session);
        }
    }

    tokio::task::spawn_blocking(move || {
        let mut results = Vec::new();

        for (i, path_str) in paths.iter().enumerate() {
            let _ = app_handle.emit(
                "denoise-batch-progress",
                serde_json::json!({
                    "current": i + 1,
                    "total": paths.len(),
                    "path": path_str
                }),
            );

            let (source_path, _) = crate::file_management::parse_virtual_path(path_str);
            let real_path = source_path.to_string_lossy().to_string();
            let is_raw = crate::formats::is_raw_file(&real_path);

            match crate::denoising::denoise_image(
                real_path.clone(),
                intensity,
                method.clone(),
                app_handle.clone(),
                if is_raw {
                    raw_session.clone()
                } else {
                    rgb_session.clone()
                },
                bayer_session.clone(),
                false,
            ) {
                Ok((image, _)) => match save_denoised(path_str, &image) {
                    Ok(output_path) => results.push(output_path.to_string_lossy().to_string()),
                    Err(e) => {
                        let _ = app_handle.emit(
                            "denoise-error",
                            format!("Failed to save {}: {}", real_path, e),
                        );
                    }
                },
                Err(e) => {
                    let _ = app_handle.emit(
                        "denoise-error",
                        format!("Failed to denoise {}: {}", real_path, e),
                    );
                }
            }
        }

        Ok(results)
    })
    .await
    .map_err(|e| format!("Batch denoising task failed: {}", e))?
}

#[tauri::command]
pub async fn save_denoised_image(
    original_path_str: String,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let denoised_image = state.denoise_result.lock().unwrap().take().ok_or_else(|| {
        "No denoised image found in memory. It might have already been saved or cleared."
            .to_string()
    })?;

    let output_path = save_denoised(&original_path_str, &denoised_image)?;
    Ok(output_path.to_string_lossy().to_string())
}

/// Saves a denoised image next to its source, with the source's metadata and edits: a linear
/// DNG for RAWs, which edits like the RAW it came from, and a PNG for other images.
fn save_denoised(path: &str, image: &DynamicImage) -> Result<PathBuf, String> {
    let (source_path, source_sidecar_path) = parse_virtual_path(path);
    let parent_dir = source_path
        .parent()
        .ok_or_else(|| "Could not determine parent directory.".to_string())?;
    let stem = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("denoised");
    let source_str = source_path.to_string_lossy().to_string();

    let output_path = if is_raw_file(&source_path) {
        let output_path = parent_dir.join(format!("{}_Denoised.dng", stem));
        let metadata = fs::read(&source_path)
            .ok()
            .and_then(|bytes| crate::exif_processing::read_raw_metadata(&bytes));
        let mut preview = image.resize(1024, 1024, image::imageops::FilterType::Triangle);
        apply_cpu_default_raw_processing(&mut preview);
        let pixels = match image.as_rgb32f() {
            Some(pixels) => std::borrow::Cow::Borrowed(pixels),
            None => std::borrow::Cow::Owned(image.to_rgb32f()),
        };
        crate::linear_dng::write(
            &output_path,
            &pixels,
            metadata.as_ref(),
            crate::as_shot::as_shot(path),
            &preview,
        )
        .map_err(|e| format!("Failed to save image: {}", e))?;
        output_path
    } else {
        let output_path = parent_dir.join(format!("{}_Denoised.png", stem));
        DynamicImage::ImageRgb8(image.to_rgb8())
            .save(&output_path)
            .map_err(|e| format!("Failed to save image: {}", e))?;
        output_path
    };

    let _ = crate::exif_processing::write_rrexif_sidecar(&source_str, &output_path);

    if source_sidecar_path.exists()
        && let Some(output_path_str) = output_path.to_str()
    {
        let (_, dest_sidecar_path) = parse_virtual_path(output_path_str);
        if let Err(e) = fs::copy(&source_sidecar_path, &dest_sidecar_path) {
            log::warn!("Failed to copy sidecar file for denoised image: {}", e);
        }
    }

    Ok(output_path)
}

fn run_bm3d(
    rgb_img: &Rgb32FImage,
    intensity: f32,
    app_handle: &AppHandle,
) -> Result<DynamicImage, String> {
    let denoised = crate::bm3d::denoise(rgb_img, intensity, &|progress| {
        let _ = app_handle.emit(
            "denoise-progress",
            format!("Denoising… {:.0}%", progress * 100.0),
        );
    });
    Ok(DynamicImage::ImageRgb32F(denoised))
}

/// Develops a RAW file with RawNIND Bayer run on its sensor data before demosaicing. Fails for
/// sensors that aren't Bayer (X-Trans, Foveon, linear DNG), leaving the caller to fall back.
fn develop_with_bayer_denoise(
    file_bytes: &[u8],
    settings: &AppSettings,
    intensity: f32,
    session: &Mutex<ort::session::Session>,
    app_handle: &AppHandle,
) -> Result<DynamicImage, String> {
    let mut denoise = |raw: &mut rawler::RawImage| {
        crate::ai_processing::run_rawnind_bayer_denoise(raw, intensity, session, |done| {
            let _ = app_handle.emit(
                "denoise-progress",
                format!("Denoising… {:.0}%", done * 100.0),
            );
        })
    };
    let developed = panic::catch_unwind(AssertUnwindSafe(|| {
        crate::raw_processing::develop_raw_image_with(
            file_bytes,
            settings.raw_highlight_compression.unwrap_or(2.5),
            settings.linear_raw_mode.clone(),
            &mut denoise,
        )
    }))
    .map_err(|_| "the RAW decoder panicked".to_string())?
    .map_err(|e| e.to_string())?;
    Ok(DynamicImage::ImageRgb32F(developed.into_rgb32f()))
}

fn denoise_image(
    path_str: String,
    intensity: f32,
    method: String,
    app_handle: AppHandle,
    ai_session: Option<Arc<Mutex<ort::session::Session>>>,
    bayer_session: Option<Arc<Mutex<ort::session::Session>>>,
    with_previews: bool,
) -> Result<(DynamicImage, serde_json::Value), String> {
    let path = Path::new(&path_str);
    if !path.exists() {
        return Err("File not found".to_string());
    }

    let started = std::time::Instant::now();
    let is_raw = is_raw_file(&path_str);
    let settings = load_settings(app_handle.clone()).unwrap_or_default();

    let _ = app_handle.emit("denoise-progress", "Loading image...");

    let file_bytes = fs::read(path).map_err(|e| e.to_string())?;

    let mut original_settings = settings.clone();
    if method == "raw9" {
        original_settings.use_apple_raw9 = Some(false);
    }
    let rawnind = (method == "ai" || method == "ai_bayer") && is_raw;
    if rawnind {
        // RawNIND is trained on noise straight out of demosaicing: skip Apple's develop.
        original_settings.use_apple_raw9 = Some(false);
    }
    if is_raw {
        // Denoised RAWs are saved as linear DNGs, which get the raw pre-pass (chroma smoothing,
        // sharpening) when they're opened, like any RAW. Denoise the image from before it.
        original_settings.raw_preprocessing_color_nr = Some(0.0);
        original_settings.raw_preprocessing_sharpening = Some(0.0);
    }

    let dynamic_img =
        load_base_image_from_bytes(&file_bytes, &path_str, false, &original_settings, None)
            .map_err(|e| e.to_string())?;

    let rgb_img_for_denoiser = dynamic_img.to_rgb32f();

    let bayer_denoised = match bayer_session {
        Some(session) if method == "ai_bayer" && is_raw => {
            let _ = app_handle.emit("denoise-progress", "Denoising (AI RawNIND Bayer)...");
            match develop_with_bayer_denoise(
                &file_bytes,
                &original_settings,
                intensity,
                &session,
                &app_handle,
            ) {
                Ok(image) => Some(image),
                Err(e) => {
                    log::warn!(
                        "RawNIND Bayer can't denoise '{}', using RawNIND instead: {}",
                        path_str,
                        e
                    );
                    None
                }
            }
        }
        _ => None,
    };

    let out_dynamic = if let Some(out) = bayer_denoised {
        out
    } else if method == "ai" || method == "ai_bayer" {
        let session_arc = match ai_session {
            Some(session) => session,
            // RawNIND Bayer only handles Bayer sensors; others fall back to RawNIND.
            None if rawnind => {
                let state = app_handle.state::<AppState>();
                tauri::async_runtime::block_on(crate::ai_processing::get_or_init_raw_denoise_model(
                    &app_handle,
                    &state.ai_state,
                    &state.ai_init_lock,
                ))
                .map_err(|e| e.to_string())?
            }
            None => return Err("AI Session not provided".to_string()),
        };
        if rawnind {
            let _ = app_handle.emit("denoise-progress", "Denoising (AI RawNIND)...");
            let denoised = crate::ai_processing::run_rawnind_denoise(
                &rgb_img_for_denoiser,
                intensity,
                &session_arc,
                |done| {
                    let _ = app_handle.emit(
                        "denoise-progress",
                        format!("Denoising… {:.0}%", done * 100.0),
                    );
                },
            )
            .map_err(|e| e.to_string())?;
            DynamicImage::ImageRgb32F(denoised)
        } else {
            // Strength mixes NIND's result with the original.
            let mut denoised = crate::ai_processing::run_ai_denoise(
                &rgb_img_for_denoiser,
                &session_arc,
                &|progress| {
                    let _ = app_handle.emit(
                        "denoise-progress",
                        format!("Denoising… {:.0}%", progress * 100.0),
                    );
                },
            )
            .map_err(|e| e.to_string())?
            .into_rgb32f();
            let strength = intensity.clamp(0.0, 1.0);
            if strength < 1.0 {
                denoised
                    .par_iter_mut()
                    .zip(rgb_img_for_denoiser.par_iter())
                    .for_each(|(d, &o)| *d = o + strength * (*d - o));
            }
            DynamicImage::ImageRgb32F(denoised)
        }
    } else if method == "raw9" {
        if !is_raw {
            return Err("Apple RAW 9 denoising only works on RAW files.".to_string());
        }
        let _ = app_handle.emit("denoise-progress", "Developing with Apple RAW 9...");
        crate::apple_raw::denoise_raw9(&file_bytes, &path_str, intensity)
            .map_err(|e| e.to_string())?
    } else {
        run_bm3d(&rgb_img_for_denoiser, intensity, &app_handle)?
    };

    if !with_previews {
        log::info!(
            "Denoised '{}' ({}) in {:.1?}",
            path_str,
            method,
            started.elapsed()
        );
        return Ok((out_dynamic, serde_json::Value::Null));
    }

    let _ = app_handle.emit("denoise-progress", "Finalizing data...");
    let _ = app_handle.emit("denoise-progress", "Generating previews...");

    let (width, height) = out_dynamic.dimensions();
    let (new_width, new_height) = if width > height {
        if width > 4000 {
            (4000, (4000.0 * height as f32 / width as f32).round() as u32)
        } else {
            (width, height)
        }
    } else {
        if height > 4000 {
            ((4000.0 * width as f32 / height as f32).round() as u32, 4000)
        } else {
            (width, height)
        }
    };
    let encode_preview = |image: DynamicImage| -> Result<String, String> {
        let image = if is_raw {
            as_first_opened(image, &settings)
        } else {
            image
        };
        let image = if new_width != width {
            image.resize(new_width, new_height, image::imageops::FilterType::Lanczos3)
        } else {
            image
        };
        let mut buf = Cursor::new(Vec::new());
        image
            .to_rgb8()
            .write_to(&mut buf, ImageFormat::Png)
            .map_err(|e| format!("Failed to encode preview: {}", e))?;
        Ok(format!(
            "data:image/png;base64,{}",
            general_purpose::STANDARD.encode(buf.get_ref())
        ))
    };
    let data_url_denoised = encode_preview(out_dynamic.clone())?;
    let data_url_orig = encode_preview(DynamicImage::ImageRgb32F(rgb_img_for_denoiser))?;

    log::info!(
        "Denoised '{}' ({}) in {:.1?}, with {:.1} MB of previews",
        path_str,
        method,
        started.elapsed(),
        (data_url_denoised.len() + data_url_orig.len()) as f64 / 1e6
    );
    let previews = serde_json::json!({
        "denoised": data_url_denoised,
        "original": data_url_orig
    });
    Ok((out_dynamic, previews))
}

/// A developed RAW (linear, before the raw pre-pass) as it looks when first opened in the editor:
/// with the raw pre-pass and the default look.
fn as_first_opened(mut image: DynamicImage, settings: &AppSettings) -> DynamicImage {
    let (color_nr, sharpening) = raw_preprocessing_amounts(settings);
    if color_nr > 0.0 || sharpening > 0.0 {
        remove_raw_artifacts_and_enhance(&mut image, color_nr, sharpening);
    }
    apply_cpu_default_raw_processing(&mut image);
    image
}
