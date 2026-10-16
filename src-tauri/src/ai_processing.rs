use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use image::imageops::{self, FilterType};
use image::{
    DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Rgb, Rgb32FImage, Rgba, RgbaImage,
};
use ndarray::{Array, Array4, IxDyn};
use ort::execution_providers::coreml::CoreMLComputeUnits;
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::Emitter;
use tauri::Manager;
use tokenizers::Tokenizer;
use tokio::sync::Mutex as TokioMutex;

// SAM 2.1 (Hiera Base+). The encoder is re-exported to run as one CoreML partition, with float16
// weights cast to float32 at load. The decoder returns all four candidate masks as 256x256 logits.
const ENCODER_URL: &str = "https://github.com/Orion-jada/RapidRAW/releases/download/sam2.1-v1/sam2.1_hiera_base_plus_encoder_coreml.onnx";
const DECODER_URL: &str = "https://github.com/Orion-jada/RapidRAW/releases/download/sam2.1-v1/sam2.1_hiera_base_plus_decoder.onnx";
const ENCODER_FILENAME: &str = "sam2.1_hiera_base_plus_encoder_coreml.onnx";
const DECODER_FILENAME: &str = "sam2.1_hiera_base_plus_decoder.onnx";
const SAM_INPUT_SIZE: u32 = 1024;
const SAM_MASK_SIZE: usize = 256;
const ENCODER_SHA256: &str = "cda7bc3644a9ceb3ccb6bc333e3352b7155cab58d71fdbe210cadbaf3eb61c65";
const DECODER_SHA256: &str = "9c9142c6ba7e4830748f96e3b3c9c5b895c5221fbc0d91eddb79b25ab1c87161";

// BiRefNet (general, 1024px), re-exported so all of it runs in one CoreML partition; weights
// are stored as float16 and cast back to float32 at load. Outputs probabilities.
const BIREFNET_URL: &str = "https://github.com/Orion-jada/RapidRAW/releases/download/birefnet-v1/birefnet_general_1024_coreml.onnx";
const BIREFNET_FILENAME: &str = "birefnet_general_1024_coreml.onnx";
const BIREFNET_INPUT_SIZE: u32 = 1024;
const BIREFNET_SHA256: &str = "5b243f8c509c05374f0c42c692e9c2dbf4c58ad151fd6531b55bfa8b4f8618e9";

// SkyWater-Seg (SegFormer-B2): background/sky/water/person segmentation trained on photos, so
// reflections in glass and bright windows stay out of the sky mask. Fixed to a 512x512 input so
// CoreML runs the whole graph; weights are stored as float16 and cast back to float32 at load.
const SKY_URL: &str = "https://github.com/Orion-jada/RapidRAW/releases/download/sky-v1/skywater_segformer_b2_512.onnx";
const SKY_FILENAME: &str = "skywater_segformer_b2_512.onnx";
const SKY_SHA256: &str = "4bfe7161e50516f84d7af32589b6d6b527dcd91ee0c892330388a268a9aeb26d";
const SKY_INPUT_SIZE: u32 = 512;
const SKY_NUM_CLASSES: usize = 4;
const SKY_CLASS: usize = 1;

// SigLIP 2 (ViT-B/16, 224px) for natural-language photo search. The vision tower is fixed to a
// 1x3x224x224 input and patched so CoreML runs all of it, with float16 weights cast back to
// float32 at load. The multilingual text tower (int8) and its tokenizer come unchanged from
// onnx-community's export, pinned to a revision.
const SEARCH_VISION_URL: &str = "https://github.com/Orion-jada/RapidRAW/releases/download/search-v1/siglip2_vision_b16_224.onnx";
const SEARCH_VISION_FILENAME: &str = "siglip2_vision_b16_224.onnx";
const SEARCH_VISION_SHA256: &str =
    "2d507cd867c4f4b4f0f315ad5d18237ec2b359b157dfa23750a200cdc9b56197";
const SEARCH_TEXT_URL: &str = "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/ba1f3b0843f24bc5417d38e19c37b287d719b2f4/onnx/text_model_quantized.onnx";
const SEARCH_TEXT_FILENAME: &str = "siglip2_text_b16_int8.onnx";
const SEARCH_TEXT_SHA256: &str = "3a0603d3a00c05a80a6ded4743c16aaac7b1e62cdcc7e362e7ce418659b96400";
const SEARCH_TOKENIZER_URL: &str = "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/ba1f3b0843f24bc5417d38e19c37b287d719b2f4/tokenizer.json";
const SEARCH_TOKENIZER_FILENAME: &str = "siglip2_tokenizer.json";
const SEARCH_TOKENIZER_SHA256: &str =
    "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322";

const CLIP_MODEL_URL: &str =
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/clip_model.onnx?download=true";
const CLIP_MODEL_FILENAME: &str = "clip_model.onnx";
const CLIP_TOKENIZER_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/clip_tokenizer.json?download=true";
const CLIP_TOKENIZER_FILENAME: &str = "clip_tokenizer.json";
const CLIP_MODEL_SHA256: &str = "57879bb1c23cdeb350d23569dd251ed4b740a96d747c529e94a2bb8040ac5d00";

const DENOISE_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/nind_denoise_utnet_684.onnx?download=true";
const DENOISE_FILENAME: &str = "nind_denoise_utnet_684.onnx";
const DENOISE_SHA256: &str = "ee3586279d514df557ff3f7dec6df37fafc51ba5d3a3435b2cc9ac2d9017e7fe";

// RawNIND UtNet2, linear Rec.2020 variant (darktable-ai's ONNX export, fixed 512x512 input).
// Denoises demosaiced raw data and outputs it at an arbitrary learned scale, so every tile is
// gain-matched to its input. Tiles overlap by RAWNIND_OVERLAP and are feathered together.
const RAWNIND_URL: &str = "https://github.com/Orion-jada/RapidRAW/releases/download/rawnind-v1/rawnind_utnet2_linear_512.onnx";
const RAWNIND_FILENAME: &str = "rawnind_utnet2_linear_512.onnx";
const RAWNIND_SHA256: &str = "df957efadcc152c007d5d3b0917bdff9e41c0d4a0efe56584ef30b36393cd181";
const RAWNIND_TILE: usize = 512;
const RAWNIND_OVERLAP: usize = 128;
const RAWNIND_BORDER: usize = 16;

const REC709_TO_REC2020: [[f32; 3]; 3] = [
    [0.627_404, 0.329_282, 0.043_313_6],
    [0.069_097, 0.919_540, 0.011_361_2],
    [0.016_391_6, 0.088_013_2, 0.895_595],
];
const REC2020_TO_REC709: [[f32; 3]; 3] = [
    [1.660_490_3, -0.587_639_1, -0.072_851_6],
    [-0.124_550, 1.132_899_9, -0.008_348],
    [-0.018_151_1, -0.100_578_7, 1.118_73],
];

// 2x super-resolution: Phhofm's 2xPublic_realplksr_dysample_layernorm_real (Apache-2.0), trained
// on photos degraded with blur, noise, JPEG and oversharpening, so it cleans those up while it
// upscales. Retargeted to a 512x512 sRGB tile and rewritten so CoreML runs the whole graph.
const UPSCALE_URL: &str = "https://github.com/Orion-jada/RapidRAW/releases/download/upscale-v1/realplksr_2x_real_512.onnx";
const UPSCALE_FILENAME: &str = "realplksr_2x_real_512.onnx";
const UPSCALE_SHA256: &str = "3fb055da243bfa717212a39e032cf510bed80315c15d72e098adc05b1f7837b4";
const UPSCALE_TILE: usize = 512;
const UPSCALE_OVERLAP: usize = 96;
const UPSCALE_BORDER: usize = 24;

const LAMA_URL: &str =
    "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/lama_fp16.onnx?download=true";
const LAMA_FILENAME: &str = "lama_fp16.onnx";
const LAMA_SHA256: &str = "2d6be6277c400d6f1b91819737f7c3da935e5c63d1b521d393be1196a2bfa82c";

const DEPTH_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/depth_anything_v2_vits.onnx?download=true";
const DEPTH_FILENAME: &str = "depth_anything_v2_vits.onnx";
const DEPTH_INPUT_SIZE: u32 = 518;
const DEPTH_SHA256: &str = "d2b11a11c1d4a12b47608fa65a17ee9a4c605b55ee1730c8e3b526304f2562be";

const NORMAL_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/moge-2-vits-normal-onnx.onnx?download=true";
const NORMAL_FILENAME: &str = "moge-2-vits-normal-onnx.onnx";
const NORMAL_MAX_SIZE: u32 = 768;
const NORMAL_NUM_TOKENS: i64 = 2500;
const NORMAL_SHA256: &str = "24eacb5dc7a2c54c7bc98f7de085ffbed79ad006ea5b664c2c2cdc02ff3a52f0";

pub struct AiModels {
    pub sam_encoder: Mutex<Session>,
    pub sam_decoder: Mutex<Session>,
    pub depth_anything: Mutex<Session>,
}

pub struct ClipModels {
    pub model: Mutex<Session>,
    pub tokenizer: Tokenizer,
}

/// SigLIP 2 image and text towers for natural-language search.
pub struct SearchModels {
    /// Starts on CPU while CoreML compiles the model for the first time, then swaps to CoreML.
    pub vision: Arc<Mutex<Session>>,
    pub text: Mutex<Session>,
    pub tokenizer: Tokenizer,
}

#[derive(Clone)]
pub struct ImageEmbeddings {
    pub path_hash: String,
    pub image_embed: Array<f32, IxDyn>,
    pub high_res_feats_0: Array<f32, IxDyn>,
    pub high_res_feats_1: Array<f32, IxDyn>,
    pub original_size: (u32, u32),
}

#[derive(Clone)]
pub struct CachedDepthMap {
    pub path_hash: String,
    pub depth_image: GrayImage,
    pub original_size: (u32, u32),
}

pub struct AiState {
    pub models: Option<Arc<AiModels>>,
    pub denoise_model: Option<Arc<Mutex<Session>>>,
    pub raw_denoise_model: Option<Arc<Mutex<Session>>>,
    pub upscale_model: Option<Arc<Mutex<Session>>>,
    pub foreground_model: Option<Arc<Mutex<Session>>>,
    pub sky_model: Option<Arc<Mutex<Session>>>,
    pub clip_models: Option<Arc<ClipModels>>,
    pub search_models: Option<Arc<SearchModels>>,
    pub lama_model: Option<Arc<Mutex<Session>>>,
    pub normal_model: Option<Arc<Mutex<Session>>>,
    pub embeddings: Option<ImageEmbeddings>,
    pub depth_map: Option<CachedDepthMap>,
}

fn edt_1d(f: &mut [f32], v: &mut [usize], z: &mut [f32], d: &mut [f32]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k = 0;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    for q in 1..n {
        let mut s = ((f[q] + (q * q) as f32) - (f[v[k]] + (v[k] * v[k]) as f32))
            / (2.0 * (q as f32 - v[k] as f32));
        while s <= z[k] {
            if k == 0 {
                break;
            }
            k -= 1;
            s = ((f[q] + (q * q) as f32) - (f[v[k]] + (v[k] * v[k]) as f32))
                / (2.0 * (q as f32 - v[k] as f32));
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f32::INFINITY;
    }
    k = 0;
    for (q, d_q) in d[..n].iter_mut().enumerate() {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let diff = q as f32 - v[k] as f32;
        *d_q = diff * diff + f[v[k]];
    }
    f.copy_from_slice(&d[..n]);
}

fn edt_2d(grid: &[bool], width: usize, height: usize) -> Vec<f32> {
    let area = width * height;
    let mut f = vec![0.0; area];
    for i in 0..area {
        f[i] = if grid[i] { 1e10 } else { 0.0 };
    }

    let max_dim = width.max(height);
    let mut v = vec![0; max_dim];
    let mut z = vec![0.0; max_dim + 1];
    let mut d = vec![0.0; max_dim];

    for y in 0..height {
        let start = y * width;
        let end = start + width;
        edt_1d(&mut f[start..end], &mut v, &mut z, &mut d);
    }

    let mut col = vec![0.0; height];
    for x in 0..width {
        for y in 0..height {
            col[y] = f[y * width + x];
        }
        edt_1d(&mut col, &mut v, &mut z, &mut d);
        for y in 0..height {
            f[y * width + x] = col[y];
        }
    }

    f.into_iter().map(|v| v.sqrt()).collect()
}

fn box_filter_horiz(src: &[f32], dest: &mut [f32], w: usize, r: usize) {
    dest.par_chunks_exact_mut(w)
        .zip(src.par_chunks_exact(w))
        .for_each(|(dest_row, src_row)| {
            let mut prefix = vec![0.0; w + 1];
            for x in 0..w {
                prefix[x + 1] = prefix[x] + src_row[x];
            }
            for (x, dest_val) in dest_row.iter_mut().enumerate() {
                let x_min = x.saturating_sub(r);
                let x_max = (x + r).min(w - 1);
                let count = (x_max - x_min + 1) as f32;
                *dest_val = (prefix[x_max + 1] - prefix[x_min]) / count;
            }
        });
}

fn transpose(src: &[f32], dest: &mut [f32], w: usize, h: usize) {
    dest.par_chunks_exact_mut(h)
        .enumerate()
        .for_each(|(x, dest_col)| {
            for y in 0..h {
                dest_col[y] = src[y * w + x];
            }
        });
}

fn box_filter_opt(
    src: &[f32],
    dest: &mut [f32],
    temp1: &mut [f32],
    temp2: &mut [f32],
    w: usize,
    h: usize,
    r: usize,
) {
    box_filter_horiz(src, temp1, w, r);
    transpose(temp1, temp2, w, h);
    box_filter_horiz(temp2, temp1, h, r);
    transpose(temp1, dest, h, w);
}

pub fn fast_guided_filter(
    guide_hr: &GrayImage,
    mask_lr: &GrayImage,
    r: usize,
    eps: f32,
) -> GrayImage {
    let (hr_w, hr_h) = guide_hr.dimensions();
    let (lr_w, lr_h) = mask_lr.dimensions();

    let w = lr_w as usize;
    let h = lr_h as usize;
    let area = w * h;

    let guide_lr = imageops::resize(guide_hr, lr_w, lr_h, FilterType::Triangle);

    let mut i_lr = vec![0.0; area];
    let mut p_lr = vec![0.0; area];
    let mut ip_lr = vec![0.0; area];
    let mut ii_lr = vec![0.0; area];

    let guide_lr_raw = guide_lr.as_raw();
    let mask_lr_raw = mask_lr.as_raw();

    i_lr.par_iter_mut()
        .zip(p_lr.par_iter_mut())
        .zip(ip_lr.par_iter_mut())
        .zip(ii_lr.par_iter_mut())
        .zip(guide_lr_raw.par_iter())
        .zip(mask_lr_raw.par_iter())
        .for_each(|(((((i_out, p_out), ip_out), ii_out), &g_val), &m_val)| {
            let i_val = g_val as f32 / 255.0;
            let p_val = m_val as f32 / 255.0;
            *i_out = i_val;
            *p_out = p_val;
            *ip_out = i_val * p_val;
            *ii_out = i_val * i_val;
        });

    let mut temp1 = vec![0.0; area];
    let mut temp2 = vec![0.0; area];

    let mut mean_i = vec![0.0; area];
    let mut mean_p = vec![0.0; area];
    let mut mean_ip = vec![0.0; area];
    let mut mean_ii = vec![0.0; area];

    box_filter_opt(&i_lr, &mut mean_i, &mut temp1, &mut temp2, w, h, r);
    box_filter_opt(&p_lr, &mut mean_p, &mut temp1, &mut temp2, w, h, r);
    box_filter_opt(&ip_lr, &mut mean_ip, &mut temp1, &mut temp2, w, h, r);
    box_filter_opt(&ii_lr, &mut mean_ii, &mut temp1, &mut temp2, w, h, r);

    let mut a = vec![0.0; area];
    let mut b = vec![0.0; area];

    a.par_iter_mut()
        .zip(b.par_iter_mut())
        .zip(mean_i.par_iter())
        .zip(mean_p.par_iter())
        .zip(mean_ip.par_iter())
        .zip(mean_ii.par_iter())
        .for_each(|(((((a_out, b_out), &m_i), &m_p), &m_ip), &m_ii)| {
            let cov_ip = m_ip - m_i * m_p;
            let var_i = m_ii - m_i * m_i;
            let a_val = cov_ip / (var_i + eps);
            *a_out = a_val;
            *b_out = m_p - a_val * m_i;
        });

    let mut mean_a = vec![0.0; area];
    let mut mean_b = vec![0.0; area];

    box_filter_opt(&a, &mut mean_a, &mut temp1, &mut temp2, w, h, r);
    box_filter_opt(&b, &mut mean_b, &mut temp1, &mut temp2, w, h, r);

    let mean_a_img = ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(lr_w, lr_h, mean_a).unwrap();
    let mean_b_img = ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(lr_w, lr_h, mean_b).unwrap();

    let mean_a_hr = imageops::resize(&mean_a_img, hr_w, hr_h, FilterType::Triangle);
    let mean_b_hr = imageops::resize(&mean_b_img, hr_w, hr_h, FilterType::Triangle);

    let mean_a_hr_raw = mean_a_hr.as_raw();
    let mean_b_hr_raw = mean_b_hr.as_raw();
    let guide_hr_raw = guide_hr.as_raw();

    let mut final_mask_raw = vec![0u8; (hr_w * hr_h) as usize];

    final_mask_raw
        .par_iter_mut()
        .zip(mean_a_hr_raw.par_iter())
        .zip(mean_b_hr_raw.par_iter())
        .zip(guide_hr_raw.par_iter())
        .for_each(|(((out, &a_val), &b_val), &i_val)| {
            let i_v = i_val as f32 / 255.0;
            let q = a_val * i_v + b_val;
            *out = (q.clamp(0.0, 1.0) * 255.0).round() as u8;
        });

    GrayImage::from_raw(hr_w, hr_h, final_mask_raw).unwrap()
}

fn get_models_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf> {
    let models_dir = app_handle.path().app_data_dir()?.join("models");
    if !models_dir.exists() {
        fs::create_dir_all(&models_dir)?;
    }
    Ok(models_dir)
}

/// Creates an ONNX Runtime session. On macOS, `coreml_units` hands the model to CoreML (GPU or
/// Neural Engine), falling back to CPU if CoreML can't compile it; `None` keeps it on CPU.
///
/// Only enable CoreML for models verified to run correctly and faster with it. Models that are
/// int8-quantized, split into many CoreML/CPU partitions, or fail inside CoreML at run time
/// (which the load-time fallback can't catch) must stay on CPU.
/// Set `RAPIDRAW_AI_CPU_ONLY=1` to force CPU for every model.
fn build_session(
    model_path: &Path,
    model_sha256: &str,
    coreml_units: Option<CoreMLComputeUnits>,
) -> Result<Session> {
    #[cfg(target_os = "macos")]
    if let Some(units) = coreml_units
        && std::env::var_os("RAPIDRAW_AI_CPU_ONLY").is_none()
    {
        match build_coreml_session(model_path, model_sha256, units) {
            Ok(session) => {
                log::info!("Loaded {} with CoreML ({:?})", model_path.display(), units);
                return Ok(session);
            }
            Err(e) => log::warn!(
                "CoreML failed for {}, falling back to CPU: {}",
                model_path.display(),
                e
            ),
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (model_sha256, coreml_units);

    Ok(Session::builder()?.commit_from_file(model_path)?)
}

/// Keyed by the model's hash so updated weights never reuse a stale compiled model.
#[cfg(target_os = "macos")]
fn coreml_cache_dir(model_path: &Path, model_sha256: &str) -> PathBuf {
    model_path
        .parent()
        .unwrap_or(Path::new("."))
        .join("coreml_cache")
        .join(&model_sha256[..16])
}

#[cfg(target_os = "macos")]
fn build_coreml_session(
    model_path: &Path,
    model_sha256: &str,
    units: CoreMLComputeUnits,
) -> Result<Session> {
    use ort::execution_providers::coreml::{CoreMLExecutionProvider, CoreMLModelFormat};

    let cache_dir = coreml_cache_dir(model_path, model_sha256);
    fs::create_dir_all(&cache_dir)?;

    let coreml = CoreMLExecutionProvider::default()
        .with_model_format(CoreMLModelFormat::MLProgram)
        .with_compute_units(units)
        .with_model_cache_dir(cache_dir.display())
        .build()
        .error_on_failure();

    Ok(Session::builder()?
        .with_execution_providers([coreml])?
        .commit_from_file(model_path)?)
}

/// Like `build_session`, for models CoreML takes over a minute to prepare on first load (and
/// again whenever macOS drops its cache). Returns a CPU session at once and swaps in the CoreML
/// one when it's ready, so nothing waits on the compile.
fn build_session_upgrading(
    model_path: &Path,
    model_sha256: &str,
    coreml_units: CoreMLComputeUnits,
) -> Result<Arc<Mutex<Session>>> {
    let session = Arc::new(Mutex::new(
        Session::builder()?.commit_from_file(model_path)?,
    ));

    #[cfg(target_os = "macos")]
    if std::env::var_os("RAPIDRAW_AI_CPU_ONLY").is_none() {
        let target = Arc::downgrade(&session);
        let path = model_path.to_path_buf();
        let sha = model_sha256.to_string();
        std::thread::spawn(
            move || match build_coreml_session(&path, &sha, coreml_units) {
                Ok(coreml) => {
                    if let Some(target) = target.upgrade() {
                        *target.lock().unwrap() = coreml;
                        log::info!("Switched {} to CoreML ({:?})", path.display(), coreml_units);
                    }
                }
                Err(e) => {
                    let _ = fs::remove_dir_all(coreml_cache_dir(&path, &sha));
                    log::warn!(
                        "CoreML failed for {}, staying on CPU: {}",
                        path.display(),
                        e
                    );
                }
            },
        );
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (model_sha256, coreml_units);

    Ok(session)
}

fn persist_downloaded_asset(dest: &Path, bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() {
        return Err(anyhow::anyhow!(
            "Downloaded asset for {} was empty",
            dest.display()
        ));
    }

    let parent = dest.parent().ok_or_else(|| {
        anyhow::anyhow!(
            "Cannot determine parent directory for downloaded asset {}",
            dest.display()
        )
    })?;
    fs::create_dir_all(parent)?;

    let file_name = dest
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid downloaded asset path {}", dest.display()))?;
    let tmp_path = dest.with_file_name(format!(".{}.download", file_name));

    {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }

    fs::rename(&tmp_path, dest).or_else(|rename_error| -> std::io::Result<()> {
        if dest.exists() {
            fs::remove_file(dest)?;
            fs::rename(&tmp_path, dest)?;
            Ok(())
        } else {
            Err(rename_error)
        }
    })?;
    Ok(())
}

async fn download_model(url: &str, dest: &Path) -> Result<()> {
    let response = reqwest::get(url).await?.error_for_status()?;
    let bytes = response.bytes().await?;
    persist_downloaded_asset(dest, &bytes)
}

fn verify_sha256(path: &Path, expected_hash: &str) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    let hash = hasher.finalize();
    let hex_hash = hex::encode(hash);
    Ok(hex_hash == expected_hash)
}

async fn download_and_verify_model(
    app_handle: &tauri::AppHandle,
    models_dir: &Path,
    filename: &str,
    url: &str,
    expected_hash: &str,
    model_name: &str,
) -> Result<()> {
    let dest_path = models_dir.join(filename);
    let is_valid = verify_sha256(&dest_path, expected_hash)?;

    if !is_valid {
        if dest_path.exists() {
            println!("Model {} has incorrect hash. Re-downloading.", model_name);
            fs::remove_file(&dest_path)?;
        }
        let _ = app_handle.emit("ai-model-download-start", model_name);
        let download_result = download_model(url, &dest_path).await;
        let _ = app_handle.emit("ai-model-download-finish", model_name);
        download_result?;

        if !verify_sha256(&dest_path, expected_hash)? {
            return Err(anyhow::anyhow!(
                "Failed to verify model {} after download. Hash mismatch.",
                model_name
            ));
        }
    }
    Ok(())
}

pub async fn get_or_init_ai_models(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<AiModels>> {
    if let Some(models) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.models.clone())
    {
        return Ok(models);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(models) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.models.clone())
    {
        return Ok(models);
    }

    let models_dir = get_models_dir(app_handle)?;

    download_and_verify_model(
        app_handle,
        &models_dir,
        ENCODER_FILENAME,
        ENCODER_URL,
        ENCODER_SHA256,
        "SAM Encoder",
    )
    .await?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        DECODER_FILENAME,
        DECODER_URL,
        DECODER_SHA256,
        "SAM Decoder",
    )
    .await?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        DEPTH_FILENAME,
        DEPTH_URL,
        DEPTH_SHA256,
        "Depth Model",
    )
    .await?;

    let _ = ort::init().with_name("AI").commit();

    let encoder_path = models_dir.join(ENCODER_FILENAME);
    let decoder_path = models_dir.join(DECODER_FILENAME);
    let depth_path = models_dir.join(DEPTH_FILENAME);

    let sam_encoder = build_session(
        &encoder_path,
        ENCODER_SHA256,
        Some(CoreMLComputeUnits::CPUAndGPU),
    )?;
    let sam_decoder = build_session(&decoder_path, DECODER_SHA256, None)?;
    let depth_anything = build_session(
        &depth_path,
        DEPTH_SHA256,
        Some(CoreMLComputeUnits::CPUAndGPU),
    )?;

    crate::register_exit_handler();

    let models = Arc::new(AiModels {
        sam_encoder: Mutex::new(sam_encoder),
        sam_decoder: Mutex::new(sam_decoder),
        depth_anything: Mutex::new(depth_anything),
    });

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.models = Some(models.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: Some(models.clone()),
            denoise_model: None,
            raw_denoise_model: None,
            upscale_model: None,
            foreground_model: None,
            sky_model: None,
            clip_models: None,
            search_models: None,
            lama_model: None,
            normal_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(models)
}

pub async fn get_or_init_denoise_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(denoise_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.denoise_model.clone())
    {
        return Ok(denoise_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(denoise_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.denoise_model.clone())
    {
        return Ok(denoise_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        DENOISE_FILENAME,
        DENOISE_URL,
        DENOISE_SHA256,
        "NIND Denoise Model",
    )
    .await?;

    let _ = ort::init().with_name("AI-Denoise").commit();
    let model_path = models_dir.join(DENOISE_FILENAME);
    let session = build_session(
        &model_path,
        DENOISE_SHA256,
        Some(CoreMLComputeUnits::CPUAndGPU),
    )?;
    let denoise_model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.denoise_model = Some(denoise_model.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: Some(denoise_model.clone()),
            raw_denoise_model: None,
            upscale_model: None,
            foreground_model: None,
            sky_model: None,
            clip_models: None,
            search_models: None,
            lama_model: None,
            normal_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(denoise_model)
}

pub async fn get_or_init_raw_denoise_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(raw_denoise_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.raw_denoise_model.clone())
    {
        return Ok(raw_denoise_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(raw_denoise_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.raw_denoise_model.clone())
    {
        return Ok(raw_denoise_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        RAWNIND_FILENAME,
        RAWNIND_URL,
        RAWNIND_SHA256,
        "RawNIND Denoise Model",
    )
    .await?;

    let _ = ort::init().with_name("AI-Denoise").commit();
    let model_path = models_dir.join(RAWNIND_FILENAME);
    let session = build_session(
        &model_path,
        RAWNIND_SHA256,
        Some(CoreMLComputeUnits::CPUAndGPU),
    )?;
    let raw_denoise_model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.raw_denoise_model = Some(raw_denoise_model.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: None,
            raw_denoise_model: Some(raw_denoise_model.clone()),
            upscale_model: None,
            foreground_model: None,
            sky_model: None,
            clip_models: None,
            search_models: None,
            lama_model: None,
            normal_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(raw_denoise_model)
}

pub async fn get_or_init_upscale_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(upscale_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.upscale_model.clone())
    {
        return Ok(upscale_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(upscale_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.upscale_model.clone())
    {
        return Ok(upscale_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        UPSCALE_FILENAME,
        UPSCALE_URL,
        UPSCALE_SHA256,
        "RealPLKSR Upscale Model",
    )
    .await?;

    let _ = ort::init().with_name("AI-Upscale").commit();
    let model_path = models_dir.join(UPSCALE_FILENAME);
    let session = build_session(
        &model_path,
        UPSCALE_SHA256,
        Some(CoreMLComputeUnits::CPUAndGPU),
    )?;
    let upscale_model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.upscale_model = Some(upscale_model.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: None,
            raw_denoise_model: None,
            upscale_model: Some(upscale_model.clone()),
            foreground_model: None,
            sky_model: None,
            clip_models: None,
            search_models: None,
            lama_model: None,
            normal_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(upscale_model)
}

pub async fn get_or_init_foreground_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(foreground_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.foreground_model.clone())
    {
        return Ok(foreground_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(foreground_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.foreground_model.clone())
    {
        return Ok(foreground_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        BIREFNET_FILENAME,
        BIREFNET_URL,
        BIREFNET_SHA256,
        "Foreground Model",
    )
    .await?;

    let _ = ort::init().with_name("AI-Foreground").commit();
    let model_path = models_dir.join(BIREFNET_FILENAME);
    let session = build_session(
        &model_path,
        BIREFNET_SHA256,
        Some(CoreMLComputeUnits::CPUAndGPU),
    )?;
    let foreground_model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.foreground_model = Some(foreground_model.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: None,
            raw_denoise_model: None,
            upscale_model: None,
            foreground_model: Some(foreground_model.clone()),
            sky_model: None,
            clip_models: None,
            search_models: None,
            lama_model: None,
            normal_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(foreground_model)
}

pub async fn get_or_init_sky_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(sky_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.sky_model.clone())
    {
        return Ok(sky_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(sky_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.sky_model.clone())
    {
        return Ok(sky_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        SKY_FILENAME,
        SKY_URL,
        SKY_SHA256,
        "Sky Model",
    )
    .await?;

    let _ = ort::init().with_name("AI-Sky").commit();
    let model_path = models_dir.join(SKY_FILENAME);
    let session = build_session(&model_path, SKY_SHA256, Some(CoreMLComputeUnits::CPUAndGPU))?;
    let sky_model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.sky_model = Some(sky_model.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: None,
            raw_denoise_model: None,
            upscale_model: None,
            foreground_model: None,
            sky_model: Some(sky_model.clone()),
            clip_models: None,
            search_models: None,
            lama_model: None,
            normal_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(sky_model)
}

pub async fn get_or_init_clip_models(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<ClipModels>> {
    if let Some(clip_models) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.clip_models.clone())
    {
        return Ok(clip_models);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(clip_models) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.clip_models.clone())
    {
        return Ok(clip_models);
    }

    let models_dir = get_models_dir(app_handle)?;

    download_and_verify_model(
        app_handle,
        &models_dir,
        CLIP_MODEL_FILENAME,
        CLIP_MODEL_URL,
        CLIP_MODEL_SHA256,
        "CLIP Model",
    )
    .await?;

    let clip_tokenizer_path = models_dir.join(CLIP_TOKENIZER_FILENAME);
    if !clip_tokenizer_path.exists() {
        let _ = app_handle.emit("ai-model-download-start", "CLIP Tokenizer");
        let download_result = download_model(CLIP_TOKENIZER_URL, &clip_tokenizer_path).await;
        let _ = app_handle.emit("ai-model-download-finish", "CLIP Tokenizer");
        download_result?;
    }

    let _ = ort::init().with_name("AI-Tagging").commit();
    let clip_model_path = models_dir.join(CLIP_MODEL_FILENAME);
    let model = Mutex::new(build_session(&clip_model_path, CLIP_MODEL_SHA256, None)?);
    let tokenizer =
        Tokenizer::from_file(clip_tokenizer_path).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    crate::register_exit_handler();

    let clip_models = Arc::new(ClipModels { model, tokenizer });

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.clip_models = Some(clip_models.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: None,
            raw_denoise_model: None,
            upscale_model: None,
            foreground_model: None,
            sky_model: None,
            clip_models: Some(clip_models.clone()),
            search_models: None,
            lama_model: None,
            normal_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(clip_models)
}

pub async fn get_or_init_search_models(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<SearchModels>> {
    let cached = || {
        ai_state_mutex
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|state| state.search_models.clone())
    };
    if let Some(search_models) = cached() {
        return Ok(search_models);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(search_models) = cached() {
        return Ok(search_models);
    }

    let models_dir = get_models_dir(app_handle)?;
    for (filename, url, sha, name) in [
        (
            SEARCH_VISION_FILENAME,
            SEARCH_VISION_URL,
            SEARCH_VISION_SHA256,
            "Search Image Model",
        ),
        (
            SEARCH_TEXT_FILENAME,
            SEARCH_TEXT_URL,
            SEARCH_TEXT_SHA256,
            "Search Text Model",
        ),
        (
            SEARCH_TOKENIZER_FILENAME,
            SEARCH_TOKENIZER_URL,
            SEARCH_TOKENIZER_SHA256,
            "Search Tokenizer",
        ),
    ] {
        download_and_verify_model(app_handle, &models_dir, filename, url, sha, name).await?;
    }

    let _ = ort::init().with_name("AI-Search").commit();
    let vision = build_session_upgrading(
        &models_dir.join(SEARCH_VISION_FILENAME),
        SEARCH_VISION_SHA256,
        CoreMLComputeUnits::CPUAndGPU,
    )?;
    // int8: stays on CPU, where a short query takes ~15 ms.
    let text = Mutex::new(build_session(
        &models_dir.join(SEARCH_TEXT_FILENAME),
        SEARCH_TEXT_SHA256,
        None,
    )?);
    let tokenizer = Tokenizer::from_file(models_dir.join(SEARCH_TOKENIZER_FILENAME))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    crate::register_exit_handler();

    let search_models = Arc::new(SearchModels {
        vision,
        text,
        tokenizer,
    });

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.search_models = Some(search_models.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: None,
            raw_denoise_model: None,
            upscale_model: None,
            foreground_model: None,
            sky_model: None,
            clip_models: None,
            search_models: Some(search_models.clone()),
            lama_model: None,
            normal_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(search_models)
}

pub async fn get_or_init_lama_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(lama_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.lama_model.clone())
    {
        return Ok(lama_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(lama_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.lama_model.clone())
    {
        return Ok(lama_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        LAMA_FILENAME,
        LAMA_URL,
        LAMA_SHA256,
        "Inpainting Model",
    )
    .await?;

    let _ = ort::init().with_name("AI-Inpainting").commit();
    let model_path = models_dir.join(LAMA_FILENAME);
    let session = build_session(&model_path, LAMA_SHA256, None)?;
    let lama_model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.lama_model = Some(lama_model.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: None,
            raw_denoise_model: None,
            upscale_model: None,
            foreground_model: None,
            sky_model: None,
            clip_models: None,
            search_models: None,
            lama_model: Some(lama_model.clone()),
            normal_model: None,
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(lama_model)
}

pub async fn get_or_init_normal_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(normal_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.normal_model.clone())
    {
        return Ok(normal_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(normal_model) = ai_state_mutex
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|state| state.normal_model.clone())
    {
        return Ok(normal_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(
        app_handle,
        &models_dir,
        NORMAL_FILENAME,
        NORMAL_URL,
        NORMAL_SHA256,
        "Normal Model",
    )
    .await?;

    let _ = ort::init().with_name("AI-Normal").commit();
    let model_path = models_dir.join(NORMAL_FILENAME);
    let session = build_session(&model_path, NORMAL_SHA256, None)?;
    let normal_model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    let mut ai_state_lock = ai_state_mutex.lock().unwrap();
    if let Some(state) = ai_state_lock.as_mut() {
        state.normal_model = Some(normal_model.clone());
    } else {
        *ai_state_lock = Some(AiState {
            models: None,
            denoise_model: None,
            raw_denoise_model: None,
            upscale_model: None,
            foreground_model: None,
            sky_model: None,
            clip_models: None,
            search_models: None,
            lama_model: None,
            normal_model: Some(normal_model.clone()),
            embeddings: None,
            depth_map: None,
        });
    }

    Ok(normal_model)
}

#[derive(Clone, Copy)]
struct TileParams {
    cs: usize,
    ucs: usize,
    overlap: usize,
    pad: usize,
}

impl TileParams {
    const fn new(cs: usize, ucs: usize, overlap: usize) -> Self {
        Self {
            cs,
            ucs,
            overlap,
            pad: (cs - ucs) / 2,
        }
    }
}

const TILE_HIGHER_QUALITY: TileParams = TileParams::new(504, 448, 12);

#[inline]
fn mirror_coord(c: i32, size: i32) -> i32 {
    if c < 0 {
        (-c).min(size - 1)
    } else if c >= size {
        (2 * size - 1 - c).max(0)
    } else {
        c
    }
}

fn extract_tile_mirror(img: &Rgb32FImage, x0: i32, y0: i32, cs: usize) -> Array4<f32> {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let mut arr = Array4::zeros((1, 3, cs, cs));
    for dy in 0..cs as i32 {
        for dx in 0..cs as i32 {
            let sx = mirror_coord(x0 + dx, w);
            let sy = mirror_coord(y0 + dy, h);
            let px = img.get_pixel(sx as u32, sy as u32);
            arr[[0, 0, dy as usize, dx as usize]] = px[0];
            arr[[0, 1, dy as usize, dx as usize]] = px[1];
            arr[[0, 2, dy as usize, dx as usize]] = px[2];
        }
    }
    arr
}

struct SeamlessBlend {
    ud0: usize,
    ud1: usize,
    ud2: usize,
    ud3: usize,
    absx0: usize,
    absy0: usize,
    fswidth: usize,
    fsheight: usize,
    overlap: usize,
}

fn apply_seamless(tile: &mut Array4<f32>, blend: &SeamlessBlend) {
    let SeamlessBlend {
        ud0,
        ud1,
        ud2,
        ud3,
        absx0,
        absy0,
        fswidth,
        fsheight,
        overlap,
    } = *blend;
    let ol = overlap;
    if absx0 > 0 {
        for c in 0..3 {
            for y in ud1..ud3 {
                for x in ud0..(ud0 + ol).min(ud2) {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
    if absy0 > 0 {
        for c in 0..3 {
            for y in ud1..(ud1 + ol).min(ud3) {
                for x in ud0..ud2 {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
    if absx0 + (ud2 - ud0) < fswidth && ol > 0 {
        let right_start = (ud2 as i32 - ol as i32).max(ud0 as i32) as usize;
        for c in 0..3 {
            for y in ud1..ud3 {
                for x in right_start..ud2 {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
    if absy0 + (ud3 - ud1) < fsheight && ol > 0 {
        let bottom_start = (ud3 as i32 - ol as i32).max(ud1 as i32) as usize;
        for c in 0..3 {
            for y in bottom_start..ud3 {
                for x in ud0..ud2 {
                    tile[[0, c, y, x]] *= 0.5;
                }
            }
        }
    }
}

fn run_native_denoise(
    img: &Rgb32FImage,
    session: &Mutex<Session>,
    accumulator: &mut [f32],
    width: usize,
    height: usize,
    app_handle: &tauri::AppHandle,
    params: TileParams,
) -> Result<()> {
    let w = width as i32;
    let h = height as i32;
    let step = params.ucs.saturating_sub(params.overlap).max(1);
    let iperhl = (width.saturating_sub(params.ucs) as f64 / step as f64).ceil() as usize;
    let ipervl = (height.saturating_sub(params.ucs) as f64 / step as f64).ceil() as usize;
    let total = (iperhl + 1) * (ipervl + 1);

    for i in 0..total {
        let yi = i / (iperhl + 1);
        let xi = i % (iperhl + 1);
        let x0 =
            params.ucs as i32 * xi as i32 - params.overlap as i32 * xi as i32 - params.pad as i32;
        let y0 =
            params.ucs as i32 * yi as i32 - params.overlap as i32 * yi as i32 - params.pad as i32;

        if i % 10 == 0 {
            let pct = (i as f32 / total as f32) * 100.0;
            let _ = app_handle.emit("denoise-progress", format!("Denoising… {:.0}%", pct));
        }

        let crop = extract_tile_mirror(img, x0, y0, params.cs);
        let input_values = crop.as_standard_layout().to_owned();
        let t_input = Tensor::from_array(input_values)?;

        let out = {
            let mut sess = session.lock().unwrap();
            let outputs = sess.run(ort::inputs![t_input])?;
            let arr = outputs[0].try_extract_array::<f32>()?.to_owned();
            arr.into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| anyhow::anyhow!("Unexpected output shape: {}", e))?
        };

        let x1pad = (0i32).max(x0 + params.cs as i32 - w) as usize;
        let y1pad = (0i32).max(y0 + params.cs as i32 - h) as usize;
        let ud0 = params.pad;
        let ud1 = params.pad;
        let ud2 = params.cs - params.pad.max(x1pad);
        let ud3 = params.cs - params.pad.max(y1pad);
        let absx0 = (x0 + params.pad as i32).max(0) as usize;
        let absy0 = (y0 + params.pad as i32).max(0) as usize;

        let mut tile = out;
        apply_seamless(
            &mut tile,
            &SeamlessBlend {
                ud0,
                ud1,
                ud2,
                ud3,
                absx0,
                absy0,
                fswidth: width,
                fsheight: height,
                overlap: params.overlap,
            },
        );

        for cy in 0..(ud3 - ud1) {
            for cx in 0..(ud2 - ud0) {
                let gx = absx0 + cx;
                let gy = absy0 + cy;
                if gx < width && gy < height {
                    let base = (gy * width + gx) * 3;
                    accumulator[base] += tile[[0, 0, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                    accumulator[base + 1] += tile[[0, 1, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                    accumulator[base + 2] += tile[[0, 2, ud1 + cy, ud0 + cx]].clamp(0.0, 1.0);
                }
            }
        }
    }
    Ok(())
}

fn accumulator_to_rgb32f(acc: &[f32], width: u32, height: u32) -> Rgb32FImage {
    let mut out = Rgb32FImage::new(width, height);
    for (i, p) in out.pixels_mut().enumerate() {
        let i3 = i * 3;
        *p = Rgb([
            acc[i3].clamp(0.0, 1.0),
            acc[i3 + 1].clamp(0.0, 1.0),
            acc[i3 + 2].clamp(0.0, 1.0),
        ]);
    }
    out
}

pub fn run_ai_denoise(
    rgb_img: &Rgb32FImage,
    intensity: f32,
    session: &Mutex<Session>,
    app_handle: &tauri::AppHandle,
) -> Result<DynamicImage> {
    let (width, height) = rgb_img.dimensions();

    let _ = app_handle.emit("denoise-progress", "Denoising (AI NIND)...");
    let mut accumulator = vec![0.0f32; width as usize * height as usize * 3];
    run_native_denoise(
        rgb_img,
        session,
        &mut accumulator,
        width as usize,
        height as usize,
        app_handle,
        TILE_HIGHER_QUALITY,
    )?;

    let mut out_img_buffer = accumulator_to_rgb32f(&accumulator, width, height);
    let strength = intensity.clamp(0.0, 1.0);
    if strength < 1.0 {
        out_img_buffer
            .par_iter_mut()
            .zip(rgb_img.par_iter())
            .for_each(|(d, &o)| *d = o + strength * (*d - o));
    }
    Ok(DynamicImage::ImageRgb32F(out_img_buffer))
}

fn mat3_mul(m: &[[f32; 3]; 3], p: &[f32]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
        m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
        m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
    ]
}

fn reflect_coord(c: isize, len: usize) -> usize {
    if len == 1 {
        return 0;
    }
    let period = 2 * (len as isize - 1);
    let m = c.rem_euclid(period);
    (if m < len as isize { m } else { period - m }) as usize
}

/// Tile origins along one axis, in image coordinates. The image is treated as mirror-padded by
/// `overlap` on both sides so every real pixel lies where some tile has full weight.
fn tile_starts(len: usize, tile: usize, overlap: usize) -> Vec<isize> {
    let last = (len + 2 * overlap).saturating_sub(tile);
    let mut starts = Vec::new();
    let mut s = 0;
    loop {
        let c = s.min(last);
        starts.push(c as isize - overlap as isize);
        if c == last {
            return starts;
        }
        s += tile - overlap;
    }
}

/// Feathering weight across a tile: zero in the outer `border` pixels (CNN edge effects), then a
/// linear ramp over the rest of the overlap, so adjacent tiles' weights sum to one.
fn tile_weight(u: usize, tile: usize, overlap: usize, border: usize) -> f32 {
    let ramp = (overlap - 2 * border) as f32;
    let u = u as f32 + 0.5;
    let rise = (u - border as f32) / ramp;
    let fall = ((tile - border) as f32 - u) / ramp;
    rise.min(fall).clamp(0.0, 1.0)
}

fn rawnind_tile_starts(len: usize) -> Vec<isize> {
    tile_starts(len, RAWNIND_TILE, RAWNIND_OVERLAP)
}

fn rawnind_tile_weight(u: usize) -> f32 {
    tile_weight(u, RAWNIND_TILE, RAWNIND_OVERLAP, RAWNIND_BORDER)
}

/// RawNIND denoise of a demosaiced raw image: linear, Rec.709 primaries, white balanced, white
/// level at 1.0. The model sees linear Rec.2020 clipped to [0, 1]; the noise it removes is added
/// back onto the unclipped input, so highlight headroom above 1.0 is kept. `strength` scales that
/// correction (0 = original, 1 = fully denoised). `progress` gets the fraction of tiles done.
pub fn run_rawnind_denoise(
    rgb_img: &Rgb32FImage,
    strength: f32,
    session: &Mutex<Session>,
    progress: impl Fn(f32),
) -> Result<Rgb32FImage> {
    const T: usize = RAWNIND_TILE;
    let (w, h) = (rgb_img.width() as usize, rgb_img.height() as usize);
    let src = rgb_img.as_raw();
    let xs = rawnind_tile_starts(w);
    let ys = rawnind_tile_starts(h);
    let weights: Vec<f32> = (0..T).map(rawnind_tile_weight).collect();

    // Per-pixel sum of weighted (denoised - input) and of the weights.
    let mut residual = vec![0.0f32; w * h * 3];
    let mut weight_sum = vec![0.0f32; w * h];
    let mut tile_in = vec![0.0f32; 3 * T * T];
    let total = xs.len() * ys.len();

    for (iy, &y0) in ys.iter().enumerate() {
        for (ix, &x0) in xs.iter().enumerate() {
            progress((iy * xs.len() + ix) as f32 / total as f32);

            let (r, rest) = tile_in.split_at_mut(T * T);
            let (g, b) = rest.split_at_mut(T * T);
            r.par_chunks_mut(T)
                .zip(g.par_chunks_mut(T))
                .zip(b.par_chunks_mut(T))
                .enumerate()
                .for_each(|(dy, ((r, g), b))| {
                    let sy = reflect_coord(y0 + dy as isize, h);
                    let row = &src[sy * w * 3..(sy + 1) * w * 3];
                    for dx in 0..T {
                        let sx = reflect_coord(x0 + dx as isize, w);
                        let p = mat3_mul(&REC709_TO_REC2020, &row[sx * 3..sx * 3 + 3]);
                        r[dx] = p[0].clamp(0.0, 1.0);
                        g[dx] = p[1].clamp(0.0, 1.0);
                        b[dx] = p[2].clamp(0.0, 1.0);
                    }
                });
            let in_mean = tile_in.iter().map(|&v| v as f64).sum::<f64>() / tile_in.len() as f64;

            let input = Array4::from_shape_vec((1, 3, T, T), tile_in.clone())?;
            let mut sess = session.lock().unwrap();
            let outputs = sess.run(ort::inputs![Tensor::from_array(input)?])?;
            let out = outputs[0].try_extract_array::<f32>()?;
            let out = out
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("RawNIND output is not contiguous"))?;
            let out_mean = out.iter().map(|&v| v as f64).sum::<f64>() / out.len() as f64;
            // A black tile has nothing to denoise, and a near-zero output mean can't be matched.
            let gain = if in_mean > 1e-6 && out_mean.abs() > 1e-12 {
                (in_mean / out_mean) as f32
            } else {
                0.0
            };

            let y_lo = y0.max(0) as usize;
            let y_hi = ((y0 + T as isize) as usize).min(h);
            let x_lo = x0.max(0) as usize;
            let x_hi = ((x0 + T as isize) as usize).min(w);
            residual
                .par_chunks_mut(w * 3)
                .zip(weight_sum.par_chunks_mut(w))
                .enumerate()
                .skip(y_lo)
                .take(y_hi - y_lo)
                .for_each(|(y, (res_row, w_row))| {
                    let dy = (y as isize - y0) as usize;
                    for x in x_lo..x_hi {
                        let dx = (x as isize - x0) as usize;
                        let wgt = weights[dy] * weights[dx];
                        if wgt == 0.0 {
                            continue;
                        }
                        w_row[x] += wgt;
                        if gain == 0.0 {
                            continue;
                        }
                        let t = dy * T + dx;
                        for c in 0..3 {
                            let i = c * T * T + t;
                            res_row[x * 3 + c] += wgt * (out[i] * gain - tile_in[i]);
                        }
                    }
                });
        }
    }

    let mut out_img = Rgb32FImage::new(w as u32, h as u32);
    out_img
        .par_chunks_mut(3)
        .zip(src.par_chunks(3))
        .zip(residual.par_chunks(3))
        .zip(weight_sum.par_iter())
        .for_each(|(((dst, p), res), &wsum)| {
            if wsum <= 0.0 {
                dst.copy_from_slice(p);
                return;
            }
            let k = strength.clamp(0.0, 1.0) / wsum;
            let mut q = mat3_mul(&REC709_TO_REC2020, p);
            for c in 0..3 {
                q[c] += k * res[c];
            }
            let back = mat3_mul(&REC2020_TO_REC709, &q);
            for c in 0..3 {
                dst[c] = back[c].max(0.0);
            }
        });
    progress(1.0);
    Ok(out_img)
}

/// Catmull-Rom 2x upsampling of one planar channel (half-pixel centres, edges clamped). Output
/// pixel 2i sits at input i - 1/4 and 2i + 1 at i + 1/4, so both use fixed four-tap weights.
fn bicubic_2x(src: &[f32], w: usize, h: usize, dst: &mut [f32]) {
    const NEAR: [f32; 4] = [-0.023_437_5, 0.226_562_5, 0.867_187_5, -0.070_312_5];
    let tap = |p: &dyn Fn(isize) -> f32, i: isize, odd: bool| {
        if odd {
            NEAR[3] * p(i - 1) + NEAR[2] * p(i) + NEAR[1] * p(i + 1) + NEAR[0] * p(i + 2)
        } else {
            NEAR[0] * p(i - 2) + NEAR[1] * p(i - 1) + NEAR[2] * p(i) + NEAR[3] * p(i + 1)
        }
    };
    let mut rows = vec![0.0f32; h * 2 * w];
    rows.par_chunks_mut(2 * w).enumerate().for_each(|(y, out)| {
        let row = &src[y * w..(y + 1) * w];
        let p = |x: isize| row[x.clamp(0, w as isize - 1) as usize];
        for (x2, o) in out.iter_mut().enumerate() {
            *o = tap(&p, (x2 / 2) as isize, x2 % 2 == 1);
        }
    });
    dst.par_chunks_mut(2 * w).enumerate().for_each(|(y2, out)| {
        for (x, o) in out.iter_mut().enumerate() {
            let p = |y: isize| rows[y.clamp(0, h as isize - 1) as usize * 2 * w + x];
            *o = tap(&p, (y2 / 2) as isize, y2 % 2 == 1);
        }
    });
}

/// Sum of the feathering weights of every tile covering each output pixel along one axis.
/// Tiles form a grid and weights are separable, so the 2-D sum is row_sum[y] * col_sum[x].
fn tile_weight_sums(starts: &[isize], len: usize, weights: &[f32], scale: usize) -> Vec<f32> {
    let mut sums = vec![0.0f32; len];
    for &s in starts {
        let s = s * scale as isize;
        for (u, &wgt) in weights.iter().enumerate() {
            let p = s + u as isize;
            if p >= 0 && (p as usize) < len {
                sums[p as usize] += wgt;
            }
        }
    }
    sums
}

/// 2x AI super-resolution of a display-referred sRGB image (values clamped to [0, 1]).
/// `strength` blends from plain bicubic (0) to the full model output (1). `progress` gets the
/// fraction of tiles done before each tile and returns false to abort.
pub fn run_ai_upscale(
    rgb_img: &Rgb32FImage,
    strength: f32,
    session: &Mutex<Session>,
    progress: impl Fn(f32) -> bool,
) -> Result<Rgb32FImage> {
    const T: usize = UPSCALE_TILE;
    const S: usize = 2;
    const TO: usize = T * S;
    let (w, h) = (rgb_img.width() as usize, rgb_img.height() as usize);
    let (ow, oh) = (w * S, h * S);
    let src = rgb_img.as_raw();
    let strength = strength.clamp(0.0, 1.0);
    let xs = tile_starts(w, T, UPSCALE_OVERLAP);
    let ys = tile_starts(h, T, UPSCALE_OVERLAP);
    // Feathering happens at output resolution: every distance doubles.
    let weights: Vec<f32> = (0..TO)
        .map(|u| tile_weight(u, TO, S * UPSCALE_OVERLAP, S * UPSCALE_BORDER))
        .collect();
    let col_sums = tile_weight_sums(&xs, ow, &weights, S);
    let row_sums = tile_weight_sums(&ys, oh, &weights, S);

    let mut acc = vec![0.0f32; ow * oh * 3];
    let mut tile_in = vec![0.0f32; 3 * T * T];
    let mut bicubic = vec![0.0f32; if strength < 1.0 { 3 * TO * TO } else { 0 }];
    let total = xs.len() * ys.len();

    for (iy, &y0) in ys.iter().enumerate() {
        for (ix, &x0) in xs.iter().enumerate() {
            if !progress((iy * xs.len() + ix) as f32 / total as f32) {
                return Err(anyhow::anyhow!("Upscaling cancelled"));
            }

            let (r, rest) = tile_in.split_at_mut(T * T);
            let (g, b) = rest.split_at_mut(T * T);
            r.par_chunks_mut(T)
                .zip(g.par_chunks_mut(T))
                .zip(b.par_chunks_mut(T))
                .enumerate()
                .for_each(|(dy, ((r, g), b))| {
                    let sy = reflect_coord(y0 + dy as isize, h);
                    let row = &src[sy * w * 3..(sy + 1) * w * 3];
                    for dx in 0..T {
                        let sx = reflect_coord(x0 + dx as isize, w);
                        r[dx] = row[sx * 3].clamp(0.0, 1.0);
                        g[dx] = row[sx * 3 + 1].clamp(0.0, 1.0);
                        b[dx] = row[sx * 3 + 2].clamp(0.0, 1.0);
                    }
                });
            if strength < 1.0 {
                for c in 0..3 {
                    bicubic_2x(
                        &tile_in[c * T * T..(c + 1) * T * T],
                        T,
                        T,
                        &mut bicubic[c * TO * TO..(c + 1) * TO * TO],
                    );
                }
            }

            let input = Array4::from_shape_vec((1, 3, T, T), tile_in.clone())?;
            let mut sess = session.lock().unwrap();
            let outputs = sess.run(ort::inputs![Tensor::from_array(input)?])?;
            let out = outputs[0].try_extract_array::<f32>()?;
            let out = out
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("Upscale output is not contiguous"))?;
            if out.len() != 3 * TO * TO {
                return Err(anyhow::anyhow!(
                    "Unexpected upscale output size {}",
                    out.len()
                ));
            }

            let (oy0, ox0) = (y0 * S as isize, x0 * S as isize);
            let y_lo = oy0.max(0) as usize;
            let y_hi = ((oy0 + TO as isize) as usize).min(oh);
            let x_lo = ox0.max(0) as usize;
            let x_hi = ((ox0 + TO as isize) as usize).min(ow);
            let bicubic = &bicubic;
            acc.par_chunks_mut(ow * 3)
                .enumerate()
                .skip(y_lo)
                .take(y_hi - y_lo)
                .for_each(|(y, acc_row)| {
                    let dy = (y as isize - oy0) as usize;
                    for x in x_lo..x_hi {
                        let dx = (x as isize - ox0) as usize;
                        let wgt = weights[dy] * weights[dx];
                        if wgt == 0.0 {
                            continue;
                        }
                        let t = dy * TO + dx;
                        for c in 0..3 {
                            let i = c * TO * TO + t;
                            let v = if strength < 1.0 {
                                bicubic[i] + strength * (out[i] - bicubic[i])
                            } else {
                                out[i]
                            };
                            acc_row[x * 3 + c] += wgt * v;
                        }
                    }
                });
        }
    }

    acc.par_chunks_mut(ow * 3)
        .zip(row_sums.par_iter())
        .for_each(|(row, &ry)| {
            for (x, px) in row.chunks_mut(3).enumerate() {
                let wsum = ry * col_sums[x];
                for v in px {
                    *v = if wsum > 0.0 {
                        (*v / wsum).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                }
            }
        });
    Rgb32FImage::from_raw(ow as u32, oh as u32, acc)
        .ok_or_else(|| anyhow::anyhow!("Failed to build upscaled image"))
}

pub fn run_lama_inpainting(
    image: &DynamicImage,
    mask: &GrayImage,
    lama_session: &Mutex<Session>,
) -> Result<RgbaImage> {
    let (w, h) = image.dimensions();

    let (mut min_x, mut min_y) = (w, h);
    let (mut max_x, mut max_y) = (0u32, 0u32);
    let mut has_mask = false;

    for (x, y, p) in mask.enumerate_pixels() {
        if p[0] > 0 {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            has_mask = true;
        }
    }

    if !has_mask {
        return Ok(image.to_rgba8());
    }

    let mask_w = max_x - min_x + 1;
    let mask_h = max_y - min_y + 1;

    let pad_x = 128.max((mask_w as f32 * 1.5) as u32);
    let pad_y = 128.max((mask_h as f32 * 1.5) as u32);

    let x0 = min_x.saturating_sub(pad_x);
    let y0 = min_y.saturating_sub(pad_y);
    let x1 = (max_x + pad_x).min(w.saturating_sub(1));
    let y1 = (max_y + pad_y).min(h.saturating_sub(1));

    let crop_w = x1 - x0 + 1;
    let crop_h = y1 - y0 + 1;

    let rgba = image.to_rgba8();

    let cropped_img = imageops::crop_imm(&rgba, x0, y0, crop_w, crop_h).to_image();
    let cropped_mask = imageops::crop_imm(mask, x0, y0, crop_w, crop_h).to_image();

    let max_dim_limit: u32 = 768;
    let needs_downscale = crop_w > max_dim_limit || crop_h > max_dim_limit;

    let (fw, fh, inf_img, inf_mask) = if needs_downscale {
        let scale = max_dim_limit as f32 / crop_w.max(crop_h) as f32;

        let scaled_w = (crop_w as f32 * scale).round().max(1.0) as u32;
        let scaled_h = (crop_h as f32 * scale).round().max(1.0) as u32;

        (
            scaled_w,
            scaled_h,
            imageops::resize(&cropped_img, scaled_w, scaled_h, FilterType::Lanczos3),
            imageops::resize(&cropped_mask, scaled_w, scaled_h, FilterType::Triangle),
        )
    } else {
        (crop_w, crop_h, cropped_img.clone(), cropped_mask.clone())
    };

    let align = 64u32;
    let mut tensor_dim = fw.max(fh);
    if tensor_dim % align != 0 {
        tensor_dim += align - (tensor_dim % align);
    }
    let tensor_dim = tensor_dim.max(align) as usize;

    let mut img_tensor = Array::<f32, _>::zeros((1, 3, tensor_dim, tensor_dim));
    let mut msk_tensor = Array::<f32, _>::zeros((1, 1, tensor_dim, tensor_dim));

    for y in 0..tensor_dim {
        for x in 0..tensor_dim {
            let sx = (x as u32).min(fw.saturating_sub(1));
            let sy = (y as u32).min(fh.saturating_sub(1));

            let p = inf_img.get_pixel(sx, sy);
            let m = inf_mask.get_pixel(sx, sy)[0];

            img_tensor[[0, 0, y, x]] = p[0] as f32 / 255.0;
            img_tensor[[0, 1, y, x]] = p[1] as f32 / 255.0;
            img_tensor[[0, 2, y, x]] = p[2] as f32 / 255.0;
            msk_tensor[[0, 0, y, x]] = if m > 0 { 1.0 } else { 0.0 };
        }
    }

    let t_img = Tensor::from_array(img_tensor.into_dyn().as_standard_layout().into_owned())?;
    let t_msk = Tensor::from_array(msk_tensor.into_dyn().as_standard_layout().into_owned())?;

    let output_tensor = {
        let mut session = lama_session.lock().unwrap();
        let outputs = session.run(ort::inputs!["image" => t_img, "mask" => t_msk])?;
        outputs[0].try_extract_array::<f32>()?.to_owned()
    };

    let mut result_inf = RgbaImage::new(fw, fh);
    for y in 0..fh {
        for x in 0..fw {
            let r = output_tensor[[0, 0, y as usize, x as usize]].clamp(0.0, 255.0) as u8;
            let g = output_tensor[[0, 1, y as usize, x as usize]].clamp(0.0, 255.0) as u8;
            let b = output_tensor[[0, 2, y as usize, x as usize]].clamp(0.0, 255.0) as u8;
            result_inf.put_pixel(x, y, Rgba([r, g, b, 255]));
        }
    }

    let result_crop = if needs_downscale {
        imageops::resize(&result_inf, crop_w, crop_h, FilterType::Lanczos3)
    } else {
        result_inf
    };

    let mut final_image = image.to_rgba8();

    for y in 0..crop_h {
        for x in 0..crop_w {
            let m = cropped_mask.get_pixel(x, y)[0];
            if m > 0 {
                let alpha = m as f32 / 255.0;
                let p = result_crop.get_pixel(x, y);
                let gx = x0 + x;
                let gy = y0 + y;
                let orig = final_image.get_pixel(gx, gy);

                let r = (p[0] as f32 * alpha + orig[0] as f32 * (1.0 - alpha)) as u8;
                let g = (p[1] as f32 * alpha + orig[1] as f32 * (1.0 - alpha)) as u8;
                let b = (p[2] as f32 * alpha + orig[2] as f32 * (1.0 - alpha)) as u8;

                final_image.put_pixel(gx, gy, Rgba([r, g, b, 255]));
            }
        }
    }

    Ok(final_image)
}

pub fn generate_image_embeddings(
    image: &DynamicImage,
    encoder: &Mutex<Session>,
) -> Result<ImageEmbeddings> {
    let (orig_width, orig_height) = image.dimensions();
    let input = Tensor::from_array(imagenet_square_tensor(image, SAM_INPUT_SIZE).into_dyn())?;

    let mut session = encoder.lock().unwrap();
    let outputs = session.run(ort::inputs![input])?;
    let extract = |name: &str| -> Result<Array<f32, IxDyn>> {
        Ok(outputs[name].try_extract_array::<f32>()?.to_owned())
    };

    Ok(ImageEmbeddings {
        path_hash: String::new(),
        image_embed: extract("image_embed")?,
        high_res_feats_0: extract("high_res_feats_0")?,
        high_res_feats_1: extract("high_res_feats_1")?,
        original_size: (orig_width, orig_height),
    })
}

/// Picks one of SAM 2's four candidate masks the way `SAM2ImagePredictor` does. A lone click is
/// ambiguous, so it takes the multimask output with the best predicted IoU. Any other prompt
/// takes the single-mask output, unless that one is unstable around the threshold.
fn select_sam2_mask<'a>(masks: &'a [f32], iou: &[f32], single_click: bool) -> &'a [f32] {
    let area = SAM_MASK_SIZE * SAM_MASK_SIZE;
    let mask = |i: usize| &masks[i * area..(i + 1) * area];
    let best_multimask = (1..4)
        .max_by(|&a, &b| iou[a].total_cmp(&iou[b]))
        .unwrap_or(1);
    if single_click {
        return mask(best_multimask);
    }

    let (inner, outer) = mask(0).iter().fold((0usize, 0usize), |(i, o), &v| {
        (i + (v > 0.05) as usize, o + (v > -0.05) as usize)
    });
    let stability = if outer > 0 {
        inner as f32 / outer as f32
    } else {
        1.0
    };
    if stability >= 0.98 {
        mask(0)
    } else {
        mask(best_multimask)
    }
}

pub fn run_sam_decoder(
    decoder: &Mutex<Session>,
    embeddings: &ImageEmbeddings,
    start_point: (f64, f64),
    end_point: (f64, f64),
    original_image: Option<&DynamicImage>,
) -> Result<GrayImage> {
    let (orig_width, orig_height) = embeddings.original_size;
    // SAM 2 sees the photo stretched to a square, so x and y scale independently.
    let scale_x = SAM_INPUT_SIZE as f64 / orig_width as f64;
    let scale_y = SAM_INPUT_SIZE as f64 / orig_height as f64;
    let n = SAM_MASK_SIZE;
    let area = n * n;
    // Centre of a low-res mask cell, in input pixels (the decoder adds the half pixel itself).
    let cell_to_input = |c: usize| (c as f32 + 0.5) * (SAM_INPUT_SIZE as f32 / n as f32) - 0.5;

    let iters = 2;

    let is_point =
        (start_point.0 - end_point.0).abs() < 1e-6 && (start_point.1 - end_point.1).abs() < 1e-6;
    let mut point_coords = Vec::new();
    let mut point_labels = Vec::new();

    if is_point {
        point_coords.push((
            (start_point.0 * scale_x) as f32,
            (start_point.1 * scale_y) as f32,
        ));
        point_labels.push(1.0f32);
    } else {
        let x1 = (start_point.0.min(end_point.0) * scale_x) as f32;
        let y1 = (start_point.1.min(end_point.1) * scale_y) as f32;
        let x2 = (start_point.0.max(end_point.0) * scale_x) as f32;
        let y2 = (start_point.1.max(end_point.1) * scale_y) as f32;
        point_coords.push((x1, y1));
        point_coords.push((x2, y2));
        point_labels.push(2.0f32);
        point_labels.push(3.0f32);
    }

    let mut mask_input = vec![0.0f32; area];
    let mut has_mask_input = 0.0f32;
    let mut logits = Vec::new();

    for i in 0..iters {
        let coords_flat: Vec<f32> = point_coords.iter().flat_map(|&(x, y)| [x, y]).collect();
        let t_point_coords = Tensor::from_array(Array::from_shape_vec(
            (1, point_coords.len(), 2),
            coords_flat,
        )?)?;
        let t_point_labels = Tensor::from_array(Array::from_shape_vec(
            (1, point_labels.len()),
            point_labels.clone(),
        )?)?;
        let t_mask_input =
            Tensor::from_array(Array::from_shape_vec((1, 1, n, n), mask_input.clone())?)?;
        let t_has_mask = Tensor::from_array(Array::from_elem(1, has_mask_input))?;

        let (masks, iou): (Vec<f32>, Vec<f32>) = {
            let mut session = decoder.lock().unwrap();
            let outputs = session.run(ort::inputs![
                Tensor::from_array(embeddings.image_embed.clone())?,
                Tensor::from_array(embeddings.high_res_feats_0.clone())?,
                Tensor::from_array(embeddings.high_res_feats_1.clone())?,
                t_point_coords,
                t_point_labels,
                t_mask_input,
                t_has_mask
            ])?;
            (
                outputs["masks"]
                    .try_extract_array::<f32>()?
                    .iter()
                    .copied()
                    .collect(),
                outputs["iou_predictions"]
                    .try_extract_array::<f32>()?
                    .iter()
                    .copied()
                    .collect(),
            )
        };
        if masks.len() != 4 * area || iou.len() != 4 {
            return Err(anyhow::anyhow!("Unexpected SAM decoder output size"));
        }

        let selected = select_sam2_mask(&masks, &iou, is_point && i == 0);

        if i == iters - 1 {
            logits = selected.to_vec();
            break;
        }

        // Second pass: re-prompt with the first mask's most interior point, the most distant
        // background point inside its bounding box, the box itself, and the mask as a prior.
        let binary_mask: Vec<bool> = selected.iter().map(|&v| v > 0.0).collect();
        let mut mask_area = 0.0;
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (n, n, 0, 0);
        for (idx, _) in binary_mask
            .iter()
            .enumerate()
            .filter(|(_, inside)| **inside)
        {
            let (x, y) = (idx % n, idx / n);
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
            mask_area += 1.0;
        }

        if mask_area == 0.0 {
            logits = selected.to_vec();
            break;
        }

        let dt_in = edt_2d(&binary_mask, n, n);
        let (pos_idx, max_in) =
            dt_in
                .iter()
                .copied()
                .enumerate()
                .fold(
                    (0, 0.0f32),
                    |best, (idx, v)| if v > best.1 { (idx, v) } else { best },
                );

        let rev_mask: Vec<bool> = binary_mask.iter().map(|&inside| !inside).collect();
        let dt_out = edt_2d(&rev_mask, n, n);
        let neg_idx = dt_out
            .iter()
            .copied()
            .enumerate()
            .filter(|(idx, _)| {
                let (x, y) = (idx % n, idx / n);
                (min_x..=max_x).contains(&x) && (min_y..=max_y).contains(&y)
            })
            .fold(
                (0, 0.0f32),
                |best, (idx, v)| if v > best.1 { (idx, v) } else { best },
            )
            .0;

        point_coords = vec![
            (cell_to_input(pos_idx % n), cell_to_input(pos_idx / n)),
            (cell_to_input(neg_idx % n), cell_to_input(neg_idx / n)),
            (cell_to_input(min_x), cell_to_input(min_y)),
            (cell_to_input(max_x), cell_to_input(max_y)),
        ];
        point_labels = vec![1.0, 0.0, 2.0, 3.0];

        let variance = (mask_area / 4.0_f32).max(1.0_f32);
        for idx in 0..area {
            mask_input[idx] = if binary_mask[idx] {
                let diff = dt_in[idx] - max_in;
                let weight = (-(diff * diff) / variance).exp();
                15.0 * if weight > 0.0 { weight } else { 1.0 }
            } else {
                -15.0
            };
        }
        has_mask_input = 1.0;
    }

    let (out_width, out_height) =
        original_image.map_or(embeddings.original_size, |img| img.dimensions());
    let binary_mask = upscale_grid(&logits, n, n, out_width, out_height, |v| {
        if v > 0.0 { 255 } else { 0 }
    });

    if let Some(img) = original_image {
        let guide_hr = img.to_luma8();

        let long_edge = out_width.max(out_height) as f32;
        let radius = (long_edge * 0.0075).clamp(8.0, 24.0);

        let feathered_coarse_mask = image::imageops::blur(&binary_mask, radius / 3.0);

        let mut refined_mask =
            fast_guided_filter(&guide_hr, &feathered_coarse_mask, radius as usize, 0.01);

        for p in refined_mask.pixels_mut() {
            let v = p[0] as f32 / 255.0;
            p[0] = (((v - 0.03) / 0.94).clamp(0.0, 1.0) * 255.0).round() as u8;
        }

        Ok(refined_mask)
    } else {
        Ok(image::imageops::blur(&binary_mask, 2.0))
    }
}

/// Sky mask from SkyWater-Seg. The model sees the whole frame stretched to its square input, and
/// a 2x2 grid of overlapping tiles from a copy about 1.6x that size, which recovers small patches
/// of sky (under arms, between branches) the whole-frame pass misses. The averaged probabilities
/// are then fitted to the photo's edges with two colour guided filters: a coarse one that turns
/// the model's blurry boundary into a clean edge, and a finer one whose coefficients are applied
/// to the full-resolution image to recover detail like antennas and spikes.
pub fn run_sky_seg_model(image: &DynamicImage, sky_session: &Mutex<Session>) -> Result<GrayImage> {
    let (width, height) = image.dimensions();
    let rgb = image.to_rgb8();
    let size = SKY_INPUT_SIZE as usize;

    let global = sky_probabilities(
        sky_session,
        imagenet_tensor(&imageops::thumbnail(&rgb, SKY_INPUT_SIZE, SKY_INPUT_SIZE)),
    )?;

    let scale = 1.6 * SKY_INPUT_SIZE as f32 / width.max(height) as f32;
    let tiled_w = ((width as f32 * scale).round() as u32).max(SKY_INPUT_SIZE);
    let tiled_h = ((height as f32 * scale).round() as u32).max(SKY_INPUT_SIZE);
    let tiled_image = imageops::thumbnail(&rgb, tiled_w, tiled_h);
    let (tw, th) = (tiled_w as usize, tiled_h as usize);
    let mut tiled = vec![0.0f32; tw * th];
    let mut counts = vec![0u8; tw * th];
    let starts = |len: usize| {
        if len > size {
            vec![0, len - size]
        } else {
            vec![0]
        }
    };
    for &y0 in &starts(th) {
        for &x0 in &starts(tw) {
            let tile = imageops::crop_imm(
                &tiled_image,
                x0 as u32,
                y0 as u32,
                SKY_INPUT_SIZE,
                SKY_INPUT_SIZE,
            )
            .to_image();
            let probabilities = sky_probabilities(sky_session, imagenet_tensor(&tile))?;
            for (y, row) in probabilities.chunks_exact(size).enumerate() {
                let offset = (y0 + y) * tw + x0;
                for (x, &v) in row.iter().enumerate() {
                    tiled[offset + x] += v;
                    counts[offset + x] += 1;
                }
            }
        }
    }
    for (v, &n) in tiled.iter_mut().zip(&counts) {
        *v /= n.max(1) as f32;
    }

    // Coarse pass at ~768px: average the two predictions and stretch them so faint detections
    // (below 0.1) drop out and confident ones (above 0.6) count as full sky.
    let (w1, h1) = fit_long_edge(width, height, 768);
    let global_1 = resample_grid(&global, size, size, w1, h1);
    let tiled_1 = resample_grid(&tiled, tw, th, w1, h1);
    let fused: Vec<f32> = global_1
        .iter()
        .zip(&tiled_1)
        .map(|(&g, &t)| ((0.5 * (g + t) - 0.1) / 0.5).clamp(0.0, 1.0))
        .collect();
    let guide_1 = rgb_planes(&imageops::thumbnail(&rgb, w1 as u32, h1 as u32));
    let (a, b) = guided_filter_rgb(&guide_1, &fused, w1, h1, 6, 1e-3);
    let coarse: Vec<f32> = (0..w1 * h1)
        .into_par_iter()
        .map(|i| {
            let q =
                a[0][i] * guide_1[0][i] + a[1][i] * guide_1[1][i] + a[2][i] * guide_1[2][i] + b[i];
            let t = ((q - 0.2) / 0.6).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        })
        .collect();

    // Fine pass at ~1536px, with its linear model applied to the full-resolution pixels.
    let (w2, h2) = fit_long_edge(width, height, 1536);
    let guide_2 = rgb_planes(&imageops::thumbnail(&rgb, w2 as u32, h2 as u32));
    let coarse_2 = resample_grid(&coarse, w1, h1, w2, h2);
    let (a, b) = guided_filter_rgb(&guide_2, &coarse_2, w2, h2, 3, 4e-3);
    Ok(apply_guided_coefficients(&rgb, &a, &b, w2, h2))
}

/// Runs SkyWater-Seg on one normalised 512x512 tensor and returns the per-pixel sky probability
/// (softmax over the four classes).
fn sky_probabilities(sky_session: &Mutex<Session>, input: Array4<f32>) -> Result<Vec<f32>> {
    let size = SKY_INPUT_SIZE as usize;
    let plane = size * size;
    let logits: Vec<f32> = {
        let mut session = sky_session.lock().unwrap();
        let outputs = session.run(ort::inputs![Tensor::from_array(input.into_dyn())?])?;
        outputs[0]
            .try_extract_array::<f32>()?
            .iter()
            .copied()
            .collect()
    };
    if logits.len() != SKY_NUM_CLASSES * plane {
        return Err(anyhow::anyhow!("Unexpected sky model output size"));
    }
    Ok((0..plane)
        .into_par_iter()
        .map(|i| {
            let max = (0..SKY_NUM_CLASSES)
                .map(|c| logits[c * plane + i])
                .fold(f32::MIN, f32::max);
            let sum: f32 = (0..SKY_NUM_CLASSES)
                .map(|c| (logits[c * plane + i] - max).exp())
                .sum();
            (logits[SKY_CLASS * plane + i] - max).exp() / sum
        })
        .collect())
}

/// Scales `width` x `height` down so its long edge is at most `long_edge`.
fn fit_long_edge(width: u32, height: u32, long_edge: u32) -> (usize, usize) {
    let scale = (long_edge as f32 / width.max(height) as f32).min(1.0);
    (
        ((width as f32 * scale).round() as usize).max(1),
        ((height as f32 * scale).round() as usize).max(1),
    )
}

/// Bilinear resize of a float grid (pixel centres aligned, edges clamped).
fn resample_grid(src: &[f32], src_w: usize, src_h: usize, w: usize, h: usize) -> Vec<f32> {
    let cols: Vec<(usize, usize, f32)> = (0..w).map(|x| bilinear_tap(x, w, src_w)).collect();
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let (y0, y1, wy) = bilinear_tap(y, h, src_h);
        let r0 = &src[y0 * src_w..(y0 + 1) * src_w];
        let r1 = &src[y1 * src_w..(y1 + 1) * src_w];
        for (o, &(x0, x1, wx)) in row.iter_mut().zip(&cols) {
            let top = r0[x0] + (r0[x1] - r0[x0]) * wx;
            let bottom = r1[x0] + (r1[x1] - r1[x0]) * wx;
            *o = top + (bottom - top) * wy;
        }
    });
    out
}

/// Source indices and weight for sampling output pixel `dst` of `dst_len` from `src_len` pixels.
fn bilinear_tap(dst: usize, dst_len: usize, src_len: usize) -> (usize, usize, f32) {
    let f = ((dst as f32 + 0.5) * src_len as f32 / dst_len as f32 - 0.5)
        .clamp(0.0, (src_len - 1) as f32);
    let i0 = f as usize;
    (i0, (i0 + 1).min(src_len - 1), f - i0 as f32)
}

fn rgb_planes(image: &image::RgbImage) -> [Vec<f32>; 3] {
    std::array::from_fn(|c| {
        image
            .as_raw()
            .par_chunks_exact(3)
            .map(|px| px[c] as f32 / 255.0)
            .collect()
    })
}

fn box_mean(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let mut dest = vec![0.0; w * h];
    let mut temp1 = vec![0.0; w * h];
    let mut temp2 = vec![0.0; w * h];
    box_filter_opt(src, &mut dest, &mut temp1, &mut temp2, w, h, r);
    dest
}

/// Colour guided filter (He et al.): fits `q = a . I + b` to `p` in every (2r+1)^2 window of the
/// RGB guide `guide`, and returns the window-averaged coefficients `a` (per channel) and `b`.
fn guided_filter_rgb(
    guide: &[Vec<f32>; 3],
    p: &[f32],
    w: usize,
    h: usize,
    r: usize,
    eps: f32,
) -> ([Vec<f32>; 3], Vec<f32>) {
    let product =
        |x: &[f32], y: &[f32]| -> Vec<f32> { x.par_iter().zip(y).map(|(a, b)| a * b).collect() };
    let mean_i: [Vec<f32>; 3] = std::array::from_fn(|c| box_mean(&guide[c], w, h, r));
    let mean_p = box_mean(p, w, h, r);
    let mean_ip: [Vec<f32>; 3] = std::array::from_fn(|c| box_mean(&product(&guide[c], p), w, h, r));
    // Upper triangle of the guide's covariance: rr, rg, rb, gg, gb, bb.
    const PAIRS: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
    let mean_ii: Vec<Vec<f32>> = PAIRS
        .iter()
        .map(|&(i, j)| box_mean(&product(&guide[i], &guide[j]), w, h, r))
        .collect();

    let coefficients: Vec<[f32; 4]> = (0..w * h)
        .into_par_iter()
        .map(|k| {
            let mi = [mean_i[0][k], mean_i[1][k], mean_i[2][k]];
            let mp = mean_p[k];
            let cov: [f32; 3] = std::array::from_fn(|c| mean_ip[c][k] - mi[c] * mp);
            let s: [f32; 6] = std::array::from_fn(|n| {
                let (i, j) = PAIRS[n];
                mean_ii[n][k] - mi[i] * mi[j] + if i == j { eps } else { 0.0 }
            });
            let a = solve_symmetric_3x3(s, cov);
            [
                a[0],
                a[1],
                a[2],
                mp - a[0] * mi[0] - a[1] * mi[1] - a[2] * mi[2],
            ]
        })
        .collect();

    let a = std::array::from_fn(|c| {
        let plane: Vec<f32> = coefficients.par_iter().map(|v| v[c]).collect();
        box_mean(&plane, w, h, r)
    });
    let b: Vec<f32> = coefficients.par_iter().map(|v| v[3]).collect();
    (a, box_mean(&b, w, h, r))
}

/// Solves `S x = v` for a symmetric 3x3 `S` given as its upper triangle (xx, xy, xz, yy, yz, zz).
fn solve_symmetric_3x3(s: [f32; 6], v: [f32; 3]) -> [f32; 3] {
    let [a, b, c, d, e, f] = s;
    let c00 = d * f - e * e;
    let c01 = c * e - b * f;
    let c02 = b * e - c * d;
    let c11 = a * f - c * c;
    let c12 = b * c - a * e;
    let c22 = a * d - b * b;
    let det = a * c00 + b * c01 + c * c02;
    if det.abs() < 1e-20 {
        return [0.0; 3];
    }
    [
        (c00 * v[0] + c01 * v[1] + c02 * v[2]) / det,
        (c01 * v[0] + c11 * v[1] + c12 * v[2]) / det,
        (c02 * v[0] + c12 * v[1] + c22 * v[2]) / det,
    ]
}

/// Evaluates a guided filter's `q = a . I + b` at full resolution, bilinearly upsampling the
/// low-resolution coefficients (the "fast guided filter").
fn apply_guided_coefficients(
    image: &image::RgbImage,
    a: &[Vec<f32>; 3],
    b: &[f32],
    grid_w: usize,
    grid_h: usize,
) -> GrayImage {
    let (width, height) = image.dimensions();
    let (w, h) = (width as usize, height as usize);
    let cols: Vec<(usize, usize, f32)> = (0..w).map(|x| bilinear_tap(x, w, grid_w)).collect();
    let pixels = image.as_raw();
    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let (y0, y1, wy) = bilinear_tap(y, h, grid_h);
        let sample = |grid: &[f32], x0: usize, x1: usize, wx: f32| {
            let r0 = &grid[y0 * grid_w..(y0 + 1) * grid_w];
            let r1 = &grid[y1 * grid_w..(y1 + 1) * grid_w];
            let top = r0[x0] + (r0[x1] - r0[x0]) * wx;
            let bottom = r1[x0] + (r1[x1] - r1[x0]) * wx;
            top + (bottom - top) * wy
        };
        let src = &pixels[y * w * 3..(y + 1) * w * 3];
        for (x, (o, &(x0, x1, wx))) in row.iter_mut().zip(&cols).enumerate() {
            let mut q = sample(b, x0, x1, wx);
            for c in 0..3 {
                q += sample(&a[c], x0, x1, wx) * (src[x * 3 + c] as f32 / 255.0);
            }
            *o = (q.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    });
    GrayImage::from_raw(width, height, out).expect("buffer matches dimensions")
}

/// Foreground mask from BiRefNet. The model expects a plain (stretched) resize to its square
/// input with ImageNet normalisation, and returns per-pixel probabilities.
pub fn run_birefnet_model(
    image: &DynamicImage,
    birefnet_session: &Mutex<Session>,
) -> Result<GrayImage> {
    let (orig_width, orig_height) = image.dimensions();
    let size = BIREFNET_INPUT_SIZE as usize;

    let t_input =
        Tensor::from_array(imagenet_square_tensor(image, BIREFNET_INPUT_SIZE).into_dyn())?;

    let probabilities: Vec<f32> = {
        let mut session = birefnet_session.lock().unwrap();
        let outputs = session.run(ort::inputs![t_input])?;
        outputs[0]
            .try_extract_array::<f32>()?
            .iter()
            .copied()
            .collect()
    };
    if probabilities.len() != size * size {
        return Err(anyhow::anyhow!("Unexpected BiRefNet output size"));
    }

    Ok(upscale_grid(
        &probabilities,
        size,
        size,
        orig_width,
        orig_height,
        |v| (v.clamp(0.0, 1.0) * 255.0).round() as u8,
    ))
}

/// Stretches `image` to a `size` x `size` square and normalises it with the ImageNet mean/std:
/// the input BiRefNet and SAM 2 both expect.
fn imagenet_square_tensor(image: &DynamicImage, size: u32) -> Array4<f32> {
    // Area-average downscale. Convert to 8-bit first: image's thumbnail mis-scales f32 pixels.
    imagenet_tensor(&imageops::thumbnail(&image.to_rgb8(), size, size))
}

/// Normalises an RGB image with the ImageNet mean/std into a 1x3xHxW tensor.
fn imagenet_tensor(rgb: &image::RgbImage) -> Array4<f32> {
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let (width, height) = rgb.dimensions();
    let mut tensor = Array4::zeros((1, 3, height as usize, width as usize));
    for (x, y, pixel) in rgb.enumerate_pixels() {
        for c in 0..3 {
            tensor[[0, c, y as usize, x as usize]] = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
        }
    }
    tensor
}

/// Bilinearly resizes a low-res grid of model outputs to a `width` x `height` mask, mapping each
/// interpolated value to a pixel with `to_pixel`. Runs in parallel over rows: several times
/// faster than `imageops::resize` for full-resolution photo masks.
fn upscale_grid(
    src: &[f32],
    src_w: usize,
    src_h: usize,
    width: u32,
    height: u32,
    to_pixel: impl Fn(f32) -> u8 + Sync,
) -> GrayImage {
    let (w, h) = (width as usize, height as usize);
    let cols: Vec<(usize, usize, f32)> = (0..w).map(|x| bilinear_tap(x, w, src_w)).collect();

    let mut out = vec![0u8; w * h];
    out.par_chunks_mut(w.max(1))
        .enumerate()
        .for_each(|(y, row)| {
            let (y0, y1, wy) = bilinear_tap(y, h, src_h);
            let r0 = &src[y0 * src_w..(y0 + 1) * src_w];
            let r1 = &src[y1 * src_w..(y1 + 1) * src_w];
            for (o, &(x0, x1, wx)) in row.iter_mut().zip(&cols) {
                let top = r0[x0] + (r0[x1] - r0[x0]) * wx;
                let bottom = r1[x0] + (r1[x1] - r1[x0]) * wx;
                *o = to_pixel(top + (bottom - top) * wy);
            }
        });
    GrayImage::from_raw(width, height, out).expect("buffer matches dimensions")
}

pub fn run_depth_anything_model(
    image: &DynamicImage,
    depth_session: &Mutex<Session>,
) -> Result<GrayImage> {
    let resized_image = image.resize(DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE, FilterType::Triangle);
    let (resized_w, resized_h) = resized_image.dimensions();
    let resized_rgb = resized_image.into_rgb8();
    let raw_pixels = resized_rgb.as_raw();

    let paste_x = ((DEPTH_INPUT_SIZE - resized_w) / 2) as usize;
    let paste_y = ((DEPTH_INPUT_SIZE - resized_h) / 2) as usize;

    let mut input_tensor: Array<f32, _> =
        Array::zeros((1, 3, DEPTH_INPUT_SIZE as usize, DEPTH_INPUT_SIZE as usize));

    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    let rw = resized_w as usize;
    let rh = resized_h as usize;

    for y in 0..rh {
        for x in 0..rw {
            let idx = (y * rw + x) * 3;
            let dest_y = y + paste_y;
            let dest_x = x + paste_x;

            input_tensor[[0, 0, dest_y, dest_x]] =
                (raw_pixels[idx] as f32 / 255.0 - mean[0]) / std[0];
            input_tensor[[0, 1, dest_y, dest_x]] =
                (raw_pixels[idx + 1] as f32 / 255.0 - mean[1]) / std[1];
            input_tensor[[0, 2, dest_y, dest_x]] =
                (raw_pixels[idx + 2] as f32 / 255.0 - mean[2]) / std[2];
        }
    }

    let input_tensor_dyn = input_tensor.into_dyn();
    let t_input = Tensor::from_array(input_tensor_dyn.as_standard_layout().into_owned())?;

    let mut session = depth_session.lock().unwrap();
    let outputs = session.run(ort::inputs![t_input])?;
    let output_tensor = outputs[0].try_extract_array::<f32>()?.to_owned();
    let out_slice = output_tensor.as_slice().unwrap();

    let usize_size = DEPTH_INPUT_SIZE as usize;

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for y in 0..rh {
        let src_y = y + paste_y;
        for x in 0..rw {
            let src_x = x + paste_x;
            let val = out_slice[src_y * usize_size + src_x];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    let range = max_val - min_val;
    let scale = if range > 1e-6 { 255.0 / range } else { 0.0 };

    let mut cropped_depth_data = Vec::with_capacity(rw * rh);

    for y in 0..rh {
        let src_y = y + paste_y;
        for x in 0..rw {
            let src_x = x + paste_x;
            let val = out_slice[src_y * usize_size + src_x];
            let pixel = if range > 1e-6 {
                ((val - min_val) * scale) as u8
            } else {
                0
            };
            cropped_depth_data.push(pixel);
        }
    }

    let depth_map = GrayImage::from_raw(resized_w, resized_h, cropped_depth_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create mask from Depth output"))?;

    Ok(depth_map)
}

fn build_normal_depth(points: &[f32], mask: &[f32]) -> Vec<u8> {
    let valid = |point: &[f32; 3], m: f32| {
        m >= 0.5 && point.iter().all(|v| v.is_finite()) && point[2] > 0.0
    };

    let mut depths: Vec<f32> = points
        .as_chunks::<3>()
        .0
        .iter()
        .zip(mask)
        .filter(|(point, m)| valid(point, **m))
        .map(|(point, _)| point[2])
        .collect();

    if depths.len() < 16 {
        return vec![255; mask.len()];
    }

    depths.sort_unstable_by(|a, b| a.total_cmp(b));
    let percentile = |p: f32| depths[((depths.len() - 1) as f32 * p) as usize];
    let near = percentile(0.005);
    let range = (percentile(0.995) - near).max(1e-6);

    points
        .as_chunks::<3>()
        .0
        .iter()
        .zip(mask)
        .map(|(point, m)| {
            if valid(point, *m) {
                (((point[2] - near) / range).clamp(0.0, 1.0) * 255.0).round() as u8
            } else {
                255
            }
        })
        .collect()
}

pub fn run_normal_model(
    image: &DynamicImage,
    normal_session: &Mutex<Session>,
) -> Result<RgbaImage> {
    let resized_image = if image.width().max(image.height()) > NORMAL_MAX_SIZE {
        image.resize(NORMAL_MAX_SIZE, NORMAL_MAX_SIZE, FilterType::Triangle)
    } else {
        image.clone()
    };
    let (resized_w, resized_h) = resized_image.dimensions();
    let resized_rgb = resized_image.into_rgb8();
    let raw_pixels = resized_rgb.as_raw();

    let rw = resized_w as usize;
    let rh = resized_h as usize;

    let mut input_tensor: Array<f32, _> = Array::zeros((1, 3, rh, rw));
    for y in 0..rh {
        for x in 0..rw {
            let idx = (y * rw + x) * 3;
            input_tensor[[0, 0, y, x]] = raw_pixels[idx] as f32 / 255.0;
            input_tensor[[0, 1, y, x]] = raw_pixels[idx + 1] as f32 / 255.0;
            input_tensor[[0, 2, y, x]] = raw_pixels[idx + 2] as f32 / 255.0;
        }
    }

    let t_image = Tensor::from_array(input_tensor.into_dyn())?;
    let t_num_tokens = Tensor::from_array(ndarray::arr0(NORMAL_NUM_TOKENS))?;

    let mut session = normal_session.lock().unwrap();
    let outputs = session.run(ort::inputs![
        "image" => t_image,
        "num_tokens" => t_num_tokens,
    ])?;

    let normal_tensor = outputs["normal"].try_extract_array::<f32>()?;
    let mask_tensor = outputs["mask"].try_extract_array::<f32>()?;
    let points_tensor = outputs["points"].try_extract_array::<f32>()?;

    if normal_tensor.shape() != [1, rh, rw, 3]
        || mask_tensor.shape() != [1, rh, rw]
        || points_tensor.shape() != [1, rh, rw, 3]
    {
        return Err(anyhow::anyhow!(
            "Unexpected normal model output shape: {:?} / {:?} / {:?}",
            normal_tensor.shape(),
            mask_tensor.shape(),
            points_tensor.shape()
        ));
    }

    let normals = normal_tensor.as_standard_layout();
    let normals = normals.as_slice().unwrap();
    let mask = mask_tensor.as_standard_layout();
    let mask = mask.as_slice().unwrap();
    let points = points_tensor.as_standard_layout();
    let depth = build_normal_depth(points.as_slice().unwrap(), mask);

    let pack = |v: f32| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;

    let mut normal_data = Vec::with_capacity(rw * rh * 4);
    for (n, d) in normals.as_chunks::<3>().0.iter().zip(depth) {
        let (nx, ny, nz) = (n[0], -n[1], -n[2]);
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len.is_finite() && len > 1e-6 {
            normal_data.extend_from_slice(&[pack(nx / len), pack(ny / len), pack(nz / len)]);
        } else {
            normal_data.extend_from_slice(&[128, 128, 255]);
        }
        normal_data.push(d);
    }

    RgbaImage::from_raw(resized_w, resized_h, normal_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create normal map from model output"))
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiSubjectMaskParameters {
    pub start_x: f64,
    pub start_y: f64,
    pub end_x: f64,
    pub end_y: f64,
    #[serde(default)]
    pub mask_data_base64: Option<String>,
    #[serde(default)]
    pub rotation: Option<f32>,
    #[serde(default)]
    pub flip_horizontal: Option<bool>,
    #[serde(default)]
    pub flip_vertical: Option<bool>,
    #[serde(default)]
    pub orientation_steps: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiSkyMaskParameters {
    #[serde(default)]
    pub mask_data_base64: Option<String>,
    #[serde(default)]
    pub rotation: Option<f32>,
    #[serde(default)]
    pub flip_horizontal: Option<bool>,
    #[serde(default)]
    pub flip_vertical: Option<bool>,
    #[serde(default)]
    pub orientation_steps: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiForegroundMaskParameters {
    #[serde(default)]
    pub mask_data_base64: Option<String>,
    #[serde(default)]
    pub rotation: Option<f32>,
    #[serde(default)]
    pub flip_horizontal: Option<bool>,
    #[serde(default)]
    pub flip_vertical: Option<bool>,
    #[serde(default)]
    pub orientation_steps: Option<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiDepthMaskParameters {
    #[serde(default)]
    pub min_depth: f32,
    #[serde(default)]
    pub max_depth: f32,
    #[serde(default)]
    pub min_fade: f32,
    #[serde(default)]
    pub max_fade: f32,
    #[serde(default)]
    pub feather: f32,
    #[serde(default)]
    pub mask_data_base64: Option<String>,
    #[serde(default)]
    pub rotation: Option<f32>,
    #[serde(default)]
    pub flip_horizontal: Option<bool>,
    #[serde(default)]
    pub flip_vertical: Option<bool>,
    #[serde(default)]
    pub orientation_steps: Option<u8>,
}
