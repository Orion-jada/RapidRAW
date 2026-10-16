//! Culling suggestions: groups of similar shots with the best one picked, and blurry photos.
//!
//! Sharpness is judged on the camera's full-size embedded JPEG for RAW files (no RAW decode),
//! at about half resolution. For each 64px tile with real structure (contrast well above the
//! noise), the edge width is the step across its edges divided by its steepest gradient: 3-5px
//! when in focus, growing with defocus. Taking the sharpest tiles finds the in-focus subject
//! even against a blurred background, and measuring x and y separately catches camera shake,
//! which leaves edges along the motion sharp.
//!
//! Similar shots are found with the AI search model's image embeddings, compared between
//! photos taken close together in time. Without the model, a perceptual hash stands in.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use image::{DynamicImage, GenericImageView, GrayImage};
use image_hasher::{HashAlg, HasherConfig, ImageHash};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

use crate::AppState;
use crate::app_settings::load_settings;
use crate::file_management::{self, parse_virtual_path};
use crate::formats::is_raw_file;
use crate::image_loader;

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CullingSettings {
    /// How alike two photos must be to be grouped, in percent.
    pub similarity_threshold: u32,
    /// Photos with a sharpness score (0-100) below this are flagged as blurry.
    pub blur_threshold: f64,
    pub group_similar: bool,
    pub filter_blurry: bool,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImageAnalysisResult {
    pub path: String,
    /// 0-1, mostly sharpness; ranks the shots within a group.
    pub quality_score: f64,
    /// 0-100; `None` when the photo has too little structure to judge.
    pub sharpness_metric: Option<f64>,
    /// Width in pixels of the sharpest edges, at about half resolution.
    pub edge_width: Option<f64>,
    /// 0-1, lower with more clipped highlights and shadows.
    pub exposure_metric: f64,
    pub width: u32,
    pub height: u32,
}

#[derive(Serialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CullGroup {
    pub representative: ImageAnalysisResult,
    pub duplicates: Vec<ImageAnalysisResult>,
}

#[derive(Serialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct CullingSuggestions {
    pub similar_groups: Vec<CullGroup>,
    pub blurry_images: Vec<ImageAnalysisResult>,
    pub failed_paths: Vec<String>,
}

#[derive(Serialize, Clone)]
struct CullingProgress {
    current: usize,
    total: usize,
    /// "analyzing", "comparing" or "grouping"; the frontend translates it.
    stage: &'static str,
}

struct ImageAnalysis {
    result: ImageAnalysisResult,
    /// Seconds since the epoch, from EXIF.
    captured_at: Option<i64>,
    hash: ImageHash,
}

const ANALYSIS_LONG_EDGE: u32 = 3000;
const TILE: usize = 64;
// A tile counts as structure when its contrast (2nd to 98th percentile) clears both of these.
const MIN_TILE_CONTRAST: f32 = 0.08;
const MIN_CONTRAST_OVER_NOISE: f32 = 10.0;
const SHARPEST_TILE_SHARE: usize = 10;
// Edge width to sharpness score: this wide or narrower scores 100, `BLURRED_EDGE_WIDTH` 0.
const SHARP_EDGE_WIDTH: f64 = 2.5;
const BLURRED_EDGE_WIDTH: f64 = 8.5;
const WEIGHT_SHARPNESS: f64 = 0.75;
const WEIGHT_EXPOSURE: f64 = 0.25;
// Embedded previews smaller than this are too coarse to judge focus; decode the RAW instead.
const MIN_PREVIEW_LONG_EDGE: u32 = 2000;

// Shots taken within `BURST_WINDOW_SECS` of each other are grouped at a slightly lower
// similarity; nothing further apart than `MAX_GROUP_GAP_SECS` is compared.
const BURST_WINDOW_SECS: i64 = 60;
const BURST_SIMILARITY_ALLOWANCE: f32 = 0.04;
const MAX_GROUP_GAP_SECS: i64 = 600;
const MAX_NEIGHBOURS: usize = 30;
// Hash distance standing in for the default similarity when there are no embeddings.
const DEFAULT_SIMILARITY: f32 = 0.92;
const HASH_DISTANCE_AT_DEFAULT: f32 = 28.0;

/// Luma at analysis resolution, sRGB-encoded in [0, 1].
struct Luma {
    data: Vec<f32>,
    width: usize,
    height: usize,
}

impl Luma {
    /// Box-downsamples by a whole factor so the long edge is at most `ANALYSIS_LONG_EDGE`.
    /// `linear` input (a developed RAW) gets a display gamma first.
    fn from_image(image: &DynamicImage, linear: bool) -> Self {
        let rgb = if linear {
            let mut float = image.to_rgb32f();
            float
                .iter_mut()
                .for_each(|v| *v = v.clamp(0.0, 1.0).powf(1.0 / 2.2));
            DynamicImage::ImageRgb32F(float).to_rgb8()
        } else {
            image.to_rgb8()
        };
        let (w, h) = rgb.dimensions();
        let factor = w.max(h).div_ceil(ANALYSIS_LONG_EDGE).max(1) as usize;
        let (width, height) = (w as usize / factor, h as usize / factor);
        let raw = rgb.as_raw();
        let stride = w as usize * 3;
        let norm = 1.0 / (255.0 * (factor * factor) as f32);

        let mut data = vec![0.0f32; width * height];
        data.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
            for (x, out) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for dy in 0..factor {
                    let line = &raw[(y * factor + dy) * stride..];
                    for dx in 0..factor {
                        let p = &line[(x * factor + dx) * 3..];
                        sum += 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
                    }
                }
                *out = sum * norm;
            }
        });
        Self {
            data,
            width,
            height,
        }
    }

    /// Immerkær's fast noise estimate (standard deviation).
    fn noise_sigma(&self) -> f32 {
        let (w, h) = (self.width, self.height);
        if w < 3 || h < 3 {
            return 0.0;
        }
        let d = &self.data;
        let sum: f64 = (1..h - 1)
            .into_par_iter()
            .map(|y| {
                let mut row = 0.0f64;
                for x in 1..w - 1 {
                    let at = |dx: usize, dy: usize| d[(y + dy - 1) * w + x + dx - 1];
                    let v = at(0, 0) - 2.0 * at(1, 0) + at(2, 0) - 2.0 * at(0, 1) + 4.0 * at(1, 1)
                        - 2.0 * at(2, 1)
                        + at(0, 2)
                        - 2.0 * at(1, 2)
                        + at(2, 2);
                    row += v.abs() as f64;
                }
                row
            })
            .sum();
        ((std::f64::consts::PI / 2.0).sqrt() * sum / (6.0 * ((w - 2) * (h - 2)) as f64)) as f32
    }

    fn gaussian(&self, sigma: f32) -> Vec<f32> {
        let radius = (3.0 * sigma).ceil() as isize;
        let mut kernel: Vec<f32> = (-radius..=radius)
            .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
            .collect();
        let total: f32 = kernel.iter().sum();
        kernel.iter_mut().for_each(|k| *k /= total);
        let (w, h) = (self.width as isize, self.height as isize);

        let mut horizontal = vec![0.0f32; self.data.len()];
        horizontal
            .par_chunks_mut(self.width)
            .enumerate()
            .for_each(|(y, row)| {
                let src = &self.data[y * self.width..(y + 1) * self.width];
                for (x, out) in row.iter_mut().enumerate() {
                    *out = kernel
                        .iter()
                        .enumerate()
                        .map(|(k, kv)| {
                            src[(x as isize + k as isize - radius).clamp(0, w - 1) as usize] * kv
                        })
                        .sum();
                }
            });
        let mut out = vec![0.0f32; self.data.len()];
        out.par_chunks_mut(self.width)
            .enumerate()
            .for_each(|(y, row)| {
                for (x, o) in row.iter_mut().enumerate() {
                    *o = kernel
                        .iter()
                        .enumerate()
                        .map(|(k, kv)| {
                            let yy = (y as isize + k as isize - radius).clamp(0, h - 1) as usize;
                            horizontal[yy * self.width + x] * kv
                        })
                        .sum();
                }
            });
        out
    }

    /// Width in pixels of the sharpest edges, in whichever of x and y is blurrier.
    fn edge_width(&self) -> Option<f64> {
        let (w, h) = (self.width, self.height);
        let (tiles_x, tiles_y) = (w / TILE, h / TILE);
        if tiles_x == 0 || tiles_y == 0 {
            return None;
        }
        let min_contrast = MIN_TILE_CONTRAST.max(MIN_CONTRAST_OVER_NOISE * self.noise_sigma());
        let smooth = self.gaussian(0.7);
        let contrast_src = self.gaussian(1.0);

        // Per tile and direction: the step across edges (90th percentile of the row or column
        // ranges) over the steepest gradient (99.5th percentile). Contrast is measured along
        // each direction so a tile of horizontal lines doesn't count as blurry in x.
        let tiles: Vec<[Option<f32>; 2]> = (0..tiles_x * tiles_y)
            .into_par_iter()
            .map(|t| {
                let (x0, y0) = ((t % tiles_x) * TILE, (t / tiles_x) * TILE);
                let at = |x: usize, y: usize| contrast_src[(y0 + y) * w + x0 + x];
                let range = |values: &mut dyn Iterator<Item = f32>| {
                    let (lo, hi) =
                        values.fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
                    hi - lo
                };
                let mut row_ranges: Vec<f32> = (0..TILE)
                    .map(|y| range(&mut (0..TILE).map(|x| at(x, y))))
                    .collect();
                let mut column_ranges: Vec<f32> = (0..TILE)
                    .map(|x| range(&mut (0..TILE).map(|y| at(x, y))))
                    .collect();
                let contrast = [
                    percentile(&mut row_ranges, 0.9),
                    percentile(&mut column_ranges, 0.9),
                ];
                if contrast.iter().all(|&c| c < min_contrast) {
                    return [None, None];
                }

                let mut gradients: [Vec<f32>; 2] = Default::default();
                for y in y0.max(1)..(y0 + TILE).min(h - 1) {
                    for x in x0.max(1)..(x0 + TILE).min(w - 1) {
                        let p = |dx: isize, dy: isize| {
                            smooth[(y as isize + dy) as usize * w + (x as isize + dx) as usize]
                        };
                        // Sobel, scaled to the step per pixel.
                        let gx = p(1, -1) + 2.0 * p(1, 0) + p(1, 1)
                            - p(-1, -1)
                            - 2.0 * p(-1, 0)
                            - p(-1, 1);
                        let gy = p(-1, 1) + 2.0 * p(0, 1) + p(1, 1)
                            - p(-1, -1)
                            - 2.0 * p(0, -1)
                            - p(1, -1);
                        gradients[0].push(gx.abs() / 8.0);
                        gradients[1].push(gy.abs() / 8.0);
                    }
                }
                let mut widths = [None, None];
                for d in 0..2 {
                    if contrast[d] >= min_contrast {
                        widths[d] =
                            Some(contrast[d] / percentile(&mut gradients[d], 0.995).max(1e-6));
                    }
                }
                widths
            })
            .collect();

        (0..2)
            .filter_map(|d| {
                let mut widths: Vec<f32> = tiles.iter().filter_map(|t| t[d]).collect();
                if widths.is_empty() {
                    return None;
                }
                widths.sort_by(f32::total_cmp);
                let keep = (widths.len() / SHARPEST_TILE_SHARE).max(1);
                Some(widths[..keep].iter().sum::<f32>() as f64 / keep as f64)
            })
            .max_by(f64::total_cmp)
    }

    /// 1 for a well-exposed photo, lower with clipped highlights (and, less so, shadows).
    fn exposure_score(&self) -> f64 {
        let n = self.data.len().max(1) as f64;
        let bright = self.data.iter().filter(|&&v| v >= 0.98).count() as f64 / n;
        let dark = self.data.iter().filter(|&&v| v <= 0.02).count() as f64 / n;
        (1.0 - 3.0 * bright - 0.5 * dark).clamp(0.0, 1.0)
    }

    fn to_gray_image(&self) -> GrayImage {
        GrayImage::from_fn(self.width as u32, self.height as u32, |x, y| {
            image::Luma([(self.data[y as usize * self.width + x as usize] * 255.0).round() as u8])
        })
    }
}

/// Nearest-rank percentile; reorders `values`.
fn percentile(values: &mut [f32], q: f32) -> f32 {
    let i = ((values.len() - 1) as f32 * q).round() as usize;
    *values.select_nth_unstable_by(i, f32::total_cmp).1
}

fn sharpness_score(edge_width: f64) -> f64 {
    (100.0 * (BLURRED_EDGE_WIDTH - edge_width) / (BLURRED_EDGE_WIDTH - SHARP_EDGE_WIDTH))
        .clamp(0.0, 100.0)
}

/// The image to judge focus on: a RAW's full-size embedded JPEG when it has one, else the
/// decoded file. The flag says whether it's linear.
fn analysis_image(
    bytes: &[u8],
    path: &str,
    settings: &crate::app_settings::AppSettings,
) -> Result<(DynamicImage, bool), String> {
    let is_raw = is_raw_file(path);
    if is_raw
        && let Some(preview) = image_loader::safe_embedded_preview_fallback(bytes, path)
        && preview.width().max(preview.height()) >= MIN_PREVIEW_LONG_EDGE
    {
        return Ok((preview, false));
    }
    let image = image_loader::load_base_image_from_bytes(bytes, path, true, settings, None)
        .map_err(|e| e.to_string())?;
    Ok((image, is_raw))
}

fn analyze_image(
    path: &str,
    hasher: &image_hasher::Hasher,
    settings: &crate::app_settings::AppSettings,
) -> Result<ImageAnalysis, String> {
    let (source_path, _) = parse_virtual_path(path);
    if file_management::is_cloud_placeholder(&source_path) {
        return Err(format!("'{}' is stored in iCloud and not downloaded", path));
    }
    let source = source_path.to_string_lossy();
    let bytes = file_management::read_file_mapped(&source_path).map_err(|e| e.to_string())?;
    let (image, linear) = analysis_image(&bytes, &source, settings)?;
    let (width, height) = image.dimensions();
    let luma = Luma::from_image(&image, linear);
    drop(image);

    let edge_width = luma.edge_width();
    let sharpness = edge_width.map(sharpness_score);
    let exposure = luma.exposure_score();
    // Photos too plain to judge rank as middling.
    let quality = WEIGHT_SHARPNESS * sharpness.unwrap_or(50.0) / 100.0 + WEIGHT_EXPOSURE * exposure;

    Ok(ImageAnalysis {
        result: ImageAnalysisResult {
            path: path.to_string(),
            quality_score: quality,
            sharpness_metric: sharpness,
            edge_width,
            exposure_metric: exposure,
            width,
            height,
        },
        captured_at: crate::exif_processing::try_get_exif_creation_date(Path::new(&*source))
            .map(|dt| dt.timestamp()),
        hash: hasher
            .hash_image(&DynamicImage::ImageLuma8(luma.to_gray_image()).thumbnail(512, 512)),
    })
}

struct DisjointSets(Vec<usize>);

impl DisjointSets {
    fn find(&mut self, i: usize) -> usize {
        let parent = self.0[i];
        if parent == i {
            return i;
        }
        let root = self.find(parent);
        self.0[i] = root;
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.0[b.max(a)] = a.min(b);
        }
    }
}

/// Groups shots of the same moment or view. Photos are compared in capture order with their
/// next few neighbours; a pair links when it's at least `threshold` alike, or nearly so and
/// taken within a minute. Returns groups of two or more, in capture order.
fn group_similar(
    analyses: &[ImageAnalysis],
    embeddings: Option<&[Option<Vec<f32>>]>,
    threshold: f32,
) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..analyses.len()).collect();
    order.sort_by(|&a, &b| {
        let key = |i: usize| (analyses[i].captured_at.is_none(), analyses[i].captured_at);
        key(a)
            .cmp(&key(b))
            .then_with(|| analyses[a].result.path.cmp(&analyses[b].result.path))
    });
    let hash_threshold = HASH_DISTANCE_AT_DEFAULT * (1.0 - threshold) / (1.0 - DEFAULT_SIMILARITY);

    let mut sets = DisjointSets((0..analyses.len()).collect());
    for (pos, &a) in order.iter().enumerate() {
        for &b in order.iter().skip(pos + 1).take(MAX_NEIGHBOURS) {
            let gap = match (analyses[a].captured_at, analyses[b].captured_at) {
                (Some(ta), Some(tb)) => Some(tb - ta),
                _ => None,
            };
            if gap.is_some_and(|g| g > MAX_GROUP_GAP_SECS) {
                break;
            }
            let in_burst = gap.is_some_and(|g| g <= BURST_WINDOW_SECS);
            let similar = match embeddings.map(|e| (&e[a], &e[b])) {
                Some((Some(ea), Some(eb))) => {
                    let cos: f32 = ea.iter().zip(eb).map(|(x, y)| x * y).sum();
                    cos >= threshold || (in_burst && cos >= threshold - BURST_SIMILARITY_ALLOWANCE)
                }
                _ => (analyses[a].hash.dist(&analyses[b].hash) as f32) <= hash_threshold,
            };
            if similar {
                sets.union(a, b);
            }
        }
    }

    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut group_of_root = std::collections::HashMap::new();
    for &i in &order {
        let root = sets.find(i);
        let slot = *group_of_root.entry(root).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[slot].push(i);
    }
    groups.retain(|g| g.len() > 1);
    groups
}

#[tauri::command]
pub async fn cull_images(
    paths: Vec<String>,
    settings: CullingSettings,
    app_handle: AppHandle,
    state: State<'_, AppState>,
) -> Result<CullingSuggestions, String> {
    if paths.is_empty() {
        return Ok(CullingSuggestions::default());
    }
    let total = paths.len();
    let _ = app_handle.emit("culling-start", total);
    let progress = |current: usize, total: usize, stage: &'static str| {
        let _ = app_handle.emit(
            "culling-progress",
            CullingProgress {
                current,
                total,
                stage,
            },
        );
    };

    let app_settings = load_settings(app_handle.clone()).unwrap_or_default();
    let handle = app_handle.clone();
    let analysis_paths = paths.clone();
    let analyses: Vec<Result<ImageAnalysis, (String, String)>> =
        tauri::async_runtime::spawn_blocking(move || {
            let hasher = HasherConfig::new()
                .hash_alg(HashAlg::DoubleGradient)
                .hash_size(16, 16)
                .to_hasher();
            let done = Arc::new(AtomicUsize::new(0));
            analysis_paths
                .par_iter()
                .map(|path| {
                    let result =
                        analyze_image(path, &hasher, &app_settings).map_err(|e| (path.clone(), e));
                    let current = done.fetch_add(1, Ordering::Relaxed) + 1;
                    let _ = handle.emit(
                        "culling-progress",
                        CullingProgress {
                            current,
                            total,
                            stage: "analyzing",
                        },
                    );
                    result
                })
                .collect()
        })
        .await
        .map_err(|e| e.to_string())?;

    let mut suggestions = CullingSuggestions::default();
    let mut analyzed = Vec::new();
    for analysis in analyses {
        match analysis {
            Ok(a) => analyzed.push(a),
            Err((path, error)) => {
                log::warn!("Culling: could not analyze {}: {}", path, error);
                suggestions.failed_paths.push(path);
            }
        }
    }

    let mut grouped = vec![false; analyzed.len()];
    if settings.group_similar && analyzed.len() > 1 {
        let embeddings = match crate::ai_processing::get_or_init_search_models(
            &app_handle,
            &state.ai_state,
            &state.ai_init_lock,
        )
        .await
        {
            Ok(models) => {
                let handle = app_handle.clone();
                let analyzed_paths: Vec<String> =
                    analyzed.iter().map(|a| a.result.path.clone()).collect();
                tauri::async_runtime::spawn_blocking(move || {
                    crate::semantic_search::image_embeddings(
                        &handle,
                        &models,
                        &analyzed_paths,
                        &|current, total| {
                            let _ = handle.emit(
                                "culling-progress",
                                CullingProgress {
                                    current,
                                    total,
                                    stage: "comparing",
                                },
                            );
                        },
                    )
                })
                .await
                .map_err(|e| e.to_string())?
                .inspect_err(|e| log::warn!("Culling: no embeddings, using hashes: {}", e))
                .ok()
            }
            Err(e) => {
                log::warn!("Culling: search model unavailable, using hashes: {}", e);
                None
            }
        };

        progress(total, total, "grouping");
        let threshold = (settings.similarity_threshold as f32 / 100.0).clamp(0.5, 0.999);
        for mut group in group_similar(&analyzed, embeddings.as_deref(), threshold) {
            group.sort_by(|&a, &b| {
                analyzed[b]
                    .result
                    .quality_score
                    .total_cmp(&analyzed[a].result.quality_score)
            });
            // The best shot stays eligible for the blur check below.
            group[1..].iter().for_each(|&i| grouped[i] = true);
            suggestions.similar_groups.push(CullGroup {
                representative: analyzed[group[0]].result.clone(),
                duplicates: group[1..]
                    .iter()
                    .map(|&i| analyzed[i].result.clone())
                    .collect(),
            });
        }
    }

    if settings.filter_blurry {
        suggestions.blurry_images = analyzed
            .iter()
            .zip(&grouped)
            .filter(|(a, grouped)| {
                !**grouped
                    && a.result
                        .sharpness_metric
                        .is_some_and(|s| s < settings.blur_threshold)
            })
            .map(|(a, _)| a.result.clone())
            .collect();
        suggestions.blurry_images.sort_by(|a, b| {
            a.sharpness_metric
                .unwrap_or(0.0)
                .total_cmp(&b.sharpness_metric.unwrap_or(0.0))
        });
    }

    let _ = app_handle.emit("culling-complete", &suggestions);
    Ok(suggestions)
}
