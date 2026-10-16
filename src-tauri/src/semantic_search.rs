//! Natural-language photo search with SigLIP 2.
//!
//! Each photo gets an image embedding, computed once from its cached thumbnail and kept in an
//! append-only file in the app cache. Entries are keyed by the thumbnail cache hash (source
//! path, modification time and adjustments), so edited photos are re-embedded. A query goes
//! through the text tower and photos are ranked by cosine similarity.

use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Result, anyhow};
use half::f16;
use image::{DynamicImage, imageops::FilterType};
use ndarray::{Array2, Array4};
use ort::value::Tensor;
use rayon::prelude::*;
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use crate::AppState;
use crate::ai_processing::SearchModels;
use crate::file_management;
use crate::image_processing::GpuContext;

const INPUT_SIZE: u32 = 224;
const EMBED_DIM: usize = 768;
const MAX_TOKENS: usize = 64;
const PAD_TOKEN: i64 = 0;
// Thumbnails are decoded in parallel a batch at a time; a newer search can take over between
// batches.
const BATCH: usize = 16;

// Scores are cosine similarities, which SigLIP keeps small: a clear match scores around
// 0.1-0.2, unrelated photos around 0.03. A photo matches if it's close to the best match and
// stands clear of a typical unrelated photo. The median of the folder estimates the latter,
// capped so a folder where most photos match (portraits, "a person") still returns them.
const BEST_MATCH_WINDOW: f32 = 0.045;
const UNRELATED_MARGIN: f32 = 0.035;
const UNRELATED_SCORE_CAP: f32 = 0.05;

const INDEX_FILENAME: &str = "semantic_index.bin";
const INDEX_MAGIC: &[u8; 8] = b"RRSIGv1\n";
// A blake3 hex key followed by the embedding as little-endian f16.
const KEY_LEN: usize = 64;
const RECORD_LEN: usize = KEY_LEN + EMBED_DIM * 2;

static SEARCH_GENERATION: AtomicU64 = AtomicU64::new(0);
static INDEX: Mutex<Option<EmbeddingIndex>> = Mutex::new(None);

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticMatch {
    path: String,
    score: f32,
}

struct EmbeddingIndex {
    file: PathBuf,
    entries: HashMap<String, Box<[f16]>>,
}

impl EmbeddingIndex {
    fn open(file: PathBuf) -> Self {
        let data = fs::read(&file).unwrap_or_default();
        let mut entries = HashMap::new();
        let mut valid_len = 0;
        if let Some(records) = data.strip_prefix(INDEX_MAGIC) {
            let (records, _) = records.as_chunks::<RECORD_LEN>();
            for record in records {
                if let Ok(key) = std::str::from_utf8(&record[..KEY_LEN]) {
                    let embedding = record[KEY_LEN..]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&b| f16::from_le_bytes(b))
                        .collect();
                    entries.insert(key.to_string(), embedding);
                }
            }
            valid_len = INDEX_MAGIC.len() + records.len() * RECORD_LEN;
        }

        // Start over on an unknown header, and drop a record cut short by a crash so later
        // appends stay aligned.
        if valid_len == 0 {
            let _ = fs::write(&file, INDEX_MAGIC);
        } else if valid_len < data.len()
            && let Ok(f) = OpenOptions::new().write(true).open(&file)
        {
            let _ = f.set_len(valid_len as u64);
        }

        Self { file, entries }
    }

    fn append(&mut self, new: Vec<(String, Box<[f16]>)>) -> Result<()> {
        let mut bytes = Vec::with_capacity(new.len() * RECORD_LEN);
        for (key, embedding) in &new {
            bytes.extend_from_slice(key.as_bytes());
            for v in embedding.iter() {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file)?
            .write_all(&bytes)?;
        self.entries.extend(new);
        Ok(())
    }
}

fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn embed_text(models: &SearchModels, query: &str) -> Result<Vec<f32>> {
    // SigLIP 2's text tower was trained on lowercase text.
    let encoding = models
        .tokenizer
        .encode(query.to_lowercase(), true)
        .map_err(|e| anyhow!(e.to_string()))?;
    let mut ids: Vec<i64> = encoding
        .get_ids()
        .iter()
        .take(MAX_TOKENS)
        .map(|&id| id as i64)
        .collect();
    ids.resize(MAX_TOKENS, PAD_TOKEN);
    let input = Tensor::from_array(Array2::from_shape_vec((1, MAX_TOKENS), ids)?)?;

    let mut session = models.text.lock().unwrap();
    let outputs = session.run(ort::inputs!["input_ids" => input])?;
    let embedding = outputs["pooler_output"].try_extract_array::<f32>()?;
    Ok(normalized(embedding.iter().copied().collect()))
}

/// Stretched to 224x224 and scaled to [-1, 1], as SigLIP was trained.
fn image_tensor(image: &DynamicImage) -> Array4<f32> {
    let rgb = image
        .resize_exact(INPUT_SIZE, INPUT_SIZE, FilterType::Triangle)
        .to_rgb8();
    let size = INPUT_SIZE as usize;
    let mut tensor = Array4::zeros((1, 3, size, size));
    for (x, y, pixel) in rgb.enumerate_pixels() {
        for c in 0..3 {
            tensor[[0, c, y as usize, x as usize]] = pixel[c] as f32 / 127.5 - 1.0;
        }
    }
    tensor
}

fn embed_image(models: &SearchModels, tensor: Array4<f32>) -> Result<Box<[f16]>> {
    let input = Tensor::from_array(tensor)?;
    let mut session = models.vision.lock().unwrap();
    let outputs = session.run(ort::inputs!["pixel_values" => input])?;
    let embedding = outputs["pooler_output"].try_extract_array::<f32>()?;
    Ok(normalized(embedding.iter().copied().collect())
        .into_iter()
        .map(f16::from_f32)
        .collect())
}

/// The small cached thumbnail is plenty for a 224px input and much faster to decode than the
/// medium one; photos without one get both generated.
fn load_thumbnail(
    path: &str,
    key: &str,
    thumb_dir: &Path,
    app_handle: &AppHandle,
    gpu_context: Option<&GpuContext>,
) -> Result<DynamicImage> {
    if let Ok(image) = image::open(thumb_dir.join(format!("{}_small.jpg", key))) {
        return Ok(image);
    }
    file_management::get_cached_or_generate_thumbnail_image(path, app_handle, gpu_context)
}

fn select_matches(mut scored: Vec<(String, f32)>) -> Vec<SemanticMatch> {
    if scored.is_empty() {
        return Vec::new();
    }
    let mut scores: Vec<f32> = scored.iter().map(|(_, s)| *s).collect();
    scores.sort_by(f32::total_cmp);
    let unrelated = scores[scores.len() / 2].min(UNRELATED_SCORE_CAP);
    let best = scores[scores.len() - 1];
    let cutoff = (unrelated + UNRELATED_MARGIN).max(best - BEST_MATCH_WINDOW);

    scored.retain(|(_, s)| *s >= cutoff);
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
        .into_iter()
        .map(|(path, score)| SemanticMatch { path, score })
        .collect()
}

fn index_keys(paths: &[String]) -> Vec<Option<String>> {
    paths
        .par_iter()
        .map(|path| file_management::get_cache_key_hash(path).filter(|k| k.len() == KEY_LEN))
        .collect()
}

/// Embeds the photos among `paths` that aren't in the index yet, a batch at a time, calling
/// `progress(done, total)` after each batch. Stops with an error once `should_stop` is true.
fn ensure_indexed(
    app_handle: &AppHandle,
    models: &SearchModels,
    paths: &[String],
    keys: &[Option<String>],
    progress: &dyn Fn(usize, usize),
    should_stop: &dyn Fn() -> bool,
) -> Result<()> {
    let started = Instant::now();
    let cache_dir = app_handle.path().app_cache_dir()?;
    fs::create_dir_all(&cache_dir)?;
    let missing: Vec<(&String, &String)> = {
        let mut index = INDEX.lock().unwrap();
        let index =
            index.get_or_insert_with(|| EmbeddingIndex::open(cache_dir.join(INDEX_FILENAME)));
        let mut queued = HashSet::new();
        paths
            .iter()
            .zip(keys)
            .filter_map(|(path, key)| Some((path, key.as_ref()?)))
            .filter(|(_, key)| !index.entries.contains_key(*key) && queued.insert(*key))
            .collect()
    };
    if missing.is_empty() {
        return Ok(());
    }

    let thumb_dir = file_management::get_thumb_cache_dir(app_handle).map_err(|e| anyhow!(e))?;
    let gpu_context =
        crate::gpu_processing::get_or_init_gpu_context(&app_handle.state::<AppState>(), app_handle)
            .ok();
    let total = missing.len();
    progress(0, total);

    for (i, batch) in missing.chunks(BATCH).enumerate() {
        if should_stop() {
            return Err(anyhow!("Superseded by a newer search"));
        }
        let tensors: Vec<(String, Option<Array4<f32>>)> = batch
            .par_iter()
            .map(|(path, key)| {
                let image = load_thumbnail(path, key, &thumb_dir, app_handle, gpu_context.as_ref());
                if let Err(e) = &image {
                    log::warn!("Search: no thumbnail for {}: {}", path, e);
                }
                ((*key).clone(), image.ok().map(|img| image_tensor(&img)))
            })
            .collect();

        let mut embedded = Vec::with_capacity(tensors.len());
        for (key, tensor) in tensors {
            if let Some(tensor) = tensor {
                embedded.push((key, embed_image(models, tensor)?));
            }
        }
        if let Some(index) = INDEX.lock().unwrap().as_mut() {
            index.append(embedded)?;
        }
        progress(((i + 1) * BATCH).min(total), total);
    }
    log::info!(
        "Search: embedded {} photos in {:.1}s",
        total,
        started.elapsed().as_secs_f32()
    );
    Ok(())
}

/// Unit-length image embeddings for `paths`, computing any that aren't cached yet; `None`
/// where a photo couldn't be read. Similar photos have a dot product close to 1.
pub fn image_embeddings(
    app_handle: &AppHandle,
    models: &SearchModels,
    paths: &[String],
    progress: &dyn Fn(usize, usize),
) -> Result<Vec<Option<Vec<f32>>>> {
    let keys = index_keys(paths);
    ensure_indexed(app_handle, models, paths, &keys, progress, &|| false)?;
    let index = INDEX.lock().unwrap();
    let entries = &index
        .as_ref()
        .ok_or_else(|| anyhow!("Search index missing"))?
        .entries;
    Ok(keys
        .iter()
        .map(|key| {
            let embedding = entries.get(key.as_ref()?)?;
            Some(embedding.iter().map(|v| v.to_f32()).collect())
        })
        .collect())
}

fn search(
    app_handle: &AppHandle,
    models: &SearchModels,
    query: &str,
    paths: Vec<String>,
    generation: u64,
) -> Result<Vec<SemanticMatch>> {
    let started = Instant::now();
    let text = embed_text(models, query)?;
    let keys = index_keys(&paths);
    ensure_indexed(
        app_handle,
        models,
        &paths,
        &keys,
        &|current, total| {
            let _ = app_handle.emit(
                "semantic-search-progress",
                serde_json::json!({ "current": current, "total": total }),
            );
        },
        &|| SEARCH_GENERATION.load(Ordering::SeqCst) != generation,
    )?;

    let scored: Vec<(String, f32)> = {
        let index = INDEX.lock().unwrap();
        let entries = &index
            .as_ref()
            .ok_or_else(|| anyhow!("Search index missing"))?
            .entries;
        paths
            .iter()
            .zip(&keys)
            .filter_map(|(path, key)| {
                let embedding = entries.get(key.as_ref()?)?;
                let score = embedding
                    .iter()
                    .zip(&text)
                    .map(|(e, t)| e.to_f32() * t)
                    .sum::<f32>();
                Some((path.clone(), score))
            })
            .collect()
    };
    let matches = select_matches(scored);
    log::info!(
        "Search {:?}: {} matches of {} photos in {} ms",
        query,
        matches.len(),
        paths.len(),
        started.elapsed().as_millis()
    );
    Ok(matches)
}

/// Ranks `paths` by how well they match `query`, best first, leaving out photos that don't
/// match. The first search in a folder embeds its photos, reporting `semantic-search-progress`.
#[tauri::command]
pub async fn semantic_search(
    query: String,
    paths: Vec<String>,
    app_handle: AppHandle,
    state: State<'_, AppState>,
) -> Result<Vec<SemanticMatch>, String> {
    let generation = SEARCH_GENERATION.fetch_add(1, Ordering::SeqCst) + 1;
    let query = query.trim().to_string();
    if query.is_empty() || paths.is_empty() {
        return Ok(Vec::new());
    }

    let models = crate::ai_processing::get_or_init_search_models(
        &app_handle,
        &state.ai_state,
        &state.ai_init_lock,
    )
    .await
    .map_err(|e| e.to_string())?;

    tauri::async_runtime::spawn_blocking(move || {
        search(&app_handle, &models, &query, paths, generation)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())
}
