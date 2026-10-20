//! Auto-enhance: chooses white balance, exposure and the tone sliders by rendering a small proxy
//! of the photo through the real GPU pipeline and solving each slider against perceptual targets.
//! Measuring the rendered result, instead of predicting it from the raw histogram, keeps the
//! answer right whichever tone mapper, curve or LUT the photo uses.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use serde_json::{Value, json};

use crate::app_state::AppState;
use crate::as_shot::AsShot;
use crate::gpu_processing::{RenderRequest, process_and_get_dynamic_image};
use crate::image_processing::{
    AutoAdjustMode, GpuContext, RenderOptions, downscale_f32_image, get_all_adjustments_from_json,
    srgb_channel_to_linear,
};
use crate::white_balance::{self, MIRED_PER_RELATIVE_UNIT, TINT_PER_RELATIVE_UNIT};

/// Long edge of the proxy that gets rendered: enough for stable statistics, small enough that
/// the ~35 renders a solve needs take a fraction of a second.
pub const PROXY_LONG_EDGE: u32 = 384;

/// The sliders auto-enhance owns. They're zeroed before solving so the result doesn't depend on
/// how the photo was edited before.
const AUTO_KEYS: [&str; 10] = [
    "exposure",
    "brightness",
    "contrast",
    "highlights",
    "shadows",
    "whites",
    "blacks",
    "vibrance",
    "temperature",
    "tint",
];

// Exposure: bring the centre-weighted mean lightness (CIE L*) toward mid-grey, but only part of
// the way for photos that are deliberately dark or bright, and less the further off they are
// (a night scene should stay a night scene).
const TARGET_LIGHTNESS: f64 = 50.0;
const KEY_PRESERVATION: f64 = 0.35;
const EXTRA_KEY_PRESERVATION: f64 = 0.45;
const MIN_TARGET_LIGHTNESS: f64 = 20.0;
const MAX_TARGET_LIGHTNESS: f64 = 60.0;
const EXPOSURE_RANGE: (f64, f64) = (-2.0, 2.0);

// Highlights: compress the brightest tones until the 99th percentile sits below this. Where
// they're clipped in the source and can't come back, settle for a mild pull instead of
// flattening the rest of the photo for nothing.
const HIGHLIGHT_PERCENTILE: f64 = 0.99;
const TARGET_HIGHLIGHT_LIGHTNESS: f64 = 93.0;
const HIGHLIGHTS_FLOOR: f64 = -60.0;
const UNRECOVERABLE_HIGHLIGHTS: f64 = -25.0;

// Shadows: gently lift deep shadows when a big part of the frame sits in them (kept small so
// silhouettes stay dark).
const DEEP_SHADOW_LIGHTNESS: f64 = 12.0;
const DEEP_SHADOW_SHARE: f64 = 0.25;
const SHADOWS_CEILING: f64 = 20.0;

// Whites and blacks: stretch the histogram so dull highlights reach toward white and the
// darkest tones toward black (only slightly lifting blacks that are already crushed).
const WHITE_PERCENTILE: f64 = 0.997;
const DULL_WHITE_LIGHTNESS: f64 = 92.0;
const TARGET_WHITE_LIGHTNESS: f64 = 95.0;
const BLACK_PERCENTILE: f64 = 0.005;
const TARGET_BLACK_LIGHTNESS: f64 = 4.0;
const WHITES_RANGE: (f64, f64) = (0.0, 40.0);
const BLACKS_RANGE: (f64, f64) = (-35.0, 5.0);

// Contrast: add some to flat photos, judged by the spread of lightness.
const FLAT_LIGHTNESS_SPREAD: f64 = 18.0;
const CONTRAST_PER_UNIT_SPREAD: f64 = 2.0;
const CONTRAST_CEILING: f64 = 25.0;

// Vibrance: bring muted photos toward a typical mean saturation.
const TARGET_SATURATION: f64 = 0.28;
const VIBRANCE_PER_UNIT_SATURATION: f64 = 100.0;
const VIBRANCE_CEILING: f64 = 25.0;

// White balance: correct casts measured on near-neutral pixels, partially (warm casts are often
// intended) and within limits, and not at all when too little of the frame looks neutral.
const MIN_NEUTRAL_SHARE: f64 = 0.03;
const FULL_CONFIDENCE_NEUTRAL_SHARE: f64 = 0.15;
const COOLING_STRENGTH: f64 = 0.5;
const WARMING_STRENGTH: f64 = 0.8;
const TINT_STRENGTH: f64 = 0.8;
const MAX_TEMPERATURE_SHIFT: f64 = 30.0;
const MAX_TINT_SHIFT: f64 = 25.0;

const BISECTION_STEPS: usize = 7;

/// Statistics of a rendered proxy, in display-referred terms.
struct RenderStats {
    /// Histogram of CIE L* in 0.1 steps.
    histogram: Vec<u32>,
    total: f64,
    weighted_mean_lightness: f64,
    lightness_spread: f64,
    mean_saturation: f64,
}

impl RenderStats {
    fn percentile(&self, p: f64) -> f64 {
        let target = (self.total * p).max(1.0);
        let mut cumulative = 0.0;
        for (i, &count) in self.histogram.iter().enumerate() {
            cumulative += count as f64;
            if cumulative >= target {
                return i as f64 / 10.0;
            }
        }
        100.0
    }

    fn share_below(&self, lightness: f64) -> f64 {
        let end = ((lightness * 10.0) as usize).min(self.histogram.len());
        self.histogram[..end].iter().map(|&c| c as f64).sum::<f64>() / self.total
    }
}

struct ProxyRenderer<'a> {
    context: &'a GpuContext,
    state: &'a tauri::State<'a, AppState>,
    proxy: DynamicImage,
    is_raw: bool,
    as_shot: AsShot,
    render: RenderOptions,
    lut: Option<Arc<crate::lut_processing::Lut>>,
    cache_key: u64,
    centre_weights: Vec<f32>,
    srgb_to_linear: [f32; 256],
}

impl ProxyRenderer<'_> {
    fn measure(&self, adjustments: &Value) -> Result<RenderStats, String> {
        let mut all =
            get_all_adjustments_from_json(adjustments, self.is_raw, self.as_shot, self.render);
        all.global.show_clipping = 0;
        let no_masks: [ImageBuffer<Luma<u8>, Vec<u8>>; 0] = [];
        let rendered = process_and_get_dynamic_image(
            self.context,
            self.state,
            &self.proxy,
            self.cache_key,
            RenderRequest {
                adjustments: all,
                mask_bitmaps: &no_masks,
                mask_cache_key: None,
                lut: self.lut.clone(),
                roi: None,
            },
            "auto_enhance",
        )?
        .to_rgb8();

        let mut histogram = vec![0u32; 1001];
        let (mut weighted_sum, mut weight_total) = (0.0f64, 0.0f64);
        let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
        let (mut saturation_sum, mut saturation_n) = (0.0f64, 0u32);
        for (pixel, &weight) in rendered.pixels().zip(&self.centre_weights) {
            let [r, g, b] = pixel.0;
            let y = 0.2126 * self.srgb_to_linear[r as usize]
                + 0.7152 * self.srgb_to_linear[g as usize]
                + 0.0722 * self.srgb_to_linear[b as usize];
            let lightness = cie_lightness(y) as f64;
            histogram[(lightness * 10.0).round().clamp(0.0, 1000.0) as usize] += 1;
            weighted_sum += lightness * weight as f64;
            weight_total += weight as f64;
            sum += lightness;
            sum_sq += lightness * lightness;

            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            if lightness > 15.0 && max > 0 {
                saturation_sum += (max - min) as f64 / max as f64;
                saturation_n += 1;
            }
        }

        let total = rendered.pixels().len() as f64;
        let mean = sum / total;
        Ok(RenderStats {
            histogram,
            total,
            weighted_mean_lightness: weighted_sum / weight_total.max(1e-9),
            lightness_spread: (sum_sq / total - mean * mean).max(0.0).sqrt(),
            mean_saturation: if saturation_n > 0 {
                saturation_sum / saturation_n as f64
            } else {
                0.0
            },
        })
    }

    /// Finds the slider value in `range` whose rendered `metric` is closest to `target`,
    /// assuming the metric moves monotonically with the slider. Also says whether the range
    /// could reach the target at all.
    fn solve(
        &self,
        adjustments: &mut Value,
        key: &str,
        range: (f64, f64),
        target: f64,
        metric: impl Fn(&RenderStats) -> f64,
    ) -> Result<(RenderStats, bool), String> {
        let (mut lo, mut hi) = range;
        adjustments[key] = json!(lo);
        let at_lo = metric(&self.measure(adjustments)?);
        adjustments[key] = json!(hi);
        let at_hi = metric(&self.measure(adjustments)?);
        let increasing = at_lo < at_hi;
        let reachable = (at_lo.min(at_hi)..=at_lo.max(at_hi)).contains(&target);
        for _ in 0..BISECTION_STEPS {
            let mid = 0.5 * (lo + hi);
            adjustments[key] = json!(mid);
            let below = metric(&self.measure(adjustments)?) < target;
            if below == increasing {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        adjustments[key] = json!(round_slider(key, 0.5 * (lo + hi)));
        Ok((self.measure(adjustments)?, reachable))
    }
}

fn cie_lightness(y: f32) -> f32 {
    let f = if y > 216.0 / 24389.0 {
        y.cbrt()
    } else {
        (24389.0 / 27.0 * y + 16.0) / 116.0
    };
    116.0 * f - 16.0
}

fn round_slider(key: &str, value: f64) -> f64 {
    if key == "exposure" {
        (value * 100.0).round() / 100.0
    } else {
        value.round()
    }
}

/// Centre-weighted metering: the middle of the frame counts about three times the corners.
fn centre_weights(width: u32, height: u32) -> Vec<f32> {
    let mut weights = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let dx = (x as f32 + 0.5) / width as f32 - 0.5;
            let dy = (y as f32 + 0.5) / height as f32 - 0.5;
            weights.push(1.0 + 2.0 * (-(dx * dx + dy * dy) / (2.0 * 0.25 * 0.25)).exp());
        }
    }
    weights
}

/// Estimates the colour cast of `proxy` (as rendered at the as-shot white balance) from its
/// near-neutral pixels, and returns the relative temperature/tint sliders that would correct it
/// along with the share of the frame that looked neutral.
fn estimate_white_balance(
    proxy: &DynamicImage,
    is_raw: bool,
    as_shot: AsShot,
) -> Option<(f64, f64, f64)> {
    let rgb = proxy.to_rgb32f();
    let pixels: Vec<[f32; 3]> = rgb
        .pixels()
        .map(|p| {
            let c = p.0.map(|v| v.max(0.0));
            if is_raw {
                c
            } else {
                c.map(srgb_channel_to_linear)
            }
        })
        .filter(|c| c.iter().all(|v| v.is_finite()))
        .collect();
    if pixels.is_empty() {
        return None;
    }

    // Skip the darkest pixels (noisy) and anything near clipping (its colour is unreliable).
    let mut luminance: Vec<f32> = pixels
        .iter()
        .map(|c| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
        .collect();
    let channel_max = pixels
        .iter()
        .flat_map(|c| c.iter())
        .copied()
        .fold(0.0f32, f32::max);
    let mut sorted = luminance.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let dark = sorted[sorted.len() / 20];
    let usable: Vec<[f32; 3]> = pixels
        .iter()
        .zip(luminance.drain(..))
        .filter(|(c, y)| *y > dark && c.iter().all(|&v| v > 1e-4 && v < 0.95 * channel_max))
        .map(|(c, _)| *c)
        .collect();
    if usable.len() < 100 {
        return None;
    }

    // Average the pixels whose chromaticity is close to the current neutral estimate, then
    // re-centre on that average with a tighter window.
    let mut neutral = [1.0f64; 3];
    let mut neutral_count = 0usize;
    for window in [0.3, 0.2, 0.12] {
        let (ref_rg, ref_bg) = (
            (neutral[0] / neutral[1]).ln(),
            (neutral[2] / neutral[1]).ln(),
        );
        let mut sum = [0.0f64; 3];
        neutral_count = 0;
        for c in &usable {
            let rg = (c[0] as f64 / c[1] as f64).ln() - ref_rg;
            let bg = (c[2] as f64 / c[1] as f64).ln() - ref_bg;
            if rg * rg + bg * bg < window * window {
                for i in 0..3 {
                    sum[i] += c[i] as f64;
                }
                neutral_count += 1;
            }
        }
        if neutral_count == 0 {
            return None;
        }
        neutral = [sum[0] / sum[1], 1.0, sum[2] / sum[1]];
    }

    let neutral_share = neutral_count as f64 / usable.len() as f64;
    let picked = white_balance::pick_white_balance(neutral, as_shot.white_balance)?;
    let mired = |kelvin: f64| 1.0e6 / kelvin;
    let temperature = (mired(as_shot.white_balance.temperature) - mired(picked.temperature))
        / MIRED_PER_RELATIVE_UNIT;
    let tint = (picked.tint - as_shot.white_balance.tint) / TINT_PER_RELATIVE_UNIT;
    Some((temperature, tint, neutral_share))
}

/// GPU texture-cache key for a photo's proxy, distinct from the keys of its full renders.
pub fn proxy_cache_key(path: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    "auto_enhance".hash(&mut hasher);
    hasher.finish()
}

/// Builds the proxy for an image that isn't open in the editor: the whole frame, downscaled.
pub fn proxy_from_image(image: &DynamicImage) -> DynamicImage {
    downscale_f32_image(image, PROXY_LONG_EDGE, PROXY_LONG_EDGE)
}

/// Solves the auto-enhance sliders for `proxy` (the photo before processing, already cropped
/// if possible) on top of `adjustments`, and returns them as adjustment JSON to merge in.
#[allow(clippy::too_many_arguments)]
pub fn compute_auto_adjustments(
    context: &GpuContext,
    state: &tauri::State<AppState>,
    proxy: DynamicImage,
    adjustments: &Value,
    is_raw: bool,
    as_shot: AsShot,
    render: RenderOptions,
    cache_key: u64,
    mode: AutoAdjustMode,
) -> Result<Value, String> {
    let start = std::time::Instant::now();
    let proxy = downscale_f32_image(&proxy, PROXY_LONG_EDGE, PROXY_LONG_EDGE);
    let (width, height) = proxy.dimensions();
    if width == 0 || height == 0 {
        return Err("Image is empty".to_string());
    }

    let mut working = if adjustments.is_object() {
        adjustments.clone()
    } else {
        crate::image_processing::default_adjustments()
    };
    let Some(map) = working.as_object_mut() else {
        return Err("Invalid adjustments".to_string());
    };
    for key in AUTO_KEYS {
        map.insert(key.to_string(), json!(0.0));
    }
    map.insert("whiteBalance".to_string(), Value::Null);
    let mut active_tools = crate::image_processing::active_tools(&working);
    for tool in ["basic", "colorPresence", "whiteBalance"] {
        active_tools.insert(tool.to_string(), json!(true));
    }
    working["activeTools"] = Value::Object(active_tools);

    let lut = working["lutPath"]
        .as_str()
        .and_then(|path| crate::lut_processing::get_or_load_lut(state, path).ok());
    let renderer = ProxyRenderer {
        context,
        state,
        proxy,
        is_raw,
        as_shot,
        render,
        lut,
        cache_key,
        centre_weights: centre_weights(width, height),
        srgb_to_linear: std::array::from_fn(|i| srgb_channel_to_linear(i as f32 / 255.0)),
    };

    if let Some((temperature, tint, neutral_share)) =
        estimate_white_balance(&renderer.proxy, is_raw, as_shot)
        && neutral_share >= MIN_NEUTRAL_SHARE
    {
        let confidence = (neutral_share / FULL_CONFIDENCE_NEUTRAL_SHARE).min(1.0);
        let warmth = if temperature < 0.0 {
            COOLING_STRENGTH
        } else {
            WARMING_STRENGTH
        };
        working["temperature"] = json!(
            (temperature * warmth * confidence)
                .clamp(-MAX_TEMPERATURE_SHIFT, MAX_TEMPERATURE_SHIFT)
                .round()
        );
        working["tint"] = json!(
            (tint * TINT_STRENGTH * confidence)
                .clamp(-MAX_TINT_SHIFT, MAX_TINT_SHIFT)
                .round()
        );
    }

    let initial = renderer.measure(&working)?;
    let offset = initial.weighted_mean_lightness - TARGET_LIGHTNESS;
    let preservation =
        KEY_PRESERVATION + EXTRA_KEY_PRESERVATION * ((offset.abs() - 15.0) / 20.0).clamp(0.0, 1.0);
    let target_lightness = (TARGET_LIGHTNESS + preservation * offset)
        .clamp(MIN_TARGET_LIGHTNESS, MAX_TARGET_LIGHTNESS);
    let (mut stats, _) = renderer.solve(
        &mut working,
        "exposure",
        EXPOSURE_RANGE,
        target_lightness,
        |s| s.weighted_mean_lightness,
    )?;

    if stats.lightness_spread < FLAT_LIGHTNESS_SPREAD {
        working["contrast"] = json!(
            ((FLAT_LIGHTNESS_SPREAD - stats.lightness_spread) * CONTRAST_PER_UNIT_SPREAD)
                .min(CONTRAST_CEILING)
                .round()
        );
        stats = renderer.measure(&working)?;
    }

    let mut pulled_highlights = false;
    if stats.percentile(HIGHLIGHT_PERCENTILE) > TARGET_HIGHLIGHT_LIGHTNESS {
        let reachable;
        (stats, reachable) = renderer.solve(
            &mut working,
            "highlights",
            (HIGHLIGHTS_FLOOR, 0.0),
            TARGET_HIGHLIGHT_LIGHTNESS,
            |s| s.percentile(HIGHLIGHT_PERCENTILE),
        )?;
        if !reachable {
            working["highlights"] = json!(UNRECOVERABLE_HIGHLIGHTS);
            stats = renderer.measure(&working)?;
        }
        pulled_highlights = true;
    }

    if stats.share_below(DEEP_SHADOW_LIGHTNESS) > DEEP_SHADOW_SHARE {
        (stats, _) = renderer.solve(
            &mut working,
            "shadows",
            (0.0, SHADOWS_CEILING),
            DEEP_SHADOW_SHARE,
            |s| s.share_below(DEEP_SHADOW_LIGHTNESS),
        )?;
    }

    if !pulled_highlights && stats.percentile(WHITE_PERCENTILE) < DULL_WHITE_LIGHTNESS {
        renderer.solve(
            &mut working,
            "whites",
            WHITES_RANGE,
            TARGET_WHITE_LIGHTNESS,
            |s| s.percentile(WHITE_PERCENTILE),
        )?;
    }

    (stats, _) = renderer.solve(
        &mut working,
        "blacks",
        BLACKS_RANGE,
        TARGET_BLACK_LIGHTNESS,
        |s| s.percentile(BLACK_PERCENTILE),
    )?;

    if stats.mean_saturation < TARGET_SATURATION {
        working["vibrance"] = json!(
            ((TARGET_SATURATION - stats.mean_saturation) * VIBRANCE_PER_UNIT_SATURATION)
                .min(VIBRANCE_CEILING)
                .round()
        );
    }

    log::info!(
        "Auto-enhance solved in {:?}: exposure {}, contrast {}, highlights {}, shadows {}, whites {}, blacks {}, vibrance {}, temperature {}, tint {}",
        start.elapsed(),
        working["exposure"],
        working["contrast"],
        working["highlights"],
        working["shadows"],
        working["whites"],
        working["blacks"],
        working["vibrance"],
        working["temperature"],
        working["tint"],
    );

    // Subtle and Strong scale the corrections, as the original auto adjust's levels do.
    let strength = match mode {
        AutoAdjustMode::Subtle => 0.5,
        AutoAdjustMode::Normal => 1.0,
        AutoAdjustMode::Strong => 1.5,
    };
    let mut result = serde_json::Map::new();
    for key in AUTO_KEYS {
        let value = match working[key].as_f64() {
            Some(value) if strength != 1.0 => {
                let limit = if key == "exposure" { 5.0 } else { 100.0 };
                json!(round_slider(key, value * strength).clamp(-limit, limit))
            }
            _ => working[key].clone(),
        };
        result.insert(key.to_string(), value);
    }
    result.insert("whiteBalance".to_string(), Value::Null);
    result.insert(
        "activeTools".to_string(),
        json!({ "basic": true, "colorPresence": true, "whiteBalance": true }),
    );
    Ok(Value::Object(result))
}

/// Merges auto-enhance results into `adjustments`, switching on the tools they use.
pub fn merge_auto_adjustments(adjustments: &mut Value, auto: &Value) {
    if !adjustments.is_object() {
        *adjustments = crate::image_processing::default_adjustments();
    }
    let mut active_tools = crate::image_processing::active_tools(adjustments);
    let (Some(target), Some(auto)) = (adjustments.as_object_mut(), auto.as_object()) else {
        return;
    };
    for (key, value) in auto {
        if key == "activeTools" {
            if let Some(tools) = value.as_object() {
                active_tools.extend(tools.clone());
            }
        } else {
            target.insert(key.clone(), value.clone());
        }
    }
    target.insert("activeTools".to_string(), Value::Object(active_tools));
}
