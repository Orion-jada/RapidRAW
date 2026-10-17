//! Linear DNGs of images RapidRAW has already developed: demosaiced, white balanced, upright,
//! linear with Rec.709 primaries and white at 1.0, as `raw_processing::develop_raw_image` returns
//! them (before the raw pre-pass). Opening one goes through the same editing pipeline as the RAW
//! it came from, so its edits carry over unchanged.
//!
//! Pixels are 16-bit floats in deflate-compressed tiles with the floating-point predictor, as in
//! Adobe's Enhanced DNGs, which keeps fine shadow steps and highlights above 1.0. The colour
//! matrix is the inverse of rawler's sRGB-to-XYZ matrix, so its calibration step leaves pixels as
//! they are. The editor applies white balance and baseline exposure relative to the file's as-shot
//! values, so the original's are kept in DNGPrivateData and read back by `as_shot`.

use crate::as_shot::AsShot;
use crate::baseline_exposure::RawExposure;
use crate::white_balance::WhiteBalance;
use anyhow::{Context, Result};
use flate2::{Compression, write::ZlibEncoder};
use glam::DMat3;
use half::f16;
use image::{DynamicImage, Rgb32FImage, imageops::FilterType};
use rawler::decoders::{Decoder, RawMetadata, WellKnownIFD};
use rawler::dng::{DNG_VERSION_V1_4, writer::DngWriter};
use rawler::formats::tiff::{
    CompressionMethod, PhotometricInterpretation, Rational, SRational, Value,
};
use rawler::imgop::xyz::{Illuminant, SRGB_TO_XYZ_D65};
use rawler::rawsource::RawSource;
use rawler::tags::{DngTag, ExifTag, TiffCommonTag};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::{Cursor, Write};
use std::path::Path;

/// DNGPrivateData starts with the name of the program that wrote it.
const PRIVATE_DATA_PREFIX: &[u8] = b"RapidRAW\0";
const TILE: usize = 256;
/// TIFF's floating-point predictor: each row's bytes split into planes, then differenced.
const PREDICTOR_FLOATING_POINT: u16 = 3;
const PREVIEW_SIZE: u32 = 1024;
const MATRIX_DENOMINATOR: i32 = 1_000_000;

#[derive(Serialize, Deserialize)]
struct PrivateData {
    white_balance: WhiteBalance,
    exposure: RawExposure,
}

/// Writes `image` (developed linear pixels, see the module docs) to `path` as a linear DNG.
/// `metadata` is the original's EXIF and `as_shot` its as-shot white balance and exposure;
/// `preview` is an sRGB rendering for the embedded preview and thumbnail.
pub fn write(
    path: &Path,
    image: &Rgb32FImage,
    metadata: Option<&RawMetadata>,
    as_shot: AsShot,
    preview: &DynamicImage,
) -> Result<()> {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let tiles_x = width.div_ceil(TILE);
    let tiles_y = height.div_ceil(TILE);
    let tiles = (0..tiles_x * tiles_y)
        .into_par_iter()
        .map(|i| encode_tile(image.as_raw(), width, height, i % tiles_x, i / tiles_x))
        .collect::<Result<Vec<_>>>()?;

    let mut buf = Cursor::new(Vec::new());
    let mut dng = DngWriter::new(&mut buf, DNG_VERSION_V1_4)?;

    let mut offsets = Vec::with_capacity(tiles.len());
    for tile in &tiles {
        offsets.push(dng.dng.write_data(tile)?);
    }
    let sizes: Vec<u32> = tiles.iter().map(|tile| tile.len() as u32).collect();
    drop(tiles);

    let mut raw = dng.subframe(0);
    let ifd = raw.ifd_mut();
    ifd.add_tag(TiffCommonTag::ImageWidth, width as u32);
    ifd.add_tag(TiffCommonTag::ImageLength, height as u32);
    ifd.add_tag(TiffCommonTag::SamplesPerPixel, 3_u16);
    ifd.add_tag(TiffCommonTag::BitsPerSample, [16_u16; 3]);
    ifd.add_tag(TiffCommonTag::SampleFormat, [3_u16; 3]); // IEEE float
    ifd.add_tag(
        TiffCommonTag::PhotometricInt,
        PhotometricInterpretation::LinearRaw,
    );
    ifd.add_tag(ExifTag::PlanarConfiguration, 1_u16);
    ifd.add_tag(TiffCommonTag::Compression, CompressionMethod::Deflate);
    ifd.add_tag(TiffCommonTag::Predictor, PREDICTOR_FLOATING_POINT);
    ifd.add_tag(TiffCommonTag::TileWidth, TILE as u32);
    ifd.add_tag(TiffCommonTag::TileLength, TILE as u32);
    ifd.add_tag(TiffCommonTag::TileOffsets, &offsets);
    ifd.add_tag(TiffCommonTag::TileByteCounts, &sizes);
    ifd.add_tag(DngTag::BlackLevelRepeatDim, [1_u16, 1]);
    ifd.add_tag(DngTag::BlackLevel, [0_u16; 3]);
    ifd.add_tag(DngTag::WhiteLevel, [1_u16; 3]); // 1.0 for float data
    raw.finalize()?;

    let preview = preview.resize(PREVIEW_SIZE, PREVIEW_SIZE, FilterType::Lanczos3);
    let mut preview_frame = dng.subframe(1);
    preview_frame.preview(&preview, 1.0)?;
    preview_frame.finalize()?;
    dng.thumbnail(&preview)?;

    dng.color_matrix(1, Illuminant::D65, xyz_to_srgb_matrix());
    dng.as_shot_neutral([Rational::new(1, 1); 3]);

    let (make, model) = match metadata {
        Some(metadata) => {
            let mut metadata = metadata.clone();
            metadata.exif.orientation = None;
            dng.load_metadata(&metadata)?;
            (metadata.make, metadata.model)
        }
        None => (String::new(), String::new()),
    };
    let root = dng.root_ifd_mut();
    if !make.is_empty() {
        root.add_tag(TiffCommonTag::Make, make.as_str());
        root.add_tag(TiffCommonTag::Model, model.as_str());
    }
    let camera = format!("{} {}", make, model);
    let camera = match camera.trim() {
        "" => "RapidRAW",
        camera => camera,
    };
    root.add_tag(DngTag::UniqueCameraModel, camera);
    // The pixels were turned upright when they were developed.
    root.add_tag(ExifTag::Orientation, 1_u16);
    root.add_tag(TiffCommonTag::Software, "RapidRAW");
    let private = PrivateData {
        white_balance: as_shot.white_balance,
        exposure: as_shot.exposure,
    };
    let mut private_data = PRIVATE_DATA_PREFIX.to_vec();
    private_data.extend(serde_json::to_vec(&private)?);
    root.add_tag_undefined(DngTag::DNGPrivateData, private_data);
    dng.close()?;

    std::fs::write(path, buf.into_inner())
        .with_context(|| format!("Couldn't write {}", path.display()))
}

/// The original's as-shot white balance and exposure, if `decoder` reads a DNG written by
/// `write`. Pixels in such a DNG are already developed, which `is_developed` tells.
pub fn as_shot(decoder: &dyn Decoder) -> Option<AsShot> {
    let root = decoder.ifd(WellKnownIFD::Root).ok()??;
    let data = match &root.get_entry(DngTag::DNGPrivateData)?.value {
        Value::Undefined(data) | Value::Byte(data) => data,
        _ => return None,
    };
    let private: PrivateData =
        serde_json::from_slice(data.strip_prefix(PRIVATE_DATA_PREFIX)?).ok()?;
    Some(AsShot {
        white_balance: private.white_balance,
        exposure: private.exposure,
    })
}

pub fn is_developed(decoder: &dyn Decoder) -> bool {
    as_shot(decoder).is_some()
}

/// Like `is_developed`, for a file's contents.
pub fn is_developed_file(path: &str, bytes: &[u8]) -> bool {
    path.to_ascii_lowercase().ends_with(".dng")
        && rawler::get_decoder(&RawSource::new_from_slice(bytes))
            .is_ok_and(|decoder| is_developed(decoder.as_ref()))
}

/// ColorMatrix1 for linear Rec.709 data: rawler calibrates with
/// `inverse(normalize(ColorMatrix * SRGB_TO_XYZ_D65))`, which this makes the identity.
fn xyz_to_srgb_matrix() -> [SRational; 9] {
    let rows = SRGB_TO_XYZ_D65.map(|row| row.map(f64::from));
    // `from_cols_array_2d` reads rows as columns; transposing twice leaves the inverse's rows.
    let inverse = DMat3::from_cols_array_2d(&rows)
        .inverse()
        .to_cols_array_2d();
    let mut out = [SRational::new(0, 1); 9];
    for (i, v) in inverse.iter().flatten().enumerate() {
        out[i] = SRational::new(
            (v * MATRIX_DENOMINATOR as f64).round() as i32,
            MATRIX_DENOMINATOR,
        );
    }
    out
}

/// One deflate-compressed tile of 16-bit floats with the floating-point predictor. Tiles past the
/// image's edge repeat its last row and column.
fn encode_tile(
    pixels: &[f32],
    width: usize,
    height: usize,
    tile_x: usize,
    tile_y: usize,
) -> Result<Vec<u8>> {
    const SAMPLES: usize = TILE * 3;
    const ROW_BYTES: usize = SAMPLES * 2;
    let mut rows = vec![0u8; TILE * ROW_BYTES];
    for (dy, row) in rows.as_chunks_mut::<ROW_BYTES>().0.iter_mut().enumerate() {
        let y = (tile_y * TILE + dy).min(height - 1);
        for dx in 0..TILE {
            let x = (tile_x * TILE + dx).min(width - 1);
            for c in 0..3 {
                let bits = f16::from_f32(pixels[(y * width + x) * 3 + c]).to_bits();
                // High bytes of the whole row first, then low bytes.
                row[dx * 3 + c] = (bits >> 8) as u8;
                row[SAMPLES + dx * 3 + c] = bits as u8;
            }
        }
        for i in (3..row.len()).rev() {
            row[i] = row[i].wrapping_sub(row[i - 3]);
        }
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&rows)?;
    Ok(encoder.finish()?)
}
