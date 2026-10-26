use std::collections::HashSet;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use image::imageops::{self, FilterType};
use image::{
    DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Rgb32FImage, RgbImage, RgbaImage,
};
use ndarray::{Array, Array4, ArrayViewD, IxDyn, s};
use ort::editor::{Graph, Model, Node, ONNX_DOMAIN, Opset, Outlet};
use ort::execution_providers::coreml::CoreMLComputeUnits;
use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};
use ort::memory::Allocator;
use ort::session::{Output, Session};
use ort::tensor::{Shape, SymbolicDimensions, TensorElementType};
use ort::value::{Tensor, TensorRef, ValueType};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::Emitter;
use tauri::Manager;
use tokenizers::Tokenizer;
use tokio::sync::Mutex as TokioMutex;

use crate::face_masking::FaceAnalysis;

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
// Gemm weights are stored transposed (transB=1), which keeps them in CoreML's weight file.
const SKY_URL: &str = "https://github.com/Orion-jada/RapidRAW/releases/download/sky-v2/skywater_segformer_b2_512.onnx";
const SKY_FILENAME: &str = "skywater_segformer_b2_512.onnx";
const SKY_SHA256: &str = "2cc52717a36a63bbba9b41cad3a4b29dfb15a9bd94e9d45cca6512e2dc98d8e4";
const SKY_INPUT_SIZE: u32 = 512;
const SKY_NUM_CLASSES: usize = 4;
const SKY_CLASS: usize = 1;

// SigLIP 2 (ViT-B/16, 224px) for natural-language photo search. The vision tower is fixed to a
// 1x3x224x224 input and patched so CoreML runs all of it, with float16 weights cast back to
// float32 at load and Gemm weights stored transposed (transB=1) to keep them in CoreML's weight
// file. The multilingual text tower (int8) and its tokenizer come unchanged from
// onnx-community's export, pinned to a revision.
const SEARCH_VISION_URL: &str = "https://github.com/Orion-jada/RapidRAW/releases/download/search-v2/siglip2_vision_b16_224.onnx";
const SEARCH_VISION_FILENAME: &str = "siglip2_vision_b16_224.onnx";
const SEARCH_VISION_SHA256: &str =
    "64aff7852f9c43198b968227f2bfb3f19d93d5455e65d5dd26cfbba43f2284f1";
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

// RawNIND UtNet2, Bayer variant (same darktable-ai package). Denoises the sensor data itself,
// before demosaicing: input is a 512x512 tile of RGGB blocks packed into 4 channels, output is
// the full-resolution 1024x1024 tile in camera RGB, again at an arbitrary learned scale.
const RAWNIND_BAYER_URL: &str = "https://github.com/Orion-jada/RapidRAW/releases/download/rawnind-v1/rawnind_utnet2_bayer_512.onnx";
const RAWNIND_BAYER_FILENAME: &str = "rawnind_utnet2_bayer_512.onnx";
const RAWNIND_BAYER_SHA256: &str =
    "da27509dab6a2915da67e988acd86cf71f9d5bbc8d1aa0ed32933578a887b901";
const RAWNIND_BAYER_TILE: usize = 512;
const RAWNIND_BAYER_OVERLAP: usize = 64;
const RAWNIND_BAYER_BORDER: usize = 8;

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

// Depth Anything 3 Mono-Large (Apache-2.0): relative depth for single photos, plus a sky head.
// Fixed to a 756x504 input so CoreML runs the whole graph (~0.2s on an M3 Max GPU). Photos of
// other shapes are stretched to it, which changes the map far less than padding them would.
// Weights are stored as float16 and cast back to float32 at load.
const DEPTH_URL: &str = "https://github.com/Orion-jada/RapidRAW/releases/download/depth3-v1/da3_mono_large_756x504.onnx";
const DEPTH_FILENAME: &str = "da3_mono_large_756x504.onnx";
const DEPTH_SHA256: &str = "eebb44105382fcc2118de9e03ab22fcdbfee549cdf332d56231d2c3160ddcdab";
const DEPTH_INPUT_WIDTH: u32 = 756;
const DEPTH_INPUT_HEIGHT: u32 = 504;
/// Sky head scores at or above this are sky, as in Depth Anything 3's own pipeline.
const DEPTH_SKY_THRESHOLD: f32 = 0.3;

// MoGe-3 ViT-L (MIT): normals and points for Relight, without MoGe-3's CUDA-only sparse
// refiner, which only refines depth. Takes any photo shape and runs on the CPU (~4s). Weights
// are stored as float16 and cast back at load. The normal map has the photo's aspect ratio with
// its long side at most NORMAL_MAX_SIZE.
const NORMAL_URL: &str =
    "https://github.com/Orion-jada/RapidRAW/releases/download/moge3-v1/moge3_vitl_normal.onnx";
const NORMAL_FILENAME: &str = "moge3_vitl_normal.onnx";
const NORMAL_MAX_SIZE: u32 = 768;
const NORMAL_NUM_TOKENS: i64 = 2500;
const NORMAL_SHA256: &str = "651ee8b995f32ef12041f91b47fe080f97aaf402c1e49355d9c18439fdfed08e";

// The same model fixed to a landscape and a portrait size, so CoreML runs all of it (~0.4s on an
// M3 Max GPU), with `metric_scale` left out. These files hold only the graph: its weights are
// ONNX external data pointing into NORMAL_FILENAME, which must sit next to them, so they're
// rebuilt whenever NORMAL_FILENAME changes. A photo is stretched to the closer one. Up to
// NORMAL_MAX_STRETCH, that moves its normals ~4° on average from the CPU model's, as much as a
// different token count does; squarer or wider photos use the CPU model.
const NORMAL_FIXED: [FixedNormalGraph; 2] = [
    FixedNormalGraph {
        width: 768,
        height: 512,
        filename: "moge3_vitl_normal_768x512.onnx",
        url: "https://github.com/Orion-jada/RapidRAW/releases/download/moge3-v1/moge3_vitl_normal_768x512.onnx",
        sha256: "866af69673867047764a88d1a350aa59dc23e7378e7bb16aa3c186c0f4717f4d",
    },
    FixedNormalGraph {
        width: 512,
        height: 768,
        filename: "moge3_vitl_normal_512x768.onnx",
        url: "https://github.com/Orion-jada/RapidRAW/releases/download/moge3-v1/moge3_vitl_normal_512x768.onnx",
        sha256: "c4c62adae689a8cda0169009858ec574612a3e510e225ad95ada61ae7b0a9776",
    },
];
/// How far a photo's aspect ratio may be stretched to fit a fixed graph: 4:3, 16:9 and 5:4 fit
/// one, 1:1 and 2:1 don't.
const NORMAL_MAX_STRETCH: f32 = 1.2;

struct FixedNormalGraph {
    width: u32,
    height: u32,
    filename: &'static str,
    url: &'static str,
    sha256: &'static str,
}

const ZOOM_MARGIN: f32 = 0.15;
const ZOOM_MAX_FRACTION: f32 = 0.6;
const ZOOM_MIN_SIDE: f32 = 1024.0;
const ZOOM_FEATHER: f32 = 0.1;

const ESAM3_IMAGE_ENCODER_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/efficientsam3_ev_m_image_encoder.onnx?download=true";
const ESAM3_IMAGE_ENCODER_FILENAME: &str = "efficientsam3_ev_m_image_encoder.onnx";
const ESAM3_IMAGE_ENCODER_SHA256: &str =
    "2ed08306579138120dbf9f1110fd31784d1c20fe1b67fa2187a0a2a6d7c82729";
const ESAM3_DECODER_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/efficientsam3_ev_m_decoder.onnx?download=true";
const ESAM3_DECODER_FILENAME: &str = "efficientsam3_ev_m_decoder.onnx";
const ESAM3_DECODER_SHA256: &str =
    "16accc22a07ee99c73ef8a14e286ec567f94c116e0d8efa1a3a6c0bc9a2596a8";
const ESAM3_TEXT_ENCODER_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/efficientsam3_ev_m_text_encoder.onnx?download=true";
const ESAM3_TEXT_ENCODER_FILENAME: &str = "efficientsam3_ev_m_text_encoder.onnx";
const ESAM3_TEXT_ENCODER_SHA256: &str =
    "3065ea6116f21f6e22bc70a0c041e5da9ae05fdec7b48df953394f5657609590";
const ESAM3_FACE_PROMPT_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/efficientsam3_face_prompt.bin?download=true";
const ESAM3_FACE_PROMPT_FILENAME: &str = "efficientsam3_face_prompt.bin";
const ESAM3_FACE_PROMPT_SHA256: &str =
    "8fdbe63ef853756b1da9cdc642c277a6e422452f3feaf35f4297e6aea3a1ffb9";
const ESAM3_INPUT_SIZE: u32 = 1008;
const ESAM3_INPUT_PADDING: u32 = 30;
const ESAM3_PROMPT_TOKENS: usize = 16;
const ESAM3_EMBED_DIM: usize = 256;
const ESAM3_EXEMPLAR_SLOTS: usize = 8;
pub const ESAM3_TEXT_SCORE_THRESHOLD: f32 = 0.3;
pub const ESAM3_FACE_SCORE_THRESHOLD: f32 = 0.3;
const ESAM3_MASK_LOW: f32 = 0.3;
const ESAM3_MASK_HIGH: f32 = 0.7;
const ESAM3_REFINE_SIZE: u32 = 1024;
const ESAM3_REFINE_RADIUS: usize = 16;
pub const ESAM3_TEXT_REFINE_EPS: f32 = 0.01;
const ESAM3_ZOOM_MIN_IOU: f32 = 0.3;

const FACE_LANDMARKS_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/mediapipe_face_landmarks.onnx?download=true";
const FACE_LANDMARKS_FILENAME: &str = "mediapipe_face_landmarks.onnx";
const FACE_LANDMARKS_SHA256: &str =
    "6524d596b4d9151b0d8d590844566261f21de5186ac38125bdcd1de6974b0bb4";
const FACE_SEGMENTER_URL: &str = "https://huggingface.co/CyberTimon/RapidRAW-Models/resolve/main/mediapipe_selfie_multiclass.onnx?download=true";
const FACE_SEGMENTER_FILENAME: &str = "mediapipe_selfie_multiclass.onnx";
const FACE_SEGMENTER_SHA256: &str =
    "a198470a4edec350a33b8182c149e417effc5e61c0506b947f226e7a81b9a08e";

type SharedSession = Arc<Mutex<Session>>;

const GPU_IDLE: u8 = 0;
const GPU_ACTIVE: u8 = 1;
const GPU_FAILED: u8 = 2;
const GPU_DISABLED: u8 = 3;

static GPU_STATUS: AtomicU8 = AtomicU8::new(GPU_IDLE);

struct ModelAsset {
    filename: &'static str,
    url: &'static str,
    sha256: Option<&'static str>,
    name: &'static str,
}

const SAM_ENCODER_ASSET: ModelAsset = ModelAsset {
    filename: ENCODER_FILENAME,
    url: ENCODER_URL,
    sha256: Some(ENCODER_SHA256),
    name: "SAM Encoder",
};
const SAM_DECODER_ASSET: ModelAsset = ModelAsset {
    filename: DECODER_FILENAME,
    url: DECODER_URL,
    sha256: Some(DECODER_SHA256),
    name: "SAM Decoder",
};
const BIREFNET_ASSET: ModelAsset = ModelAsset {
    filename: BIREFNET_FILENAME,
    url: BIREFNET_URL,
    sha256: Some(BIREFNET_SHA256),
    name: "Foreground Model",
};
const SKY_ASSET: ModelAsset = ModelAsset {
    filename: SKY_FILENAME,
    url: SKY_URL,
    sha256: Some(SKY_SHA256),
    name: "Sky Model",
};
const ESAM3_IMAGE_ENCODER_ASSET: ModelAsset = ModelAsset {
    filename: ESAM3_IMAGE_ENCODER_FILENAME,
    url: ESAM3_IMAGE_ENCODER_URL,
    sha256: Some(ESAM3_IMAGE_ENCODER_SHA256),
    name: "EfficientSAM3 Encoder",
};
const ESAM3_DECODER_ASSET: ModelAsset = ModelAsset {
    filename: ESAM3_DECODER_FILENAME,
    url: ESAM3_DECODER_URL,
    sha256: Some(ESAM3_DECODER_SHA256),
    name: "EfficientSAM3 Decoder",
};
const ESAM3_TEXT_ENCODER_ASSET: ModelAsset = ModelAsset {
    filename: ESAM3_TEXT_ENCODER_FILENAME,
    url: ESAM3_TEXT_ENCODER_URL,
    sha256: Some(ESAM3_TEXT_ENCODER_SHA256),
    name: "EfficientSAM3 Text Encoder",
};
const ESAM3_FACE_PROMPT_ASSET: ModelAsset = ModelAsset {
    filename: ESAM3_FACE_PROMPT_FILENAME,
    url: ESAM3_FACE_PROMPT_URL,
    sha256: Some(ESAM3_FACE_PROMPT_SHA256),
    name: "Face Prompt",
};
const FACE_LANDMARKS_ASSET: ModelAsset = ModelAsset {
    filename: FACE_LANDMARKS_FILENAME,
    url: FACE_LANDMARKS_URL,
    sha256: Some(FACE_LANDMARKS_SHA256),
    name: "Face Landmarks Model",
};
const FACE_SEGMENTER_ASSET: ModelAsset = ModelAsset {
    filename: FACE_SEGMENTER_FILENAME,
    url: FACE_SEGMENTER_URL,
    sha256: Some(FACE_SEGMENTER_SHA256),
    name: "Face Segmentation Model",
};
const DEPTH_ASSET: ModelAsset = ModelAsset {
    filename: DEPTH_FILENAME,
    url: DEPTH_URL,
    sha256: Some(DEPTH_SHA256),
    name: "Depth Model",
};
const CLIP_MODEL_ASSET: ModelAsset = ModelAsset {
    filename: CLIP_MODEL_FILENAME,
    url: CLIP_MODEL_URL,
    sha256: Some(CLIP_MODEL_SHA256),
    name: "CLIP Model",
};
const CLIP_TOKENIZER_ASSET: ModelAsset = ModelAsset {
    filename: CLIP_TOKENIZER_FILENAME,
    url: CLIP_TOKENIZER_URL,
    sha256: None,
    name: "CLIP Tokenizer",
};
const DENOISE_ASSET: ModelAsset = ModelAsset {
    filename: DENOISE_FILENAME,
    url: DENOISE_URL,
    sha256: Some(DENOISE_SHA256),
    name: "NIND Denoise Model",
};
const RAWNIND_ASSET: ModelAsset = ModelAsset {
    filename: RAWNIND_FILENAME,
    url: RAWNIND_URL,
    sha256: Some(RAWNIND_SHA256),
    name: "RawNIND Denoise Model",
};
const RAWNIND_BAYER_ASSET: ModelAsset = ModelAsset {
    filename: RAWNIND_BAYER_FILENAME,
    url: RAWNIND_BAYER_URL,
    sha256: Some(RAWNIND_BAYER_SHA256),
    name: "RawNIND Bayer Denoise Model",
};
const UPSCALE_ASSET: ModelAsset = ModelAsset {
    filename: UPSCALE_FILENAME,
    url: UPSCALE_URL,
    sha256: Some(UPSCALE_SHA256),
    name: "Upscale Model",
};
const SEARCH_VISION_ASSET: ModelAsset = ModelAsset {
    filename: SEARCH_VISION_FILENAME,
    url: SEARCH_VISION_URL,
    sha256: Some(SEARCH_VISION_SHA256),
    name: "Search Image Model",
};
const SEARCH_TEXT_ASSET: ModelAsset = ModelAsset {
    filename: SEARCH_TEXT_FILENAME,
    url: SEARCH_TEXT_URL,
    sha256: Some(SEARCH_TEXT_SHA256),
    name: "Search Text Model",
};
const SEARCH_TOKENIZER_ASSET: ModelAsset = ModelAsset {
    filename: SEARCH_TOKENIZER_FILENAME,
    url: SEARCH_TOKENIZER_URL,
    sha256: Some(SEARCH_TOKENIZER_SHA256),
    name: "Search Tokenizer",
};
const LAMA_ASSET: ModelAsset = ModelAsset {
    filename: LAMA_FILENAME,
    url: LAMA_URL,
    sha256: Some(LAMA_SHA256),
    name: "Inpainting Model",
};
const NORMAL_ASSET: ModelAsset = ModelAsset {
    filename: NORMAL_FILENAME,
    url: NORMAL_URL,
    sha256: Some(NORMAL_SHA256),
    name: "Normal Model",
};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AiMaskModel {
    Subject,
    Foreground,
    Sky,
    Text,
    Face,
    Depth,
}

impl AiMaskModel {
    fn assets(self) -> &'static [ModelAsset] {
        match self {
            AiMaskModel::Subject => &[SAM_ENCODER_ASSET, SAM_DECODER_ASSET],
            AiMaskModel::Foreground => &[BIREFNET_ASSET],
            AiMaskModel::Sky => &[SKY_ASSET],
            AiMaskModel::Text => &[
                ESAM3_IMAGE_ENCODER_ASSET,
                ESAM3_DECODER_ASSET,
                ESAM3_TEXT_ENCODER_ASSET,
                CLIP_TOKENIZER_ASSET,
            ],
            AiMaskModel::Face => &[
                ESAM3_IMAGE_ENCODER_ASSET,
                ESAM3_DECODER_ASSET,
                ESAM3_FACE_PROMPT_ASSET,
                FACE_LANDMARKS_ASSET,
                FACE_SEGMENTER_ASSET,
            ],
            AiMaskModel::Depth => &[DEPTH_ASSET],
        }
    }
}

pub struct Esam3Features {
    pub path_hash: String,
    pub fpn: [Array<f32, IxDyn>; 3],
    pub pos: Array<f32, IxDyn>,
}

pub struct Esam3Prompt {
    features: Array<f32, IxDyn>,
    padding: Array<f32, IxDyn>,
}

pub struct AiModels {
    models_dir: PathBuf,
    use_gpu: bool,
    downloaded: Mutex<HashSet<AiMaskModel>>,
    sam_encoder: Mutex<Option<SharedSession>>,
    sam_decoder: Mutex<Option<SharedSession>>,
    birefnet: Mutex<Option<SharedSession>>,
    sky: Mutex<Option<SharedSession>>,
    esam3_image_encoder: Mutex<Option<SharedSession>>,
    esam3_decoder: Mutex<Option<SharedSession>>,
    esam3_text_encoder: Mutex<Option<SharedSession>>,
    esam3_face_prompt: Mutex<Option<Arc<Esam3Prompt>>>,
    clip_tokenizer: Mutex<Option<Arc<Tokenizer>>>,
    face_landmarks: Mutex<Option<SharedSession>>,
    face_segmenter: Mutex<Option<SharedSession>>,
    depth_anything: Mutex<Option<SharedSession>>,
}

impl AiModels {
    fn session(
        &self,
        slot: &Mutex<Option<SharedSession>>,
        filename: &str,
        use_gpu: bool,
    ) -> Result<SharedSession> {
        let mut slot = slot.lock().unwrap();
        if let Some(session) = slot.as_ref() {
            return Ok(Arc::clone(session));
        }
        let session = Arc::new(Mutex::new(build_session(
            &self.models_dir.join(filename),
            SessionKind::OneShot,
            self.use_gpu && use_gpu,
            None,
        )?));
        *slot = Some(Arc::clone(&session));
        Ok(session)
    }

    /// Loads one of this fork's models the first time it's used: on macOS with CoreML, which
    /// it's verified to run correctly and faster with, and on the CPU elsewhere. `upgrading`
    /// starts a model CoreML takes seconds to prepare on the CPU, until CoreML is ready.
    fn coreml_session(
        &self,
        slot: &Mutex<Option<SharedSession>>,
        asset: &ModelAsset,
        upgrading: bool,
    ) -> Result<SharedSession> {
        let mut slot = slot.lock().unwrap();
        if let Some(session) = slot.as_ref() {
            return Ok(Arc::clone(session));
        }
        let path = self.models_dir.join(asset.filename);
        let sha256 = asset
            .sha256
            .expect("models compiled for CoreML have a known hash");
        let units = self.use_gpu.then_some(CoreMLComputeUnits::CPUAndGPU);
        let session = if upgrading {
            build_session_upgrading(&path, sha256, units)?
        } else {
            Arc::new(Mutex::new(build_coreml_or_cpu_session(
                &path,
                sha256,
                units,
                SessionKind::OneShot,
            )?))
        };
        *slot = Some(Arc::clone(&session));
        Ok(session)
    }

    pub fn sam_encoder(&self) -> Result<SharedSession> {
        self.coreml_session(&self.sam_encoder, &SAM_ENCODER_ASSET, false)
    }

    pub fn sam_decoder(&self) -> Result<SharedSession> {
        self.session(&self.sam_decoder, DECODER_FILENAME, false)
    }

    pub fn birefnet(&self) -> Result<SharedSession> {
        self.coreml_session(&self.birefnet, &BIREFNET_ASSET, false)
    }

    pub fn sky(&self) -> Result<SharedSession> {
        self.coreml_session(&self.sky, &SKY_ASSET, false)
    }

    pub fn esam3_image_encoder(&self) -> Result<SharedSession> {
        self.session(
            &self.esam3_image_encoder,
            ESAM3_IMAGE_ENCODER_FILENAME,
            gpu_allowed_on_platform(),
        )
    }

    pub fn esam3_decoder(&self) -> Result<SharedSession> {
        self.session(
            &self.esam3_decoder,
            ESAM3_DECODER_FILENAME,
            gpu_allowed_on_platform(),
        )
    }

    pub fn depth_anything(&self) -> Result<SharedSession> {
        self.coreml_session(&self.depth_anything, &DEPTH_ASSET, true)
    }

    pub fn face_landmarks(&self) -> Result<SharedSession> {
        self.session(&self.face_landmarks, FACE_LANDMARKS_FILENAME, false)
    }

    pub fn face_segmenter(&self) -> Result<SharedSession> {
        self.session(&self.face_segmenter, FACE_SEGMENTER_FILENAME, false)
    }

    pub fn esam3_face_prompt(&self) -> Result<Arc<Esam3Prompt>> {
        self.stored_prompt(&self.esam3_face_prompt, ESAM3_FACE_PROMPT_FILENAME)
    }

    fn stored_prompt(
        &self,
        slot: &Mutex<Option<Arc<Esam3Prompt>>>,
        filename: &str,
    ) -> Result<Arc<Esam3Prompt>> {
        let mut slot = slot.lock().unwrap();
        if let Some(prompt) = slot.as_ref() {
            return Ok(Arc::clone(prompt));
        }
        let bytes = fs::read(self.models_dir.join(filename))?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&chunk| f32::from_le_bytes(chunk))
            .collect();
        let feature_len = ESAM3_PROMPT_TOKENS * ESAM3_EMBED_DIM;
        if values.len() != feature_len + ESAM3_PROMPT_TOKENS {
            return Err(anyhow::anyhow!("Invalid prompt embedding {}", filename));
        }
        let prompt = Arc::new(Esam3Prompt {
            features: Array::from_shape_vec(
                IxDyn(&[ESAM3_PROMPT_TOKENS, 1, ESAM3_EMBED_DIM]),
                values[..feature_len].to_vec(),
            )?,
            padding: Array::from_shape_vec(
                IxDyn(&[1, ESAM3_PROMPT_TOKENS]),
                values[feature_len..].to_vec(),
            )?,
        });
        *slot = Some(Arc::clone(&prompt));
        Ok(prompt)
    }

    pub fn esam3_text_prompt(&self, text: &str) -> Result<Esam3Prompt> {
        let tokenizer = {
            let mut slot = self.clip_tokenizer.lock().unwrap();
            match slot.as_ref() {
                Some(tokenizer) => Arc::clone(tokenizer),
                None => {
                    let tokenizer = Arc::new(
                        Tokenizer::from_file(self.models_dir.join(CLIP_TOKENIZER_FILENAME))
                            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
                    );
                    *slot = Some(Arc::clone(&tokenizer));
                    tokenizer
                }
            }
        };
        let encoding = tokenizer
            .encode(text.trim(), true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let ids = encoding.get_ids();
        let mut tokens = Array::<i64, _>::zeros((1, ESAM3_PROMPT_TOKENS));
        let kept = ids.len().min(ESAM3_PROMPT_TOKENS);
        for (i, &id) in ids[..kept].iter().enumerate() {
            tokens[[0, i]] = id as i64;
        }
        if let Some(&end) = ids.last()
            && ids.len() > ESAM3_PROMPT_TOKENS
        {
            tokens[[0, ESAM3_PROMPT_TOKENS - 1]] = end as i64;
        }

        let session = self.session(&self.esam3_text_encoder, ESAM3_TEXT_ENCODER_FILENAME, false)?;
        let mut session = session.lock().unwrap();
        let outputs = session.run(ort::inputs!["tokens" => Tensor::from_array(tokens)?])?;
        Ok(Esam3Prompt {
            features: outputs["language_features"]
                .try_extract_array::<f32>()?
                .to_owned(),
            padding: outputs["language_padding"]
                .try_extract_array::<f32>()?
                .to_owned(),
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SessionKind {
    OneShot,
    Persistent,
}

#[cfg(target_os = "windows")]
fn high_performance_dml_device_id() -> Option<i32> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
        IDXGIAdapter1, IDXGIFactory6,
    };

    unsafe {
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let preferred: IDXGIAdapter1 = factory
            .EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
            .ok()?;
        let luid = preferred.GetDesc1().ok()?.AdapterLuid;
        (0..)
            .map_while(|i| factory.EnumAdapters1(i).ok())
            .position(|adapter| {
                adapter
                    .GetDesc1()
                    .is_ok_and(|desc| desc.AdapterLuid == luid)
            })
            .map(|index| index as i32)
    }
}

#[cfg(target_os = "windows")]
fn gpu_execution_provider() -> Option<ExecutionProviderDispatch> {
    Some(
        ort::execution_providers::DirectMLExecutionProvider::default()
            .with_device_id(high_performance_dml_device_id().unwrap_or(0))
            .build()
            .error_on_failure(),
    )
}

#[cfg(target_os = "macos")]
fn gpu_execution_provider() -> Option<ExecutionProviderDispatch> {
    use ort::execution_providers::coreml::{CoreMLComputeUnits, CoreMLModelFormat};

    Some(
        ort::execution_providers::CoreMLExecutionProvider::default()
            .with_model_format(CoreMLModelFormat::MLProgram)
            .with_compute_units(CoreMLComputeUnits::All)
            .build()
            .error_on_failure(),
    )
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn gpu_execution_provider() -> Option<ExecutionProviderDispatch> {
    None
}

fn gpu_allowed_on_platform() -> bool {
    !cfg!(target_os = "macos")
}

type GpuModelPatch = fn() -> ort::Result<(Model, Vec<Output>)>;

fn i64_initializer(values: &[i64]) -> ort::Result<Tensor<i64>> {
    let mut tensor = Tensor::<i64>::new(&Allocator::default(), [values.len()])?;
    tensor.extract_tensor_mut().1.copy_from_slice(values);
    Ok(tensor)
}

fn denoise_crop_as_slice() -> ort::Result<(Model, Vec<Output>)> {
    let output_type = ValueType::Tensor {
        ty: TensorElementType::Float32,
        shape: Shape::new([1, 3, -1, -1]),
        dimension_symbols: SymbolicDimensions::empty(4),
    };

    let mut graph = Graph::new()?;
    graph.add_initializer("crop_starts", i64_initializer(&[2, 2])?, false)?;
    graph.add_initializer("crop_ends", i64_initializer(&[-2, -2])?, false)?;
    graph.add_initializer("crop_axes", i64_initializer(&[2, 3])?, false)?;
    graph.add_node(Node::new(
        "Slice",
        ONNX_DOMAIN,
        "crop_slice",
        ["conv2d_9", "crop_starts", "crop_ends", "crop_axes"],
        ["denoised"],
        [],
    )?)?;
    graph.set_outputs([Outlet::new("denoised", &output_type)?])?;

    let mut model = Model::new([Opset::new(ONNX_DOMAIN, 18)?])?;
    model.add_graph(graph)?;
    Ok((
        model,
        vec![Output {
            name: "denoised".to_string(),
            output_type,
        }],
    ))
}

fn create_session(
    model_path: &Path,
    kind: SessionKind,
    gpu_provider: Option<ExecutionProviderDispatch>,
    gpu_patch: Option<GpuModelPatch>,
) -> Result<Session> {
    let use_gpu = gpu_provider.is_some();
    let cpu_provider = CPUExecutionProvider::default()
        .with_arena_allocator(kind == SessionKind::Persistent)
        .build();
    let providers: Vec<_> = gpu_provider.into_iter().chain([cpu_provider]).collect();

    let mut builder = Session::builder()?;
    if use_gpu || kind == SessionKind::OneShot {
        builder = builder.with_memory_pattern(false)?;
    }
    if use_gpu && kind == SessionKind::OneShot {
        builder = builder.with_config_entry("ep.dml.disable_memory_arena", "1")?;
    }
    let builder = builder.with_execution_providers(providers)?;
    match gpu_patch {
        Some(patch) => {
            let (model, outputs) = patch()?;
            let mut session = builder.edit_from_file(model_path)?;
            session.apply_model(&model)?;
            let mut session = session.into_session()?;
            session.outputs = outputs;
            Ok(session)
        }
        None => Ok(builder.commit_from_file(model_path)?),
    }
}

fn build_session(
    model_path: &Path,
    kind: SessionKind,
    use_gpu: bool,
    gpu_patch: Option<GpuModelPatch>,
) -> Result<Session> {
    if use_gpu
        && GPU_STATUS.load(Ordering::Acquire) < GPU_FAILED
        && let Some(gpu_provider) = gpu_execution_provider()
    {
        match create_session(model_path, kind, Some(gpu_provider), gpu_patch) {
            Ok(session) => {
                let _ = GPU_STATUS.compare_exchange(
                    GPU_IDLE,
                    GPU_ACTIVE,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
                return Ok(session);
            }
            Err(e) => log::warn!(
                "GPU acceleration unavailable for {}, falling back to CPU: {}",
                model_path.display(),
                e
            ),
        }
    }
    create_session(model_path, kind, None, None)
}

pub fn gpu_acceleration_enabled(app_handle: &tauri::AppHandle) -> bool {
    GPU_STATUS.load(Ordering::Acquire) < GPU_FAILED
        && crate::app_settings::load_settings(app_handle.clone())
            .map(|settings| settings.ai_gpu_acceleration)
            .unwrap_or(true)
}

pub fn gpu_inference<T>(result: ort::Result<T>) -> Result<T> {
    result.map_err(|e| {
        if GPU_STATUS
            .compare_exchange(GPU_ACTIVE, GPU_FAILED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            log::warn!(
                "GPU inference failed, using the CPU for AI models from now on: {}",
                e
            );
        }
        e.into()
    })
}

pub fn reset_ai_sessions(ai_state_mutex: &Mutex<Option<AiState>>) {
    if let Some(ai_state) = ai_state_mutex.lock().unwrap().as_mut() {
        ai_state.release_sessions();
    }
    GPU_STATUS.store(GPU_IDLE, Ordering::Release);
}

/// Models that have been replaced, as the file names they were saved under and the SHA-256 their
/// CoreML cache is keyed by.
const SUPERSEDED_MODELS: &[(&[&str], &str)] = &[
    // SAM 1 (ViT-B), replaced by SAM 2.1
    (
        &["sam_vit_b_01ec64_encoder.onnx"],
        "16ab73d9c824886f0de2938c19df22fb9ec3deebfd0de58e65177e479213d7d1",
    ),
    (
        &["sam_vit_b_01ec64_decoder.onnx"],
        "85d0d672cf5b7fe763edcde429e5533e62f674af4b15c7d688b7673b0ef00bf7",
    ),
    // U²-Net foreground, replaced by BiRefNet
    (
        &["u2net.onnx"],
        "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491",
    ),
    // U²-Net sky, replaced by SkyWater-Seg
    (
        &["skyseg_u2net.onnx", "skyseg-u2net.onnx"],
        "ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39",
    ),
    // Depth Anything V2 Small, replaced by Depth Anything 3 Mono-Large
    (
        &["depth_anything_v2_vits.onnx"],
        "d2b11a11c1d4a12b47608fa65a17ee9a4c605b55ee1730c8e3b526304f2562be",
    ),
    // MoGe-2 ViT-S, replaced by MoGe-3 ViT-L
    (
        &["moge-2-vits-normal-onnx.onnx"],
        "24eacb5dc7a2c54c7bc98f7de085ffbed79ad006ea5b664c2c2cdc02ff3a52f0",
    ),
];

/// Compiled CoreML models live in this folder of the models folder, one subfolder per model hash.
/// It's renamed whenever the way models are handed to CoreML changes, so every model is compiled
/// afresh; the folders in `OLD_COREML_CACHES` are deleted at startup.
const COREML_CACHE: &str = "coreml_cache_v2";

/// `coreml_cache` holds models compiled with MatMul + Add fused, at up to 4.8 GB each.
const OLD_COREML_CACHES: &[&str] = &["coreml_cache"];

/// Frees the disk space taken by models in `SUPERSEDED_MODELS` and by old CoreML caches. Nothing
/// else is touched.
pub fn remove_superseded_models(app_handle: &tauri::AppHandle) {
    if let Ok(data_dir) = app_handle.path().app_data_dir() {
        remove_superseded_models_in(&data_dir.join("models"));
    }
}

fn remove_superseded_models_in(models_dir: &Path) {
    let superseded = SUPERSEDED_MODELS.iter().flat_map(|(filenames, sha256)| {
        filenames
            .iter()
            .map(|filename| models_dir.join(filename))
            .chain([models_dir.join(COREML_CACHE).join(&sha256[..16])])
    });
    let old_caches = OLD_COREML_CACHES.iter().map(|dir| models_dir.join(dir));
    for path in superseded.chain(old_caches) {
        let removed = match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.is_dir() => fs::remove_dir_all(&path),
            Ok(_) => fs::remove_file(&path),
            Err(_) => continue,
        };
        match removed {
            Ok(()) => log::info!("Removed superseded {}", path.display()),
            Err(e) => log::warn!("Failed to remove {}: {}", path.display(), e),
        }
    }
}

/// Creates a session for one of this fork's models. On macOS, `coreml_units` hands the model to
/// CoreML (GPU or Neural Engine), falling back to CPU if CoreML can't compile it; `None` keeps it
/// on CPU, as do other platforms.
///
/// Only enable CoreML for models verified to run correctly and faster with it. Models that are
/// int8-quantized, split into many CoreML/CPU partitions, or fail inside CoreML at run time
/// (which the load-time fallback can't catch) must stay on CPU.
/// Set `RAPIDRAW_AI_CPU_ONLY=1` to force CPU for every model.
fn build_coreml_or_cpu_session(
    model_path: &Path,
    model_sha256: &str,
    coreml_units: Option<CoreMLComputeUnits>,
    kind: SessionKind,
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

    create_session(model_path, kind, None, None)
}

/// Keyed by the model's hash so updated weights never reuse a stale compiled model.
#[cfg(target_os = "macos")]
fn coreml_cache_dir(model_path: &Path, model_sha256: &str) -> PathBuf {
    model_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(COREML_CACHE)
        .join(&model_sha256[..16])
}

/// A failed build deletes the cache, so the next attempt compiles from scratch instead of
/// reusing a broken one.
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

    // ORT fuses MatMul + Add into Gemm, and its CoreML EP passes a Gemm weight to CoreML by
    // transposing it and writing it into the model as text instead of into weight.bin. For a
    // ViT that's gigabytes of cache and minutes of compiling. Unfused, CoreML gets matmul + add
    // with the weights in weight.bin, and the output and speed are identical. (The value is a
    // `;`-separated list.)
    let session = Session::builder()?
        .with_config_entry(
            "optimization.disable_specified_optimizers",
            "MatMulAddFusion",
        )?
        .with_execution_providers([coreml])?
        .commit_from_file(model_path);
    match session {
        Ok(session) => {
            remove_coreml_sources(&cache_dir);
            Ok(session)
        }
        Err(e) => {
            let _ = fs::remove_dir_all(&cache_dir);
            Err(e.into())
        }
    }
}

/// ORT keeps the CoreML model it wrote (`Manifest.json`, `Data`) next to the compiled copy it
/// runs (`compiled_model.mlmodelc`), and once the compiled copy exists it never reads the source
/// again, so deleting it halves the cache. `model` itself must stay: it's how ORT tells the
/// model is cached.
#[cfg(target_os = "macos")]
fn remove_coreml_sources(cache_dir: &Path) {
    let partitions = fs::read_dir(cache_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|model| fs::read_dir(model.path()).ok())
        .flatten()
        .flatten();
    for partition in partitions {
        let package = partition.path().join("model");
        if !package.join("compiled_model.mlmodelc").is_dir() {
            continue;
        }
        for source in ["Manifest.json", "Data", "tmp"] {
            let path = package.join(source);
            let _ = match fs::symlink_metadata(&path) {
                Ok(metadata) if metadata.is_dir() => fs::remove_dir_all(&path),
                Ok(_) => fs::remove_file(&path),
                Err(_) => continue,
            };
        }
    }
}

/// Like `build_coreml_or_cpu_session`, for big models CoreML takes several seconds to prepare on first load
/// (and again whenever macOS drops its cache). Returns a CPU session at once and swaps in the
/// CoreML one when it's ready, so nothing waits on the compile.
fn build_session_upgrading(
    model_path: &Path,
    model_sha256: &str,
    coreml_units: Option<CoreMLComputeUnits>,
) -> Result<Arc<Mutex<Session>>> {
    let session = Arc::new(Mutex::new(create_session(
        model_path,
        SessionKind::Persistent,
        None,
        None,
    )?));

    #[cfg(target_os = "macos")]
    if let Some(coreml_units) = coreml_units
        && std::env::var_os("RAPIDRAW_AI_CPU_ONLY").is_none()
    {
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

/// The CoreML compute units this fork's models run with on macOS, unless GPU acceleration is
/// turned off in the settings.
fn coreml_units(app_handle: &tauri::AppHandle) -> Option<CoreMLComputeUnits> {
    gpu_acceleration_enabled(app_handle).then_some(CoreMLComputeUnits::CPUAndGPU)
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

#[derive(Default)]
pub struct AiState {
    pub models: Option<Arc<AiModels>>,
    pub denoise_model: Option<Arc<Mutex<Session>>>,
    pub raw_denoise_model: Option<Arc<Mutex<Session>>>,
    pub raw_bayer_denoise_model: Option<Arc<Mutex<Session>>>,
    pub upscale_model: Option<Arc<Mutex<Session>>>,
    pub clip_models: Option<Arc<ClipModels>>,
    pub search_models: Option<Arc<SearchModels>>,
    pub lama_model: Option<Arc<Mutex<Session>>>,
    pub normal_model: Option<Arc<NormalModel>>,
    pub embeddings: Option<ImageEmbeddings>,
    pub esam3_features: Option<Arc<Esam3Features>>,
    pub face_analysis: Option<Arc<FaceAnalysis>>,
    pub depth_map: Option<CachedDepthMap>,
}

impl AiState {
    pub fn release_sessions(&mut self) {
        self.models = None;
        self.denoise_model = None;
        self.raw_denoise_model = None;
        self.raw_bayer_denoise_model = None;
        self.upscale_model = None;
        self.clip_models = None;
        self.search_models = None;
        self.lama_model = None;
        self.normal_model = None;
    }
}

fn cached_ai_state<T>(
    ai_state_mutex: &Mutex<Option<AiState>>,
    get: impl Fn(&AiState) -> Option<T>,
) -> Option<T> {
    let mut ai_state = ai_state_mutex.lock().unwrap();
    if GPU_STATUS
        .compare_exchange(
            GPU_FAILED,
            GPU_DISABLED,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_ok()
        && let Some(ai_state) = ai_state.as_mut()
    {
        ai_state.release_sessions();
    }
    ai_state.as_ref().and_then(get)
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

    let guide_lr = if (lr_w, lr_h) == (hr_w, hr_h) {
        std::borrow::Cow::Borrowed(guide_hr)
    } else {
        std::borrow::Cow::Owned(imageops::resize(guide_hr, lr_w, lr_h, FilterType::Triangle))
    };

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

    drop(guide_lr);

    let mut temp1 = vec![0.0; area];
    let mut temp2 = vec![0.0; area];

    let box_filter = |src: Vec<f32>, temp1: &mut [f32], temp2: &mut [f32]| {
        let mut mean = vec![0.0; area];
        box_filter_opt(&src, &mut mean, temp1, temp2, w, h, r);
        mean
    };
    let mean_i = box_filter(i_lr, &mut temp1, &mut temp2);
    let mean_p = box_filter(p_lr, &mut temp1, &mut temp2);
    let mut a = box_filter(ip_lr, &mut temp1, &mut temp2);
    let mut b = box_filter(ii_lr, &mut temp1, &mut temp2);

    a.par_iter_mut()
        .zip(b.par_iter_mut())
        .zip(mean_i.par_iter())
        .zip(mean_p.par_iter())
        .for_each(|(((a_out, b_out), &m_i), &m_p)| {
            let (m_ip, m_ii) = (*a_out, *b_out);
            let cov_ip = m_ip - m_i * m_p;
            let var_i = m_ii - m_i * m_i;
            let a_val = cov_ip / (var_i + eps);
            *a_out = a_val;
            *b_out = m_p - a_val * m_i;
        });

    let (mut mean_a, mut mean_b) = (mean_i, mean_p);
    box_filter_opt(&a, &mut mean_a, &mut temp1, &mut temp2, w, h, r);
    box_filter_opt(&b, &mut mean_b, &mut temp1, &mut temp2, w, h, r);
    drop((a, b, temp1, temp2));

    let mean_a_img = ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(lr_w, lr_h, mean_a).unwrap();
    let mean_b_img = ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(lr_w, lr_h, mean_b).unwrap();

    let upscale = |img: ImageBuffer<Luma<f32>, Vec<f32>>| {
        if (lr_w, lr_h) == (hr_w, hr_h) {
            img.into_raw()
        } else {
            imageops::resize(&img, hr_w, hr_h, FilterType::Triangle).into_raw()
        }
    };
    let mean_a_hr_raw = upscale(mean_a_img);
    let mean_b_hr_raw = upscale(mean_b_img);
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

fn bilinear_sample(i: usize, scale: f32, n: usize) -> (usize, usize, f32) {
    let f = ((i as f32 + 0.5) * scale - 0.5).max(0.0);
    let i0 = (f.floor() as usize).min(n - 1);
    let i1 = (i0 + 1).min(n - 1);
    (i0, i1, (f - i0 as f32).clamp(0.0, 1.0))
}

pub fn fast_color_guided_filter(
    guide_hr: &RgbImage,
    mask_lr: &GrayImage,
    r: usize,
    eps: f32,
) -> GrayImage {
    let (hr_w, hr_h) = guide_hr.dimensions();
    let (lr_w, lr_h) = mask_lr.dimensions();
    let (w, h) = (lr_w as usize, lr_h as usize);
    let area = w * h;

    let guide_lr = imageops::resize(guide_hr, lr_w, lr_h, FilterType::Triangle);
    let guide: [Vec<f32>; 3] =
        std::array::from_fn(|c| guide_lr.pixels().map(|p| p[c] as f32 / 255.0).collect());
    drop(guide_lr);
    let p: Vec<f32> = mask_lr.as_raw().iter().map(|&v| v as f32 / 255.0).collect();

    let mut temp1 = vec![0.0; area];
    let mut temp2 = vec![0.0; area];
    let mut box_filter = |src: &[f32]| {
        let mut mean = vec![0.0; area];
        box_filter_opt(src, &mut mean, &mut temp1, &mut temp2, w, h, r);
        mean
    };
    let product = |a: &[f32], b: &[f32]| -> Vec<f32> {
        a.par_iter().zip(b.par_iter()).map(|(x, y)| x * y).collect()
    };

    let mean_i = guide.each_ref().map(|c| box_filter(c));
    let mean_p = box_filter(&p);
    let mean_ip: [Vec<f32>; 3] = std::array::from_fn(|c| box_filter(&product(&guide[c], &p)));
    let mean_ii = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        .map(|(i, j)| box_filter(&product(&guide[i], &guide[j])));
    drop((guide, p));

    let coefficients: Vec<[f32; 4]> = (0..area)
        .into_par_iter()
        .map(|k| {
            let m = [mean_i[0][k], mean_i[1][k], mean_i[2][k]];
            let cov = [0, 1, 2].map(|c| mean_ip[c][k] - m[c] * mean_p[k]);
            let s00 = mean_ii[0][k] - m[0] * m[0] + eps;
            let s01 = mean_ii[1][k] - m[0] * m[1];
            let s02 = mean_ii[2][k] - m[0] * m[2];
            let s11 = mean_ii[3][k] - m[1] * m[1] + eps;
            let s12 = mean_ii[4][k] - m[1] * m[2];
            let s22 = mean_ii[5][k] - m[2] * m[2] + eps;
            let c00 = s11 * s22 - s12 * s12;
            let c01 = s02 * s12 - s01 * s22;
            let c02 = s01 * s12 - s02 * s11;
            let c11 = s00 * s22 - s02 * s02;
            let c12 = s01 * s02 - s00 * s12;
            let c22 = s00 * s11 - s01 * s01;
            let det = s00 * c00 + s01 * c01 + s02 * c02;
            let a0 = (c00 * cov[0] + c01 * cov[1] + c02 * cov[2]) / det;
            let a1 = (c01 * cov[0] + c11 * cov[1] + c12 * cov[2]) / det;
            let a2 = (c02 * cov[0] + c12 * cov[1] + c22 * cov[2]) / det;
            [a0, a1, a2, mean_p[k] - a0 * m[0] - a1 * m[1] - a2 * m[2]]
        })
        .collect();
    drop((mean_i, mean_p, mean_ip, mean_ii));

    let mean_coefficients: [Vec<f32>; 4] = std::array::from_fn(|i| {
        box_filter(&coefficients.iter().map(|c| c[i]).collect::<Vec<f32>>())
    });
    drop(coefficients);

    let (hr_w, hr_h) = (hr_w as usize, hr_h as usize);
    let x_samples: Vec<(usize, usize, f32)> = (0..hr_w)
        .map(|x| bilinear_sample(x, w as f32 / hr_w as f32, w))
        .collect();
    let mut final_mask_raw = vec![0u8; hr_w * hr_h];

    final_mask_raw
        .par_chunks_exact_mut(hr_w)
        .zip(guide_hr.as_raw().par_chunks_exact(hr_w * 3))
        .enumerate()
        .for_each(|(y, (out_row, guide_row))| {
            let (y0, y1, ty) = bilinear_sample(y, h as f32 / hr_h as f32, h);
            for (x, (out, pixel)) in out_row
                .iter_mut()
                .zip(guide_row.as_chunks::<3>().0)
                .enumerate()
            {
                let (x0, x1, tx) = x_samples[x];
                let sample = |values: &[f32]| {
                    let top = values[y0 * w + x0] * (1.0 - tx) + values[y0 * w + x1] * tx;
                    let bottom = values[y1 * w + x0] * (1.0 - tx) + values[y1 * w + x1] * tx;
                    top * (1.0 - ty) + bottom * ty
                };
                let q = (0..3)
                    .map(|c| sample(&mean_coefficients[c]) * pixel[c] as f32 / 255.0)
                    .sum::<f32>()
                    + sample(&mean_coefficients[3]);
                *out = (q.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        });

    GrayImage::from_raw(hr_w as u32, hr_h as u32, final_mask_raw).unwrap()
}

fn get_models_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf> {
    let models_dir = app_handle.path().app_data_dir()?.join("models");
    if !models_dir.exists() {
        fs::create_dir_all(&models_dir)?;
    }
    Ok(models_dir)
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

async fn download_model(
    app_handle: &tauri::AppHandle,
    asset: &ModelAsset,
    dest: &Path,
) -> Result<()> {
    let mut response = reqwest::get(asset.url).await?.error_for_status()?;
    let total = response.content_length().unwrap_or(0);
    let mut bytes = Vec::with_capacity(total as usize);
    let mut last_progress = Instant::now();
    while let Some(chunk) = response.chunk().await? {
        bytes.extend_from_slice(&chunk);
        if total > 0 && last_progress.elapsed() >= Duration::from_millis(200) {
            last_progress = Instant::now();
            let _ = app_handle.emit(
                "ai-model-download-progress",
                serde_json::json!({ "name": asset.name, "current": bytes.len(), "total": total }),
            );
        }
    }
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
    asset: &ModelAsset,
) -> Result<()> {
    let dest_path = models_dir.join(asset.filename);
    let is_valid = match asset.sha256 {
        Some(expected_hash) => verify_sha256(&dest_path, expected_hash)?,
        None => dest_path.exists(),
    };

    if !is_valid {
        if dest_path.exists() {
            println!("Model {} has incorrect hash. Re-downloading.", asset.name);
            fs::remove_file(&dest_path)?;
        }
        let _ = app_handle.emit("ai-model-download-start", asset.name);
        let download_result = async {
            download_model(app_handle, asset, &dest_path).await?;
            if let Some(expected_hash) = asset.sha256
                && !verify_sha256(&dest_path, expected_hash)?
            {
                return Err(anyhow::anyhow!(
                    "Failed to verify model {} after download. Hash mismatch.",
                    asset.name
                ));
            }
            Ok(())
        }
        .await;
        let _ = app_handle.emit(
            "ai-model-download-finish",
            serde_json::json!({
                "name": asset.name,
                "error": download_result.as_ref().err().map(|e| e.to_string()),
            }),
        );
        download_result?;
    }
    Ok(())
}

pub async fn get_or_init_ai_models(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
    model: AiMaskModel,
) -> Result<Arc<AiModels>> {
    let is_downloaded = |models: &AiModels| models.downloaded.lock().unwrap().contains(&model);

    if let Some(models) = cached_ai_state(ai_state_mutex, |state| state.models.clone())
        && is_downloaded(&models)
    {
        return Ok(models);
    }

    let _guard = ai_init_lock.lock().await;

    let models = match cached_ai_state(ai_state_mutex, |state| state.models.clone()) {
        Some(models) => models,
        None => {
            let _ = ort::init().with_name("AI").commit();
            crate::register_exit_handler();

            let models = Arc::new(AiModels {
                models_dir: get_models_dir(app_handle)?,
                use_gpu: gpu_acceleration_enabled(app_handle),
                downloaded: Mutex::new(HashSet::new()),
                sam_encoder: Mutex::new(None),
                sam_decoder: Mutex::new(None),
                birefnet: Mutex::new(None),
                sky: Mutex::new(None),
                esam3_image_encoder: Mutex::new(None),
                esam3_decoder: Mutex::new(None),
                esam3_text_encoder: Mutex::new(None),
                esam3_face_prompt: Mutex::new(None),
                clip_tokenizer: Mutex::new(None),
                face_landmarks: Mutex::new(None),
                face_segmenter: Mutex::new(None),
                depth_anything: Mutex::new(None),
            });
            ai_state_mutex
                .lock()
                .unwrap()
                .get_or_insert_with(AiState::default)
                .models = Some(models.clone());
            models
        }
    };

    if !is_downloaded(&models) {
        for asset in model.assets() {
            download_and_verify_model(app_handle, &models.models_dir, asset).await?;
        }
        models.downloaded.lock().unwrap().insert(model);
    }

    Ok(models)
}

pub async fn get_or_init_denoise_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(denoise_model) =
        cached_ai_state(ai_state_mutex, |state| state.denoise_model.clone())
    {
        return Ok(denoise_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(denoise_model) =
        cached_ai_state(ai_state_mutex, |state| state.denoise_model.clone())
    {
        return Ok(denoise_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(app_handle, &models_dir, &DENOISE_ASSET).await?;

    let _ = ort::init().with_name("AI-Denoise").commit();
    let model_path = models_dir.join(DENOISE_FILENAME);
    // On macOS it runs with CoreML, compiled and cached as this fork's models are.
    let session = if cfg!(target_os = "macos") {
        build_coreml_or_cpu_session(
            &model_path,
            DENOISE_SHA256,
            coreml_units(app_handle),
            SessionKind::Persistent,
        )?
    } else {
        build_session(
            &model_path,
            SessionKind::Persistent,
            gpu_acceleration_enabled(app_handle),
            Some(denoise_crop_as_slice),
        )?
    };
    let denoise_model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    ai_state_mutex
        .lock()
        .unwrap()
        .get_or_insert_with(AiState::default)
        .denoise_model = Some(denoise_model.clone());

    Ok(denoise_model)
}

pub async fn get_or_init_raw_denoise_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(model) = cached_ai_state(ai_state_mutex, |state| state.raw_denoise_model.clone()) {
        return Ok(model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(model) = cached_ai_state(ai_state_mutex, |state| state.raw_denoise_model.clone()) {
        return Ok(model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(app_handle, &models_dir, &RAWNIND_ASSET).await?;

    let _ = ort::init().with_name("AI-Denoise").commit();
    let session = build_coreml_or_cpu_session(
        &models_dir.join(RAWNIND_FILENAME),
        RAWNIND_SHA256,
        coreml_units(app_handle),
        SessionKind::Persistent,
    )?;
    let model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    ai_state_mutex
        .lock()
        .unwrap()
        .get_or_insert_with(AiState::default)
        .raw_denoise_model = Some(model.clone());

    Ok(model)
}

pub async fn get_or_init_raw_bayer_denoise_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(model) = cached_ai_state(ai_state_mutex, |state| {
        state.raw_bayer_denoise_model.clone()
    }) {
        return Ok(model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(model) = cached_ai_state(ai_state_mutex, |state| {
        state.raw_bayer_denoise_model.clone()
    }) {
        return Ok(model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(app_handle, &models_dir, &RAWNIND_BAYER_ASSET).await?;

    let _ = ort::init().with_name("AI-Denoise").commit();
    let session = build_coreml_or_cpu_session(
        &models_dir.join(RAWNIND_BAYER_FILENAME),
        RAWNIND_BAYER_SHA256,
        coreml_units(app_handle),
        SessionKind::Persistent,
    )?;
    let model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    ai_state_mutex
        .lock()
        .unwrap()
        .get_or_insert_with(AiState::default)
        .raw_bayer_denoise_model = Some(model.clone());

    Ok(model)
}

pub async fn get_or_init_upscale_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(model) = cached_ai_state(ai_state_mutex, |state| state.upscale_model.clone()) {
        return Ok(model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(model) = cached_ai_state(ai_state_mutex, |state| state.upscale_model.clone()) {
        return Ok(model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(app_handle, &models_dir, &UPSCALE_ASSET).await?;

    let _ = ort::init().with_name("AI-Upscale").commit();
    let session = build_coreml_or_cpu_session(
        &models_dir.join(UPSCALE_FILENAME),
        UPSCALE_SHA256,
        coreml_units(app_handle),
        SessionKind::Persistent,
    )?;
    let model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    ai_state_mutex
        .lock()
        .unwrap()
        .get_or_insert_with(AiState::default)
        .upscale_model = Some(model.clone());

    Ok(model)
}

pub async fn get_or_init_clip_models(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<ClipModels>> {
    if let Some(clip_models) = cached_ai_state(ai_state_mutex, |state| state.clip_models.clone()) {
        return Ok(clip_models);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(clip_models) = cached_ai_state(ai_state_mutex, |state| state.clip_models.clone()) {
        return Ok(clip_models);
    }

    let models_dir = get_models_dir(app_handle)?;

    download_and_verify_model(app_handle, &models_dir, &CLIP_MODEL_ASSET).await?;
    download_and_verify_model(app_handle, &models_dir, &CLIP_TOKENIZER_ASSET).await?;
    let clip_tokenizer_path = models_dir.join(CLIP_TOKENIZER_FILENAME);

    let _ = ort::init().with_name("AI-Tagging").commit();
    let clip_model_path = models_dir.join(CLIP_MODEL_FILENAME);
    let model = Mutex::new(build_session(
        &clip_model_path,
        SessionKind::Persistent,
        gpu_allowed_on_platform() && gpu_acceleration_enabled(app_handle),
        None,
    )?);
    let tokenizer =
        Tokenizer::from_file(clip_tokenizer_path).map_err(|e| anyhow::anyhow!(e.to_string()))?;

    crate::register_exit_handler();

    let clip_models = Arc::new(ClipModels { model, tokenizer });

    ai_state_mutex
        .lock()
        .unwrap()
        .get_or_insert_with(AiState::default)
        .clip_models = Some(clip_models.clone());

    Ok(clip_models)
}

pub async fn get_or_init_search_models(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<SearchModels>> {
    if let Some(search_models) =
        cached_ai_state(ai_state_mutex, |state| state.search_models.clone())
    {
        return Ok(search_models);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(search_models) =
        cached_ai_state(ai_state_mutex, |state| state.search_models.clone())
    {
        return Ok(search_models);
    }

    let models_dir = get_models_dir(app_handle)?;
    for asset in [
        &SEARCH_VISION_ASSET,
        &SEARCH_TEXT_ASSET,
        &SEARCH_TOKENIZER_ASSET,
    ] {
        download_and_verify_model(app_handle, &models_dir, asset).await?;
    }

    let _ = ort::init().with_name("AI-Search").commit();
    let vision = build_session_upgrading(
        &models_dir.join(SEARCH_VISION_FILENAME),
        SEARCH_VISION_SHA256,
        coreml_units(app_handle),
    )?;
    // int8: stays on CPU, where a short query takes ~15 ms.
    let text = Mutex::new(build_coreml_or_cpu_session(
        &models_dir.join(SEARCH_TEXT_FILENAME),
        SEARCH_TEXT_SHA256,
        None,
        SessionKind::Persistent,
    )?);
    let tokenizer = Tokenizer::from_file(models_dir.join(SEARCH_TOKENIZER_FILENAME))
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    crate::register_exit_handler();

    let search_models = Arc::new(SearchModels {
        vision,
        text,
        tokenizer,
    });

    ai_state_mutex
        .lock()
        .unwrap()
        .get_or_insert_with(AiState::default)
        .search_models = Some(search_models.clone());

    Ok(search_models)
}

pub async fn get_or_init_lama_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<Mutex<Session>>> {
    if let Some(lama_model) = cached_ai_state(ai_state_mutex, |state| state.lama_model.clone()) {
        return Ok(lama_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(lama_model) = cached_ai_state(ai_state_mutex, |state| state.lama_model.clone()) {
        return Ok(lama_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(app_handle, &models_dir, &LAMA_ASSET).await?;

    let _ = ort::init().with_name("AI-Inpainting").commit();
    let model_path = models_dir.join(LAMA_FILENAME);
    let session = build_session(&model_path, SessionKind::OneShot, false, None)?;
    let lama_model = Arc::new(Mutex::new(session));

    crate::register_exit_handler();

    ai_state_mutex
        .lock()
        .unwrap()
        .get_or_insert_with(AiState::default)
        .lama_model = Some(lama_model.clone());

    Ok(lama_model)
}

/// MoGe-3 for Relight. Sessions load when a photo needs them, and only one is kept, as each holds
/// about 2 GB: switching between landscape and portrait photos reloads the compiled CoreML model,
/// which takes under a second.
pub struct NormalModel {
    models_dir: PathBuf,
    /// Whether the fixed graphs may run with CoreML (GPU acceleration is on).
    use_coreml: bool,
    /// The session, and the index in NORMAL_FIXED of the graph it runs (`None`: any-size model).
    loaded: Mutex<Option<(Option<usize>, Session)>>,
    /// Fixed graphs that couldn't be downloaded or compiled; their photos use the CPU model.
    unavailable: [AtomicBool; NORMAL_FIXED.len()],
}

impl NormalModel {
    /// The fixed graph to run a `width` x `height` photo through, if CoreML can.
    fn fixed_graph_for(&self, width: u32, height: u32) -> Option<usize> {
        if !cfg!(target_os = "macos")
            || !self.use_coreml
            || std::env::var_os("RAPIDRAW_AI_CPU_ONLY").is_some()
        {
            return None;
        }
        fixed_normal_graph(width, height).filter(|&i| !self.unavailable[i].load(Ordering::Relaxed))
    }

    /// Runs `f` with the session for `graph`, loading it in place of the last one if needed. If
    /// CoreML can't compile a fixed graph, the any-size model runs instead, then and from then on.
    fn run<R>(
        &self,
        graph: Option<usize>,
        f: impl FnOnce(Option<&FixedNormalGraph>, &mut Session) -> Result<R>,
    ) -> Result<R> {
        let mut loaded = self.loaded.lock().unwrap();
        if loaded.as_ref().map(|(g, _)| *g) != Some(graph) {
            // Free the old session before the new one loads.
            *loaded = None;
            *loaded = Some(self.load(graph)?);
        }
        let (graph, session) = loaded.as_mut().unwrap();
        f(graph.map(|i| &NORMAL_FIXED[i]), session)
    }

    fn load(&self, graph: Option<usize>) -> Result<(Option<usize>, Session)> {
        #[cfg(target_os = "macos")]
        if let Some(index) = graph {
            let path = self.models_dir.join(NORMAL_FIXED[index].filename);
            match build_coreml_session(
                &path,
                NORMAL_FIXED[index].sha256,
                CoreMLComputeUnits::CPUAndGPU,
            ) {
                Ok(session) => {
                    log::info!("Loaded {} with CoreML", path.display());
                    return Ok((graph, session));
                }
                Err(e) => {
                    log::warn!(
                        "CoreML failed for {}, using the CPU model: {}",
                        path.display(),
                        e
                    );
                    self.unavailable[index].store(true, Ordering::Relaxed);
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = graph;

        let session = create_session(
            &self.models_dir.join(NORMAL_FILENAME),
            SessionKind::OneShot,
            None,
            None,
        )?;
        Ok((None, session))
    }
}

/// Index of the fixed graph closest to a `width` x `height` photo's aspect ratio, if it's within
/// NORMAL_MAX_STRETCH of it.
fn fixed_normal_graph(width: u32, height: u32) -> Option<usize> {
    let aspect = width as f32 / height.max(1) as f32;
    NORMAL_FIXED
        .iter()
        .map(|g| {
            let graph_aspect = g.width as f32 / g.height as f32;
            (aspect / graph_aspect).max(graph_aspect / aspect)
        })
        .enumerate()
        .filter(|&(_, stretch)| stretch <= NORMAL_MAX_STRETCH)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(index, _)| index)
}

pub async fn get_or_init_normal_model(
    app_handle: &tauri::AppHandle,
    ai_state_mutex: &Mutex<Option<AiState>>,
    ai_init_lock: &TokioMutex<()>,
) -> Result<Arc<NormalModel>> {
    if let Some(normal_model) = cached_ai_state(ai_state_mutex, |state| state.normal_model.clone())
    {
        return Ok(normal_model);
    }

    let _guard = ai_init_lock.lock().await;

    if let Some(normal_model) = cached_ai_state(ai_state_mutex, |state| state.normal_model.clone())
    {
        return Ok(normal_model);
    }

    let models_dir = get_models_dir(app_handle)?;
    download_and_verify_model(app_handle, &models_dir, &NORMAL_ASSET).await?;

    // The fixed graphs are only for CoreML. Without them, photos use the CPU model.
    let use_coreml = coreml_units(app_handle).is_some();
    let unavailable = std::array::from_fn(|_| AtomicBool::new(!cfg!(target_os = "macos")));
    if cfg!(target_os = "macos") && use_coreml {
        for (fixed, unavailable) in NORMAL_FIXED.iter().zip(&unavailable) {
            let asset = ModelAsset {
                filename: fixed.filename,
                url: fixed.url,
                sha256: Some(fixed.sha256),
                name: NORMAL_ASSET.name,
            };
            if let Err(e) = download_and_verify_model(app_handle, &models_dir, &asset).await {
                log::warn!(
                    "Couldn't get {}, using the CPU model: {}",
                    fixed.filename,
                    e
                );
                unavailable.store(true, Ordering::Relaxed);
            }
        }
    }

    let _ = ort::init().with_name("AI-Normal").commit();
    let normal_model = Arc::new(NormalModel {
        models_dir,
        use_coreml,
        loaded: Mutex::new(None),
        unavailable,
    });

    crate::register_exit_handler();

    ai_state_mutex
        .lock()
        .unwrap()
        .get_or_insert_with(AiState::default)
        .normal_model = Some(normal_model.clone());

    Ok(normal_model)
}

#[derive(Clone, Copy)]
struct TileParams {
    cs: usize,
    margin: usize,
    overlap: usize,
}

const DENOISE_TILE: TileParams = TileParams {
    cs: 504,
    margin: 12,
    overlap: 64,
};

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

fn tile_origins(len: usize, params: TileParams) -> Vec<i32> {
    let valid = params.cs - 2 * params.margin;
    let step = (valid - params.overlap).max(1);
    let mut origins = Vec::new();
    let mut start = 0;
    loop {
        origins.push(start as i32 - params.margin as i32);
        if start + valid >= len {
            break;
        }
        start += step;
    }
    origins
}

fn tile_blend_weights(params: TileParams) -> Vec<f32> {
    let valid = params.cs - 2 * params.margin;
    let ramp = |d: usize| ((d as f32 + 0.5) / params.overlap.max(1) as f32).min(1.0);
    (0..valid)
        .map(|i| {
            let t = ramp(i).min(ramp(valid - 1 - i));
            t * t * (3.0 - 2.0 * t)
        })
        .collect()
}

fn run_native_denoise(
    img: &Rgb32FImage,
    session: &Mutex<Session>,
    params: TileParams,
    on_progress: &dyn Fn(f32),
) -> Result<Rgb32FImage> {
    let (width, height) = (img.width() as usize, img.height() as usize);
    let valid = params.cs - 2 * params.margin;
    let weights = tile_blend_weights(params);
    let xs = tile_origins(width, params);
    let ys = tile_origins(height, params);
    let total = xs.len() * ys.len();

    let mut accumulator = vec![0.0f32; width * height * 3];
    let mut weight_sum = vec![0.0f32; width * height];

    for (i, (&y0, &x0)) in ys
        .iter()
        .flat_map(|y| xs.iter().map(move |x| (y, x)))
        .enumerate()
    {
        on_progress(i as f32 / total as f32);

        let crop = extract_tile_mirror(img, x0, y0, params.cs);
        let t_input = Tensor::from_array(crop.as_standard_layout().to_owned())?;

        let tile = {
            let mut sess = session.lock().unwrap();
            let outputs = gpu_inference(sess.run(ort::inputs![t_input]))?;
            outputs[0]
                .try_extract_array::<f32>()?
                .to_owned()
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| anyhow::anyhow!("Unexpected output shape: {}", e))?
        };

        for ty in 0..valid {
            let gy = y0 + (params.margin + ty) as i32;
            if gy < 0 || gy >= height as i32 {
                continue;
            }
            for tx in 0..valid {
                let gx = x0 + (params.margin + tx) as i32;
                if gx < 0 || gx >= width as i32 {
                    continue;
                }
                let w = weights[ty] * weights[tx];
                let idx = gy as usize * width + gx as usize;
                for c in 0..3 {
                    let v = tile[[0, c, params.margin + ty, params.margin + tx]].clamp(0.0, 1.0);
                    accumulator[idx * 3 + c] += v * w;
                }
                weight_sum[idx] += w;
            }
        }
    }

    let mut out = Rgb32FImage::new(width as u32, height as u32);
    out.par_chunks_exact_mut(3)
        .zip(accumulator.par_chunks_exact(3))
        .zip(weight_sum.par_iter())
        .for_each(|((px, acc), &w)| {
            let inv = if w > 0.0 { 1.0 / w } else { 0.0 };
            for c in 0..3 {
                px[c] = (acc[c] * inv).clamp(0.0, 1.0);
            }
        });
    Ok(out)
}

pub fn run_ai_denoise(
    rgb_img: &Rgb32FImage,
    session: &Mutex<Session>,
    on_progress: &dyn Fn(f32),
) -> Result<DynamicImage> {
    let out = run_native_denoise(rgb_img, session, DENOISE_TILE, on_progress)?;
    Ok(DynamicImage::ImageRgb32F(out))
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

/// Offset (row, col) within the repeating 2x2 CFA where a block reads R G / G B. `None` for
/// anything but a 2x2 Bayer pattern with red and blue on the diagonal.
fn rggb_offset(cfa: &rawler::CFA) -> Option<(usize, usize)> {
    use rawler::cfa::CFAColor;
    if cfa.width != 2 || cfa.height != 2 {
        return None;
    }
    let is = |row: usize, col: usize, color: CFAColor| cfa.color_at(row, col) == color as usize;
    [(0, 0), (0, 1), (1, 0), (1, 1)]
        .into_iter()
        .find(|&(oy, ox)| {
            is(oy, ox, CFAColor::RED)
                && is(oy, ox + 1, CFAColor::GREEN)
                && is(oy + 1, ox, CFAColor::GREEN)
                && is(oy + 1, ox + 1, CFAColor::BLUE)
        })
}

/// RawNIND denoise of Bayer sensor data, in place and in raw units, before demosaicing. Each
/// photosite is normalised to [0, 1] with its own black and white level, as the model was
/// trained. The model returns full-resolution camera RGB; each photosite takes its own colour
/// from it (re-mosaicing, as darktable does), so demosaicing and everything after it stay the
/// same. Photosites at or near clipping are left alone. `strength` blends from the original (0)
/// to fully denoised (1). Fails on non-Bayer sensors, leaving the data untouched.
pub fn run_rawnind_bayer_denoise(
    raw: &mut rawler::RawImage,
    strength: f32,
    session: &Mutex<Session>,
    progress: impl Fn(f32),
) -> Result<()> {
    const T: usize = RAWNIND_BAYER_TILE;
    const T2: usize = 2 * T;

    let rawler::rawimage::RawPhotometricInterpretation::Cfa(config) = &raw.photometric else {
        anyhow::bail!("not a colour filter array raw");
    };
    let (oy, ox) = rggb_offset(&config.cfa)
        .ok_or_else(|| anyhow::anyhow!("CFA pattern '{}' is not Bayer", config.cfa.name))?;
    let width = raw.width;
    let (ax, ay, aw, ah) = match raw.active_area {
        Some(r) => (r.p.x, r.p.y, r.d.w, r.d.h),
        None => (0, 0, raw.width, raw.height),
    };
    // First row and column of a whole R G / G B block inside the active area.
    let r0 = ay + (oy + 2 - ay % 2) % 2;
    let c0 = ax + (ox + 2 - ax % 2) % 2;
    let ph = (ay + ah).saturating_sub(r0) / 2;
    let pw = (ax + aw).saturating_sub(c0) / 2;
    if raw.cpp != 1 || ph < 2 || pw < 2 || ax + aw > raw.width || ay + ah > raw.height {
        anyhow::bail!("unsupported sensor layout");
    }
    let rawler::RawImageData::Integer(data) = &mut raw.data else {
        anyhow::bail!("floating-point sensor data");
    };
    if data.len() != raw.width * raw.height {
        anyhow::bail!("sensor data size doesn't match its dimensions");
    }

    // Black level and range for each block site: R, G1, G2, B.
    let blacks = raw.blacklevel.as_bayer_array();
    let whites = raw.whitelevel.as_bayer_array();
    let mut black = [0.0f32; 4];
    let mut range = [0.0f32; 4];
    for k in 0..4 {
        let site = ((oy + k / 2) % 2) * 2 + (ox + k % 2) % 2;
        black[k] = blacks[site];
        range[k] = (whites[site] - blacks[site]).max(1.0);
    }

    // Normalised photosites, one RGGB block per packed pixel: [ph][pw][4].
    let mut packed = vec![0.0f32; ph * pw * 4];
    packed
        .par_chunks_mut(pw * 4)
        .enumerate()
        .for_each(|(py, row)| {
            let top = (r0 + 2 * py) * width + c0;
            let bottom = top + width;
            for px in 0..pw {
                let raw_values = [
                    data[top + 2 * px],
                    data[top + 2 * px + 1],
                    data[bottom + 2 * px],
                    data[bottom + 2 * px + 1],
                ];
                for k in 0..4 {
                    row[px * 4 + k] = (raw_values[k] as f32 - black[k]) / range[k];
                }
            }
        });

    let xs = tile_starts(pw, T, RAWNIND_BAYER_OVERLAP);
    let ys = tile_starts(ph, T, RAWNIND_BAYER_OVERLAP);
    let weights: Vec<f32> = (0..T)
        .map(|u| tile_weight(u, T, RAWNIND_BAYER_OVERLAP, RAWNIND_BAYER_BORDER))
        .collect();

    // Per-photosite sum of weighted denoised values, and per-block sum of the weights.
    let mut acc = vec![0.0f32; ph * pw * 4];
    let mut weight_sum = vec![0.0f32; ph * pw];
    let mut tile_in = vec![0.0f32; 4 * T * T];
    let total = xs.len() * ys.len();
    let mut failed = 0;

    for (iy, &y0) in ys.iter().enumerate() {
        for (ix, &x0) in xs.iter().enumerate() {
            progress((iy * xs.len() + ix) as f32 / total as f32);

            for (k, plane) in tile_in.chunks_mut(T * T).enumerate() {
                plane.par_chunks_mut(T).enumerate().for_each(|(dy, row)| {
                    let sy = reflect_coord(y0 + dy as isize, ph);
                    let src = &packed[sy * pw * 4..(sy + 1) * pw * 4];
                    for (dx, v) in row.iter_mut().enumerate() {
                        let sx = reflect_coord(x0 + dx as isize, pw);
                        *v = src[sx * 4 + k].clamp(0.0, 1.0);
                    }
                });
            }
            let in_mean = tile_in.iter().map(|&v| v as f64).sum::<f64>() / tile_in.len() as f64;

            let input = Array4::from_shape_vec((1, 4, T, T), tile_in.clone())?;
            let mut sess = session.lock().unwrap();
            let outputs = sess.run(ort::inputs![Tensor::from_array(input)?])?;
            let out = outputs[0].try_extract_array::<f32>()?;
            let out = out
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("RawNIND output is not contiguous"))?;
            if out.len() != 3 * T2 * T2 {
                anyhow::bail!("unexpected RawNIND Bayer output size {}", out.len());
            }
            // Each photosite's own colour from the model's camera RGB: R, G1, G2, B.
            let mosaic = |dy: usize, dx: usize| {
                let at = |c: usize, y: usize, x: usize| out[c * T2 * T2 + y * T2 + x];
                [
                    at(0, 2 * dy, 2 * dx),
                    at(1, 2 * dy, 2 * dx + 1),
                    at(1, 2 * dy + 1, 2 * dx),
                    at(2, 2 * dy + 1, 2 * dx + 1),
                ]
            };
            let out_mean = (0..T * T)
                .into_par_iter()
                .map(|t| mosaic(t / T, t % T).iter().map(|&v| v as f64).sum::<f64>())
                .sum::<f64>()
                / (4 * T * T) as f64;
            // Matched like for like, as in `run_rawnind_denoise`. A black tile stays black.
            let gain = if in_mean <= 1e-6 {
                0.0
            } else if out_mean.is_finite() && out_mean.abs() > 1e-12 {
                (in_mean / out_mean) as f32
            } else {
                failed += 1;
                continue;
            };

            let y_lo = y0.max(0) as usize;
            let y_hi = ((y0 + T as isize) as usize).min(ph);
            let x_lo = x0.max(0) as usize;
            let x_hi = ((x0 + T as isize) as usize).min(pw);
            acc.par_chunks_mut(pw * 4)
                .zip(weight_sum.par_chunks_mut(pw))
                .enumerate()
                .skip(y_lo)
                .take(y_hi - y_lo)
                .for_each(|(y, (acc_row, w_row))| {
                    let dy = (y as isize - y0) as usize;
                    for x in x_lo..x_hi {
                        let dx = (x as isize - x0) as usize;
                        let wgt = weights[dy] * weights[dx];
                        if wgt == 0.0 {
                            continue;
                        }
                        w_row[x] += wgt;
                        let m = mosaic(dy, dx);
                        for k in 0..4 {
                            acc_row[x * 4 + k] += wgt * gain * m[k];
                        }
                    }
                });
        }
    }
    if failed == total {
        anyhow::bail!("RawNIND Bayer produced no usable output");
    }
    if failed > 0 {
        log::warn!("RawNIND Bayer: skipped {failed} of {total} tiles with invalid output");
    }

    let strength = strength.clamp(0.0, 1.0);
    data[r0 * width..(r0 + 2 * ph) * width]
        .par_chunks_mut(2 * width)
        .enumerate()
        .for_each(|(py, rows)| {
            let (top, bottom) = rows.split_at_mut(width);
            for px in 0..pw {
                let p = py * pw + px;
                let wsum = weight_sum[p];
                if wsum <= 0.0 {
                    continue;
                }
                let c = c0 + 2 * px;
                let mut values = [top[c], top[c + 1], bottom[c], bottom[c + 1]];
                for (k, value) in values.iter_mut().enumerate() {
                    let original = packed[p * 4 + k];
                    // Clipped photosites hold no noise worth removing, only lost detail.
                    if original >= 0.99 {
                        continue;
                    }
                    let denoised = acc[p * 4 + k] / wsum;
                    let v = original + strength * (denoised - original);
                    *value = (black[k] + v * range[k])
                        .round()
                        .clamp(0.0, u16::MAX as f32) as u16;
                }
                top[c] = values[0];
                top[c + 1] = values[1];
                bottom[c] = values[2];
                bottom[c + 1] = values[3];
            }
        });
    progress(1.0);
    Ok(())
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

const LAMA_PIXEL_BUDGET: f32 = 768.0 * 768.0;
const LAMA_PATCH_RADIUS: i32 = 3;
const LAMA_PATCH_MATCH_ITERS: u32 = 6;

fn dilate_mask(mask: &[bool], w: usize, h: usize, r: usize) -> Vec<bool> {
    let mut horiz = vec![false; w * h];
    horiz.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let src = &mask[y * w..(y + 1) * w];
        let mut prefix = vec![0u32; w + 1];
        for x in 0..w {
            prefix[x + 1] = prefix[x] + src[x] as u32;
        }
        for (x, out) in row.iter_mut().enumerate() {
            *out = prefix[(x + r + 1).min(w)] > prefix[x.saturating_sub(r)];
        }
    });
    let mut dilated = vec![false; w * h];
    dilated.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(h - 1);
        for (x, out) in row.iter_mut().enumerate() {
            *out = (y0..=y1).any(|yy| horiz[yy * w + x]);
        }
    });
    dilated
}

fn catmull_rom_sample(img: &[[f32; 3]], w: usize, h: usize, x: f32, y: f32) -> [f32; 3] {
    let weights = |t: f32| {
        let t2 = t * t;
        let t3 = t2 * t;
        [
            -0.5 * t3 + t2 - 0.5 * t,
            1.5 * t3 - 2.5 * t2 + 1.0,
            -1.5 * t3 + 2.0 * t2 + 0.5 * t,
            0.5 * t3 - 0.5 * t2,
        ]
    };
    let (fx, fy) = (x.floor(), y.floor());
    let wx = weights(x - fx);
    let wy = weights(y - fy);
    let mut acc = [0.0f32; 3];
    for (j, wyj) in wy.iter().enumerate() {
        let sy = (fy as i32 + j as i32 - 1).clamp(0, h as i32 - 1) as usize;
        for (i, wxi) in wx.iter().enumerate() {
            let sx = (fx as i32 + i as i32 - 1).clamp(0, w as i32 - 1) as usize;
            let p = img[sy * w + sx];
            let wgt = wyj * wxi;
            acc[0] += p[0] * wgt;
            acc[1] += p[1] * wgt;
            acc[2] += p[2] * wgt;
        }
    }
    acc
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn patch_distance(
    target: &[[f32; 3]],
    source: &[[f32; 3]],
    w: usize,
    h: usize,
    t: (i32, i32),
    s: (i32, i32),
    limit: f32,
) -> f32 {
    let mut d = 0.0f32;
    for dy in -LAMA_PATCH_RADIUS..=LAMA_PATCH_RADIUS {
        let ty = (t.1 + dy).clamp(0, h as i32 - 1) as usize;
        let srow = (s.1 + dy) as usize * w;
        for dx in -LAMA_PATCH_RADIUS..=LAMA_PATCH_RADIUS {
            let tx = (t.0 + dx).clamp(0, w as i32 - 1) as usize;
            let a = target[ty * w + tx];
            let b = source[srow + (s.0 + dx) as usize];
            let (e0, e1, e2) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
            d += e0 * e0 + e1 * e1 + e2 * e2;
        }
        if d >= limit {
            return d;
        }
    }
    d
}

fn lama_patch_match(
    target: &[[f32; 3]],
    source: &[[f32; 3]],
    w: usize,
    h: usize,
    is_target: &[bool],
    is_source: &[bool],
) -> Option<Vec<Option<(i32, i32)>>> {
    let sources: Vec<(i32, i32)> = (0..w * h)
        .filter(|&i| is_source[i])
        .map(|i| ((i % w) as i32, (i / w) as i32))
        .collect();
    let targets: Vec<usize> = (0..w * h).filter(|&i| is_target[i]).collect();
    if sources.is_empty() || targets.is_empty() {
        return None;
    }

    let valid = |p: (i32, i32)| {
        p.0 >= 0
            && p.1 >= 0
            && (p.0 as usize) < w
            && (p.1 as usize) < h
            && is_source[p.1 as usize * w + p.0 as usize]
    };
    let random_source = |seed: u64| sources[(seed % sources.len() as u64) as usize];

    let mut nnf = vec![(0i32, 0i32); w * h];
    let mut cost = vec![f32::MAX; w * h];
    let init: Vec<((i32, i32), f32)> = targets
        .par_iter()
        .map(|&i| {
            let t = ((i % w) as i32, (i / w) as i32);
            let s = random_source(splitmix64(i as u64));
            (s, patch_distance(target, source, w, h, t, s, f32::MAX))
        })
        .collect();
    for (&i, (s, c)) in targets.iter().zip(init) {
        nnf[i] = s;
        cost[i] = c;
    }

    for iter in 0..LAMA_PATCH_MATCH_ITERS {
        let jump = 1i32 << (LAMA_PATCH_MATCH_ITERS - 1 - iter).min(3);
        let updates: Vec<((i32, i32), f32)> = targets
            .par_iter()
            .map(|&i| {
                let t = ((i % w) as i32, (i / w) as i32);
                let mut best = nnf[i];
                let mut best_cost = cost[i];
                let consider = |c: (i32, i32), best: &mut (i32, i32), best_cost: &mut f32| {
                    if valid(c) && c != *best {
                        let d = patch_distance(target, source, w, h, t, c, *best_cost);
                        if d < *best_cost {
                            *best = c;
                            *best_cost = d;
                        }
                    }
                };

                for (dx, dy) in [(-jump, 0), (jump, 0), (0, -jump), (0, jump)] {
                    let (nx, ny) = (t.0 + dx, t.1 + dy);
                    if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                        let n = ny as usize * w + nx as usize;
                        if is_target[n] {
                            consider((nnf[n].0 - dx, nnf[n].1 - dy), &mut best, &mut best_cost);
                        }
                    }
                }

                let mut seed = splitmix64(i as u64 ^ ((iter as u64 + 1) << 40));
                let mut radius = w.max(h) as i32;
                while radius >= 1 {
                    seed = splitmix64(seed);
                    let span = (2 * radius + 1) as u64;
                    let c = (
                        best.0 + (seed % span) as i32 - radius,
                        best.1 + ((seed >> 32) % span) as i32 - radius,
                    );
                    consider(c, &mut best, &mut best_cost);
                    radius /= 2;
                }
                consider(random_source(splitmix64(seed)), &mut best, &mut best_cost);

                (best, best_cost)
            })
            .collect();
        for (&i, (s, c)) in targets.iter().zip(updates) {
            nnf[i] = s;
            cost[i] = c;
        }
    }

    Some((0..w * h).map(|i| is_target[i].then_some(nnf[i])).collect())
}

pub fn run_lama_inpainting(
    image: &DynamicImage,
    mask: &GrayImage,
    lama_session: &Mutex<Session>,
) -> Result<(RgbaImage, u32, u32)> {
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
        return Ok((image.to_rgba8(), 0, 0));
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

    let cropped_img = image.crop_imm(x0, y0, crop_w, crop_h).to_rgba8();
    let cropped_mask = imageops::crop_imm(mask, x0, y0, crop_w, crop_h).to_image();

    let scale = (LAMA_PIXEL_BUDGET / (crop_w as f32 * crop_h as f32))
        .sqrt()
        .min(1.0);
    let fw = ((crop_w as f32 * scale).round() as u32).max(1);
    let fh = ((crop_h as f32 * scale).round() as u32).max(1);
    let needs_downscale = fw != crop_w || fh != crop_h;

    let (inf_img, inf_mask) = if needs_downscale {
        (
            imageops::resize(&cropped_img, fw, fh, FilterType::Lanczos3),
            imageops::resize(&cropped_mask, fw, fh, FilterType::Triangle),
        )
    } else {
        (cropped_img.clone(), cropped_mask.clone())
    };

    let align = 32u32;
    let tensor_w = fw.div_ceil(align) * align;
    let tensor_h = fh.div_ceil(align) * align;

    let mut img_tensor = Array::<f32, _>::zeros((1, 3, tensor_h as usize, tensor_w as usize));
    let mut msk_tensor = Array::<f32, _>::zeros((1, 1, tensor_h as usize, tensor_w as usize));

    for y in 0..tensor_h {
        for x in 0..tensor_w {
            let p = inf_img.get_pixel(x.min(fw - 1), y.min(fh - 1));
            let m = inf_mask.get_pixel(x.min(fw - 1), y.min(fh - 1))[0];
            let (y, x) = (y as usize, x as usize);

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

    let (iw, ih) = (fw as usize, fh as usize);
    let target: Vec<[f32; 3]> = (0..iw * ih)
        .map(|i| {
            let (x, y) = (i % iw, i / iw);
            [0, 1, 2].map(|c| (output_tensor[[0, c, y, x]] / 255.0).clamp(0.0, 1.0))
        })
        .collect();
    let source: Vec<[f32; 3]> = inf_img
        .pixels()
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();

    let nnf = if needs_downscale {
        let hole: Vec<bool> = inf_mask.pixels().map(|p| p[0] > 0).collect();
        let is_target = dilate_mask(&hole, iw, ih, 1);
        let blocked = dilate_mask(&hole, iw, ih, LAMA_PATCH_RADIUS as usize + 2);
        let r = LAMA_PATCH_RADIUS;
        let is_source: Vec<bool> = (0..iw * ih)
            .map(|i| {
                let (x, y) = ((i % iw) as i32, (i / iw) as i32);
                !blocked[i] && x >= r && y >= r && x < iw as i32 - r && y < ih as i32 - r
            })
            .collect();
        lama_patch_match(&target, &source, iw, ih, &is_target, &is_source)
    } else {
        None
    };

    let sx = fw as f32 / crop_w as f32;
    let sy = fh as f32 / crop_h as f32;
    let mut final_crop = cropped_img.clone();

    final_crop
        .par_chunks_mut(crop_w as usize * 4)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..crop_w as usize {
                let m = cropped_mask.get_pixel(x as u32, y as u32)[0];
                if m == 0 {
                    continue;
                }
                let u = (x as f32 + 0.5) * sx - 0.5;
                let v = (y as f32 + 0.5) * sy - 0.5;
                let mut value = catmull_rom_sample(&target, iw, ih, u, v);

                if let Some(nnf) = &nnf {
                    let (u0, v0) = (u.floor(), v.floor());
                    let (fu, fv) = (u - u0, v - v0);
                    let mut groups = [((0i32, 0i32), 0.0f32, [0.0f32; 3]); 4];
                    let mut group_count = 0;
                    for (cu, cv, wgt) in [
                        (u0, v0, (1.0 - fu) * (1.0 - fv)),
                        (u0 + 1.0, v0, fu * (1.0 - fv)),
                        (u0, v0 + 1.0, (1.0 - fu) * fv),
                        (u0 + 1.0, v0 + 1.0, fu * fv),
                    ] {
                        let cu = (cu as i32).clamp(0, iw as i32 - 1);
                        let cv = (cv as i32).clamp(0, ih as i32 - 1);
                        let Some((su, sv)) = nnf[cv as usize * iw + cu as usize] else {
                            continue;
                        };
                        let qx = x as i32 + ((su - cu) as f32 / sx).round() as i32;
                        let qy = y as i32 + ((sv - cv) as f32 / sy).round() as i32;
                        if qx < 0 || qy < 0 || qx >= crop_w as i32 || qy >= crop_h as i32 {
                            continue;
                        }
                        if cropped_mask.get_pixel(qx as u32, qy as u32)[0] > 0 {
                            continue;
                        }
                        let o = cropped_img.get_pixel(qx as u32, qy as u32);
                        let low = catmull_rom_sample(
                            &source,
                            iw,
                            ih,
                            (qx as f32 + 0.5) * sx - 0.5,
                            (qy as f32 + 0.5) * sy - 0.5,
                        );
                        let detail = [0, 1, 2].map(|c| o[c] as f32 / 255.0 - low[c]);
                        let shift = (qx - x as i32, qy - y as i32);
                        match groups[..group_count].iter_mut().find(|g| g.0 == shift) {
                            Some(g) => g.1 += wgt,
                            None => {
                                groups[group_count] = (shift, wgt, detail);
                                group_count += 1;
                            }
                        }
                    }
                    let norm = groups[..group_count]
                        .iter()
                        .map(|g| g.1 * g.1)
                        .sum::<f32>()
                        .sqrt();
                    if norm > 0.0 {
                        for (_, wgt, detail) in &groups[..group_count] {
                            for c in 0..3 {
                                value[c] += wgt * detail[c] / norm;
                            }
                        }
                    }
                }

                let alpha = m as f32 / 255.0;
                let px = &mut row[x * 4..x * 4 + 3];
                for c in 0..3 {
                    px[c] = (value[c].clamp(0.0, 1.0) * 255.0 * alpha
                        + px[c] as f32 * (1.0 - alpha))
                        .round() as u8;
                }
            }
        });

    Ok((final_crop, x0, y0))
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

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub fn run_esam3_image_encoder(
    image: &DynamicImage,
    encoder_session: &Mutex<Session>,
) -> Result<Esam3Features> {
    let size = ESAM3_INPUT_SIZE as usize;
    let padding = ESAM3_INPUT_PADDING as i64;
    let inner = ESAM3_INPUT_SIZE - 2 * ESAM3_INPUT_PADDING;
    let resized_rgb = image
        .resize_exact(inner, inner, FilterType::Triangle)
        .into_rgb8();

    let reflect = |i: usize| -> u32 {
        let (i, n) = (i as i64 - padding, inner as i64);
        let i = if i < 0 { -i } else { i };
        (if i >= n { 2 * n - 2 - i } else { i }) as u32
    };
    let mut input_tensor = Array4::<f32>::zeros((1, 3, size, size));
    for y in 0..size {
        let src_y = reflect(y);
        for x in 0..size {
            let pixel = resized_rgb.get_pixel(reflect(x), src_y);
            for c in 0..3 {
                input_tensor[[0, c, y, x]] = pixel[c] as f32 / 127.5 - 1.0;
            }
        }
    }

    let mut session = encoder_session.lock().unwrap();
    let outputs =
        gpu_inference(session.run(ort::inputs!["image" => Tensor::from_array(input_tensor)?]))?;
    let extract = |name: &str| -> Result<Array<f32, IxDyn>> {
        Ok(outputs[name].try_extract_array::<f32>()?.to_owned())
    };

    Ok(Esam3Features {
        path_hash: String::new(),
        fpn: [extract("fpn_0")?, extract("fpn_1")?, extract("fpn_2")?],
        pos: extract("pos_2")?,
    })
}

fn run_esam3_queries<T>(
    decoder_session: &Mutex<Session>,
    features: &Esam3Features,
    prompt: &Esam3Prompt,
    read: impl FnOnce(Vec<f32>, ArrayViewD<f32>, ArrayViewD<f32>) -> T,
) -> Result<T> {
    let exemplar_boxes = Array::<f32, _>::zeros((ESAM3_EXEMPLAR_SLOTS, 1, 4));
    let exemplar_labels = Array::<i64, _>::zeros((ESAM3_EXEMPLAR_SLOTS, 1));
    let exemplar_mask = Array::<f32, _>::ones((1, ESAM3_EXEMPLAR_SLOTS));

    let mut session = decoder_session.lock().unwrap();
    let outputs = gpu_inference(session.run(ort::inputs![
        "fpn_0" => TensorRef::from_array_view(&features.fpn[0])?,
        "fpn_1" => TensorRef::from_array_view(&features.fpn[1])?,
        "fpn_2" => TensorRef::from_array_view(&features.fpn[2])?,
        "pos_2" => TensorRef::from_array_view(&features.pos)?,
        "language_features" => TensorRef::from_array_view(&prompt.features)?,
        "language_padding" => TensorRef::from_array_view(&prompt.padding)?,
        "exemplar_boxes" => Tensor::from_array(exemplar_boxes)?,
        "exemplar_labels" => Tensor::from_array(exemplar_labels)?,
        "exemplar_mask" => Tensor::from_array(exemplar_mask)?,
    ]))?;

    let presence = outputs["presence_logit"]
        .try_extract_array::<f32>()?
        .iter()
        .next()
        .map_or(0.0, |&v| sigmoid(v));
    let scores = outputs["pred_logits"]
        .try_extract_array::<f32>()?
        .iter()
        .map(|&logit| sigmoid(logit) * presence)
        .collect();
    Ok(read(
        scores,
        outputs["pred_boxes"].try_extract_array::<f32>()?,
        outputs["pred_masks"].try_extract_array::<f32>()?,
    ))
}

pub fn run_esam3_boxes(
    decoder_session: &Mutex<Session>,
    features: &Esam3Features,
    prompt: &Esam3Prompt,
    score_threshold: f32,
) -> Result<Vec<(f32, [f32; 4])>> {
    let mut detections =
        run_esam3_queries(decoder_session, features, prompt, |scores, boxes, _| {
            scores
                .iter()
                .enumerate()
                .filter(|&(_, &score)| score > score_threshold)
                .map(|(query, &score)| {
                    let size = ESAM3_INPUT_SIZE as f32;
                    let padding = ESAM3_INPUT_PADDING as f32;
                    let inner = size - 2.0 * padding;
                    let [cx, cy, w, h]: [f32; 4] = std::array::from_fn(|i| {
                        let v = boxes[[0, query, i]] * size;
                        if i < 2 {
                            (v - padding) / inner
                        } else {
                            v / inner
                        }
                    });
                    (
                        score,
                        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                    )
                })
                .collect::<Vec<_>>()
        })?;
    detections.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(detections)
}

pub fn run_esam3_decoder(
    decoder_session: &Mutex<Session>,
    features: &Esam3Features,
    prompt: &Esam3Prompt,
    image: &DynamicImage,
    score_threshold: f32,
    refine_eps: f32,
) -> Result<Option<GrayImage>> {
    let (coarse_mask, mask_w, mask_h) =
        run_esam3_queries(decoder_session, features, prompt, |scores, _, masks| {
            let (mask_h, mask_w) = (masks.shape()[2], masks.shape()[3]);
            let mut coarse_mask: Option<Vec<f32>> = None;
            for (query, &score) in scores.iter().enumerate() {
                if score <= score_threshold {
                    continue;
                }
                let combined = coarse_mask.get_or_insert_with(|| vec![0.0; mask_h * mask_w]);
                let query_mask = masks.slice(s![0, query, .., ..]);
                for (out, &value) in combined.iter_mut().zip(query_mask.iter()) {
                    *out = out.max(sigmoid(value));
                }
            }
            (coarse_mask, mask_w, mask_h)
        })?;

    let Some(coarse_mask) = coarse_mask else {
        return Ok(None);
    };

    let coarse_mask = GrayImage::from_raw(
        mask_w as u32,
        mask_h as u32,
        coarse_mask
            .iter()
            .map(|&v| {
                let v = ((v - ESAM3_MASK_LOW) / (ESAM3_MASK_HIGH - ESAM3_MASK_LOW)).clamp(0.0, 1.0);
                (v * 255.0).round() as u8
            })
            .collect(),
    )
    .ok_or_else(|| anyhow::anyhow!("Failed to create mask from EfficientSAM3 output"))?;

    let input_mask = imageops::resize(
        &coarse_mask,
        ESAM3_INPUT_SIZE,
        ESAM3_INPUT_SIZE,
        FilterType::Triangle,
    );
    let inner = ESAM3_INPUT_SIZE - 2 * ESAM3_INPUT_PADDING;
    let image_mask = imageops::crop_imm(
        &input_mask,
        ESAM3_INPUT_PADDING,
        ESAM3_INPUT_PADDING,
        inner,
        inner,
    )
    .to_image();

    let (orig_width, orig_height) = image.dimensions();
    let scale = (ESAM3_REFINE_SIZE as f32 / orig_width.max(orig_height) as f32).min(1.0);
    let refine_w = ((orig_width as f32 * scale).round() as u32).max(1);
    let refine_h = ((orig_height as f32 * scale).round() as u32).max(1);
    let refine_mask = imageops::resize(&image_mask, refine_w, refine_h, FilterType::Triangle);

    Ok(Some(fast_color_guided_filter(
        &image.to_rgb8(),
        &refine_mask,
        ESAM3_REFINE_RADIUS,
        refine_eps,
    )))
}

pub fn run_esam3_zoomed_decoder(
    encoder_session: &Mutex<Session>,
    decoder_session: &Mutex<Session>,
    features: &Esam3Features,
    prompt: &Esam3Prompt,
    image: &DynamicImage,
    score_threshold: f32,
    refine_eps: f32,
) -> Result<Option<GrayImage>> {
    let Some(mut mask) = run_esam3_decoder(
        decoder_session,
        features,
        prompt,
        image,
        score_threshold,
        refine_eps,
    )?
    else {
        return Ok(None);
    };

    let (width, height) = image.dimensions();
    let grid_size = ZOOM_MIN_SIDE as u32;
    let grid = imageops::resize(&mask, grid_size, grid_size, FilterType::Triangle);
    if let Some((x, y, crop_width, crop_height)) = zoom_region(&grid, width, height) {
        let crop = image.crop_imm(x, y, crop_width, crop_height);
        let crop_features = run_esam3_image_encoder(&crop, encoder_session)?;
        if let Some(detail) = run_esam3_decoder(
            decoder_session,
            &crop_features,
            prompt,
            &crop,
            score_threshold,
            refine_eps,
        )? && mask_iou(&mask, &detail, x, y) >= ESAM3_ZOOM_MIN_IOU
        {
            blend_feathered(&mut mask, &detail, x, y);
        }
    }

    Ok(Some(mask))
}

fn mask_iou(mask: &GrayImage, detail: &GrayImage, x0: u32, y0: u32) -> f32 {
    let (mut intersection, mut union) = (0u64, 0u64);
    for (u, v, refined) in detail.enumerate_pixels() {
        let base = mask.get_pixel(x0 + u, y0 + v)[0] >= 128;
        let refined = refined[0] >= 128;
        intersection += (base && refined) as u64;
        union += (base || refined) as u64;
    }
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

fn blend_feathered(mask: &mut GrayImage, detail: &GrayImage, x0: u32, y0: u32) {
    let (width, height) = mask.dimensions();
    let (crop_width, crop_height) = detail.dimensions();
    let feather = (crop_width.min(crop_height) as f32 * ZOOM_FEATHER).max(1.0);
    let ramp = |start: u32, length: u32, extent: u32| -> Vec<f32> {
        (0..length)
            .map(|i| {
                let head = if start == 0 { f32::MAX } else { i as f32 + 0.5 };
                let tail = if start + length == extent {
                    f32::MAX
                } else {
                    (length - i) as f32 - 0.5
                };
                (head.min(tail) / feather).min(1.0)
            })
            .collect()
    };
    let weights_x = ramp(x0, crop_width, width);
    let weights_y = ramp(y0, crop_height, height);

    for (v, weight_y) in weights_y.iter().enumerate() {
        for (u, weight_x) in weights_x.iter().enumerate() {
            let weight = weight_x.min(*weight_y);
            let base = mask.get_pixel(x0 + u as u32, y0 + v as u32)[0] as f32;
            let refined = detail.get_pixel(u as u32, v as u32)[0] as f32;
            mask.put_pixel(
                x0 + u as u32,
                y0 + v as u32,
                Luma([(base + (refined - base) * weight).round() as u8]),
            );
        }
    }
}

fn zoom_region(coarse: &GrayImage, width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (u32::MAX, u32::MAX, 0, 0);
    for (x, y, pixel) in coarse.enumerate_pixels() {
        if pixel[0] >= 128 {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x + 1);
            max_y = max_y.max(y + 1);
        }
    }
    if min_x == u32::MAX {
        return None;
    }

    let scale_x = width as f32 / coarse.width() as f32;
    let scale_y = height as f32 / coarse.height() as f32;
    let long_side = width.max(height) as f32;
    let subject_side = ((max_x - min_x) as f32 * scale_x).max((max_y - min_y) as f32 * scale_y);
    let side = (subject_side * (1.0 + 2.0 * ZOOM_MARGIN))
        .max(ZOOM_MIN_SIDE)
        .min(long_side);
    if side > ZOOM_MAX_FRACTION * long_side {
        return None;
    }

    let place = |min: u32, max: u32, scale: f32, extent: u32| {
        let center = (min + max) as f32 * 0.5 * scale;
        let length = side.min(extent as f32);
        let start = (center - length * 0.5).clamp(0.0, extent as f32 - length);
        (start as u32, (length as u32).min(extent - start as u32))
    };
    let (x, crop_width) = place(min_x, max_x, scale_x, width);
    let (y, crop_height) = place(min_y, max_y, scale_y, height);
    Some((x, y, crop_width, crop_height))
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

/// Depth Anything 3's relative depth as disparity (near is bright, sky is black), the way
/// Depth Anything V2 reported it, so depth masks, lens blur and fog read it as before. The map has
/// the photo's aspect ratio with its long side at the model's 756px.
pub fn run_depth_anything_model(
    image: &DynamicImage,
    depth_session: &Mutex<Session>,
) -> Result<GrayImage> {
    let (width, height) = (DEPTH_INPUT_WIDTH as usize, DEPTH_INPUT_HEIGHT as usize);
    let t_input = Tensor::from_array(
        imagenet_tensor(&imageops::thumbnail(
            &image.to_rgb8(),
            DEPTH_INPUT_WIDTH,
            DEPTH_INPUT_HEIGHT,
        ))
        .into_dyn(),
    )?;

    let (depth, sky): (Vec<f32>, Vec<f32>) = {
        let mut session = depth_session.lock().unwrap();
        let outputs = session.run(ort::inputs![t_input])?;
        let depth = outputs["depth"].try_extract_array::<f32>()?;
        let sky = outputs["sky"].try_extract_array::<f32>()?;
        (
            depth.iter().copied().collect(),
            sky.iter().copied().collect(),
        )
    };
    if depth.len() != width * height || sky.len() != depth.len() {
        return Err(anyhow::anyhow!("Unexpected Depth Anything output size"));
    }

    let (orig_width, orig_height) = image.dimensions();
    let fit = DEPTH_INPUT_WIDTH as f32 / orig_width.max(orig_height).max(1) as f32;
    Ok(disparity_map(
        depth,
        &sky,
        (width, height),
        (
            ((orig_width as f32 * fit).round() as u32).max(1),
            ((orig_height as f32 * fit).round() as u32).max(1),
        ),
    ))
}

/// Turns Depth Anything 3's depth and sky outputs into an 8-bit disparity map at the output size.
fn disparity_map(
    mut depth: Vec<f32>,
    sky: &[f32],
    (width, height): (usize, usize),
    (out_width, out_height): (u32, u32),
) -> GrayImage {
    // Like Depth Anything 3 itself: the sky is as far as the farthest of the rest of the photo.
    let mut ground: Vec<f32> = depth
        .iter()
        .zip(sky)
        .filter(|&(_, &s)| s < DEPTH_SKY_THRESHOLD)
        .map(|(&d, _)| d)
        .collect();
    if ground.len() > 10 && depth.len() - ground.len() > 10 {
        let far = crate::culling::percentile(&mut ground, 0.99);
        for (d, &s) in depth.iter_mut().zip(sky) {
            *d = if s < DEPTH_SKY_THRESHOLD {
                d.min(far)
            } else {
                far
            };
        }
    }

    let disparity: Vec<f32> = depth.iter().map(|&d| 1.0 / d.max(1e-6)).collect();
    // Stretch between the 0.5th and 99.5th percentiles, so a few stray pixels can't flatten it.
    let mut sorted = disparity.clone();
    let low = crate::culling::percentile(&mut sorted, 0.005);
    let high = crate::culling::percentile(&mut sorted, 0.995);
    let scale = if high - low > 1e-6 {
        255.0 / (high - low)
    } else {
        0.0
    };

    upscale_grid(&disparity, width, height, out_width, out_height, |v| {
        ((v - low) * scale).clamp(0.0, 255.0).round() as u8
    })
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

/// MoGe-3's geometry for a photo, on a grid with the photo's aspect ratio and its long side at
/// most NORMAL_MAX_SIZE. Points and normals are in camera space: x right, y down, z forward.
pub struct NormalGeometry {
    pub width: u32,
    pub height: u32,
    /// Three values per pixel. Only meaningful where `mask` is at least 0.5.
    pub points: Vec<f32>,
    /// Three values per pixel, not quite unit length after resampling.
    pub normals: Vec<f32>,
    /// One value per pixel, 0 to 1: below 0.5 there's no surface, as in the sky.
    pub mask: Vec<f32>,
}

impl NormalGeometry {
    /// Packs the geometry and a texture for the 3D view: width, height, texture width and
    /// texture height as little-endian u32s, then the points and normals as f32s, the mask as
    /// one byte per pixel and the texture as RGBA. The f32s start on 4-byte boundaries.
    pub fn to_view_bytes(&self, texture: &RgbaImage) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(
            16 + (self.points.len() + self.normals.len()) * 4
                + self.mask.len()
                + texture.as_raw().len(),
        );
        for value in [self.width, self.height, texture.width(), texture.height()] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in self.points.iter().chain(&self.normals) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend(
            self.mask
                .iter()
                .map(|&m| (m.clamp(0.0, 1.0) * 255.0).round() as u8),
        );
        bytes.extend_from_slice(texture.as_raw());
        bytes
    }
}

pub fn run_normal_model(image: &DynamicImage, normal_model: &NormalModel) -> Result<RgbaImage> {
    let geometry = run_normal_geometry(image, normal_model)?;
    let depth = build_normal_depth(&geometry.points, &geometry.mask);

    let pack = |v: f32| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;

    let mut normal_data = Vec::with_capacity(geometry.mask.len() * 4);
    for (n, d) in geometry.normals.as_chunks::<3>().0.iter().zip(depth) {
        let (nx, ny, nz) = (n[0], -n[1], -n[2]);
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len.is_finite() && len > 1e-6 {
            normal_data.extend_from_slice(&[pack(nx / len), pack(ny / len), pack(nz / len)]);
        } else {
            normal_data.extend_from_slice(&[128, 128, 255]);
        }
        normal_data.push(d);
    }

    RgbaImage::from_raw(geometry.width, geometry.height, normal_data)
        .ok_or_else(|| anyhow::anyhow!("Failed to create normal map from model output"))
}

pub fn run_normal_geometry(
    image: &DynamicImage,
    normal_model: &NormalModel,
) -> Result<NormalGeometry> {
    let resized_image = if image.width().max(image.height()) > NORMAL_MAX_SIZE {
        image.resize(NORMAL_MAX_SIZE, NORMAL_MAX_SIZE, FilterType::Triangle)
    } else {
        image.clone()
    };
    let (resized_w, resized_h) = resized_image.dimensions();
    let rw = resized_w as usize;
    let rh = resized_h as usize;

    let graph = normal_model.fixed_graph_for(resized_w, resized_h);
    let (normals, mask, points) = normal_model.run(graph, |graph, session| {
        // A fixed graph gets the photo stretched to its size.
        let input = match graph {
            Some(g) => resized_image
                .resize_exact(g.width, g.height, FilterType::Triangle)
                .into_rgb8(),
            None => resized_image.to_rgb8(),
        };
        let (iw, ih) = (input.width() as usize, input.height() as usize);
        let input_tensor = Array4::from_shape_fn((1, 3, ih, iw), |(_, c, y, x)| {
            input.as_raw()[(y * iw + x) * 3 + c] as f32 / 255.0
        });
        let t_image = Tensor::from_array(input_tensor.into_dyn())?;

        let outputs = match graph {
            Some(_) => session.run(ort::inputs!["image" => t_image])?,
            None => session.run(ort::inputs![
                "image" => t_image,
                "num_tokens" => Tensor::from_array(ndarray::arr0(NORMAL_NUM_TOKENS))?,
            ])?,
        };

        let normal_tensor = outputs["normal"].try_extract_array::<f32>()?;
        let mask_tensor = outputs["mask"].try_extract_array::<f32>()?;
        let points_tensor = outputs["points"].try_extract_array::<f32>()?;

        if normal_tensor.shape() != [1, ih, iw, 3]
            || mask_tensor.shape() != [1, ih, iw]
            || points_tensor.shape() != [1, ih, iw, 3]
        {
            return Err(anyhow::anyhow!(
                "Unexpected normal model output shape: {:?} / {:?} / {:?}",
                normal_tensor.shape(),
                mask_tensor.shape(),
                points_tensor.shape()
            ));
        }

        // Back onto the photo's own grid.
        let to_photo = |values: Vec<f32>, channels: usize| {
            if (iw, ih) == (rw, rh) {
                return values;
            }
            let planes: Vec<Vec<f32>> = (0..channels)
                .map(|c| {
                    let plane: Vec<f32> =
                        values.iter().skip(c).step_by(channels).copied().collect();
                    resample_grid(&plane, iw, ih, rw, rh)
                })
                .collect();
            (0..rw * rh * channels)
                .map(|i| planes[i % channels][i / channels])
                .collect()
        };
        Ok((
            to_photo(normal_tensor.iter().copied().collect(), 3),
            to_photo(mask_tensor.iter().copied().collect(), 1),
            to_photo(points_tensor.iter().copied().collect(), 3),
        ))
    })?;

    Ok(NormalGeometry {
        width: resized_w,
        height: resized_h,
        points,
        normals,
        mask,
    })
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub box_space_width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub box_space_height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_source: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_source: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiTextMaskParameters {
    #[serde(default)]
    pub prompt: String,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_source: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct AiFaceMaskParameters {
    #[serde(default)]
    pub parts: Vec<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_source: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_source: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_source: Option<String>,
}

#[cfg(test)]
mod depth_map_tests {
    use super::*;

    #[test]
    fn near_is_bright_sky_is_black_and_the_map_keeps_the_photo_shape() {
        let (width, height) = (60, 40);
        // Depth grows towards the top of the frame; the top quarter is sky.
        let depth: Vec<f32> = (0..width * height)
            .map(|i| 1.0 + 4.0 * (1.0 - (i / width) as f32 / height as f32))
            .collect();
        let sky: Vec<f32> = (0..width * height)
            .map(|i| if i / width < height / 4 { 0.9 } else { 0.0 })
            .collect();

        let map = disparity_map(depth, &sky, (width, height), (40, 60));
        assert_eq!(map.dimensions(), (40, 60));
        assert_eq!(map.get_pixel(20, 0)[0], 0, "sky is the farthest");
        assert_eq!(
            map.get_pixel(20, 59)[0],
            255,
            "the bottom row is the nearest"
        );
        assert!(map.get_pixel(20, 30)[0] > map.get_pixel(20, 20)[0]);
    }
}

#[cfg(test)]
mod model_cache_tests {
    use super::*;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rapidraw-{}-{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn startup_removes_superseded_models_and_old_coreml_caches_only() {
        let models = scratch_dir("superseded-test");
        let (old_filenames, old_sha256) = SUPERSEDED_MODELS[0];
        let old_cache = models.join(COREML_CACHE).join(&old_sha256[..16]);
        let current_cache = models.join(COREML_CACHE).join(&DEPTH_SHA256[..16]);
        let unfixed_cache = models.join(OLD_COREML_CACHES[0]).join(&DEPTH_SHA256[..16]);
        for dir in [&old_cache, &current_cache, &unfixed_cache] {
            fs::create_dir_all(dir).unwrap();
        }
        fs::write(models.join(old_filenames[0]), b"old").unwrap();
        fs::write(models.join(DEPTH_FILENAME), b"current").unwrap();

        remove_superseded_models_in(&models);

        assert!(!models.join(old_filenames[0]).exists());
        assert!(!old_cache.exists());
        assert!(!models.join(OLD_COREML_CACHES[0]).exists());
        assert!(models.join(DEPTH_FILENAME).exists());
        assert!(current_cache.exists());
        fs::remove_dir_all(&models).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn only_compiled_coreml_models_lose_their_source() {
        let cache = scratch_dir("coreml-sources-test");
        let compiled = cache.join("123/0_dynamic_mlprogram/model");
        let uncompiled = cache.join("123/1_dynamic_mlprogram/model");
        for package in [&compiled, &uncompiled] {
            fs::create_dir_all(package.join("Data/com.microsoft.OnnxRuntime/weights")).unwrap();
            fs::write(package.join("Manifest.json"), b"{}").unwrap();
        }
        fs::create_dir_all(compiled.join("compiled_model.mlmodelc/weights")).unwrap();
        fs::write(cache.join("123/model.txt"), b"model path").unwrap();

        remove_coreml_sources(&cache);

        assert!(!compiled.join("Data").exists());
        assert!(!compiled.join("Manifest.json").exists());
        assert!(compiled.join("compiled_model.mlmodelc/weights").is_dir());
        assert!(
            uncompiled.join("Data").is_dir() && uncompiled.join("Manifest.json").exists(),
            "a model that isn't compiled yet still needs its source"
        );
        assert!(cache.join("123/model.txt").exists());
        fs::remove_dir_all(&cache).unwrap();
    }
}

#[cfg(test)]
mod normal_model_tests {
    use super::*;

    #[test]
    fn view_bytes_hold_sizes_then_points_normals_mask_and_texture() {
        let geometry = NormalGeometry {
            width: 2,
            height: 1,
            points: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            normals: vec![0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
            mask: vec![1.0, 0.0],
        };
        let texture = RgbaImage::from_raw(1, 1, vec![10, 20, 30, 255]).unwrap();
        let bytes = geometry.to_view_bytes(&texture);

        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let f32_at = |i: usize| f32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        assert_eq!([u32_at(0), u32_at(4), u32_at(8), u32_at(12)], [2, 1, 1, 1]);
        assert_eq!(f32_at(16), 1.0);
        assert_eq!(f32_at(16 + 5 * 4), 6.0);
        assert_eq!(f32_at(16 + 8 * 4), -1.0);
        assert_eq!(&bytes[64..66], &[255, 0]);
        assert_eq!(&bytes[66..], &[10, 20, 30, 255]);
    }

    #[test]
    fn photos_use_the_fixed_graph_of_their_orientation_within_the_stretch_limit() {
        let (landscape, portrait) = (Some(0), Some(1));
        assert_eq!(NORMAL_FIXED[0].width, 768);
        assert_eq!(NORMAL_FIXED[1].height, 768);
        for (width, height, expected) in [
            (768, 512, landscape),
            (512, 768, portrait),
            (768, 576, landscape), // 4:3
            (576, 768, portrait),
            (768, 432, landscape), // 16:9
            (432, 768, portrait),
            (5000, 4000, landscape), // 5:4, stretched by exactly NORMAL_MAX_STRETCH
            (768, 768, None),
            (768, 384, None), // 2:1
        ] {
            assert_eq!(
                fixed_normal_graph(width, height),
                expected,
                "{width}x{height}"
            );
        }
    }
}
