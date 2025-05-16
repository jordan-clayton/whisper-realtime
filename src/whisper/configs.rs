use std::num::NonZeroUsize;
use std::str::FromStr;

use strum::{AsRefStr, Display, EnumString, FromRepr, IntoStaticStr};
use whisper_rs;

use crate::utils::constants;
use crate::whisper::model::{DefaultModelType, Model};

/// Versioned wrapper for supported whisper-realtime configuration types.
/// Can be serialized (using serde or otherwise) to persist settings.
/// V1: Legacy, whisper and realtime configurations.
/// V2: Current whisper-only configurations.
/// RealtimeV1: A complete realtime-configuration. Composed of V2 and RealtimeConfigs
///
/// Note: RealtimeConfigs is not included directly; it cannot be used alone for transcription
///
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug)]
pub enum Configs {
    V1(WhisperConfigsV1),
    V2(WhisperConfigsV2),
    RealtimeV1(WhisperRealtimeConfigs),
}

impl Configs {
    /// Used to migrate (older) V1 configs to RealtimeV1 configs with the original model data
    /// path preserved. Individual paths are now stored in Model.
    /// This only affects V1; V2 and RealtimeV1 will just return the same object.
    ///
    /// If only V2 configurations are needed, chain with .into_v2()
    pub fn migrate_v1_preserve_model_path(self, model_path: &std::path::Path) -> Self {
        match self {
            Configs::V1(v1) => {
                Self::RealtimeV1(v1.into_realtime_v1_with_models_directory(model_path))
            }
            Configs::V2(_) => self,
            Configs::RealtimeV1(_) => self,
        }
    }

    /// Converts configuration to WhisperRealtimeConfigurations (RealtimeV1).
    /// Realtime configurations are preserved from V1.
    /// If called on V2, RealtimeConfigurations::default() is supplied
    pub fn into_realtime_v1(self) -> Self {
        match self {
            Configs::V1(v1) => Self::RealtimeV1(v1.into_realtime_v1()),
            Configs::V2(v2) => Self::RealtimeV1(v2.into_realtime_v1()),
            Configs::RealtimeV1(_) => self,
        }
    }

    /// Converts configuration to WhisperConfigsV2.
    /// Note: Migrating from V1 or RealtimeV1 to V2 is lossy; realtime configurations are not part
    /// of WhisperConfigs V2
    pub fn into_v2(self) -> Self {
        match self {
            Configs::V1(v1) => Self::V2(v1.into_v2()),
            Configs::V2(_) => self,
            Configs::RealtimeV1(r_v1) => Self::V2(r_v1.into_whisper_v2()),
        }
    }
}

/// A configurations type that holds a subset of useful configurations for whisper-rs::FullParam,
/// a transcription model, and a flag to indicate whether the GPU should be used to run the transcription.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug)]
pub struct WhisperConfigsV2 {
    // Whisper FullParams data
    // The number of thread s to use for HW decoding.
    n_threads: std::ffi::c_int,
    // As a prompt for the decoder, defaults to 16384
    max_past_prompt_tokens: std::ffi::c_int,
    sampling_strategy: WhisperSamplingStrategy,
    translate: bool,
    language: Option<Language>,
    // Don't use past transcriptions as an initial prompt for the decoder.
    use_no_context: bool,
    // Whisper Context data
    model: Model,
    use_gpu: bool,
}

impl WhisperConfigsV2 {
    pub fn new() -> Self {
        Self {
            n_threads: 0,
            max_past_prompt_tokens: 0,
            sampling_strategy: WhisperSamplingStrategy::Greedy { best_of: 0 },
            translate: false,
            language: None,
            use_gpu: false,
            use_no_context: false,
            model: Default::default(),
        }
    }

    pub fn with_n_threads(mut self, num_threads: usize) -> Self {
        self.n_threads = num_threads as std::ffi::c_int;
        self
    }
    pub fn with_max_past_prompt_tokens(mut self, num_prompt_tokens: usize) -> Self {
        self.max_past_prompt_tokens = num_prompt_tokens as std::ffi::c_int;
        self
    }
    pub fn set_translate(mut self, set_translate: bool) -> Self {
        self.translate = set_translate;
        self
    }
    pub fn with_language(mut self, language: Option<Language>) -> Self {
        self.language = language;
        self
    }
    pub fn set_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }
    pub fn set_use_no_context(mut self, use_no_context: bool) -> Self {
        self.use_no_context = use_no_context;
        self
    }

    pub fn with_sampling_strategy(mut self, sampling_strategy: WhisperSamplingStrategy) -> Self {
        self.sampling_strategy = sampling_strategy;
        self
    }
    pub fn with_model(mut self, model: Model) -> Self {
        self.model = model;
        self
    }

    pub fn n_threads(&self) -> usize {
        self.n_threads as usize
    }
    pub fn max_past_prompt_tokens(&self) -> usize {
        self.max_past_prompt_tokens as usize
    }
    pub fn translate(&self) -> bool {
        self.translate
    }
    pub fn language(&self) -> &Option<Language> {
        &self.language
    }
    pub fn using_gpu(&self) -> bool {
        self.use_gpu
    }
    pub fn using_no_context(&self) -> bool {
        self.use_no_context
    }
    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn into_realtime_v1(self) -> WhisperRealtimeConfigs {
        WhisperRealtimeConfigs::new().with_whisper_configs(self)
    }

    /// For constructing a FullParams object ready to be passed to WhisperState::full()
    /// Note: these configurations do not cover FullParams in entirety.
    /// See https://docs.rs/whisper-rs/latest/whisper_rs/struct.FullParams.html and set other fields accordingly.
    pub fn to_whisper_full_params(&self) -> whisper_rs::FullParams {
        let mut params = match self.sampling_strategy {
            WhisperSamplingStrategy::Greedy { best_of } => {
                whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy {
                    best_of: best_of as std::ffi::c_int,
                })
            }
            WhisperSamplingStrategy::BeamSearch {
                beam_size,
                patience,
            } => whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::BeamSearch {
                beam_size: beam_size as std::ffi::c_int,
                patience: patience as std::ffi::c_float,
            }),
        };

        params.set_n_threads(self.n_threads);
        params.set_n_max_text_ctx(self.max_past_prompt_tokens);
        params.set_translate(self.translate);
        if let Some(lang) = self.language {
            params.set_language(Some(lang.into()))
        }
        params.set_no_context(self.use_no_context);
        params
    }
}

impl Default for WhisperConfigsV2 {
    fn default() -> Self {
        let n_threads = std::cmp::min(
            4,
            std::thread::available_parallelism()
                .unwrap_or(NonZeroUsize::new(2).unwrap())
                .get(),
        );

        let max_prompt_tokens = 16384;

        Self::new()
            .with_n_threads(n_threads)
            .with_max_past_prompt_tokens(max_prompt_tokens)
            .with_sampling_strategy(WhisperSamplingStrategy::Greedy { best_of: 1 })
            .set_gpu(true)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug)]
pub enum WhisperSamplingStrategy {
    Greedy { best_of: usize },
    BeamSearch { beam_size: usize, patience: f32 },
}

/// A configurations component that holds relevant configurations for tweaking realtime transcription.
/// All timeouts/audio lengths are measured in milliseconds
// TODO: return to this struct and include VAD configurations if allowing multiple.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug)]
pub struct RealtimeConfigs {
    // in Milliseconds.
    realtime_timeout: u128,
    audio_sample_len: usize,
    vad_sample_len: usize,
    phrase_timeout: usize,
    voice_probability_threshold: f32,
}

impl RealtimeConfigs {
    pub fn new() -> Self {
        Self {
            realtime_timeout: 0,
            audio_sample_len: 0,
            vad_sample_len: 0,
            phrase_timeout: 0,
            voice_probability_threshold: 0.,
        }
    }
    pub fn with_realtime_timeout(mut self, realtime_timeout: u128) -> Self {
        self.realtime_timeout = realtime_timeout;
        self
    }
    pub fn with_audio_sample_len(mut self, len_ms: usize) -> Self {
        self.audio_sample_len = len_ms;
        self
    }
    pub fn with_vad_sample_len(mut self, len_ms: usize) -> Self {
        self.vad_sample_len = len_ms;
        self
    }
    pub fn with_phrase_timeout(mut self, len_ms: usize) -> Self {
        self.phrase_timeout = len_ms;
        self
    }
    pub fn with_voice_probability_threshold(mut self, p_threshold: f32) -> Self {
        self.voice_probability_threshold = p_threshold;
        self
    }
    pub fn realtime_timeout(&self) -> u128 {
        self.realtime_timeout
    }
    pub fn audio_sample_len(&self) -> usize {
        self.audio_sample_len
    }
    pub fn vad_sample_len(&self) -> usize {
        self.vad_sample_len
    }
    pub fn phrase_timeout(&self) -> usize {
        self.phrase_timeout
    }
    pub fn voice_probability_threshold(&self) -> f32 {
        self.voice_probability_threshold
    }
}

impl Default for RealtimeConfigs {
    fn default() -> Self {
        Self::new()
            // 1 hour
            .with_realtime_timeout(constants::REALTIME_AUDIO_TIMEOUT)
            // 10 seconds / 10 000 ms
            .with_audio_sample_len(constants::AUDIO_SAMPLE_MS)
            // .3 seconds / 300 ms
            .with_vad_sample_len(constants::VAD_SAMPLE_MS)
            // 3 seconds / 3000 ms
            .with_phrase_timeout(constants::PHRASE_TIMEOUT)
            // 0.65
            .with_voice_probability_threshold(constants::VOICE_PROBABILITY_THRESHOLD)
    }
}

// TODO: language codes
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(
    Copy,
    Clone,
    Debug,
    PartialOrd,
    PartialEq,
    AsRefStr,
    IntoStaticStr,
    FromRepr,
    Display,
    EnumString,
)]
#[strum(serialize_all = "lowercase")]
pub enum Language {
    En,
}

/// A composite configurations type holding all relevant configurations for running realtime
/// transcription.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug)]
pub struct WhisperRealtimeConfigs {
    whisper: WhisperConfigsV2,
    realtime: RealtimeConfigs,
}

impl WhisperRealtimeConfigs {
    pub fn new() -> Self {
        Self {
            whisper: WhisperConfigsV2::new(),
            realtime: RealtimeConfigs::new(),
        }
    }
    pub fn with_whisper_configs(mut self, w_configs: WhisperConfigsV2) -> Self {
        self.whisper = w_configs;
        self
    }
    pub fn with_realtime_configs(mut self, r_configs: RealtimeConfigs) -> Self {
        self.realtime = r_configs;
        self
    }
    pub fn whisper_configs(&self) -> &WhisperConfigsV2 {
        &self.whisper
    }
    pub fn realtime_configs(&self) -> &RealtimeConfigs {
        &self.realtime
    }
    pub fn into_whisper_v2(self) -> WhisperConfigsV2 {
        self.whisper
    }

    pub fn into_realtime(self) -> RealtimeConfigs {
        self.realtime
    }
}

impl Default for WhisperRealtimeConfigs {
    fn default() -> Self {
        Self::new()
            .with_whisper_configs(WhisperConfigsV2::default())
            .with_realtime_configs(RealtimeConfigs::default())
    }
}

// LEGACY
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug)]
pub struct WhisperConfigsV1 {
    pub n_threads: std::ffi::c_int,
    pub set_translate: bool,
    pub language: Option<String>,
    // This is more or less a necessity for realtime.
    pub use_gpu: bool,

    pub model: DefaultModelType,

    // in milliseconds.
    pub realtime_timeout: u128,
    pub audio_sample_ms: usize,
    pub vad_sample_ms: usize,
    pub phrase_timeout: usize,

    pub voice_probability_threshold: f32,
    // To determine high-pass
    pub naive_vad_freq_threshold: f64,
    // To determine VAD for naive impl.
    pub naive_vad_energy_threshold: f64,
    pub naive_window_len: f64,
    pub naive_window_step: f64,
    // Stdout only.
    pub print_special: bool,
    pub print_progress: bool,
    pub print_realtime: bool,
    pub print_timestamps: bool,
}

impl WhisperConfigsV1 {
    pub fn into_v2(self) -> WhisperConfigsV2 {
        let model = self.model.to_model();
        self.into_v2_with_model(model)
    }

    pub fn into_v2_with_models_directory(
        self,
        models_directory: &std::path::Path,
    ) -> WhisperConfigsV2 {
        let model = self.model.to_model_with_path_prefix(models_directory);
        self.into_v2_with_model(model)
    }
    pub fn into_realtime_v1(self) -> WhisperRealtimeConfigs {
        let realtime_configs = self.to_realtime_configs();
        let whisper_configs = self.into_v2();
        WhisperRealtimeConfigs::new()
            .with_realtime_configs(realtime_configs)
            .with_whisper_configs(whisper_configs)
    }

    pub fn into_realtime_v1_with_models_directory(
        self,
        models_directory: &std::path::Path,
    ) -> WhisperRealtimeConfigs {
        let realtime_configs = self.to_realtime_configs();
        let whisper_configs = self.into_v2_with_models_directory(models_directory);
        WhisperRealtimeConfigs::new()
            .with_realtime_configs(realtime_configs)
            .with_whisper_configs(whisper_configs)
    }

    fn into_v2_with_model(self, model: Model) -> WhisperConfigsV2 {
        let language = self
            .language
            .as_ref()
            .map(|lang| Language::from_str(lang).unwrap());
        WhisperConfigsV2::default()
            .with_n_threads(self.n_threads as usize)
            .set_translate(self.set_translate)
            .with_language(language)
            .set_gpu(self.use_gpu)
            // To avoid losing the stored model type
            // Note: this does not preserve the data directory and will need to be handled.
            .with_model(model)
    }

    // Extracts the realtime-related information from WhisperConfigsV1 without consuming self,
    // so that self may be fully consumed into WhisperConfigsV2 (with or without a model prefix path)
    fn to_realtime_configs(&self) -> RealtimeConfigs {
        RealtimeConfigs::new()
            .with_audio_sample_len(self.audio_sample_ms)
            .with_vad_sample_len(self.vad_sample_ms)
            .with_realtime_timeout(self.realtime_timeout)
            .with_phrase_timeout(self.phrase_timeout)
            .with_voice_probability_threshold(self.voice_probability_threshold)
    }
}

impl Default for WhisperConfigsV1 {
    fn default() -> Self {
        let n_threads = match std::thread::available_parallelism() {
            Ok(n) => {
                let min = n.get();
                std::cmp::min(min, 4) as std::ffi::c_int
            }
            Err(_) => 2,
        };

        Self {
            n_threads,
            set_translate: false,
            language: Some(String::from("en")),
            use_gpu: cfg!(feature = "_gpu"),
            model: DefaultModelType::default(),
            // Currently set to 1 hr
            realtime_timeout: constants::REALTIME_AUDIO_TIMEOUT,
            audio_sample_ms: constants::AUDIO_SAMPLE_MS,
            vad_sample_ms: constants::VAD_SAMPLE_MS,
            phrase_timeout: constants::PHRASE_TIMEOUT,
            naive_vad_energy_threshold: constants::VOICE_ENERGY_THRESHOLD,
            naive_window_len: constants::VAD_WIN_LEN,
            naive_window_step: constants::VAD_WIN_HOP,
            voice_probability_threshold: constants::VOICE_PROBABILITY_THRESHOLD,
            naive_vad_freq_threshold: constants::VAD_FREQ_THRESHOLD,
            print_special: false,
            print_progress: false,
            print_realtime: false,
            print_timestamps: false,
        }
    }
}
