use crate::constants;
use crate::model;

#[cfg_attr(feature = "use_serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug)]
pub struct Configs {
    pub n_threads: std::ffi::c_int,
    pub set_translate: bool,
    pub set_language: Option<String>,
    // This is more or less a necessity for realtime in debug mode.
    // CPU realtime has not yet been tested.
    pub use_gpu: bool,
    pub model: model::ModelType,

    // in milliseconds.
    pub realtime_timeout: u128,
    pub audio_sample_ms: usize,
    pub vad_sample_ms: usize,
    pub phrase_timeout: usize,

    pub voice_threshold: f32,
    // Stdout only.
    pub print_special: bool,
    pub print_progress: bool,
    pub print_realtime: bool,
    pub print_timestamps: bool,
}

impl Default for Configs {
    fn default() -> Self {
        Self {
            n_threads: 4,
            set_translate: false,
            set_language: Some(String::from("en")),
            use_gpu: cfg!(feature = "_gpu"),
            model: model::ModelType::default(),
            // Currently set to 1 hr
            realtime_timeout: constants::REALTIME_AUDIO_TIMEOUT,
            audio_sample_ms: constants::AUDIO_SAMPLE_MS,
            vad_sample_ms: constants::VAD_SAMPLE_MS,
            phrase_timeout: constants::PHRASE_TIMEOUT,
            voice_threshold: constants::VOICE_THRESHOLD,
            print_special: false,
            print_progress: false,
            print_realtime: false,
            print_timestamps: false,
        }
    }
}
