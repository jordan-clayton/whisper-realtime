use crate::utils::constants;
use crate::whisper::model;

// TODO: get rid of pub and properly encapsulate Config parameters
// TODO: migration to new schema: consider using a versioning Enum or some sort to handle migration somewhat gracefully.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug)]
pub struct Configs {
    pub n_threads: std::ffi::c_int,
    pub set_translate: bool,
    pub language: Option<String>,
    // This is more or less a necessity for realtime.
    pub use_gpu: bool,

    // EXPERIMENTAL:
    // pub speed_up: bool,
    // TODO: this will cause problems with the new implementation, use model in ConfigsV2: the sequel
    // Model derives serialize/deserialize, so this should hold a model that should persist
    pub model: model::DefaultModelType,

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

impl Default for Configs {
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
            // use_vad: true,
            // speed_up: false,
            model: model::DefaultModelType::default(),
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
