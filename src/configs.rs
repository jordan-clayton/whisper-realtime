use serde::{Deserialize, Serialize};

use crate::model::ModelType;

// TODO: separate WhisperRS configs & add Realtime Configs
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Configs {
    pub n_threads: std::ffi::c_int,
    pub set_translate: bool,
    pub set_language: Option<String>,
    // This is more or less a necessity for realtime
    pub use_gpu: bool,
    pub model: ModelType,

    // in milliseconds.
    pub realtime_timeout: u128,
    // Stdout only.
    pub print_special: bool,
    pub print_progress: bool,
    pub print_realtime: bool,
    pub print_timestamps: bool,
}

impl Default for Configs {
    fn default() -> Self {
        Configs {
            n_threads: 4,
            set_translate: false,
            set_language: Some(String::from("en")),
            // TODO: this should be cfg![]
            use_gpu: true,
            model: ModelType::default(),
            // Currently set to 10 mins.
            realtime_timeout: crate::constants::REALTIME_AUDIO_TIMEOUT,
            print_special: false,
            print_progress: false,
            print_realtime: false,
            print_timestamps: false,
        }
    }
}
