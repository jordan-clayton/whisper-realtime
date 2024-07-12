use serde::{Deserialize, Serialize};

use crate::model::ModelType;

#[derive(Serialize, Deserialize, Debug)]
pub struct GUIPreferences {}
impl Default for GUIPreferences {
    fn default() -> Self {
        GUIPreferences {}
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Configs {
    // TODO: This needs to gooo
    pub input_device_name: Option<String>,
    pub n_threads: std::ffi::c_int,
    pub set_translate: bool,
    pub set_language: Option<String>,
    pub use_gpu: bool,
    pub model: ModelType,
    // Stdout only.
    pub print_special: bool,
    pub print_progress: bool,
    pub print_realtime: bool,
    pub print_timestamps: bool,
}

impl Default for Configs {
    fn default() -> Self {
        Configs {
            input_device_name: Some(String::from("default")),
            n_threads: 4,
            set_translate: false,
            set_language: Some(String::from("en")),
            use_gpu: true,
            model: ModelType::default(),
            print_special: false,
            print_progress: false,
            print_realtime: false,
            print_timestamps: false,
        }
    }
}
