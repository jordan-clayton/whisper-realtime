//TODO: Implement serialization
use crate::model::Model;

pub struct Preferences {
    pub gui_prefs: GUIPreferences,
    pub trans_prefs: TranscriberPreferences,
}

impl Default for Preferences {
    fn default() -> Self {
        Preferences {
            gui_prefs: GUIPreferences::default(),
            trans_prefs: TranscriberPreferences::default(),
        }
    }
}

struct GUIPreferences {}
impl Default for GUIPreferences {
    fn default() -> Self {
        GUIPreferences {}
    }
}
struct TranscriberPreferences {
    pub n_threads: std::ffi::c_int,
    pub set_translate: bool,
    pub set_language: Option<&'static str>,
    pub use_gpu: bool,
    pub model: Model,
    // Stdout only.
    pub print_special: bool,
    pub print_progress: bool,
    pub print_realtime: bool,
    pub print_timestamps: bool,
}

impl Default for TranscriberPreferences {
    fn default() -> Self {
        TranscriberPreferences {
            n_threads: 4,
            set_translate: false,
            set_language: Some("en"),
            use_gpu: true,
            model: Model::default(),
            print_special: false,
            print_progress: false,
            print_realtime: false,
            print_timestamps: false,
        }
    }
}
