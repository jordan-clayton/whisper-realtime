use directories::ProjectDirs;

use crate::preferences::{Configs, GUIPreferences};

// TODO build a download trait & generalize these functions
// Download should be async

pub fn load_configs(proj_dir: &ProjectDirs) -> Configs {
    let p_dir = proj_dir.clone();
    let mut buf = p_dir.config_local_dir().to_path_buf();
    buf.push("wg_configs.bson");
    let wg_configs_path = buf.as_path();

    let wg_configs_deserialized = std::fs::read(wg_configs_path);
    match wg_configs_deserialized {
        Ok(v) => {
            let wg_config_doc =
                bson::Document::from_reader(&mut v.as_slice()).expect("bson deserialize failed");
            bson::from_document(wg_config_doc).unwrap()
        }
        Err(_e) => Configs::default(),
    }
}
pub fn save_configs(proj_dir: &ProjectDirs, wg_configs: &Configs) {
    let p_dir = proj_dir.clone();
    let wg_configs_dir = p_dir.config_local_dir();

    if !wg_configs_dir.exists() {
        std::fs::create_dir_all(wg_configs_dir).expect("failed to create config directory");
    }

    let mut buf = p_dir.config_local_dir().to_path_buf();
    buf.push("wg_configs.bson");
    let wg_configs_path = buf.as_path();

    let wg_configs_serialized = bson::to_document(&wg_configs).expect("bson failed");
    let mut buf = Vec::new();
    wg_configs_serialized.to_writer(&mut buf).unwrap();
    std::fs::write(wg_configs_path, buf).expect("failed to write transcriber configs")
}

pub fn load_prefs(proj_dir: &ProjectDirs) -> GUIPreferences {
    let p_dir = proj_dir.clone();
    let mut buf = p_dir.preference_dir().to_path_buf();
    buf.push("wg_prefs.bson");
    let wg_prefs_path = buf.as_path();

    let wg_prefs_deserialized = std::fs::read(wg_prefs_path);
    match wg_prefs_deserialized {
        Ok(v) => {
            let wg_prefs_doc =
                bson::Document::from_reader(&mut v.as_slice()).expect("bson deserialize failed");
            bson::from_document(wg_prefs_doc).unwrap()
        }
        Err(_e) => GUIPreferences::default(),
    }
}

pub fn save_prefs(proj_dir: &ProjectDirs, wg_prefs: &GUIPreferences) {
    let p_dir = proj_dir.clone();
    let wg_prefs_dir = proj_dir.config_local_dir();

    if !wg_prefs_dir.exists() {
        std::fs::create_dir_all(wg_prefs_dir).expect("failed to create preferences directory");
    }
    let mut buf = p_dir.preference_dir().to_path_buf();
    buf.push("wg_prefs.bson");
    let wg_prefs_path = buf.as_path();

    let wg_prefs_serialized = bson::to_document(&wg_prefs).expect("bson failed");
    let mut buf = Vec::new();
    wg_prefs_serialized.to_writer(&mut buf).unwrap();
    std::fs::write(wg_prefs_path, buf).expect("failed to write preferences")
}
