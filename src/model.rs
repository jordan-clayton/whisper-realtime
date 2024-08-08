use std::fs;

// TODO: CoreML support.
pub struct Model {
    pub model_type: ModelType,
    data_directory: std::path::PathBuf,
}

impl Default for Model {
    fn default() -> Self {
        let mut path = std::env::current_dir().unwrap();
        path.push("data");
        Model {
            model_type: ModelType::default(),
            data_directory: path,
        }
    }
}

impl Model {
    pub fn new() -> Self {
        Self::default()
    }

    // This defaults to the local directory/data
    pub fn new_with_model_type(m_type: ModelType) -> Self {
        let mut path = std::env::current_dir().unwrap();
        path.push("data");

        Model {
            model_type: m_type,
            data_directory: path,
        }
    }

    pub fn new_with_data_dir(path: std::path::PathBuf) -> Self {
        Model {
            model_type: ModelType::default(),
            data_directory: path,
        }
    }

    pub fn new_with_type_and_dir(m_type: ModelType, path: std::path::PathBuf) -> Self {
        Model {
            model_type: m_type,
            data_directory: path,
        }
    }
    pub fn url(&self) -> String {
        let mut url =
            String::from("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-");
        url.push_str(self.model_file_name());
        url
    }

    pub fn model_directory(&self) -> std::path::PathBuf {
        let mut buf = self.data_directory.clone();
        buf.push("models");
        buf
    }

    pub fn file_path(&self) -> std::path::PathBuf {
        let mut buf = self.model_directory();
        let file_name = self.model_file_name();
        buf.push(file_name);
        buf.clone()
    }

    pub fn model_file_name(&self) -> &str {
        match self.model_type {
            ModelType::TinyEn => "tiny.en.bin",
            ModelType::Tiny => "tiny.bin",
            ModelType::BaseEn => "base.en.bin",
            ModelType::Base => "base.bin",
            ModelType::SmallEn => "small.en.bin",
            ModelType::Small => "small.bin",
            ModelType::MediumEn => "medium.en.bin",
            ModelType::Medium => "medium.bin",
            ModelType::LargeV1 => "large-v1.bin",
            ModelType::LargeV2 => "large-v2.bin",
            ModelType::LargeV3 => "large.bin",
        }
    }

    // NOTE: this doesn't handle malformed files.
    pub fn is_downloaded(&self) -> bool {
        let path_buf = self.file_path();
        let path = path_buf.as_path();

        let path = fs::metadata(path);
        match path {
            Ok(p) => p.is_file(),
            Err(_) => false,
        }
    }

    pub fn delete(&self) {
        if !self.is_downloaded() {
            return;
        }

        let path_buf = self.file_path();
        let model_path = path_buf.as_path();
        fs::remove_file(model_path).expect("failed to delete mod");
    }
}

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Copy, Clone, Debug)]
pub enum ModelType {
    TinyEn,
    Tiny,
    BaseEn,
    Base,
    SmallEn,
    Small,
    MediumEn,
    Medium,
    LargeV1,
    LargeV2,
    LargeV3,
}

impl Default for ModelType {
    fn default() -> Self {
        ModelType::TinyEn
    }
}
