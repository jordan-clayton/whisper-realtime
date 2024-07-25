use log::info;

// TODO: async downloading & GUI refactor.
pub struct Model {
    pub model_type: ModelType,
    data_directory: std::path::PathBuf,
}

impl Default for Model {
    fn default() -> Self {
        // let p_dir =
        //     directories::ProjectDirs::from("com", "Jordan", "WhisperGUI").expect("no home folder");
        // let path = p_dir.data_dir().to_path_buf();

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

    pub fn new_with_model_type(m_type: ModelType) -> Self {
        let p_dir =
            directories::ProjectDirs::from("com", "Jordan", "WhisperGUI").expect("no home folder");
        let path = p_dir.data_dir().to_path_buf();
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
        match self.model_type {
            ModelType::TinyEn => {
                url.push_str("tiny.en.bin");
            }
            ModelType::Tiny => {
                url.push_str("tiny.bin");
            }
            ModelType::BaseEn => {
                url.push_str("base.en.bin");
            }
            ModelType::Base => {
                url.push_str("base.bin");
            }
            ModelType::SmallEn => {
                url.push_str("small.en.bin");
            }
            ModelType::Small => {
                url.push_str("small.bin");
            }
            ModelType::MediumEn => {
                url.push_str("medium.en.bin");
            }
            ModelType::Medium => {
                url.push_str("medium.bin");
            }
            ModelType::LargeV1 => {
                url.push_str("large-v1.bin");
            }
            ModelType::LargeV2 => {
                url.push_str("large-v2.bin");
            }
            ModelType::LargeV3 => {
                url.push_str("large-v3.bin");
            }
        };
        url
    }

    pub fn model_directory(&self) -> std::path::PathBuf {
        let mut buf = self.data_directory.clone();
        buf.push("models");
        buf
    }
    pub fn file_path(&self) -> std::path::PathBuf {
        let mut buf = self.model_directory();
        match self.model_type {
            ModelType::TinyEn => {
                buf.push("tiny.en.bin");
            }
            ModelType::Tiny => {
                buf.push("tiny.bin");
            }
            ModelType::BaseEn => {
                buf.push("base.en.bin");
            }

            ModelType::Base => {
                buf.push("base.bin");
            }
            ModelType::SmallEn => {
                buf.push("small.en.bin");
            }
            ModelType::Small => {
                buf.push("small.bin");
            }
            ModelType::MediumEn => {
                buf.push("medium.en.bin");
            }
            ModelType::Medium => {
                buf.push("medium.bin");
            }
            ModelType::LargeV1 => {
                buf.push("large-v1.bin");
            }
            ModelType::LargeV2 => {
                buf.push("large-v2.bin");
            }
            // LargeV3 is large -> latest mod
            ModelType::LargeV3 => {
                buf.push("large.bin");
            }
        };
        buf.clone()
    }

    pub fn is_downloaded(&self) -> bool {
        let path_buf = self.file_path();
        let path = path_buf.as_path();

        let path = std::fs::metadata(path);
        match path {
            Ok(_) => path.unwrap().is_file(),
            Err(_) => false,
        }
    }

    // TODO: Get the progress on this.

    // Ideally, this should be run on a thread.
    pub fn download(&self) {
        if self.is_downloaded() {
            return;
        }

        // Create the models directory if it doesn't exist
        let model_directory_buf = self.model_directory();
        let m_dir = model_directory_buf.as_path();

        if !m_dir.exists() {
            std::fs::create_dir_all(m_dir).expect("failed to create models directory");
        }

        let url = self.url();

        info!("Downloading mod {}", url);

        let resp = ureq::get(url.as_str()).call().expect("download failed");
        let len: usize = resp
            .header("Content-Length")
            .expect("request returned zero-length")
            .parse()
            .unwrap_or_default();

        let mut bytes: Vec<u8> = Vec::with_capacity(len);
        resp.into_reader()
            .read_to_end(&mut bytes)
            .expect("failed to serialize mod data");

        info!("Downloaded mod {}", url);
        let path_buf = self.file_path();
        let model_path = path_buf.as_path();
        std::fs::write(model_path, bytes).expect("failed to save mod");
    }

    pub fn delete(&self) {
        if !self.is_downloaded() {
            return;
        }

        let path_buf = self.file_path();
        let model_path = path_buf.as_path();
        std::fs::remove_file(model_path).expect("failed to delete mod");
    }
}

#[cfg_attr(feature = "use_serde", derive(serde::Deserialize, serde::Serialize))]
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
