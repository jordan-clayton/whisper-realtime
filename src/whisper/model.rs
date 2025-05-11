use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "integrity")]
use reqwest::blocking;
#[cfg(feature = "integrity")]
use serde_json;
#[cfg(feature = "integrity")]
use sha2::{Digest, Sha256};
use strum::{
    Display, EnumCount, EnumIs, EnumIter, EnumString, FromRepr, IntoStaticStr, VariantArray,
    VariantNames,
};

#[cfg(feature = "integrity")]
use crate::utils::errors::WhisperRealtimeError;

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug)]
pub struct Model {
    name: String,
    file_name: String,
    path_prefix: PathBuf,
    #[cfg(feature = "integrity")]
    checksum_verified: bool,
}

impl Model {
    pub fn new() -> Self {
        Self {
            name: Default::default(),
            file_name: Default::default(),
            path_prefix: Default::default(),
            #[cfg(feature = "integrity")]
            checksum_verified: false,
        }
    }

    pub fn new_with_parameters(name: &str, file_name: &str, path_prefix: &Path) -> Self {
        Self::new()
            .with_name(name)
            .with_file_name(file_name)
            .with_path_prefix(path_prefix)
    }

    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_owned();
        self
    }

    pub fn with_file_name(mut self, file_name: &str) -> Self {
        self.file_name = file_name.to_owned();
        #[cfg(feature = "integrity")]
        {
            self.checksum_verified = false;
        }
        self
    }

    pub fn with_path_prefix(mut self, path_prefix: &Path) -> Self {
        self.path_prefix = path_prefix.to_path_buf();
        #[cfg(feature = "integrity")]
        {
            self.checksum_verified = false;
        }
        self
    }
    pub fn name(&self) -> &str {
        self.name.as_str()
    }
    pub fn file_name(&self) -> &str {
        self.file_name.as_str()
    }

    pub fn file_path(&self) -> PathBuf {
        self.path_prefix.join(self.file_name.as_str())
    }

    #[cfg(feature = "integrity")]
    pub fn checksum_verified(&self) -> bool {
        self.checksum_verified.clone()
    }

    /// This canonicalizes the file path and checks the directory for an existing file.
    /// It does not verify file integrity
    pub fn exists_in_directory(&self) -> bool {
        return match fs::metadata(self.file_path().as_path()) {
            Ok(m) => m.is_file(),
            Err(_) => false,
        };
    }

    /// For verifying file integrity against a user-provided checksum.
    /// Since a Model file can come from anywhere, responsibility falls upon the user to ensure
    /// integrity. This method provides a mechanism to carry out that responsibility.
    /// Returns true on a checksum match.
    /// Returns a WhisperRealtimeError when the file does not exist.
    ///
    ///
    /// NOTE: This may be refactored into a utility later if the use cases arise
    #[cfg(feature = "integrity")]
    fn compare_sha256(&self, checksum: &str) -> Result<bool, WhisperRealtimeError> {
        // Compute the checksum on the file
        let mut file = fs::File::open(self.file_path())?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        let hash = hasher.finalize();

        // Convert to byte string and compare checksums
        let byte_str = format!("{:x}", hash);
        Ok(checksum.to_lowercase() == byte_str)
    }

    #[cfg(feature = "integrity")]
    pub fn verify_checksum(&mut self, checksum: &str) -> Result<bool, WhisperRealtimeError> {
        if !self.exists_in_directory() {
            self.checksum_verified = false;
            return Ok(false);
        }

        match self.compare_sha256(checksum) {
            Ok(equal) => {
                self.checksum_verified = equal;
                Ok(equal)
            }
            Err(e) => {
                self.checksum_verified = false;
                Err(e)
            }
        }
    }
}

impl Default for Model {
    fn default() -> Self {
        Model::new()
    }
}

// TODO: remove the old struct and implementation
pub struct OldModel {
    pub model_type: DefaultModelType,
    data_directory: PathBuf,
}

impl Default for OldModel {
    fn default() -> Self {
        let mut path = std::env::current_dir().unwrap();
        path.push("data");
        OldModel {
            model_type: DefaultModelType::default(),
            data_directory: path,
        }
    }
}

impl OldModel {
    pub fn new() -> Self {
        Self::default()
    }

    // This defaults to the local directory/data
    pub fn new_with_model_type(m_type: DefaultModelType) -> Self {
        let mut path = std::env::current_dir().unwrap();
        path.push("data");

        OldModel {
            model_type: m_type,
            data_directory: path,
        }
    }
    pub fn new_with_data_dir(path: std::path::PathBuf) -> Self {
        Self {
            model_type: DefaultModelType::default(),
            data_directory: path,
        }
    }

    pub fn with_data_dir(self, path: std::path::PathBuf) -> Self {
        Self {
            model_type: self.model_type,
            data_directory: path,
        }
    }

    pub fn new_with_type_and_dir(m_type: DefaultModelType, path: std::path::PathBuf) -> Self {
        Self {
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

    pub fn model_directory(&self) -> PathBuf {
        let mut buf = self.data_directory.clone();
        buf.push("models");
        buf
    }

    pub fn file_path(&self) -> PathBuf {
        let mut buf = self.model_directory();
        let file_name = self.model_file_name();
        buf.push(file_name);
        buf.clone()
    }

    pub fn model_file_name(&self) -> &'static str {
        self.model_type.to_file_name()
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
#[derive(
    Copy,
    Clone,
    Debug,
    PartialOrd,
    PartialEq,
    Ord,
    Eq,
    EnumCount,
    EnumIter,
    EnumString,
    Display,
    IntoStaticStr,
    EnumIs,
    FromRepr,
    VariantArray,
    VariantNames,
)]
pub enum DefaultModelType {
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

impl DefaultModelType {
    pub fn to_file_name(&self) -> &'static str {
        match self {
            DefaultModelType::TinyEn => "tiny.en.bin",
            DefaultModelType::Tiny => "tiny.bin",
            DefaultModelType::BaseEn => "base.en.bin",
            DefaultModelType::Base => "base.bin",
            DefaultModelType::SmallEn => "small.en.bin",
            DefaultModelType::Small => "small.bin",
            DefaultModelType::MediumEn => "medium.en.bin",
            DefaultModelType::Medium => "medium.bin",
            DefaultModelType::LargeV1 => "large-v1.bin",
            DefaultModelType::LargeV2 => "large-v2.bin",
            DefaultModelType::LargeV3 => "large-v3.bin",
        }
    }

    /// To canonicalize the huggingface url for downloading the appropriate ggml model.
    pub fn url(&self) -> String {
        let file_name = self.to_file_name();
        const URL_PREFIX: &'static str =
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-";

        [URL_PREFIX, file_name].concat()
    }

    /// To canonicalize the huggingface url for downloading the appropriate coreml zip
    /// This will need to be extracted into the same directory as an accompanying model
    /// NOTE: Do not strip away the top level directory when extracting:
    /// Whisper-rs (Whisper.cpp) expects the -encoder.mlmodelc directory to exist within the same
    /// directory as its accompanying model.
    /// eg. models/ggml-base.en-encoder.mlmodelc AND models/ggml-base.en.bin
    #[cfg(feature = "coreml")]
    pub fn coreml_zip_url(&self) -> String {
        let url = self.url();
        url.replace(".bin", "-encoder.mlmodelc.zip")
    }

    /// This makes a blocking request to grab the checksum for each of the provided default models
    /// This should be called on a separate thread if you do not want the program to hang.
    /// Async requests are not provided at this time and will only be implemented should the demand
    /// be present.
    /// Upon failing to retrieve the JSON, or the checksum, this will return a WhisperRealtimeError::DownloadError
    #[cfg(feature = "integrity")]
    pub fn get_checksum(
        &self,
        client: Option<&blocking::Client>,
    ) -> Result<String, WhisperRealtimeError> {
        let file_name = self.to_file_name();
        const URL: &str = "https://huggingface.co/api/models/ggerganov/whisper.cpp";

        let json: serde_json::Value = match client {
            None => blocking::get(URL),
            Some(r_client) => r_client.get(URL).send(),
        }?
        .json()?;
        let checksum = json["siblings"]
            .as_array()
            .ok_or(WhisperRealtimeError::DownloadError(
                "Failed to get checksum".to_owned(),
            ))?
            .iter()
            .find(|entry| entry["rfilename"] == file_name)
            .and_then(|entry| entry["sha256"].as_str())
            .ok_or(WhisperRealtimeError::DownloadError(
                "Checksum not found".to_owned(),
            ))?;

        Ok(checksum.to_owned())
    }
}

impl Default for DefaultModelType {
    fn default() -> Self {
        DefaultModelType::TinyEn
    }
}
