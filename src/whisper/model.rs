use std::convert::AsRef;
use std::fs;
use std::path::{Path, PathBuf};

#[cfg(feature = "integrity")]
use reqwest::blocking;
#[cfg(feature = "integrity")]
use sha1::Sha1;
#[cfg(feature = "integrity")]
use sha2::{Digest, Sha256};
use strum::{
    AsRefStr, Display, EnumCount, EnumIs, EnumIter, EnumString, FromRepr, IntoStaticStr,
    VariantArray, VariantNames,
};

#[cfg(feature = "integrity")]
use crate::utils::errors::WhisperRealtimeError;
#[cfg(feature = "integrity")]
use crate::whisper::integrity_utils::{
    checksums_need_updating, ChecksumStatus, get_model_checksum, get_new_checksums,
    serialize_new_checksums, write_latest_repo_checksum_to_disk,
};

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
        &self.file_name
    }

    pub fn path_prefix(&self) -> &Path {
        self.path_prefix.as_path()
    }

    pub fn file_path(&self) -> PathBuf {
        self.path_prefix.join(&self.file_name)
    }

    #[cfg(feature = "integrity")]
    pub fn checksum_verified(&self) -> bool {
        self.checksum_verified
    }

    /// This canonicalizes the file path and checks the directory for an existing file.
    /// It does not verify file integrity
    pub fn exists_in_directory(&self) -> bool {
        return match fs::metadata(self.file_path().as_path()) {
            Ok(m) => m.is_file(),
            Err(_) => false,
        };
    }

    #[allow(dead_code)]
    #[cfg(feature = "integrity")]
    fn compare_sha256(&self, checksum: &str) -> Result<bool, WhisperRealtimeError> {
        // Compute the checksum on the file
        let mut file = fs::File::open(self.file_path())?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        let hash = hasher.finalize();

        // Convert to byte string and compare checksums
        let byte_string = format!("{:x}", hash);
        let byte_str = byte_string.as_str();
        Ok(checksum.to_lowercase() == byte_str)
    }

    /// For verifying file integrity against a user-provided checksum.
    /// Since a Model file can come from anywhere, responsibility falls upon the user to ensure
    /// integrity. This method provides a mechanism to carry out that responsibility.
    /// Returns true on a checksum match.
    /// Returns a WhisperRealtimeError when the file does not exist.
    ///
    ///
    #[cfg(feature = "integrity")]
    fn compare_sha1(&self, checksum: &str) -> Result<bool, WhisperRealtimeError> {
        // Compute the checksum on the file
        let mut file = fs::File::open(self.file_path())?;
        let mut hasher = Sha1::new();
        std::io::copy(&mut file, &mut hasher)?;
        let hash = hasher.finalize();

        // Convert to byte string and compare checksums
        let byte_string = format!("{:x}", hash);
        let byte_str = byte_string.as_str();
        Ok(checksum.to_lowercase() == byte_str)
    }

    #[cfg(feature = "integrity")]
    pub fn verify_checksum(&mut self, checksum: &Checksum) -> Result<bool, WhisperRealtimeError> {
        if !self.exists_in_directory() {
            self.checksum_verified = false;
            return Ok(false);
        }
        let is_equal = match checksum {
            Checksum::Sha1(c) => self.compare_sha1(c),
            Checksum::Sha256(c) => self.compare_sha256(c),
        };

        match is_equal {
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
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(
    Copy,
    Clone,
    Debug,
    PartialOrd,
    PartialEq,
    Ord,
    Eq,
    AsRefStr,
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
            DefaultModelType::TinyEn => "ggml-tiny.en.bin",
            DefaultModelType::Tiny => "ggml-tiny.bin",
            DefaultModelType::BaseEn => "ggml-base.en.bin",
            DefaultModelType::Base => "ggml-base.bin",
            DefaultModelType::SmallEn => "ggml-small.en.bin",
            DefaultModelType::Small => "ggml-small.bin",
            DefaultModelType::MediumEn => "ggml-medium.en.bin",
            DefaultModelType::Medium => "ggml-medium.bin",
            DefaultModelType::LargeV1 => "ggml-large-v1.bin",
            DefaultModelType::LargeV2 => "ggml-large-v2.bin",
            DefaultModelType::LargeV3 => "ggml-large-v3.bin",
        }
    }
    #[cfg(feature = "integrity")]
    fn to_sha_key(&self) -> &'static str {
        match self {
            DefaultModelType::TinyEn => "tiny.en",
            DefaultModelType::Tiny => "tiny",
            DefaultModelType::BaseEn => "base.en",
            DefaultModelType::Base => "base",
            DefaultModelType::SmallEn => "small.en",
            DefaultModelType::Small => "small",
            DefaultModelType::MediumEn => "medium.en",
            DefaultModelType::Medium => "medium",
            DefaultModelType::LargeV1 => "large-v1",
            DefaultModelType::LargeV2 => "large-v2",
            DefaultModelType::LargeV3 => "large-v3",
        }
    }

    pub fn to_model(&self) -> Model {
        Model::new()
            .with_name(self.as_ref())
            .with_file_name(self.to_file_name())
    }

    pub fn to_model_with_path_prefix(&self, prefix: &Path) -> Model {
        let file_name = self.to_file_name();
        Model::new_with_parameters(self.as_ref(), file_name, prefix)
    }

    /// To canonicalize the huggingface url for downloading the appropriate ggml model.
    pub fn url(&self) -> String {
        let file_name = self.to_file_name();
        const URL_PREFIX: &'static str =
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";

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
        model_directory: &Path,
        client: Option<&blocking::Client>,
    ) -> Result<String, WhisperRealtimeError> {
        let key = self.to_sha_key();
        let needs_updating = checksums_need_updating(model_directory, client);

        // Handle the current status of the stored repository checksum
        match needs_updating {
            ChecksumStatus::UpToDate(_) => {}
            ChecksumStatus::NeedsUpdating(c) => {
                let checksums = get_new_checksums(client)?;
                serialize_new_checksums(&checksums, model_directory)?;
                write_latest_repo_checksum_to_disk(c.as_str(), model_directory)?;
            }
            // Treat an Unknown checksum status as a total failure and escape early
            ChecksumStatus::Unknown => {
                return Err(WhisperRealtimeError::DownloadError(
                    "Failed to get checksum due to network failure".to_string(),
                ))
            }
        }
        // Grab the latest model checksum
        let model_checksum = get_model_checksum(model_directory, key)?.ok_or(
            WhisperRealtimeError::ParameterError(format!("Failed to find mapping for: {}", key)),
        )?;
        Ok(model_checksum)
    }
}

impl Default for DefaultModelType {
    fn default() -> Self {
        DefaultModelType::TinyEn
    }
}

#[cfg(feature = "integrity")]
pub enum Checksum<'a> {
    Sha1(&'a str),
    Sha256(&'a str),
}
