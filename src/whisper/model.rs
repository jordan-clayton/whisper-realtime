use crate::utils::errors::RibbleWhisperError;
#[cfg(feature = "integrity")]
use crate::whisper::integrity_utils::{
    checksums_need_updating, get_model_checksum, get_new_checksums, serialize_new_checksums,
    write_latest_repo_checksum_to_disk, ChecksumStatus,
};
#[cfg(feature = "integrity")]
use reqwest::blocking;
#[cfg(feature = "integrity")]
use sha1::Sha1;
#[cfg(feature = "integrity")]
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use strum::{
    AsRefStr, Display, EnumCount, EnumIs, EnumIter, EnumString, FromRepr, IntoStaticStr,
    VariantArray, VariantNames,
};

/// A Type alias representing a model's ID, (e.g. based on a hash)
pub type ModelId = u64;

// TODO: document -> non-concurrent trait impl.
pub trait ModelBank {
    fn model_directory(&self) -> &Path;
    fn insert_model(&mut self, model: Model) -> Result<ModelId, RibbleWhisperError>;
    // TODO: document
    // Depending on the ways in which a model is stored/replaced, it's ModelId may change.
    // Thus, the ModelId should be returned upon a successful update.
    fn replace_model(
        &mut self,
        model_id: ModelId,
        model: Model,
    ) -> Result<ModelId, RibbleWhisperError>;

    // Depending on the ways in which a model is stored/mutated, it's ModelId may change.
    // Thus, the ModelId should be returned upon a successful update.
    fn update_model_parameters(
        &mut self,
        model_id: ModelId,
        name: Option<String>,
        file_name: Option<String>,
    ) -> Result<ModelId, RibbleWhisperError>;

    #[cfg(feature = "integrity")]
    fn verify_checksum(
        &mut self,
        model_id: ModelId,
        checksum: &Checksum,
    ) -> Result<bool, RibbleWhisperError>;
    fn model_exists_in_storage(&self, model_id: ModelId) -> Result<bool, RibbleWhisperError>;
    fn retrieve_model(&self, model_id: ModelId) -> Option<&Model>;
    fn remove_model(&mut self, model_id: ModelId) -> Result<ModelId, RibbleWhisperError>;
}

// TODO: document. Same as ModelBank but imposes interior mutability + concurrency
pub trait ConcurrentModelBank: Send + Sync {
    fn model_directory(&self) -> &Path;
    fn insert_model(&self, model: Model) -> Result<ModelId, RibbleWhisperError>;
    // TODO: document
    // Depending on the ways in which a model is stored/replaced, it's ModelId may change.
    // Thus, the ModelId should be returned upon a successful update.
    fn replace_model(&self, model_id: ModelId, model: Model)
    -> Result<ModelId, RibbleWhisperError>;

    // Depending on the ways in which a model is stored/replaced, it's ModelId may change.
    // Thus, the ModelId should be returned upon a successful update.
    fn update_model_parameters(
        &self,
        model_id: ModelId,
        name: Option<String>,
        file_name: Option<String>,
    ) -> Result<ModelId, RibbleWhisperError>;

    #[cfg(feature = "integrity")]
    fn verify_checksum(
        &self,
        model_id: ModelId,
        checksum: &Checksum,
    ) -> Result<bool, RibbleWhisperError>;
    fn model_exists_in_storage(&self, model_id: ModelId) -> Result<bool, RibbleWhisperError>;
    fn retrieve_model(&self, model_id: ModelId) -> Option<&Model>;
    fn remove_model(&self, model_id: ModelId) -> Result<ModelId, RibbleWhisperError>;
}

// TODO: document -> limited scope API for things that require getting paths for whisper models.
// i.e. the transcribers so that the path doesn't need to be explicitly passed around.
pub trait ModelRetriever {
    fn retrieve_model_path(&self, model_id: ModelId) -> Option<PathBuf>;
}

// TODO: document - this is a very bare-bones implementation, but it's sufficient for getting things running
// This is mainly used for testing, but it can be exposed for use.
pub struct DefaultModelBank {
    model_directory: PathBuf,
    models: HashMap<ModelId, Model>,
}

impl DefaultModelBank {
    pub fn new() -> Self {
        let path = std::env::current_dir().unwrap().join("data").join("models");
        let default_models = [
            DefaultModelType::Tiny,
            DefaultModelType::TinyEn,
            DefaultModelType::Small,
            DefaultModelType::SmallEn,
            DefaultModelType::Medium,
            DefaultModelType::MediumEn,
        ];

        let models = default_models
            .iter()
            .map(|model_type| {
                let mut hasher = DefaultHasher::new();
                model_type.hash(&mut hasher);
                (hasher.finish(), model_type.to_model())
            })
            .collect::<HashMap<ModelId, Model>>();

        Self {
            model_directory: path,
            models,
        }
    }
    pub fn get_model_id(&self, model_type: DefaultModelType) -> ModelId {
        let mut hasher = DefaultHasher::new();
        model_type.hash(&mut hasher);
        hasher.finish()
    }
}

impl ModelBank for DefaultModelBank {
    fn model_directory(&self) -> &Path {
        self.model_directory.as_path()
    }

    fn insert_model(&mut self, model: Model) -> Result<ModelId, RibbleWhisperError> {
        let mut hasher = DefaultHasher::new();
        model.file_name().hash(&mut hasher);
        let model_id = hasher.finish();
        // This returns None if the previous bucket was not occupied, otherwise
        // it returns the previous value.
        self.models.insert(model_id, model);
        Ok(model_id)
    }

    fn replace_model(
        &mut self,
        model_id: ModelId,
        model: Model,
    ) -> Result<ModelId, RibbleWhisperError> {
        self.remove_model(model_id)?;
        self.insert_model(model)
    }

    fn update_model_parameters(
        &mut self,
        _model_id: ModelId,
        _name: Option<String>,
        _file_name: Option<String>,
    ) -> Result<ModelId, RibbleWhisperError> {
        todo!("Implement this for testing purposes if needed.")
    }

    #[cfg(feature = "integrity")]
    fn verify_checksum(
        &mut self,
        model_id: ModelId,
        checksum: &Checksum,
    ) -> Result<bool, RibbleWhisperError> {
        let exists = self.model_exists_in_storage(model_id)?;

        let model = self
            .models
            .get_mut(&model_id)
            .ok_or(RibbleWhisperError::ParameterError(
                "Invalid model key supplied to test bank.".to_string(),
            ))?;

        if !exists {
            model.checksum_verified = false;
            Ok(false)
        } else {
            model.verify_checksum(self.model_directory.as_path(), checksum)
        }
    }

    // Fails on an IO permissions error
    fn model_exists_in_storage(&self, model_id: ModelId) -> Result<bool, RibbleWhisperError> {
        let model = self.models.get(&model_id);
        if model.is_none() {
            return Ok(false);
        }
        let model = model.unwrap();
        let file_path = self.model_directory.join(model.file_name());
        match fs::metadata(&file_path) {
            Ok(m) => Ok(m.is_file()),
            Err(e) => match e.kind() {
                std::io::ErrorKind::NotFound => Ok(false),
                _ => Err(e.into()),
            },
        }
    }

    fn retrieve_model(&self, model_id: ModelId) -> Option<&Model> {
        self.models.get(&model_id)
    }

    fn remove_model(&mut self, model_id: ModelId) -> Result<ModelId, RibbleWhisperError> {
        let model = self
            .models
            .get(&model_id)
            .ok_or(RibbleWhisperError::ParameterError(
                "Invalid model id supplied to test bank".to_string(),
            ))?;

        if self.model_exists_in_storage(model_id)? {
            let file_path = self.model_directory.join(model.file_name());
            fs::remove_file(&file_path)?;
        }
        self.models.remove(&model_id);
        Ok(model_id)
    }
}

impl ModelRetriever for DefaultModelBank {
    fn retrieve_model_path(&self, model_id: ModelId) -> Option<PathBuf> {
        self.models
            .get(&model_id)
            .and_then(|model| Some(self.model_directory.join(model.file_name())))
    }
}

/// Encapsulates a compatible whisper model
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone)]
pub struct Model {
    name: String,
    file_name: String,
    #[cfg(feature = "integrity")]
    checksum_verified: bool,
}

impl Model {
    pub fn new() -> Self {
        Self {
            name: Default::default(),
            file_name: Default::default(),
            #[cfg(feature = "integrity")]
            checksum_verified: false,
        }
    }

    /// Constructs a model with the provided (user-facing) name, filename, and path prefix (model directory).
    pub fn new_with_parameters(name: String, file_name: String) -> Self {
        Self::new().with_name(name).with_file_name(file_name)
    }

    /// Sets the model's user-facing name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    /// Sets the model's filename.
    /// NOTE: if this file_name gets changed (to point to a different file), it can no longer be
    /// verified.
    /// NOTE: depending on
    pub fn with_file_name(mut self, file_name: String) -> Self {
        self.file_name = file_name;
        #[cfg(feature = "integrity")]
        {
            self.checksum_verified = false;
        }
        self
    }

    /// Gets the model's user-friendly name
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Gets the model's filename
    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    /// Gets the model's (checksum) verified status.
    /// Requires the integrity feature flag to be set.
    #[cfg(feature = "integrity")]
    pub fn checksum_verified(&self) -> bool {
        self.checksum_verified
    }

    // NOTE:
    // It should be the responsibility of the Model Bank to handle and mediate all changes to models.
    // This is reflected in the interface of ModelBank.

    // However, it is the case that all prior self-contained model integrity mechanisms still work,
    // they are still exposed and can be used for the interim.

    #[cfg(feature = "integrity")]
    fn compare_sha256(
        &self,
        checksum: &str,
        model_directory: &Path,
    ) -> Result<bool, RibbleWhisperError> {
        // Compute the checksum on the file
        let mut file = fs::File::open(model_directory.join(self.file_name()))?;
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)?;
        let hash = hasher.finalize();

        // Convert to byte string and compare checksums
        let byte_string = format!("{:x}", hash);
        let byte_str = byte_string.as_str();
        Ok(checksum.to_lowercase() == byte_str)
    }

    // TODO: -> move to model bank, args: id + checksum
    #[cfg(feature = "integrity")]
    fn compare_sha1(
        &self,
        checksum: &str,
        model_directory: &Path,
    ) -> Result<bool, RibbleWhisperError> {
        // TODO: -> move to model bank
        // Compute the checksum on the file
        let mut file = fs::File::open(model_directory.join(self.file_name()))?;
        let mut hasher = Sha1::new();
        std::io::copy(&mut file, &mut hasher)?;
        let hash = hasher.finalize();

        // Convert to byte string and compare checksums
        let byte_string = format!("{:x}", hash);
        let byte_str = byte_string.as_str();
        Ok(checksum.to_lowercase() == byte_str)
    }

    /// # Deprecated: These will eventually become utility functions which can be used in a [ModelBank] or similar to handle checksums.
    /// For verifying file integrity against a user-provided checksum.
    /// Since a Model file can come from anywhere, responsibility falls upon the user to ensure
    /// integrity. This method provides a mechanism to carry out that responsibility.
    /// # Arguments:
    /// * checksum (Sha1/2)
    /// # Returns
    /// * Ok(matches) on success, Err on I/O error, or failure to compute the checksum.
    #[cfg(feature = "integrity")]
    pub fn verify_checksum(
        &mut self,
        model_directory: &Path,
        checksum: &Checksum,
    ) -> Result<bool, RibbleWhisperError> {
        let is_equal = match checksum {
            Checksum::Sha1(c) => self.compare_sha1(c, model_directory),
            Checksum::Sha256(c) => self.compare_sha256(c, model_directory),
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

/// Encapsulates a series of base models available for download and use with WhisperRealtime
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(
    Copy,
    Clone,
    Debug,
    PartialOrd,
    PartialEq,
    Ord,
    Eq,
    Hash,
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

// TODO: rethink this.
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

    /// Constructs a model object and sets the path prefix to the current working directory
    pub fn to_model(&self) -> Model {
        Model::new()
            .with_name(self.to_string())
            .with_file_name(self.to_file_name().to_string())
    }

    /// Canonicalizes a download url to retrieve the model from huggingface.
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

    /// Makes a blocking request to check the sha1 checksums for each of the provided default models
    /// If the cached checksums are out of date, this will download and update the cache.
    /// This should be called on a separate thread if you do not want the program to hang.
    /// Async requests are not provided at this time and will only be implemented should the demand
    /// be present.
    /// Requires the integrity feature flag to be set.
    /// # Arguments:
    /// * records_directory: The path to the stored model checksums
    /// * client: An optional reference to a (blocking) Reqwest client. Supply None if client-reuse is not a concern.
    /// # Returns:
    /// * Ok(checksum) on successfully retrieving the cached checksum
    /// * Err on network failure, or failure to retrieve the model checksum
    #[cfg(feature = "integrity")]
    pub fn get_checksum(
        &self,
        records_directory: &Path,
        client: Option<&blocking::Client>,
    ) -> Result<String, RibbleWhisperError> {
        let key = self.to_sha_key();
        let needs_updating = checksums_need_updating(records_directory, client);

        // Handle the current status of the stored repository checksum
        match needs_updating {
            ChecksumStatus::UpToDate(_) => {}
            ChecksumStatus::NeedsUpdating(c) => {
                let checksums = get_new_checksums(client)?;
                serialize_new_checksums(&checksums, records_directory)?;
                write_latest_repo_checksum_to_disk(c.as_str(), records_directory)?;
            }
            // Treat an Unknown checksum status as a total failure and escape early
            ChecksumStatus::Unknown => {
                return Err(RibbleWhisperError::DownloadError(
                    "Failed to get checksum due to network failure".to_string(),
                ));
            }
        }
        // Grab the latest model checksum
        let model_checksum = get_model_checksum(records_directory, key)?.ok_or(
            RibbleWhisperError::ParameterError(format!("Failed to find mapping for: {}", key)),
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
