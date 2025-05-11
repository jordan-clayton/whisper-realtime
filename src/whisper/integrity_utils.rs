use std::{fs, io};
use std::collections::HashMap;
use std::path::Path;

use lazy_static::lazy_static;
use regex::Regex;
use reqwest::blocking;

use crate::utils::errors::WhisperRealtimeError;

// Stores a string literal of the most up-to-date commit hash of whisper.cpp's huggingface repository
const LATEST_CHECKSUM: &str = "latest_checksum";
// Stores a pretty-print JSON mapping models to their provided sha1
const CHECKSUM_FILE: &str = "checksum.json";
const REPO_URL: &str = "https://huggingface.co/api/models/ggerganov/whisper.cpp";
// As of this implementation, the checksums are made available in README.md
const README_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/raw/main/README.md";

// TODO: migrate to once_cell
lazy_static! {
    // (Very specific) Regex: Used to extract the required fields from README.md
    // |whitespace*(non-whitespace+)whitespace*|, First field: Model name, first capture group
    // |whitespace*[parse up to pipe]whitespace*|, Second field: File size, no capture
    // |whitespace*(sha1 digest)whitespace*|, Optionally consumes ` literal, Third Field: Sha1, second capture group
    pub static ref CHECKSUM_RE: Regex =
        Regex::new(r"\|\s*(\S+)\s*\|\s*[^|]+?\s*\|\s*`?([a-fA-F0-9]{40})`?\s*\|")
            .expect("Failed to build checksum RE");
}

pub enum ChecksumStatus {
    UpToDate(String),
    NeedsUpdating(String),
    // Likely due to a network error.
    Unknown,
}

impl ChecksumStatus {
    pub fn is_known(&self) -> bool {
        matches!(
            self,
            ChecksumStatus::UpToDate(_) | ChecksumStatus::NeedsUpdating(_)
        )
    }
}

/// This grabs the latest commit checksum from the huggingface repository
/// and compares it with the one stored in the models directory.
/// The ChecksumStatus will hold a copy of the latest checksum sha string if
/// it has known status.
pub fn checksums_need_updating(
    models_directory: &Path,
    client: Option<&blocking::Client>,
) -> ChecksumStatus {
    let latest_checksum = match get_latest_repo_checksum(client) {
        Ok(c) => c,
        Err(_) => return ChecksumStatus::Unknown,
    };

    let path = models_directory.join(LATEST_CHECKSUM);
    if !path.is_file() {
        return ChecksumStatus::NeedsUpdating(latest_checksum);
    }

    // If this returns an (IO) error, this means the file does not exist and should thus be updated.
    let stored_checksum = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => return ChecksumStatus::NeedsUpdating(latest_checksum),
    };

    let needs_updating = latest_checksum != stored_checksum;

    if needs_updating {
        ChecksumStatus::NeedsUpdating(latest_checksum)
    } else {
        ChecksumStatus::UpToDate(latest_checksum)
    }
}

/// Requests and parses README.md and returns a map of model names to sha256 checksums.
/// This does not yet filter out all non-default model types.
pub fn get_new_checksums(
    client: Option<&blocking::Client>,
) -> Result<HashMap<String, String>, WhisperRealtimeError> {
    let response = match client {
        None => blocking::get(README_URL),
        Some(r_client) => r_client.get(README_URL).send(),
    }?;

    let body = response.text()?;
    let mut checksums = HashMap::new();

    // Matches lines by RegEx to capture a suitable key/value pairing
    for line in body.lines() {
        if let Some(c) = CHECKSUM_RE.captures(line) {
            let model = c[1].to_string();
            let sha = c[2].to_string();
            checksums.insert(model, sha);
        }
    }
    Ok(checksums)
}

/// Serializes a checksum map into a pretty JSON file.
pub fn serialize_new_checksums(
    checksums: &HashMap<String, String>,
    model_directory: &Path,
) -> Result<(), WhisperRealtimeError> {
    let path = model_directory.join(CHECKSUM_FILE);
    let outfile = fs::File::create(path)?;
    let writer = io::BufWriter::new(outfile);

    // Write out pretty JSON
    serde_json::to_writer_pretty(writer, &checksums)?;
    Ok(())
}

/// Stores the current whisper.cpp huggingface checksum to disk.
pub fn write_latest_repo_checksum_to_disk(
    new_checksum: &str,
    model_directory: &Path,
) -> Result<(), WhisperRealtimeError> {
    let outfile_path = model_directory.join(LATEST_CHECKSUM);
    fs::write(outfile_path, new_checksum)?;
    Ok(())
}

/// Retrieves a model's checksum from checksum.json (if it exists).
/// This will return an error on an IO failure, or on failure to parse the json string.
pub fn get_model_checksum(
    model_directory: &Path,
    model_key: &str,
) -> Result<Option<String>, WhisperRealtimeError> {
    let path = model_directory.join(CHECKSUM_FILE);
    let in_file = fs::File::open(path)?;
    let reader = io::BufReader::new(in_file);
    let json: HashMap<String, String> = serde_json::from_reader(reader)?;
    // In-case the sha string contains string-literal quotation marks, strip them.
    let checksum = json.get(model_key).map(|v| v.to_string().replace("\"", ""));
    Ok(checksum)
}

/// Grabs the latest whisper.cpp commit Sha from HuggingFace's model API to be used as a mechanism
/// for determining when to update the model checksums.
fn get_latest_repo_checksum(
    client: Option<&blocking::Client>,
) -> Result<String, WhisperRealtimeError> {
    let json: serde_json::Value = match client {
        None => blocking::get(REPO_URL),
        Some(r_client) => r_client.get(REPO_URL).send(),
    }?
    .json()?;

    let latest_checksum = json["sha"]
        .as_str()
        .ok_or(WhisperRealtimeError::DownloadError(
            "Response JSON missing expecting 'sha'".to_owned(),
        ))?;
    Ok(latest_checksum.to_owned())
}
