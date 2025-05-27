// Since this is not considered to be within the scope of the core library, these will only be addressed
// should there be a need.
// Roadmap:
// - implement proper unit testing using mocked directories to avoid relying on integration tests
//   and encountering filesystem instability.
// - fuzz-testing the regex.
#[cfg(test)]
#[cfg(feature = "integrity")]
mod model_integrity_tests {
    use std::collections::HashMap;

    use whisper_realtime::whisper::integrity_utils::CHECKSUM_RE;
    use whisper_realtime::whisper::model::{Checksum, DefaultModelType};

    fn delete_model(file_path: &std::path::Path) -> std::io::Result<()> {
        std::fs::remove_file(file_path)
    }

    #[test]
    fn test_regex() {
        let mut checksums = HashMap::new();

        let test_line =
            "| Model               | Disk    | SHA                                        |
| ------------------- | ------- | ------------------------------------------ |
| tiny                | 75 MiB  | `bd577a113a864445d4c299885e0cb97d4ba92b5f` |
| tiny-q5_1           | 31 MiB  | `2827a03e495b1ed3048ef28a6a4620537db4ee51` |
| tiny.en             | 75 MiB  | `c78c86eb1a8faa21b369bcd33207cc90d64ae9df` |
| large-v1            | 2.9 GiB | `b1caaf735c4cc1429223d5a74f0f4d0b9b59a299` |";
        for line in test_line.lines() {
            if let Some(c) = CHECKSUM_RE.captures(line) {
                let model = c[1].to_string();
                let sha = c[2].to_string();
                checksums.insert(model, sha);
            }
        }

        assert!(
            !checksums.is_empty(),
            "Zero length: Regex failed to parse text"
        );
        assert!(
            checksums.contains_key("tiny"),
            "Tiny Key missing: Regex failed to parse text"
        );
        assert!(
            checksums.contains_key("tiny-q5_1"),
            "Tiny q5 Key missing: Regex failed to parse text"
        );

        let actual_tiny = checksums.get("tiny").unwrap();
        let expected_tiny = "bd577a113a864445d4c299885e0cb97d4ba92b5f";
        let actual_tinyq5 = checksums.get("tiny-q5_1").unwrap();
        let expected_tinyq5 = "2827a03e495b1ed3048ef28a6a4620537db4ee51";
        let actual_tiny_en = checksums.get("tiny.en").unwrap();
        let expected_tiny_en = "c78c86eb1a8faa21b369bcd33207cc90d64ae9df";
        let actual_large_v1 = checksums.get("large-v1").unwrap();
        let expected_large_v1 = "b1caaf735c4cc1429223d5a74f0f4d0b9b59a299";

        assert_eq!(
            actual_tiny, expected_tiny,
            "Value failure, tiny. Test: {}, Expected: {}",
            actual_tiny, expected_tiny
        );
        assert_eq!(
            actual_tinyq5, expected_tinyq5,
            "Value failure, tiny-q5_1. Test: {}, Expected: {}",
            actual_tinyq5, expected_tinyq5
        );

        assert_eq!(
            actual_tiny_en, expected_tiny_en,
            "Value failure, tiny.en. Test: {}, Expected: {}",
            actual_tiny_en, expected_tiny_en
        );

        assert_eq!(
            actual_large_v1, expected_large_v1,
            "Value failure, large-v1. Test: {}, Expected: {}",
            actual_large_v1, expected_large_v1,
        );
    }
    #[test]
    fn test_integrity_check_not_downloaded() {
        let model_path = std::env::current_dir().unwrap().join("data").join("models");
        let model_type = DefaultModelType::LargeV1;
        let mut model = model_type.to_model().with_path_prefix(model_path.as_path());
        let file_path = model.file_path();

        if model.exists_in_directory() {
            let deleted = delete_model(file_path.as_path());
            assert!(
                deleted.is_ok(),
                "Failed to delete model: {}",
                deleted.unwrap_err()
            )
        }

        assert!(
            !model.exists_in_directory(),
            "Model still exists in the directory"
        );

        // Get the checksum.
        let checksum = model_type.get_checksum(model_path.as_path(), None);
        assert!(checksum.is_ok(), "{}", checksum.unwrap_err());
        let checksum = checksum.unwrap();
        let c = Checksum::Sha1(checksum.as_str());
        let verified = model.verify_checksum(&c);
        assert!(verified.is_ok(), "File error: {}", verified.unwrap_err());

        assert!(
            !verified.unwrap(),
            "Checksum test passed despite file not existing. Test checksum: {}",
            checksum
        )
    }
    #[test]
    fn test_integrity_check_downloaded() {
        let model_path = std::env::current_dir().unwrap().join("data").join("models");
        let model_type = DefaultModelType::default();
        let mut model = model_type.to_model().with_path_prefix(model_path.as_path());

        assert!(
            model.exists_in_directory(),
            "Failed to find model in data directory"
        );

        // Get the checksum.
        let checksum = model_type.get_checksum(model_path.as_path(), None);
        assert!(checksum.is_ok(), "{}", checksum.unwrap_err());
        let checksum = checksum.unwrap();
        let c = Checksum::Sha1(checksum.as_str());

        let verified = model.verify_checksum(&c);
        assert!(verified.is_ok(), "File error: {}", verified.unwrap_err());

        assert!(
            verified.unwrap(),
            "Checksum test failed despite file existing. Test checksum: {}",
            checksum
        )
    }
}
