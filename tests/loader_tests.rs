mod common;
#[cfg(test)]
mod loader_tests {
    use ribble_whisper::audio::loading::{
        audio_file_num_frames, load_audio_file, load_normalized_audio_file,
    };

    #[test]
    fn test_num_frames() {
        let path = "tests/audio_files/128896__joshenanigans__sentence-recitation.wav";

        let expected_n_frames = audio_file_num_frames(path);
        assert!(
            expected_n_frames.is_ok(),
            "Failed to get num_frames: {}",
            expected_n_frames.unwrap_err()
        );

        let expected_n_frames = expected_n_frames.unwrap() as usize;

        // Set up an accumulator.
        let mut n_frames = 0;
        let mut n_frames_normalized = 0;

        let progress_callback = Some(|decoded| {
            n_frames += decoded;
        });
        let progress_callback_normalized = Some(|decoded| {
            n_frames_normalized += decoded;
        });

        let loaded_audio = load_audio_file(path, progress_callback);
        let loaded_audio_normalized =
            load_normalized_audio_file(path, progress_callback_normalized);

        assert!(
            loaded_audio.is_ok(),
            "Audio loading (non-normalized) failed: {}",
            loaded_audio.err().unwrap()
        );

        assert!(
            loaded_audio_normalized.is_ok(),
            "Audio loading (normalized) failed: {}",
            loaded_audio_normalized.err().unwrap()
        );

        assert_eq!(
            expected_n_frames, n_frames,
            "Wrong number of frames. Expected: {}, n_frames: {}",
            expected_n_frames, n_frames
        );
        assert_eq!(
            expected_n_frames, n_frames_normalized,
            "Wrong number of frames. Expected: {}, n_frames_normalized: {}",
            expected_n_frames, n_frames_normalized
        );
    }
}
