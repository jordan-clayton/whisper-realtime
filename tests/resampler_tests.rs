// TODO: refactor this once the transcriber API is cleaned up
#[cfg(test)]
#[cfg(feature = "resampler")]
mod resampler_test {
    // Note: You will need to supply your own audio file and modify the tests accordingly.
    // These tests are primarily to be used internally to ensure the resampling API is correct,
    // as such, audio files have not been supplied for use.

    use std::sync::{Arc, Mutex};
    use std::sync::atomic::AtomicBool;

    use hound::{SampleFormat, WavSpec, WavWriter};

    use whisper_realtime::audio::loading::load_normalized_audio_file;
    use whisper_realtime::audio::resampler::file_needs_normalizing;
    use whisper_realtime::transcriber::static_transcriber::{
        StaticTranscriber, SupportedAudioSample, SupportedChannels,
    };
    use whisper_realtime::transcriber::traits::Transcriber;
    use whisper_realtime::utils::constants;
    use whisper_realtime::whisper::configs::Configs;
    use whisper_realtime::whisper::model::{DefaultModelType, OldModel};

    // Tests the resampling from a file path, which will also implicitly using the track handle
    #[test]
    fn test_needs_resampling() {
        let needs_normalizing = file_needs_normalizing(
            "tests/audio_files/128896__joshenanigans__sentence-recitation.wav",
        )
        .unwrap();

        assert!(needs_normalizing);
    }

    #[test]
    fn test_needs_resampling_mp3() {
        let needs_normalizing = file_needs_normalizing("tests/audio_files/test_mp3.mp3").unwrap();

        assert!(needs_normalizing);
    }
    // Loads some audio at 44.1 khz, resamples it to 16kHz, then writes it to an output file.
    // The audio will need to be checked manually to ensure the integrity
    #[test]
    #[ignore]
    fn test_resample() {
        let audio = load_normalized_audio_file(
            "tests/audio_files/128896__joshenanigans__sentence-recitation.wav",
            None::<fn(usize)>,
        )
        .unwrap();
        let wav_spec = WavSpec {
            bits_per_sample: 32,
            channels: 1,
            sample_format: SampleFormat::Float,
            sample_rate: constants::WHISPER_SAMPLE_RATE as u32,
        };

        let mut writer = WavWriter::create("tests/audio_files/resampled.wav", wav_spec).unwrap();
        let SupportedAudioSample::F32(samples) = audio else {
            unreachable!()
        };
        for sample in samples.iter() {
            writer.write_sample(*sample).unwrap();
        }
        writer.finalize().unwrap();
    }

    // Loads some audio at 44.1 khz, resamples it to 16kHz, then sends the audio to whisper to
    // transcribe. The audio is simple enough such that the transcription should be 1:1
    // A successful transcription means the resampling is correct.
    #[test]
    #[ignore]
    fn test_resample_whisper() {
        let expected_transcription =
            "Mary has many dreams but can't touch Tennessee by way of flight.";

        let audio = load_normalized_audio_file(
            "tests/audio_files/128896__joshenanigans__sentence-recitation.wav",
            None::<fn(usize)>,
        )
        .unwrap();
        let t_audio = Arc::new(Mutex::new(audio));

        let mut configs = Configs::default();
        // This presumes a model is already downloaded. Handle accordingly.
        configs.model = DefaultModelType::MediumEn;
        let c_configs = Arc::new(configs);

        let mut static_transcriber = StaticTranscriber::new_with_configs(
            t_audio.clone(),
            None,
            c_configs.clone(),
            SupportedChannels::MONO,
        );
        // Set up whisper
        let mut proj_dir = std::env::current_dir().unwrap();
        proj_dir.push("data");
        let model =
            OldModel::new_with_type_and_dir(DefaultModelType::MediumEn, proj_dir.to_path_buf());

        let whisper_ctx_params = whisper_rs::WhisperContextParameters::default();
        let ctx = whisper_rs::WhisperContext::new_with_params(
            model.file_path().to_str().unwrap(),
            whisper_ctx_params,
        )
        .expect("Failed to load model.");

        let mut state = ctx.create_state().expect("Failed to create state");
        let run_transcription = Arc::new(AtomicBool::new(true));
        // Transcribe the audio
        let transcription =
            static_transcriber.process_audio(&mut state, run_transcription, None::<fn(i32)>);

        assert_eq!(transcription, expected_transcription);
    }

    #[test]
    #[ignore]
    fn test_resample_whisper_from_mp3() {
        let expected_transcription =
            "Mary has many dreams but can't touch Tennessee by way of flight.";

        let audio = load_normalized_audio_file("tests/audio_files/test_mp3.mp3", None::<fn(usize)>)
            .unwrap();
        let t_audio = Arc::new(Mutex::new(audio));

        let mut configs = Configs::default();
        // This presumes a model is already downloaded. Handle accordingly.
        configs.model = DefaultModelType::MediumEn;
        let c_configs = Arc::new(configs);

        let mut static_transcriber = StaticTranscriber::new_with_configs(
            t_audio.clone(),
            None,
            c_configs.clone(),
            SupportedChannels::MONO,
        );
        // Set up whisper
        let mut proj_dir = std::env::current_dir().unwrap();
        proj_dir.push("data");
        let model =
            OldModel::new_with_type_and_dir(DefaultModelType::MediumEn, proj_dir.to_path_buf());

        let whisper_ctx_params = whisper_rs::WhisperContextParameters::default();
        let ctx = whisper_rs::WhisperContext::new_with_params(
            model.file_path().to_str().unwrap(),
            whisper_ctx_params,
        )
        .expect("Failed to load model.");

        let mut state = ctx.create_state().expect("Failed to create state");
        let run_transcription = Arc::new(AtomicBool::new(true));
        // Transcribe the audio
        let transcription =
            static_transcriber.process_audio(&mut state, run_transcription, None::<fn(i32)>);

        assert_eq!(transcription, expected_transcription);
    }
}
