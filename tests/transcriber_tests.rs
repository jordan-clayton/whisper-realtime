mod transcriber_tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, LazyLock};
    use std::thread::scope;

    use ribble_whisper::audio::audio_ring_buffer::AudioRingBuffer;
    use ribble_whisper::audio::loading::load_normalized_audio_file;
    use ribble_whisper::audio::WhisperAudioSample;
    use ribble_whisper::transcriber::realtime_transcriber::RealtimeTranscriberBuilder;
    use ribble_whisper::transcriber::vad::{Silero, VAD};
    use ribble_whisper::transcriber::{
        redirect_whisper_logging_to_hooks, Transcriber, WhisperOutput,
    };
    use ribble_whisper::utils;
    use ribble_whisper::utils::constants;
    use ribble_whisper::whisper::configs::WhisperRealtimeConfigs;
    use ribble_whisper::whisper::model::DefaultModelType;

    // Prepare an audio sample with a known output to try and make conditions as replicable as
    // possible
    static AUDIO_SAMPLE: LazyLock<Box<[f32]>> = LazyLock::new(|| {
        let sample = load_normalized_audio_file(
            "tests/audio_files/128896__joshenanigans__sentence-recitation.wav",
            None::<fn(usize)>,
        )
        .expect("Test audio should load without issue.");
        let audio = match sample {
            WhisperAudioSample::I16(_) => unreachable!(),
            WhisperAudioSample::F32(audio) => audio,
        };

        // Extract only the voiced frames to try and prevent accidental early silence.
        // Use Silero for accuracy
        let mut vad =
            Silero::try_new_whisper_offline_default().expect("Silero should build without issues");
        let out = vad.extract_voiced_frames(&audio);
        assert!(!out.is_empty());
        out
    });

    #[test]
    fn test_realtime_transcriber() {
        let proj_dir = std::env::current_dir().unwrap().join("data").join("models");
        let model_type = DefaultModelType::Medium;

        let model = model_type.to_model_with_path_prefix(proj_dir.as_path());

        assert!(
            model.exists_in_directory(),
            "Whisper medium has not been downloaded."
        );

        let configs = WhisperRealtimeConfigs::default()
            .with_n_threads(8)
            .with_model(model.clone())
            // Also, optionally set flash attention.
            // Generally keep this on for a performance gain.
            .set_flash_attention(true);

        let (text_sender, text_receiver) = utils::get_channel(constants::INPUT_BUFFER_CAPACITY);
        let mut vad = Silero::try_new_whisper_realtime_default()
            .expect("Silero VAD expected to build without issue");
        let sample_up_to =
            ((configs.vad_sample_len() as f64 / 1000f64) * constants::WHISPER_SAMPLE_RATE) as usize;

        // Silero needs to be warmed up before it can predictably detect audio.
        let mut detected_audio = vad.voice_detected(&AUDIO_SAMPLE[..sample_up_to]);
        if !detected_audio {
            detected_audio = vad.voice_detected(&AUDIO_SAMPLE[..sample_up_to]);
        }
        assert!(detected_audio, "Failed to detect audio after warming up");

        let audio_ring_buffer = AudioRingBuffer::default();
        let (mut transcriber, handle) = RealtimeTranscriberBuilder::<Silero>::new()
            .with_configs(configs.clone())
            .with_audio_buffer(&audio_ring_buffer)
            .with_output_sender(text_sender)
            .with_voice_activity_detector(vad)
            .build()
            .expect("RealtimeTranscriber expected to build without issues.");

        // The returned string should resemble this one with fewer than roughly 2-3 edits.
        let expected_offline_transcription =
            "Mary has many dreams but can't touch Tennessee by way of flight";

        // Prevent logging to stderr
        redirect_whisper_logging_to_hooks();
        // Break the audio sample into chunks of size constants::AUDIO_CHUNK_SIZE to simulate default
        // audio input
        let chunks = AUDIO_SAMPLE.chunks(constants::AUDIO_BUFFER_SIZE as usize);
        let run_transcription = Arc::new(AtomicBool::new(true));

        let transcribed = scope(|s| {
            let t_thread_run_transcription = Arc::clone(&run_transcription);
            let d_thread_run_transcription = Arc::clone(&run_transcription);
            let _a_thread = s.spawn(move || {
                // Clear the audio_ring_buffer
                audio_ring_buffer.clear();

                // Wait until the transcriber is ready to simulate actual speech
                while !handle.ready() {}
                assert!(chunks.len() > 0);
                for chunk in chunks {
                    audio_ring_buffer.push_audio(chunk)
                }
            });
            let t_thread = s.spawn(move || transcriber.process_audio(t_thread_run_transcription));
            // Simple thread to just drain the audio - this will sleep until it receives output from
            // the transcriber
            let _d_thread = s.spawn(move || {
                let offline_output_length = expected_offline_transcription.len();
                let epsilon = 5usize;

                // repeatedly drain the audio buffer to prevent a memory panic, and also set the
                // exit condition
                while let Ok(out) = text_receiver.recv() {
                    let message = match out {
                        WhisperOutput::ConfirmedTranscription(message) => message,
                        WhisperOutput::CurrentSegments(segments) => segments.join("").to_string(),
                        WhisperOutput::ControlPhrase(_) => "".to_string(),
                    };
                    let current_len = message.len();
                    if current_len > offline_output_length - epsilon {
                        d_thread_run_transcription.store(false, Ordering::Release);
                    }
                }
            });
            // Return the output of the transcriber thread.
            t_thread.join()
        });
        assert!(
            transcribed.is_ok(),
            "Transcription thread panicked: {:?}.",
            transcribed.unwrap_err()
        );

        let transcription = transcribed.unwrap();
        assert!(
            transcription.is_ok(),
            "Transcription returned an error: {}",
            transcription.unwrap_err()
        );

        let transcription = transcription.unwrap();

        // At most, there should only be around 2 edits (inserted punctuation)
        // Anymore and there's something up with the transcription, or the model is hallucinating
        let max_edit_distance = 2usize;

        let edit_distance = strsim::levenshtein(&transcription, expected_offline_transcription);

        assert!(
            edit_distance <= max_edit_distance,
            "Failed to output reasonable match.\nEdit distance: {}, Max distance: {}\n, Output: {}, Expected: {}",
            edit_distance,
            max_edit_distance,
            transcription,
            expected_offline_transcription
        )
    }
}
