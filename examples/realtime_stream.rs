use std::io::{stdout, Write};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::scope;

use indicatif::{ProgressBar, ProgressStyle};
use parking_lot::Mutex;

use ribble_whisper::audio::audio_ring_buffer::AudioRingBuffer;
use ribble_whisper::audio::microphone::{AudioBackend, FanoutMicCapture, MicCapture};
use ribble_whisper::audio::recorder::UseArc;
use ribble_whisper::audio::{AudioChannelConfiguration, WhisperAudioSample};
use ribble_whisper::downloader::downloaders::sync_download_request;
#[cfg(feature = "downloader")]
use ribble_whisper::downloader::SyncDownload;
use ribble_whisper::transcriber::offline_transcriber::OfflineTranscriberBuilder;
use ribble_whisper::transcriber::realtime_transcriber::RealtimeTranscriberBuilder;
use ribble_whisper::transcriber::vad::{Silero, WebRtc};
use ribble_whisper::transcriber::{redirect_whisper_logging_to_hooks, Transcriber};
use ribble_whisper::transcriber::{
    CallbackTranscriber, WhisperCallbacks, WhisperControlPhrase, WhisperOutput,
};
use ribble_whisper::utils;
use ribble_whisper::utils::callback::{ProgressCallback, StaticProgressCallback};
use ribble_whisper::utils::constants;
use ribble_whisper::whisper::configs::WhisperRealtimeConfigs;
use ribble_whisper::whisper::model;
use ribble_whisper::whisper::model::Model;

fn main() {
    // Get a model. If not already downloaded, this will also download the model.
    let model = prepare_model();
    // Set the number of threads according to your hardware.
    // If you can allocate around 7-8, do so as this tends to be more performant.
    let configs = WhisperRealtimeConfigs::default()
        .with_n_threads(8)
        .with_model(model)
        // Also, optionally set flash attention.
        // (Generally keep this on for a performance gain with gpu processing).
        .set_flash_attention(true);

    let audio_ring_buffer = AudioRingBuffer::<f32>::default();
    let (audio_sender, audio_receiver) =
        utils::get_channel::<Arc<[f32]>>(constants::INPUT_BUFFER_CAPACITY);
    let (text_sender, text_receiver) = utils::get_channel(constants::INPUT_BUFFER_CAPACITY);

    // Note: Any VAD<T> + Send can be used.
    let vad = Silero::try_new_whisper_realtime_default()
        .expect("Earshot realtime VAD expected to build without issue");

    // Transcriber
    let (mut transcriber, transcriber_handle) = RealtimeTranscriberBuilder::<WebRtc>::new()
        .with_configs(configs.clone())
        .with_audio_buffer(&audio_ring_buffer)
        .with_output_sender(text_sender)
        .with_voice_activity_detector(vad)
        .build()
        .expect("RealtimeTranscriber expected to build without issues.");

    // Optional store + re-transcription.
    // Get input for stdin.
    print!("Would you like to store audio and re-transcribe once realtime has finished? y/n: ");
    stdout().flush().unwrap();

    let mut confirm_re_transcribe = String::new();

    let offline_audio_buffer: Arc<Mutex<Option<Vec<f32>>>> = Arc::new(Mutex::new(None));
    if let Ok(_) = std::io::stdin().read_line(&mut confirm_re_transcribe) {
        let confirm = confirm_re_transcribe.trim().to_lowercase();
        if "y" == confirm {
            println!("Confirmed. Recording audio.\n");
            let mut guard = offline_audio_buffer.lock();
            *guard = Some(vec![])
        }
    }

    let t_offline_audio_buffer = Arc::clone(&offline_audio_buffer);

    let run_transcription = Arc::new(AtomicBool::new(true));
    let c_handler_run_transcription = Arc::clone(&run_transcription);

    // Use CTRL-C to stop the transcription.
    ctrlc::set_handler(move || {
        println!("Interrupt received");
        c_handler_run_transcription.store(false, Ordering::SeqCst);
    })
    .expect("failed to set SIGINT handler");

    // Set up the Audio Backend.
    let audio_backend = AudioBackend::new().expect("Audio backend should build without issue");
    let mic: FanoutMicCapture<f32, UseArc> = audio_backend
        .build_whisper_fanout_default(audio_sender)
        .expect("Mic handle expected to build without issue.");

    let transcriber_thread = scope(|s| {
        let a_thread_run_transcription = Arc::clone(&run_transcription);
        let t_thread_run_transcription = Arc::clone(&run_transcription);
        let p_thread_run_transcription = Arc::clone(&run_transcription);

        // Block Whisper.cpp from logging to stdout/stderr and instead redirect to an optional logging hook.
        redirect_whisper_logging_to_hooks();
        mic.play();
        // Read data from the AudioBackend and write to the ringbuffer + (optional) static audio
        let _audio_thread = s.spawn(move || {
            // Just hog the mutex because there's only one writer.
            let mut offline_buffer = t_offline_audio_buffer.lock();
            while a_thread_run_transcription.load(Ordering::Acquire) {
                match audio_receiver.recv() {
                    Ok(audio_data) => {
                        // If the transcriber is not yet loaded, just consume the audio
                        if !transcriber_handle.ready() {
                            continue;
                        }
                        // Otherwise, fan out to the ring-buffer (for transcrption) and the optional
                        // offline buffer
                        audio_ring_buffer.push_audio(&audio_data);
                        if let Some(buffer) = offline_buffer.as_mut() {
                            buffer.extend_from_slice(&audio_data);
                        }
                    }
                    Err(_) => {
                        a_thread_run_transcription.store(false, Ordering::Release);
                    }
                }
            }

            // Drop the guard to release the mutex.
            drop(offline_buffer)
        });

        // Move the transcriber off to a thread to handle processing audio
        let transcription_thread =
            s.spawn(move || transcriber.process_audio(t_thread_run_transcription));

        // Update the UI with the newly transcribed data
        let print_thread = s.spawn(move || {
            let mut latest_control_message = WhisperControlPhrase::GettingReady;
            let mut latest_confirmed = String::default();
            let mut latest_segments: Vec<String> = vec![];
            while p_thread_run_transcription.load(Ordering::Acquire) {
                match text_receiver.recv() {
                    Ok(output) => match output {
                        // This is the most up-to-date full string transcription
                        WhisperOutput::ConfirmedTranscription(text) => {
                            latest_confirmed = text;
                        }
                        WhisperOutput::CurrentSegments(segments) => latest_segments = segments,

                        WhisperOutput::ControlPhrase(message) => {
                            latest_control_message = message;
                        }
                    },
                    Err(_) => p_thread_run_transcription.store(false, Ordering::Release),
                }

                clear_stdout();
                println!("Latest Control Message: {}\n", latest_control_message);
                println!("Transcription:\n");
                // Print the remaining current working set of segments.
                print!("{}", latest_confirmed);
                // Print the remaining current working set of segments.
                for segment in latest_segments.iter() {
                    print!("{}", segment);
                }

                stdout().flush().expect("Stdout should clear normally.");
            }

            // Drain the last of the segments into the final string
            let last_text = latest_segments.drain(..);
            latest_confirmed.extend(last_text);
            latest_confirmed
        });

        // Send both outputs for comparison in stdout.
        (transcription_thread.join(), print_thread.join())
    });

    mic.pause();

    let transcription = transcriber_thread
        .0
        .expect("Transcription thread should not panic.")
        .expect("Transcription should run properly.");
    let rt_transcription = transcriber_thread
        .1
        .expect("Print thread should not panic.");

    // Comparison
    // clear_stdout();
    println!("\nFinal Transcription (print thread):");
    println!("{}", &rt_transcription);

    println!();

    println!("Final Transcription (returned):");
    println!("{}", &transcription);

    // Offline audio (re) transcription:
    if let Some(buffer) = offline_audio_buffer.lock().take() {
        // Use silero to prune out the silence
        let vad = Silero::try_new_whisper_offline_default()
            .expect("Silero expected to build with whisper defaults");
        // Consume the configs into whisper v2 (or reuse)
        let s_configs = configs.into_whisper_v2_configs();
        let mut offline_transcriber = OfflineTranscriberBuilder::<Silero>::new()
            .with_configs(s_configs)
            .with_audio(WhisperAudioSample::F32(buffer.into_boxed_slice()))
            .with_channel_configurations(AudioChannelConfiguration::Mono)
            .with_voice_activity_detector(vad)
            .build()
            .expect("OfflineTranscriber expected to build with no issues.");

        // Initiate a progress bar
        let pb = ProgressBar::new(100);
        pb.set_style(ProgressStyle::default_bar()
            .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {percent} ({eta})").unwrap()
            .progress_chars("#>-")
        );

        pb.set_message("Transcription Progress: ");
        let pb_c = pb.clone();

        // Reset run_transcription
        let run_offline_transcription = Arc::clone(&run_transcription);
        run_offline_transcription.store(true, Ordering::Release);
        let progress_closure = move |p| {
            pb_c.set_position(p as u64);
        };
        let static_progress_callback = StaticProgressCallback::new(progress_closure);

        let callbacks = WhisperCallbacks {
            progress: Some(static_progress_callback),
        };

        let transcription =
            offline_transcriber.process_with_callbacks(run_offline_transcription, callbacks);
        pb.finish_and_clear();
        println!("\nOffline re-transcription:");
        if let Err(e) = &transcription {
            eprintln!("{}", e);
            return;
        }
        println!("{}", &transcription.unwrap());
    };
}

// Downloads the model if it doesn't exist within CWD/data/models.
// If the path does not exist already, this will create the full path upon downloading the model.
fn prepare_model() -> Model {
    let proj_dir = std::env::current_dir().unwrap().join("data").join("models");

    // GPU acceleration is currently required to run larger models in realtime.
    let model_type = if cfg!(feature = "_gpu") {
        model::DefaultModelType::Medium
    } else {
        model::DefaultModelType::Small
    };

    let model = model_type.to_model_with_path_prefix(proj_dir.as_path());

    if !model.exists_in_directory() {
        println!("Downloading model:");
        stdout().flush().unwrap();

        // Downloading.
        let url = model_type.url();

        let sync_downloader = sync_download_request(url.as_str());
        if let Err(e) = sync_downloader.as_ref() {
            eprintln!("{}", e);
        }

        let sync_downloader = sync_downloader.unwrap();

        // Initiate a progress bar.
        let pb = ProgressBar::new(sync_downloader.total_size() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})").unwrap()
            .progress_chars("#>-")
        );
        pb.set_message(format!("Downloading {}", url));

        let pb_c = pb.clone();

        // Set the progress callback
        let progress_callback_closure = move |n: usize| {
            pb_c.set_position(n as u64);
        };

        let progress_callback = ProgressCallback::new(progress_callback_closure);
        let mut sync_downloader = sync_downloader.with_progress_callback(progress_callback);

        let download = sync_downloader.download(model.file_path().as_path(), model.file_name());
        assert!(download.is_ok());
        assert!(model.exists_in_directory());
    }
    model
}

fn clear_stdout() {
    Command::new("cls")
        .status()
        .or_else(|_| Command::new("clear").status())
        .expect("Failed to clear stdout");
}
