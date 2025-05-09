use std::io::{stdout, Write};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(feature = "crossbeam"))]
use std::sync::mpsc::{channel, sync_channel};
use std::thread::scope;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};
use whisper_rs::install_logging_hooks;

use whisper_realtime::audio::audio_ring_buffer::AudioRingBuffer;
use whisper_realtime::audio::microphone::AudioBackend;
#[cfg(feature = "downloader")]
use whisper_realtime::downloader::request;
use whisper_realtime::downloader::traits::SyncDownload;
use whisper_realtime::transcriber::{realtime_transcriber, static_transcriber};
use whisper_realtime::transcriber::traits::Transcriber;
use whisper_realtime::utils::callback::ProgressCallback;
use whisper_realtime::utils::constants;
use whisper_realtime::whisper::{configs, model};

fn main() {
    // Download the model.
    let mut proj_dir = std::env::current_dir().unwrap();
    proj_dir.push("data");
    let mut model = model::Model::new_with_data_dir(proj_dir.to_path_buf());

    // GPU acceleration is currently required to run larger models in realtime.
    model.model_type = if cfg!(feature = "_gpu") {
        model::ModelType::Medium
    } else {
        model::ModelType::Small
    };

    if !model.is_downloaded() {
        println!("Downloading model:");
        stdout().flush().unwrap();

        // Downloading.

        let url = model.url();
        let url = url.as_str();
        let client = reqwest::blocking::Client::new();

        let sync_downloader = request::sync_download_request(&client, url);
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

        // File Path:
        let file_directory = model.model_directory();
        let file_directory = file_directory.as_path();
        let file_name = model.model_file_name();

        let download = sync_downloader.download(file_directory, file_name);
        assert!(download.is_ok());
        assert!(model.is_downloaded());
    }

    let model = Arc::new(model);
    let c_model = model.clone();

    // Configs ptr -> This should just be the default.
    let configs: Arc<configs::Configs> = Arc::new(configs::Configs::default());
    let c_configs = configs.clone();

    // Audio buffer
    let audio: AudioRingBuffer<f32> =
        AudioRingBuffer::new_with_size(constants::INPUT_BUFFER_CAPACITY, None)
            .expect("Audio length needs to be non-zero");
    let audio_p = Arc::new(audio);
    let audio_p_mic = audio_p.clone();

    // Input sender
    #[cfg(not(feature = "crossbeam"))]
    let (a_sender, a_receiver) = sync_channel(constants::INPUT_BUFFER_CAPACITY);
    #[cfg(feature = "crossbeam")]
    let (a_sender, a_receiver) = crossbeam::channel::bounded(constants::INPUT_BUFFER_CAPACITY);

    // Output sender
    #[cfg(not(feature = "crossbeam"))]
    let (o_sender, o_receiver) = channel();
    #[cfg(feature = "crossbeam")]
    let (o_sender, o_receiver) = crossbeam::channel::unbounded();

    let o_sender_p = o_sender.clone();

    // State flags.
    let is_running = Arc::new(AtomicBool::new(true));
    let c_is_running = is_running.clone();
    let c_is_running_audio_receiver = is_running.clone();
    let c_is_running_data_receiver = is_running.clone();
    let c_interrupt_is_running = is_running.clone();

    // Use CTRL-C to stop the transcription.
    ctrlc::set_handler(move || {
        println!("Interrupt received");
        c_interrupt_is_running.store(false, Ordering::SeqCst);
    })
    .expect("failed to set SIGINT handler");

    let audio_backend = AudioBackend::new();
    if let Err(e) = audio_backend.as_ref() {
        eprintln!("{}", e);
    }

    let audio_backend = audio_backend.unwrap();
    let microphone = audio_backend
        .build_microphone(a_sender)
        .with_sample_rate(Some(constants::WHISPER_SAMPLE_RATE as i32))
        .with_num_channels(Some(1))
        .with_sample_size(Some(1024));

    let mic_stream = microphone.build();
    if let Err(e) = mic_stream.as_ref() {
        eprintln!("{}", e);
    }

    let mic_stream = mic_stream.unwrap();

    // Model params
    let mut whisper_ctx_params = whisper_rs::WhisperContextParameters::default();
    whisper_ctx_params.use_gpu = c_configs.use_gpu;

    let model_path = c_model.file_path();
    let model_path = model_path.as_path();
    let ctx = whisper_rs::WhisperContext::new_with_params(
        model_path.to_str().expect("failed to unwrap path"),
        whisper_ctx_params,
    )
    .expect("Failed to load model");

    let mut state = ctx.create_state().expect("failed to create state");

    let mut text_output_buffer: Vec<String> = vec![];

    // Get the VAD strategy.
    let mut confirm_buffer = String::new();

    // Optional store + re-transcription.
    // Get input for stdin.
    print!("Would you like to store audio and re-transcribe once realtime has finished? y/n: ");
    stdout().flush().unwrap();

    let read_success = std::io::stdin().read_line(&mut confirm_buffer);

    let static_audio_buffer: Option<Vec<f32>> = match read_success {
        Ok(_) => {
            confirm_buffer = confirm_buffer.trim().to_lowercase();
            if confirm_buffer == "y" {
                println!("Confirmed. Recording audio.\n");
                Some(vec![])
            } else {
                None
            }
        }
        Err(_) => None,
    };

    let static_audio_buffer_p = Arc::new(Mutex::new(static_audio_buffer));
    let static_audio_buffer_c = static_audio_buffer_p.clone();

    let transcriber_thread = scope(|s| {
        // Hide the logging away from stdout
        install_logging_hooks();
        mic_stream.resume();
        let _audio_thread = s.spawn(move || {
            let mut t_static_audio_buffer = static_audio_buffer_c
                .lock()
                .expect("Failed to get audio storage mutex");

            loop {
                if !c_is_running_audio_receiver.load(Ordering::Acquire) {
                    break;
                }
                let output = a_receiver.recv();

                // Check for RecvError -> No senders
                match output {
                    Ok(audio_data) => {
                        audio_p_mic.push_audio(&audio_data);

                        if let Some(a_vec) = t_static_audio_buffer.as_mut() {
                            // This might be a borrow problem.
                            a_vec.extend_from_slice(&audio_data);
                        }
                    }
                    // When there are no more senders, this loop will break.
                    Err(_e) => {
                        c_is_running_audio_receiver.store(false, Ordering::SeqCst);
                        break;
                    }
                }
            }
        });

        let transcription_thread = s.spawn(move || {
            let mut transcriber = realtime_transcriber::RealtimeTranscriber::new_with_configs(
                audio_p, o_sender_p, c_configs, None,
            );
            let output = transcriber.process_audio(&mut state, c_is_running, None::<fn(i32)>);
            output
        });

        let print_thread = s.spawn(move || {
            loop {
                if !c_is_running_data_receiver.load(Ordering::Acquire) {
                    break;
                }

                let output = o_receiver.recv();

                // Check for RecvError -> No senders
                match output {
                    Ok(op) => match op {
                        Ok(text_pkg) => {
                            if text_pkg.1 {
                                text_output_buffer.push(text_pkg.0);
                            } else {
                                let last_index = text_output_buffer.len() - 1;
                                text_output_buffer[last_index] = text_pkg.0;
                            }
                        }
                        Err(err) => {
                            eprintln!("{}", err);
                            c_is_running_data_receiver.store(false, Ordering::SeqCst);
                        }
                    },
                    // When there are no more senders, this loop will break.
                    Err(e) => {
                        eprintln!("RecvError: {}", e);
                        c_is_running_data_receiver.store(false, Ordering::SeqCst);
                        break;
                    }
                }
                clear_stdout();

                println!("Transcription:");
                for text_chunk in &text_output_buffer {
                    print!("{}", text_chunk);
                }
                stdout().flush().unwrap();
            }
            text_output_buffer.join("").clone()
        });
        let tt = transcription_thread.join();
        let pt = print_thread.join();
        (tt, pt)
    });

    mic_stream.pause();

    let transcription = transcriber_thread.0.expect("transcription thread panicked");
    let rt_transcription = transcriber_thread.1.expect("print thread panicked");

    // Comparison

    clear_stdout();
    println!("Final Transcription (print thread):");
    println!("{}", &rt_transcription);

    println!();

    println!("Final Transcription (returned):");
    println!("{}", &transcription);

    // Static audio transcription:

    let static_audio_buffer = static_audio_buffer_p
        .lock()
        .expect("Failed to get static audio mutex");

    if let Some(audio_recording) = static_audio_buffer.to_owned() {
        let audio_recording = Arc::new(Mutex::new(static_transcriber::SupportedAudioSample::F32(
            audio_recording,
        )));
        let configs = configs.clone();

        let mut static_transcriber = static_transcriber::StaticTranscriber::new_with_configs(
            audio_recording,
            None,
            configs,
            static_transcriber::SupportedChannels::MONO,
        );

        // new state for static re-transcription.
        let mut state = ctx
            .create_state()
            .expect("failed to create state for static re-transcription");

        println!("Offline Transcribing...");

        // Initiate a progress bar.
        let pb = ProgressBar::new(100);
        pb.set_style(ProgressStyle::default_bar()
            .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {percent} ({eta})").unwrap()
            .progress_chars("#>-")
        );

        pb.set_message("Transcription Progress: ");
        pb.enable_steady_tick(Duration::from_millis(10));

        let pb_c = pb.clone();
        let run_transcription = Arc::new(AtomicBool::new(true));

        let output = static_transcriber.process_audio(
            &mut state,
            run_transcription,
            Some(move |p| {
                pb_c.set_position(p as u64);
            }),
        );
        pb.finish_and_clear();
        println!("\nOffline re-transcription:");
        println!("{}", &output);
    }
}

fn clear_stdout() {
    Command::new("cls")
        .status()
        .or_else(|_| Command::new("clear").status())
        .expect("Failed to clear stdout");
}
