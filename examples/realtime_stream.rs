use std::io::{stdout, Write};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender, sync_channel};
use std::thread::scope;

use sdl2::audio::AudioDevice;

use whisper_realtime::{configs, constants, microphone, model};
use whisper_realtime::audio_ring_buffer::AudioRingBuffer;
use whisper_realtime::errors::WhisperRealtimeError;
use whisper_realtime::recorder::Recorder;
use whisper_realtime::transcriber::realtime_transcriber;
use whisper_realtime::transcriber::static_transcriber;
use whisper_realtime::transcriber::transcriber::Transcriber;

fn main() {
    // Download the model.
    let mut proj_dir = std::env::current_dir().unwrap();
    proj_dir.push("data");
    let mut model = model::Model::new_with_data_dir(proj_dir.to_path_buf());
    model.model_type = model::ModelType::MediumEn;
    if !model.is_downloaded() {
        println!("Downloading model:");
        model.download();
        println!("Model downloaded");
    }

    let model = Arc::new(model);
    let c_model = model.clone();

    // Configs ptr -> This should just be the default.
    let configs: Arc<configs::Configs> = Arc::new(configs::Configs::default());
    let c_configs = configs.clone();

    // Audio buffer
    let audio: AudioRingBuffer<f32> = AudioRingBuffer::new(constants::INPUT_BUFFER_CAPACITY);
    let audio_p = Arc::new(audio);
    let audio_p_mic = audio_p.clone();

    // Input sender
    let (a_sender, a_receiver) = sync_channel(constants::INPUT_BUFFER_CAPACITY);

    // Output sender
    let (o_sender, o_receiver) = channel();
    let o_sender_p: Arc<Sender<Result<(String, bool), WhisperRealtimeError>>> = Arc::new(o_sender);

    // State flags.
    let is_ready = Arc::new(AtomicBool::new(true));
    let c_is_ready = is_ready.clone();
    let is_running = Arc::new(AtomicBool::new(true));
    let c_is_running = is_running.clone();
    let c_is_running_recorder_thread = is_running.clone();
    let c_is_running_audio_receiver = is_running.clone();
    let c_is_running_data_receiver = is_running.clone();
    let c_interrupt_is_running = is_running.clone();

    // Use CTRL-C to stop the transcription.
    ctrlc::set_handler(move || {
        println!("Interrupt received");
        c_interrupt_is_running.store(false, Ordering::SeqCst);
    })
    .expect("failed to set SIGINT handler");

    // SDL
    let ctx = sdl2::init().expect("Failed to initialize sdl");
    let audio_subsystem = ctx.audio().expect("Failed to initialize audio");
    let desired_audio_spec = microphone::get_desired_audio_spec(
        Some(constants::WHISPER_SAMPLE_RATE as i32),
        Some(1),
        Some(1024),
    );

    // Setup
    let mic_stream: AudioDevice<Recorder<f32>> = microphone::build_audio_stream(
        &audio_subsystem,
        &desired_audio_spec,
        a_sender,
        c_is_running_recorder_thread,
    );

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

    // Optional store + re-transcription.
    // Get input for stdin.
    println!();
    print!("Would you like to store audio and re-transcribe once realtime has finished? y/n: ");
    stdout().flush().unwrap();
    let mut confirm_buffer = String::new();
    let read_success = std::io::stdin().read_line(&mut confirm_buffer);

    let static_audio_buffer: Option<Vec<f32>> = if let Ok(_b) = read_success {
        confirm_buffer = confirm_buffer.trim().to_lowercase();
        if confirm_buffer == "y" {
            println!("Confirmed. Recording audio.");
            Some(vec![])
        } else {
            None
        }
    } else {
        None
    };

    let static_audio_buffer_p = Arc::new(Mutex::new(static_audio_buffer));
    let static_audio_buffer_c = static_audio_buffer_p.clone();

    let transcriber_thread = scope(|s| {
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
                    Ok(mut audio_data) => {
                        audio_p_mic.push_audio(&mut audio_data);
                        if t_static_audio_buffer.is_some() {
                            let a_vec = t_static_audio_buffer.as_mut().unwrap();
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
                audio_p,
                o_sender_p,
                c_is_running,
                c_is_ready,
                c_configs,
            );
            let output = transcriber.process_audio(&mut state);
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
                            eprintln!("TranscriptionError: {}", err.cause());
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

    if static_audio_buffer.is_some() {
        let audio_recording = static_audio_buffer.to_owned().unwrap();

        let audio_recording = Arc::new(Mutex::new(audio_recording));
        let configs = configs.clone();

        let mut static_transcriber =
            static_transcriber::StaticTranscriber::new_with_configs(audio_recording, configs);

        // new state for static re-transcription.
        let mut state = ctx
            .create_state()
            .expect("failed to create state for static re-transcription");

        let output = static_transcriber.process_audio(&mut state);
        clear_stdout();
        println!("Static re-transcription:");
        println!("{}", &output);
    }
}

fn clear_stdout() {
    Command::new("cls")
        .status()
        .or_else(|_| Command::new("clear").status())
        .expect("Failed to clear stdout");
}
