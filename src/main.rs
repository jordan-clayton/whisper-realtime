use std::io::{stdout, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::thread::scope;

use directories::ProjectDirs;
use sdl2::audio::{AudioCallback, AudioDevice};

use crate::audio_ring_buffer::AudioRingBuffer;
use crate::errors::TranscriptionError;
use crate::recorder::Recorder;
use crate::traits::Transcriber;

mod audio_ring_buffer;
mod constants;
mod errors;
mod microphone;
mod model;
mod preferences;
mod recorder;
mod serialize;
mod traits;
mod transcriber;

fn main() {
    let proj_dir = ProjectDirs::from("com", "Jordan", "WhisperGUI").expect("No home folder");
    let mut wg_configs: preferences::Configs = serialize::load_configs(&proj_dir);
    let mut wg_prefs: preferences::GUIPreferences = serialize::load_prefs(&proj_dir);

    // TODO: remove this.
    let device = String::from("jack");
    if wg_configs.input_device_name.is_none()
        || wg_configs.input_device_name.clone().unwrap() != device
    {
        wg_configs.input_device_name = Some(device);
    }

    // Download the model.
    let data_dir = proj_dir.data_local_dir();

    let mut model = model::Model::new_with_data_dir(data_dir.to_path_buf());
    model.model_type = model::ModelType::MediumEn;
    // Large model
    if !model.is_downloaded() {
        println!("Downloading model:");
        model.download();
        println!("Model downloaded");
    }

    // Not sure whether to clone or just move
    let model = Arc::new(model);
    let c_model = model.clone();

    // Configs ptr
    let configs: Arc<preferences::Configs> = Arc::new(wg_configs.clone());
    let c_configs = configs.clone();

    // Audio buffer

    let audio: AudioRingBuffer<f32> = AudioRingBuffer::new(constants::INPUT_BUFFER_CAPACITY);
    let audio_p = Arc::new(audio);
    let mut audio_p_mic = audio_p.clone();

    // Input sender
    let (a_sender, a_receiver) = channel();

    // Output sender
    let (o_sender, o_receiver) = channel();
    let o_sender_p: Arc<Sender<Result<String, TranscriptionError>>> = Arc::new(o_sender);

    // State flags.
    let is_ready = Arc::new(AtomicBool::new(true));
    let c_is_ready = is_ready.clone();
    let is_running = Arc::new(AtomicBool::new(true));
    let c_is_running = is_running.clone();
    let c_is_running_audio_receiver = is_running.clone();
    let c_is_running_data_receiver = is_running.clone();
    let c_interrupt_is_running = is_running.clone();

    // Register SIGINT handler to stop the transcription.
    ctrlc::set_handler(move || {
        println!("Interrupt received");
        c_interrupt_is_running.store(false, Ordering::SeqCst);
    })
    .expect("failed to set SIGINT handler");

    // SDL context
    let ctx = sdl2::init().expect("Failed to initialize sdl");
    let audio_subsystem = ctx.audio().expect("Failed to initialize audio");
    let desired_audio_spec = microphone::get_desired_audio_spec(
        Some(constants::SAMPLE_RATE as i32),
        Some(1),
        Some(1024),
    );

    // Setup
    let mic_stream: AudioDevice<Recorder<f32>> =
        microphone::build_audio_stream(&audio_subsystem, &desired_audio_spec, a_sender);

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

    let transcription_thread = scope(|s| {
        mic_stream.resume();
        let _audio_thread = s.spawn(move || loop {
            // Wait on a message.
            let output = a_receiver.recv();

            // Check for RecvError -> No senders
            // Issue borrowing as mutable. Figure this out.
            match output {
                Ok(mut audio_data) => {
                    // Push the audio to the ringbuffer.
                    // TODO: refactor to "push_audio"
                    // TODO: finish refactor to atomics to avoid need for mut.
                    audio_p_mic.push_audio(&mut audio_data);
                }
                // When there are no more senders, this loop will break.
                Err(_e) => {
                    c_is_running_audio_receiver.store(false, Ordering::SeqCst);
                    break;
                }
            }
        });

        let transcription_thread = s.spawn(move || {
            let mut transcriber = transcriber::RealtimeTranscriber::new_with_configs(
                audio_p,
                o_sender_p,
                c_is_running,
                c_is_ready,
                c_configs,
            );
            let output = transcriber.process_audio(&mut state);
            output
        });

        let _print_thread = s.spawn(move || {
            loop {
                // Wait on a message.
                let output = o_receiver.recv();

                // Check for RecvError -> No senders
                match output {
                    Ok(op) => match op {
                        Ok(text) => {
                            print!("{}", &text);
                        }
                        Err(err) => {
                            eprintln!("TranscriptionError: {}", err.cause());
                        }
                    },
                    // When there are no more senders, this loop will break.
                    Err(_e) => {
                        c_is_running_data_receiver.store(false, Ordering::SeqCst);
                        break;
                    }
                }
                stdout().flush().unwrap();
            }
        });
        transcription_thread.join()
    });

    mic_stream.pause();

    let transcription = transcription_thread.expect("transcription thread panicked");

    println!();

    println!("Final Transcription (returned):");
    println!("{}", &transcription);

    serialize::save_configs(&proj_dir, &wg_configs);
    serialize::save_prefs(&proj_dir, &wg_prefs)
}
