use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bbqueue_sync::BBBuffer;
use cpal::traits::StreamTrait;
use directories::ProjectDirs;

use crate::traits::Transcriber;

mod constants;
mod microphone;
mod model;
mod preferences;
mod ring_buffer;
mod serialize;
mod traits;
mod transcriber;

static MIC_DATA_BUFFER: BBBuffer<{ constants::INPUT_BUFFER_CAPACITY }> = BBBuffer::new();
static MIC_ERROR_BUFFER: BBBuffer<{ constants::CPAL_ERROR_BUFFER_CAPACITY }> = BBBuffer::new();
static TRANSCRIPTION_BUFFER: BBBuffer<{ constants::OUTPUT_BUFFER_CAPACITY }> = BBBuffer::new();
static TRANSCRIPTION_ERROR_BUFFER: BBBuffer<{ constants::ERROR_BUFFER_CAPACITY }> = BBBuffer::new();

fn main() {
    let proj_dir = ProjectDirs::from("com", "Jordan", "WhisperGUI").expect("No home folder");
    let mut wg_configs: preferences::Configs = serialize::load_configs(&proj_dir);
    let mut wg_prefs: preferences::GUIPreferences = serialize::load_prefs(&proj_dir);

    //TODO: migrate testing code once GUI
    // println!("Hello World");

    // Download the model.

    let data_dir = proj_dir.data_local_dir();

    let model = model::Model::new_with_data_dir(data_dir.to_path_buf());

    // Rn, going with default, tiny model.
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

    // Mic Channels
    let (mic_producer, mic_consumer) = MIC_DATA_BUFFER
        .try_split()
        .expect("failed to create input buffer");
    let (mic_err_producer, mic_err_consumer) = MIC_ERROR_BUFFER
        .try_split()
        .expect("failed to create input error buffer");

    let mic_p = Arc::new(mic_producer);
    let c_mic_p = mic_p.clone();
    let mic_c = Arc::new(mic_consumer);
    let c_mic_c = mic_c.clone();
    let mic_err_p = Arc::new(mic_err_producer);
    let c_mic_err_p = mic_err_p.clone();
    let mic_err_c = Arc::new(mic_err_consumer);
    let c_mic_err_c = mic_err_c.clone();

    // Output buffers
    let (trans_producer, trans_consumer) = TRANSCRIPTION_BUFFER
        .try_split()
        .expect("failed to create output buffer");
    let (trans_err_producer, trans_err_consumer) = TRANSCRIPTION_ERROR_BUFFER
        .try_split()
        .expect("failed to create output error buffer");

    let trans_p = Arc::new(trans_producer);
    let c_trans_p = trans_p.clone();
    let trans_c = Arc::new(trans_consumer);
    let c_trans_c = trans_c.clone();
    let trans_err_p = Arc::new(trans_err_producer);
    let c_trans_err_p = trans_err_p.clone();
    let trans_err_c = Arc::new(trans_err_consumer);
    let c_trans_err_c = trans_err_c.clone();

    // State flags.
    let is_ready = Arc::new(AtomicBool::new(true));
    let c_is_ready = is_ready.clone();
    let is_running = Arc::new(AtomicBool::new(false));
    let c_is_running = is_running.clone();

    let c_interrupt_is_running = is_running.clone();

    // Register SIGINT handler to stop the transcription.

    ctrlc::set_handler(move || {
        c_interrupt_is_running.store(false, Ordering::Relaxed);
    })
    .expect("failed to set SIGINT handler");

    let transcription_thread = std::thread::spawn(move || {
        let mut whisper_ctx_params = whisper_rs::WhisperContextParameters::default();
        whisper_ctx_params.use_gpu = c_configs.use_gpu;

        let model_path = c_model.file_path();
        let model_path = model_path.as_path();
        let ctx = whisper_rs::WhisperContext::new_with_params(
            model_path.to_str().expect("failed to unwrap path"),
            whisper_ctx_params,
        )
        .expect("Failed to load model");

        let state = ctx.create_state().expect("failed to create state");

        let (stream, sample_format) =
            microphone::create_microphone_stream(c_mic_p, c_mic_err_p, None, None);

        stream.play().expect("failed to start mic stream");
        let mut transcriber = transcriber::RealtimeTranscriber::new_with_configs(
            state,
            &sample_format,
            c_mic_c,
            c_mic_err_c,
            c_trans_p,
            c_trans_err_p,
            c_is_running,
            c_is_ready,
            c_configs,
        );

        // THIS WORKS!
        let transcriber = Box::new(transcriber);
        let mut t_ptr = Box::into_raw(transcriber);

        let output = unsafe {
            let transcriber = t_ptr.as_mut().unwrap();
            let s = transcriber.process_audio();
            let b = Box::from_raw(t_ptr);
            drop(b);
            s
        };

        drop(stream);
        output.to_owned()
    });

    // Normally this would go on a separate thread, this is just for testing.
    let mut output_buffer: Vec<String> = vec![];
    println!("Transcription: ");
    loop {
        let running = is_running.load(Ordering::Relaxed);
        if !running {
            break;
        }

        // Handle errors - This might need to grab in chunks of cpal::StreamError size.
        // THIS MIGHT PANIC.
        let err_g = c_trans_err_c.read();
        if let Ok(err_grant) = err_g {
            let err_buf = err_grant.buf();
            let used_bytes = err_buf.len();
            let mut err: Vec<u8> = vec![];
            err.extend_from_slice(err_buf);
            let err = err.as_mut_ptr() as *mut cpal::StreamError;
            eprintln!("{:?}", err);
            is_running.store(false, Ordering::Relaxed);
            err_grant.release(used_bytes);
        }

        // Handle input - not currently flushing stdout
        let audio_g = c_trans_c.read();
        if let Ok(audio) = audio_g {
            let input_buf = audio.buf();
            let used_bytes = input_buf.len();
            let mut byte_string: Vec<u8> = vec![];
            byte_string.extend_from_slice(input_buf);
            let text_chunk = String::from_utf8(byte_string).expect("failed to parse bytestring");
            print!("{}", text_chunk);
            output_buffer.push(text_chunk);
            std::io::stdout().flush().ok().unwrap();
            audio.release(used_bytes)
        }
    }

    let transcription = transcription_thread.join();

    // I'm not 100% sure about this yet.
    // -> Should probably be some sort of error.
    // Lol. This needs to be by reference & figure out some way to empty the bUUUuffer.

    // Free the buffers here - factor into a function?.
    // Free the audio data buffer
    let mic_read = mic_c.read();
    if let Ok(mic_grant) = mic_read {
        let used_bytes = mic_grant.len();
        mic_grant.release(used_bytes);
    }

    // Free the mic error buffer
    let mic_error_read = mic_err_c.read();
    if let Ok(mic_error_grant) = mic_error_read {
        let used_bytes = mic_error_grant.len();
        mic_error_grant.release(used_bytes);
    }

    // Free the thread output buffer
    let trans_data_read = trans_c.read();
    if let Ok(mut trans_grant) = trans_data_read {
        let input_buf = trans_grant.buf().clone();
        let mut byte_string: Vec<u8> = vec![];
        byte_string.extend_from_slice(input_buf);
        let text_chunk = String::from_utf8(byte_string).expect("failed to parse bytestring");
        print!("{}", text_chunk);
        output_buffer.push(text_chunk);
        std::io::stdout().flush().unwrap();
        trans_grant.to_release(input_buf.len());
    }

    // Free the err buffer.
    let trans_err_read = trans_err_c.read();
    if let Ok(mut t_e_grant) = trans_err_read {
        let err_buf = t_e_grant.buf();
        let used_bytes = err_buf.len();
        let mut err: Vec<u8> = vec![];
        err.extend_from_slice(err_buf);
        let err = err.as_mut_ptr() as *mut cpal::StreamError;
        eprintln!("{:?}", err);
        t_e_grant.to_release(used_bytes);
    }

    let final_transcription = output_buffer.join("");

    // let _ = MIC_DATA_BUFFER.try_release(mic_producer, mic_consumer);
    // let _ = MIC_ERROR_BUFFER.try_release(mic_err_producer, mic_err_consumer);
    // let _ = TRANSCRIPTION_BUFFER.try_release(trans_producer, trans_consumer);
    // let _ = TRANSCRIPTION_ERROR_BUFFER.try_release(trans_err_producer, trans_err_consumer);

    println!();
    println!("Final Transcription: ");
    println!("{}", final_transcription);

    println!();

    match transcription {
        Ok(s) => {
            println!("Final Transcription (returned):");
            println!("{}", s);
        }
        Err(_e) => {}
    }

    serialize::save_configs(&proj_dir, &wg_configs);
    serialize::save_prefs(&proj_dir, &wg_prefs)
}
