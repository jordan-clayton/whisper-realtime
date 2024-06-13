use bbqueue_sync::BBBuffer;
use directories::ProjectDirs;

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
    println!("Hello World");

    serialize::save_configs(&proj_dir, &wg_configs);
    serialize::save_prefs(&proj_dir, &wg_prefs)
}
