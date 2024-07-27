use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use sdl2::audio::{AudioCallback, AudioFormatNum};

pub struct Recorder<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> {
    pub sender: std::sync::mpsc::SyncSender<Vec<T>>,
    pub is_running: Arc<AtomicBool>,
}

impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> AudioCallback for Recorder<T> {
    type Channel = T;

    fn callback(&mut self, input: &mut [T]) {
        let success = self.sender.send(input.to_vec());
        if let Err(e) = success {
            // TODO: figure out how to bubble this up. Add SendError to error enum.
            eprintln!("SendError: {}", e);
            self.is_running.store(false, Ordering::SeqCst);
        }
    }
}
