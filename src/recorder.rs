use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;

use sdl2::audio::{AudioCallback, AudioFormatNum};

pub struct Recorder<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> {
    pub sender: SyncSender<Vec<T>>,
    pub is_running: Arc<AtomicBool>,
}

impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> Recorder<T> {
    pub fn new(sender: SyncSender<Vec<T>>, is_running: Arc<AtomicBool>) -> Self {
        Self { sender, is_running }
    }
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
