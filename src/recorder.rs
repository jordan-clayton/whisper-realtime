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
        // Errors only happen when all receivers are disconnected.
        if let Err(_) = success {
            // Errors only happen when all receivers are disconnected.
            self.is_running.store(false, Ordering::SeqCst);
        }
    }
}
