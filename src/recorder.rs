use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use sdl2::audio::{AudioCallback, AudioDevice, AudioFormatNum};

pub struct Recorder<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> {
    pub sender: std::sync::mpsc::SyncSender<Vec<T>>,
    pub is_running: Arc<AtomicBool>,
}

impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> AudioCallback for Recorder<T> {
    type Channel = T;

    fn callback(&mut self, input: &mut [T]) {
        // This needs to block if/when a channel is full
        // Pushing data to the ringbuffer blocks on the mutex and this doesn't block.
        //
        let success = self.sender.send(input.to_vec());

        if let Err(e) = success {
            eprintln!("SendError: {}", e);
            self.is_running.store(false, Ordering::SeqCst);
        }
    }
}

pub struct MicWrapper<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> {
    pub mic: AudioDevice<Recorder<T>>,
}

unsafe impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> Sync for MicWrapper<T> {}
unsafe impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> Send for MicWrapper<T> {}

impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> MicWrapper<T> {
    pub fn resume(&self) {
        self.mic.resume();
    }
    pub fn pause(&self) {
        self.mic.pause();
    }
}
