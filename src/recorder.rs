use sdl2::audio::{AudioCallback, AudioDevice, AudioFormatNum};

pub struct Recorder<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> {
    pub sender: std::sync::mpsc::Sender<Vec<T>>,
}

impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> AudioCallback for Recorder<T> {
    type Channel = T;

    fn callback(&mut self, input: &mut [T]) {
        // TODO: this occasionally panics.
        // This needs to block if/when a channel is full
        // Pushing data to the ringbuffer blocks on the mutex and this doesn't block.
        self.sender
            .send(input.to_vec())
            .expect("Failed to send data");
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
