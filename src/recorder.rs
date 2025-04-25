// TODO! Refactor this: send an ARC'd slice; there's no need to incur the Vec penalty
#[cfg(not(feature = "crossbeam"))]
use std::sync::mpsc::SyncSender;

use sdl2::audio::{AudioCallback, AudioFormatNum};

#[cfg(not(feature = "crossbeam"))]
pub struct Recorder<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> {
    pub sender: SyncSender<Vec<T>>,
}

#[cfg(not(feature = "crossbeam"))]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> Recorder<T> {
    pub fn new(sender: SyncSender<Vec<T>>) -> Self {
        Self { sender }
    }
}

#[cfg(not(feature = "crossbeam"))]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> AudioCallback for Recorder<T> {
    type Channel = T;

    fn callback(&mut self, input: &mut [T]) {
        let _ = self.sender.send(input.to_vec());
    }
}

#[cfg(feature = "crossbeam")]
pub struct Recorder<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> {
    pub sender: crossbeam::channel::Sender<Vec<T>>,
}

#[cfg(feature = "crossbeam")]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> Recorder<T> {
    pub fn new(sender: crossbeam::channel::Sender<Vec<T>>) -> Self {
        Self { sender }
    }
}

#[cfg(feature = "crossbeam")]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> AudioCallback for Recorder<T> {
    type Channel = T;

    fn callback(&mut self, input: &mut [T]) {
        let _ = self.sender.send(input.to_vec());
    }
}
