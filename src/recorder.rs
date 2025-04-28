use std::sync::Arc;
// TODO: Test Arc<[T]>
// TODO: Remove blocking; drain oldest samples on full capacity
#[cfg(not(feature = "crossbeam"))]
use std::sync::mpsc::SyncSender;

use sdl2::audio::{AudioCallback, AudioFormatNum};

// VEC RECORDER
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

#[cfg(not(feature = "crossbeam"))]
pub struct SliceRecorder<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> {
    pub sender: SyncSender<Arc<[T]>>,
}

#[cfg(not(feature = "crossbeam"))]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> SliceRecorder<T> {
    pub fn new(sender: SyncSender<Arc<[T]>>) -> Self {
        Self { sender }
    }
}

#[cfg(not(feature = "crossbeam"))]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> AudioCallback
    for SliceRecorder<T>
{
    type Channel = T;

    fn callback(&mut self, input: &mut [T]) {
        let out = Arc::new(*input);
        let _ = self.sender.send(out.clone());
    }
}

#[cfg(feature = "crossbeam")]
pub struct SliceRecorder<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> {
    pub sender: crossbeam::channel::Sender<Arc<[T]>>,
}

#[cfg(feature = "crossbeam")]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> SliceRecorder<T> {
    pub fn new(sender: crossbeam::channel::Sender<Arc<[T]>>) -> Self {
        Self { sender }
    }
}

#[cfg(feature = "crossbeam")]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> AudioCallback
    for SliceRecorder<T>
{
    type Channel = T;

    fn callback(&mut self, input: &mut [T]) {
        let out = Arc::new(*input);
        let _ = self.sender.send(out.clone());
    }
}
