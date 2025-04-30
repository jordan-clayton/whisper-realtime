use std::sync::Arc;
#[cfg(not(feature = "crossbeam"))]
use std::sync::mpsc::SyncSender;

use sdl2::audio::{AudioCallback, AudioFormatNum};

/// Recorder: trait object to be passed to sdl2's AudioSubsystem.
/// Incoming audio data is passed through a message channel to obtain the data elsewhere
/// If you do not require vectors, use SliceRecorder
///
/// SliceRecorder: same as above, using Arc<[T]> to share data for slightly better performance.
///
/// In multi-threaded applications, the bottleneck will be in the work done by other threads, so
/// use either struct as they suit your needs.

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

// SLICE RECORDER
#[cfg(not(feature = "crossbeam"))]
pub struct SliceRecorder<T: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static> {
    pub sender: SyncSender<Arc<[T]>>,
}

#[cfg(not(feature = "crossbeam"))]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static> SliceRecorder<T> {
    pub fn new(sender: SyncSender<Arc<[T]>>) -> Self {
        Self { sender }
    }
}

#[cfg(not(feature = "crossbeam"))]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static> AudioCallback
    for SliceRecorder<T>
{
    type Channel = T;

    fn callback(&mut self, input: &mut [T]) {
        let out: Arc<[T]> = Arc::from(&input[..]);
        let _ = self.sender.send(out.clone());
    }
}

#[cfg(feature = "crossbeam")]
pub struct SliceRecorder<T: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static> {
    pub sender: crossbeam::channel::Sender<Arc<[T]>>,
}

#[cfg(feature = "crossbeam")]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static> SliceRecorder<T> {
    pub fn new(sender: crossbeam::channel::Sender<Arc<[T]>>) -> Self {
        Self { sender }
    }
}

#[cfg(feature = "crossbeam")]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static> AudioCallback
    for SliceRecorder<T>
{
    type Channel = T;

    fn callback(&mut self, input: &mut [T]) {
        let out: Arc<[T]> = Arc::from(&input[..]);
        let _ = self.sender.send(out.clone());
    }
}
