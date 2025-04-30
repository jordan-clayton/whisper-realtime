use std::sync::Arc;
#[cfg(not(feature = "crossbeam"))]
use std::sync::mpsc::SyncSender;

use sdl2::audio::{AudioCallback, AudioFormatNum};

/// AudioRecorder*: trait object to be passed to sdl2's AudioSubsystem.
///
/// Incoming audio data is passed through a message channel to obtain the data elsewhere,
/// The VecSender sends as Vec<T>, the SliceSender sends as Arc<[T]>
/// For single-thread prefer SliceSender for performance.
/// In multithreaded applications, the bottleneck will be the work done by other threads, so
/// use either struct as they suit your needs. SliceSender will still be noticeably more performant,
/// especially so for lighter work threads.

// AudioRecorderVecSender
#[cfg(not(feature = "crossbeam"))]
pub struct AudioRecorderVecSender<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> {
    pub sender: SyncSender<Vec<T>>,
}

#[cfg(not(feature = "crossbeam"))]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> AudioRecorderVecSender<T> {
    pub fn new(sender: SyncSender<Vec<T>>) -> Self {
        Self { sender }
    }
}

#[cfg(not(feature = "crossbeam"))]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> AudioCallback
    for AudioRecorderVecSender<T>
{
    type Channel = T;

    fn callback(&mut self, input: &mut [T]) {
        let _ = self.sender.send(input.to_vec());
    }
}

#[cfg(feature = "crossbeam")]
pub struct AudioRecorderVecSender<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> {
    pub sender: crossbeam::channel::Sender<Vec<T>>,
}

#[cfg(feature = "crossbeam")]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> AudioRecorderVecSender<T> {
    pub fn new(sender: crossbeam::channel::Sender<Vec<T>>) -> Self {
        Self { sender }
    }
}

#[cfg(feature = "crossbeam")]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + 'static> AudioCallback
    for AudioRecorderVecSender<T>
{
    type Channel = T;

    fn callback(&mut self, input: &mut [T]) {
        let _ = self.sender.send(input.to_vec());
    }
}

// AudioRecorderSliceSender
#[cfg(not(feature = "crossbeam"))]
pub struct AudioRecorderSliceSender<
    T: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static,
> {
    pub sender: SyncSender<Arc<[T]>>,
}

#[cfg(not(feature = "crossbeam"))]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static>
    AudioRecorderSliceSender<T>
{
    pub fn new(sender: SyncSender<Arc<[T]>>) -> Self {
        Self { sender }
    }
}

#[cfg(not(feature = "crossbeam"))]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static> AudioCallback
    for AudioRecorderSliceSender<T>
{
    type Channel = T;

    fn callback(&mut self, input: &mut [T]) {
        let out: Arc<[T]> = Arc::from(&input[..]);
        let _ = self.sender.send(out.clone());
    }
}

#[cfg(feature = "crossbeam")]
pub struct AudioRecorderSliceSender<
    T: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static,
> {
    pub sender: crossbeam::channel::Sender<Arc<[T]>>,
}

#[cfg(feature = "crossbeam")]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static>
    AudioRecorderSliceSender<T>
{
    pub fn new(sender: crossbeam::channel::Sender<Arc<[T]>>) -> Self {
        Self { sender }
    }
}

#[cfg(feature = "crossbeam")]
impl<T: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static> AudioCallback
    for AudioRecorderSliceSender<T>
{
    type Channel = T;

    fn callback(&mut self, input: &mut [T]) {
        let out: Arc<[T]> = Arc::from(&input[..]);
        let _ = self.sender.send(out.clone());
    }
}
