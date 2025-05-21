use std::marker::PhantomData;
use std::sync::Arc;

use sdl2::audio::{AudioCallback, AudioFormatNum};

use crate::utils::Sender;

/// This is a workaround for trait aliasing until nightly moves to stable.
pub trait RecorderSample: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static {}
impl<T: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static> RecorderSample for T {}

pub trait AudioInputAdapter<T: RecorderSample> {
    type SenderOutput: Send + Clone + 'static;
    fn convert(input: &[T]) -> Self::SenderOutput;
}
#[derive(Copy, Clone)]
pub struct UseVec;
#[derive(Copy, Clone)]
pub struct UseArc;

impl<T: RecorderSample> AudioInputAdapter<T> for UseVec {
    type SenderOutput = Vec<T>;
    fn convert(input: &[T]) -> Self::SenderOutput {
        input.to_vec()
    }
}
impl<T: RecorderSample> AudioInputAdapter<T> for UseArc {
    type SenderOutput = Arc<[T]>;
    fn convert(input: &[T]) -> Self::SenderOutput {
        Arc::from(input)
    }
}

/// AudioRecorder: trait object to be passed to sdl2's AudioSubsystem.
///
/// Incoming audio data is passed through a message channel to obtain the data elsewhere,
/// For single-thread prefer UseArc for performance unless vectors are absolutely required.
/// In multithreaded applications the bottleneck will always be the work done by other threads;
/// use the sender that suits your needs.
///
/// In all cases, UseArc will be noticeably faster for sending out data.

pub struct AudioRecorder<T: RecorderSample, AC: AudioInputAdapter<T>> {
    sender: Sender<AC::SenderOutput>,
    _marker: PhantomData<AC>,
}

impl<T: RecorderSample, AC: AudioInputAdapter<T> + Send> AudioRecorder<T, AC> {
    pub fn new(sender: Sender<AC::SenderOutput>) -> Self {
        Self {
            sender,
            _marker: PhantomData,
        }
    }
}

impl<T: RecorderSample, AC: AudioInputAdapter<T> + Send> AudioCallback for AudioRecorder<T, AC> {
    type Channel = T;
    fn callback(&mut self, input: &mut [T]) {
        if let Err(e) = self.sender.send(AC::convert(input)) {
            eprintln!("Channel disconnected: {}", e);
        }
    }
}

impl<T: RecorderSample> AudioRecorder<T, UseVec> {
    pub fn new_vec(sender: Sender<Vec<T>>) -> Self {
        Self::new(sender)
    }
}

impl<T: RecorderSample> AudioRecorder<T, UseArc> {
    pub fn new_arc(sender: Sender<Arc<[T]>>) -> Self {
        Self::new(sender)
    }
}
