use std::marker::PhantomData;
use std::sync::Arc;

use crate::audio::audio_ring_buffer::AudioRingBuffer;
use crate::utils::Sender;
use sdl2::audio::{AudioCallback, AudioFormatNum};

// This is a workaround for trait aliasing until the feature reaches stable.
pub trait RecorderSample: Default + Copy + AudioFormatNum + Send + Sync + 'static {}
impl<T: Default + Copy + AudioFormatNum + Send + Sync + 'static> RecorderSample for T {}

/// Trait object to manage converting recorded audio into a sendable form across a message channel
/// For use with [FanoutRecorder]
pub trait AudioInputAdapter {
    type SenderInput: RecorderSample;
    type SenderOutput: Send + Clone + 'static;
    fn convert(input: &[Self::SenderInput]) -> Self::SenderOutput;
}

/// ZST type object that implements [AudioInputAdapter] using `Vec<T>`
#[derive(Default, Copy, Clone)]
pub struct UseVec<T: RecorderSample>(PhantomData<T>);
/// ZST type object that implements [AudioInputAdapter] using `Arc<[T]>`
#[derive(Default, Copy, Clone)]
pub struct UseArc<T: RecorderSample>(PhantomData<T>);

impl<T: RecorderSample> AudioInputAdapter for UseVec<T> {
    type SenderInput = T;
    type SenderOutput = Vec<T>;
    fn convert(input: &[T]) -> Self::SenderOutput {
        input.to_vec()
    }
}

impl<T: RecorderSample> AudioInputAdapter for UseArc<T> {
    type SenderInput = T;
    type SenderOutput = Arc<[T]>;
    fn convert(input: &[T]) -> Self::SenderOutput {
        Arc::from(input)
    }
}

/// Implements [AudioCallback] to handle passing audio samples across a message channel.
/// This does not write directly into a [AudioRingBuffer]
/// to allow flexibility in the implementation and additional processing
/// (e.g. write to a temporary file, etc.).
/// You will need to explicitly write to the audio_ring_buffer used by a
/// [crate::transcriber::realtime_transcriber::RealtimeTranscriber].
/// See: examples/realtime_stream.rs for how to do this.
///
/// For single-thread prefer UseArc for performance unless vectors are absolutely required.
/// In multithreaded applications the bottleneck will always be the work done by other threads;
/// use the sender that suits your needs.
///
/// In all cases, UseArc will be noticeably faster for sending out data.

pub struct FanoutRecorder<AC: AudioInputAdapter> {
    sender: Sender<AC::SenderOutput>,
    _marker: PhantomData<AC>,
}

impl<AC: AudioInputAdapter + Send> FanoutRecorder<AC> {
    pub fn new(sender: Sender<AC::SenderOutput>) -> Self {
        Self {
            sender,
            _marker: PhantomData,
        }
    }
}

impl<AC: AudioInputAdapter + Send> AudioCallback for FanoutRecorder<AC> {
    type Channel = AC::SenderInput;
    fn callback(&mut self, input: &mut [Self::Channel]) {
        if let Err(e) = self.sender.send(AC::convert(input)) {
            eprintln!("Channel disconnected: {}", e);
        }
    }
}

impl<T: RecorderSample> FanoutRecorder<UseVec<T>> {
    pub fn new_vec(sender: Sender<Vec<T>>) -> Self {
        Self::new(sender)
    }
}

impl<T: RecorderSample> FanoutRecorder<UseArc<T>> {
    pub fn new_arc(sender: Sender<Arc<[T]>>) -> Self {
        Self::new(sender)
    }
}

/// Implements [AudioCallback] to write audio samples directly into a ringbuffer
/// that can be read by a [crate::transcriber::realtime_transcriber::RealtimeTranscriber]
pub struct ClosedLoopRecorder<T: RecorderSample> {
    buffer: AudioRingBuffer<T>,
}

impl<T: RecorderSample> ClosedLoopRecorder<T> {
    pub fn new(ring_buffer: AudioRingBuffer<T>) -> Self {
        Self {
            buffer: ring_buffer,
        }
    }
}

impl<T: RecorderSample> AudioCallback for ClosedLoopRecorder<T> {
    type Channel = T;
    fn callback(&mut self, input: &mut [T]) {
        self.buffer.push_audio(input);
    }
}
