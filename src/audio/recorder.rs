use std::marker::PhantomData;
use std::sync::Arc;

use crate::audio::audio_ring_buffer::AudioRingBuffer;
use crate::utils::Sender;
use sdl2::audio::{AudioCallback, AudioFormatNum};

// This is a workaround for trait aliasing until the feature reaches stable.
pub trait RecorderSample: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static {}
impl<T: Default + Clone + Copy + AudioFormatNum + Send + Sync + 'static> RecorderSample for T {}

/// Trait object to manage converting recorded audio into a sendable form across a message channel
/// For use with [crate::audio::recorder::FanoutRecorder]
pub trait AudioInputAdapter<T: RecorderSample> {
    type SenderOutput: Send + Clone + 'static;
    fn convert(input: &[T]) -> Self::SenderOutput;
}

/// ZST type object that implements [crate::audio::recorder::AudioInputAdapter] using `Vec<T>`
#[derive(Copy, Clone)]
pub struct UseVec;
/// ZST type object that implements [crate::audio::recorder::AudioInputAdapter] using `Arc<[T]>`
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

/// Implements [sdl2::audio::AudioCallback] to handle passing audio samples across a message channel.
/// This does not write directly into a [crate::audio::audio_ring_buffer::AudioRingBuffer]
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

pub struct FanoutRecorder<T: RecorderSample, AC: AudioInputAdapter<T>> {
    sender: Sender<AC::SenderOutput>,
    _marker: PhantomData<AC>,
}

impl<T: RecorderSample, AC: AudioInputAdapter<T> + Send> FanoutRecorder<T, AC> {
    pub fn new(sender: Sender<AC::SenderOutput>) -> Self {
        Self {
            sender,
            _marker: PhantomData,
        }
    }
}

impl<T: RecorderSample, AC: AudioInputAdapter<T> + Send> AudioCallback for FanoutRecorder<T, AC> {
    type Channel = T;
    fn callback(&mut self, input: &mut [T]) {
        if let Err(e) = self.sender.send(AC::convert(input)) {
            eprintln!("Channel disconnected: {}", e);
        }
    }
}

impl<T: RecorderSample> FanoutRecorder<T, UseVec> {
    pub fn new_vec(sender: Sender<Vec<T>>) -> Self {
        Self::new(sender)
    }
}

impl<T: RecorderSample> FanoutRecorder<T, UseArc> {
    pub fn new_arc(sender: Sender<Arc<[T]>>) -> Self {
        Self::new(sender)
    }
}

/// Implements [sdl2::audio::AudioCallback] to write audio samples directly into a ringbuffer
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
