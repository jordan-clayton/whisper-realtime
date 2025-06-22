use std::marker::PhantomData;
use std::sync::Arc;

use crate::audio::audio_ring_buffer::AudioRingBuffer;
use crate::audio::pcm::{FromPcmS16, IntoPcmS16};
use crate::utils::constants::{CONVERT_MONO_TO_STEREO, CONVERT_STEREO_TO_MONO};
use crate::utils::errors::RibbleWhisperError;
use crate::utils::Sender;
use sdl2::audio::{AudioCallback, AudioFormatNum};

// This is a workaround for trait aliasing until the feature reaches stable.
pub trait RecorderSample:
    Default
    + Copy
    + AudioFormatNum
    + std::ops::Add<Output = Self>
    + std::ops::Div<Output = Self>
    + From<u8>
    + Send
    + Sync
    + 'static
{
}
impl<
    T: Default
        + Copy
        + AudioFormatNum
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>
        + From<u8>
        + Send
        + Sync
        + 'static,
> RecorderSample for T
{
}

/// Trait object to manage converting recorded audio into a sendable form across a message channel
/// For use with [FanoutRecorder]
pub trait AudioInputAdapter {
    type SenderInput: RecorderSample;
    type SenderOutput: Send + Clone + 'static;
    fn convert(input: &[Self::SenderInput]) -> Self::SenderOutput;
}

// TODO: document this please
// Since SDL runs audio single-threaded, it's okay for this to be mut self.
// The Fallback struct that uses this will be given to SDL, so it can't be shared.
pub trait StereoMonoConvertible<T: RecorderSample> {
    // conversion = bitfield to match on for stereo/mono conversion
    fn with_conversions(self, conversion: u8) -> Result<Self, RibbleWhisperError>
    where
        Self: Sized;
    fn convert(&mut self, input: &[T]);
    // TODO: rename this to something a little clearer once implementation hammered out.
    fn get_converted(&self) -> &[T];
}

// TODO: modify visibility and document
pub mod sealed_fallback {
    // Float audio
    pub trait ValidToF32FallbackPair {}
    impl ValidToF32FallbackPair for super::FallbackUseArc<f32, f32> {}
    impl ValidToF32FallbackPair for super::FallbackUseArc<u16, f32> {}

    impl ValidToF32FallbackPair for super::FallbackUseVec<f32, f32> {}
    impl ValidToF32FallbackPair for super::FallbackUseVec<u16, f32> {}

    // Integer audio
    pub trait ValidToS16FallbackPair {}
    impl ValidToS16FallbackPair for super::FallbackUseArc<i16, i16> {}
    impl ValidToS16FallbackPair for super::FallbackUseArc<f32, i16> {}

    impl ValidToS16FallbackPair for super::FallbackUseVec<i16, i16> {}
    impl ValidToS16FallbackPair for super::FallbackUseVec<f32, i16> {}
}

/// ZST type object that implements [AudioInputAdapter] using `Vec<T>`
#[derive(Default, Copy, Clone)]
pub struct UseVec<T: RecorderSample>(PhantomData<T>);
/// ZST type object that implements [AudioInputAdapter] using `Arc<[T]>`
#[derive(Default, Copy, Clone)]
pub struct UseArc<T: RecorderSample>(PhantomData<T>);

// TODO: Document -> fallback structs to handle converting to the proper audio format/channel configurations to avoid SDL errors.
#[derive(Default, Copy, Clone)]
pub struct FallbackUseArc<T1: RecorderSample, T2: RecorderSample> {
    _input_phantom: PhantomData<T1>,
    _output_phantom: PhantomData<T2>,
}

#[derive(Copy, Clone)]
pub struct FallbackUseVec<T1: RecorderSample, T2: RecorderSample> {
    _input_phantom: PhantomData<T1>,
    _output_phantom: PhantomData<T2>,
}

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

impl AudioInputAdapter for FallbackUseArc<f32, f32> {
    type SenderInput = f32;
    type SenderOutput = Arc<[f32]>;

    fn convert(input: &[Self::SenderInput]) -> Self::SenderOutput {
        Arc::from(input)
    }
}
impl AudioInputAdapter for FallbackUseArc<i16, f32> {
    type SenderInput = i16;
    type SenderOutput = Arc<[f32]>;

    fn convert(input: &[Self::SenderInput]) -> Self::SenderOutput {
        let output = input
            .iter()
            .cloned()
            .map(|signal| f32::from_pcm_s16(signal))
            .collect::<Vec<_>>();
        Arc::from(output)
    }
}
impl AudioInputAdapter for FallbackUseArc<i16, i16> {
    type SenderInput = i16;
    type SenderOutput = Arc<[i16]>;

    fn convert(input: &[Self::SenderInput]) -> Self::SenderOutput {
        Arc::from(input)
    }
}

impl AudioInputAdapter for FallbackUseArc<f32, i16> {
    type SenderInput = f32;
    type SenderOutput = Arc<[i16]>;

    fn convert(input: &[Self::SenderInput]) -> Self::SenderOutput {
        let output = input
            .iter()
            .cloned()
            .map(|sample| sample.into_pcm_s16())
            .collect::<Vec<_>>();
        Arc::from(output)
    }
}

impl AudioInputAdapter for FallbackUseVec<f32, f32> {
    type SenderInput = f32;
    type SenderOutput = Vec<f32>;

    fn convert(input: &[Self::SenderInput]) -> Self::SenderOutput {
        input.to_vec()
    }
}
impl AudioInputAdapter for FallbackUseVec<i16, f32> {
    type SenderInput = i16;
    type SenderOutput = Vec<f32>;

    fn convert(input: &[Self::SenderInput]) -> Self::SenderOutput {
        input
            .iter()
            .cloned()
            .map(|signal| f32::from_pcm_s16(signal))
            .collect::<Vec<_>>()
    }
}
impl AudioInputAdapter for FallbackUseVec<i16, i16> {
    type SenderInput = i16;
    type SenderOutput = Vec<i16>;

    fn convert(input: &[Self::SenderInput]) -> Self::SenderOutput {
        input.to_vec()
    }
}

impl AudioInputAdapter for FallbackUseVec<f32, i16> {
    type SenderInput = f32;
    type SenderOutput = Vec<i16>;

    fn convert(input: &[Self::SenderInput]) -> Self::SenderOutput {
        input
            .iter()
            .cloned()
            .map(|sample| sample.into_pcm_s16())
            .collect()
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

// TODO: document -> can/should be used for fallback implementations to avoid SDL errors.
// Very limited use, might even make it completely private and just document the trait
// It's only going to be used in the fallback builder functions.
pub struct StereoMonoConverter<T: RecorderSample> {
    working_buffer: Vec<T>,
    conversion: u8,
}

impl<T: RecorderSample> StereoMonoConverter<T> {
    // NOTE: Try to set this to the correct buffer size at the start
    // the working buffer will correct its size so it's not required, but heavily encouraged.
    pub fn new(buffer_size: usize) -> Self {
        Self {
            working_buffer: Vec::with_capacity(buffer_size),
            conversion: 0,
        }
    }
}

impl<T: RecorderSample + std::ops::Add<Output = T> + std::ops::Div<Output = T> + From<u8>>
    StereoMonoConvertible<T> for StereoMonoConverter<T>
{
    fn with_conversions(mut self, conversion: u8) -> Result<Self, RibbleWhisperError> {
        let invalid = CONVERT_STEREO_TO_MONO | CONVERT_MONO_TO_STEREO;
        if conversion >= invalid {
            Err(RibbleWhisperError::ParameterError(
                "Cannot convert both stereo-to-mono and mono-to-stereo".to_string(),
            ))
        } else {
            self.conversion = conversion;
            Ok(self)
        }
    }

    fn convert(&mut self, input: &[T]) {
        self.working_buffer.clear();
        match self.conversion {
            CONVERT_MONO_TO_STEREO => {
                self.working_buffer
                    .resize(input.len() * 2, Default::default());
                let iter = input
                    .iter()
                    .cloned()
                    .map(|sample| [sample, sample])
                    .flatten();

                for (i, sample) in iter.enumerate() {
                    self.working_buffer[i] = sample;
                }
            }
            CONVERT_STEREO_TO_MONO => {
                self.working_buffer
                    .resize(input.len() / 2, Default::default());

                debug_assert!(
                    input.len().is_multiple_of(2),
                    "Input buffer somehow not a multiple of 2"
                );

                let iter = input
                    .chunks(2)
                    // NOTE: this is lossy when T is i16
                    .map(|chunk| (chunk[0] + chunk[1]) / T::from(2));
                for (i, chunk) in iter.enumerate() {
                    self.working_buffer[i] = chunk;
                }
            }
            0 => {
                self.working_buffer.resize(input.len(), Default::default());
                self.working_buffer.copy_from_slice(input);
            }
            _ => unreachable!("The converter should always have a valid bitmask."),
        }
    }

    fn get_converted(&self) -> &[T] {
        self.working_buffer.as_slice()
    }
}

// TODO: document this
pub struct FallbackFanoutRecorder<AC, S>
where
    AC: AudioInputAdapter,
    S: StereoMonoConvertible<AC::SenderInput> + Send,
{
    stereo_mono_converter: S,
    sender: Sender<AC::SenderOutput>,
    _marker: PhantomData<AC>,
}

impl<T, AC, S> FallbackFanoutRecorder<AC, S>
where
    T: RecorderSample,
    AC: AudioInputAdapter<SenderInput = T, SenderOutput = Arc<[f32]>>
        + sealed_fallback::ValidToF32FallbackPair
        + Send,
    S: StereoMonoConvertible<T> + Send,
{
    pub fn new_arc_fp(sender: Sender<Arc<[f32]>>, stereo_mono_converter: S) -> Self {
        Self {
            stereo_mono_converter,
            sender,
            _marker: Default::default(),
        }
    }
}
impl<T, AC, S> FallbackFanoutRecorder<AC, S>
where
    T: RecorderSample,
    AC: AudioInputAdapter<SenderInput = T, SenderOutput = Arc<[i16]>>
        + sealed_fallback::ValidToS16FallbackPair
        + Send,
    S: StereoMonoConvertible<T> + Send,
{
    pub fn new_arc_int(sender: Sender<Arc<[i16]>>, stereo_mono_converter: S) -> Self {
        Self {
            stereo_mono_converter,
            sender,
            _marker: Default::default(),
        }
    }
}

impl<T, AC, S> FallbackFanoutRecorder<AC, S>
where
    T: RecorderSample,
    AC: AudioInputAdapter<SenderInput = T, SenderOutput = Vec<f32>>
        + sealed_fallback::ValidToF32FallbackPair
        + Send,
    S: StereoMonoConvertible<T> + Send,
{
    pub fn new_vec_fp(sender: Sender<Vec<f32>>, stereo_mono_converter: S) -> Self {
        Self {
            stereo_mono_converter,
            sender,
            _marker: Default::default(),
        }
    }
}
impl<T, AC, S> FallbackFanoutRecorder<AC, S>
where
    T: RecorderSample,
    AC: AudioInputAdapter<SenderInput = T, SenderOutput = Vec<i16>>
        + sealed_fallback::ValidToS16FallbackPair
        + Send,
    S: StereoMonoConvertible<T> + Send,
{
    pub fn new_vec_int(sender: Sender<Vec<i16>>, stereo_mono_converter: S) -> Self {
        Self {
            stereo_mono_converter,
            sender,
            _marker: Default::default(),
        }
    }
}

impl<AC, S> AudioCallback for FallbackFanoutRecorder<AC, S>
where
    AC: AudioInputAdapter + Send,
    S: StereoMonoConvertible<AC::SenderInput> + Send,
{
    type Channel = AC::SenderInput;
    fn callback(&mut self, input: &mut [Self::Channel]) {
        self.stereo_mono_converter.convert(input);

        if let Err(e) = self
            .sender
            .send(AC::convert(self.stereo_mono_converter.get_converted()))
        {
            eprintln!("Channel disconnected: {}", e);
        }
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

// TODO: document this properly -- fallback implementation to avoid SDL problems.
pub struct FallbackClosedLoopRecorder<T1, T2, S>
where
    T1: RecorderSample,
    T2: RecorderSample,
    S: StereoMonoConvertible<T1>,
{
    stereo_mono_converter: S,
    buffer: AudioRingBuffer<T2>,
    _marker: PhantomData<T1>,
}

impl<T1, T2, S> FallbackClosedLoopRecorder<T1, T2, S>
where
    T1: RecorderSample,
    T2: RecorderSample,
    S: StereoMonoConvertible<T1>,
{
    pub fn new(ring_buffer: AudioRingBuffer<T2>, stereo_mono_converter: S) -> Self {
        Self {
            stereo_mono_converter,
            buffer: ring_buffer,
            _marker: Default::default(),
        }
    }
}

impl<S: StereoMonoConvertible<f32> + Send> AudioCallback
    for FallbackClosedLoopRecorder<f32, f32, S>
{
    type Channel = f32;
    fn callback(&mut self, input: &mut [Self::Channel]) {
        self.stereo_mono_converter.convert(input);
        self.buffer
            .push_audio(self.stereo_mono_converter.get_converted());
    }
}
impl<S: StereoMonoConvertible<f32> + Send> AudioCallback
    for FallbackClosedLoopRecorder<f32, i16, S>
{
    type Channel = f32;

    fn callback(&mut self, input: &mut [Self::Channel]) {
        self.stereo_mono_converter.convert(input);
        let out = self
            .stereo_mono_converter
            .get_converted()
            .iter()
            .cloned()
            .map(|signal| signal.into_pcm_s16())
            .collect::<Vec<_>>();
        self.buffer.push_audio(&out);
    }
}
impl<S: StereoMonoConvertible<i16> + Send> AudioCallback
    for FallbackClosedLoopRecorder<i16, i16, S>
{
    type Channel = i16;
    fn callback(&mut self, input: &mut [Self::Channel]) {
        self.stereo_mono_converter.convert(input);
        self.buffer
            .push_audio(self.stereo_mono_converter.get_converted());
    }
}
impl<S: StereoMonoConvertible<i16> + Send> AudioCallback
    for FallbackClosedLoopRecorder<i16, f32, S>
{
    type Channel = i16;
    fn callback(&mut self, input: &mut [Self::Channel]) {
        self.stereo_mono_converter.convert(input);
        let out = self
            .stereo_mono_converter
            .get_converted()
            .iter()
            .cloned()
            .map(|signal| f32::from_pcm_s16(signal))
            .collect::<Vec<_>>();
        self.buffer.push_audio(&out);
    }
}
