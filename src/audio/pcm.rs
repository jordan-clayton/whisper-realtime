/// A simple trait for round-trip conversion between i16 PCM audio and the original signal format
pub trait PcmS16Convertible: IntoPcmS16 + FromPcmS16 {}

impl PcmS16Convertible for i16 {}
impl PcmS16Convertible for u8 {}
impl PcmS16Convertible for f32 {}
impl PcmS16Convertible for f64 {}

/// To handle and convert various numeric audio formats into i16 PCM for use with WebRtc
/// and wherever required.
pub trait IntoPcmS16 {
    fn into_pcm_s16(self) -> i16;
}

impl IntoPcmS16 for i16 {
    fn into_pcm_s16(self) -> i16 {
        self
    }
}

impl IntoPcmS16 for u8 {
    fn into_pcm_s16(self) -> i16 {
        (self as i16 - 128) << 8
    }
}

impl IntoPcmS16 for f32 {
    fn into_pcm_s16(self) -> i16 {
        (self.clamp(-1., 1.) * (i16::MAX as f32)).clamp(i16::MIN as f32, i16::MAX as f32) as i16
    }
}

impl IntoPcmS16 for f64 {
    fn into_pcm_s16(self) -> i16 {
        (self.clamp(-1., 1.) * i16::MAX as f64).clamp(i16::MIN as f64, i16::MAX as f64) as i16
    }
}

/// To handle and convert various numeric audio formats from i16 PCM back to the original type
pub trait FromPcmS16: Sized {
    fn from_pcm_s16(sample: i16) -> Self;
}

impl FromPcmS16 for i16 {
    fn from_pcm_s16(sample: i16) -> Self {
        sample
    }
}

impl FromPcmS16 for u8 {
    fn from_pcm_s16(sample: i16) -> Self {
        ((sample >> 8) + 128).clamp(0, u8::MAX as i16) as u8
    }
}

impl FromPcmS16 for f32 {
    fn from_pcm_s16(sample: i16) -> Self {
        sample as f32 / i16::MAX as f32
    }
}
impl FromPcmS16 for f64 {
    fn from_pcm_s16(sample: i16) -> Self {
        sample as f64 / i16::MAX as f64
    }
}
