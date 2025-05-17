/// To handle different audio types,
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
