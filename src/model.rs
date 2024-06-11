// TODO: Pathing to store models.
#[derive(Copy, Clone)]
pub enum Model {
    TinyEn,
    Tiny,
    BaseEn,
    Base,
    SmallEn,
    Small,
    MediumEn,
    Medium,
    LargeV1,
    LargeV2,
    LargeV3,
}

impl Model {
    // TODO: modify this once downloading is finished.
    pub fn file_path(&self) -> &str {
        match self {
            Self::TinyEn => "/assets/models/tiny.en.bin",
            Self::Tiny => "/assets/models/tiny.bin",
            Self::BaseEn => "/assets/models/base.en.bin",
            Self::Base => "/assets/models/base.bin",
            Self::SmallEn => "/assets/models/small.en.bin",
            Self::Small => "/assets/models/small.bin",
            Self::MediumEn => "/assets/models/medium.en.bin",
            Self::Medium => "/assets/models/medium.bin",
            Self::LargeV1 => "/assets/models/large-v1.bin",
            Self::LargeV2 => "/assets/models/large-v2.bin",
            Self::LargeV3 => "/assets/models/large-v3.bin",
        }
    }
    pub fn is_downloaded(&self) -> bool {
        let path = std::fs::metadata(self.file_path());
        match path {
            Ok(_) => path.unwrap().is_file(),
            Err(_) => false,
        }
    }
    pub fn download(&self) {
        if self.is_downloaded() {
            return;
        }
        todo!("Implement downloading, background thread");
    }

    pub fn delete(&self) {
        if !self.is_downloaded() {
            return;
        }
        todo!("implement delete");
    }
}

impl Default for Model {
    fn default() -> Self {
        Model::TinyEn
    }
}
