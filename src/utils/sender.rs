/// Type alias to handle channel configurations
#[cfg(not(feature = "crossbeam"))]
pub type Sender<T> = std::sync::mpsc::SyncSender<T>;
#[cfg(feature = "crossbeam")]
pub type Sender<T> = crossbeam::channel::Sender<T>;
