// TODO: Downloading trait.
// TODO: File load trait.
// TODO: Deleting models trait.
// TODO: Serialization trait.

pub trait Transcriber<'a> {
    fn new() -> Self;

    fn init(&mut self, prefs: Option<crate::preferences::Preferences>);

    fn convert_audio(&mut self);

    fn process_audio(&mut self) -> String;

    fn renew_ctx_and_state(&'a mut self);

    fn set_prefs(&mut self, prefs: &crate::preferences::Preferences);
    fn default_prefs(&mut self);
}

pub trait Queue<T> {
    fn new() -> Self;
    fn with_capacity(size: usize) -> Self;

    fn resize(&mut self, new_len: usize);
    fn resize_with(&mut self, new_len: usize, generator: impl FnMut() -> T);
    fn pop(&mut self) -> T;
    fn push(&mut self, item: &T);

    fn peek(&self) -> T;
    fn len(&self) -> usize;
    fn capacity(&self) -> usize;
    fn is_empty(&self) -> bool;
}
