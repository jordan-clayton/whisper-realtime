use std::marker::PhantomData;

/// Encapsulates optional callbacks
pub trait Callback {
    type Argument;
    fn call(&mut self, arg: Self::Argument);
}

pub trait ShortCircuitCallback {
    type Argument;
    fn should_run_callback(&mut self) -> bool;
    fn call(&mut self, arg: Self::Argument);
}

/// Encapsulates a basic FnMut(T) callback for functions that accept optional callbacks.
#[repr(C)]
pub struct RibbleWhisperCallback<T, CB: FnMut(T)> {
    callback: CB,
    _marker: PhantomData<T>,
}
impl<T, CB: FnMut(T)> RibbleWhisperCallback<T, CB> {
    pub fn new(callback: CB) -> Self {
        Self {
            callback,
            _marker: PhantomData,
        }
    }
}

impl<T, CB: FnMut(T)> Callback for RibbleWhisperCallback<T, CB> {
    type Argument = T;
    fn call(&mut self, arg: T) {
        (self.callback)(arg);
    }
}

/// This is the static equivalent of  [RibbleWhisperCallback]
/// Encouraged for use when `'static` lifetimes are required, (e.g. OfflineWhisperProgressCallback).
/// It is not strictly necessary to use this over ProgressCallback, but it may help with
/// locating and debugging lifetime errors.
#[repr(C)]
pub struct StaticRibbleWhisperCallback<T, CB: FnMut(T) + 'static> {
    callback: CB,
    _marker: PhantomData<T>,
}

impl<T, CB: FnMut(T) + 'static> StaticRibbleWhisperCallback<T, CB> {
    pub fn new(callback: CB) -> Self {
        Self {
            callback,
            _marker: PhantomData,
        }
    }
}

impl<T, CB: FnMut(T) + 'static> Callback for StaticRibbleWhisperCallback<T, CB> {
    type Argument = T;
    fn call(&mut self, arg: T) {
        (self.callback)(arg);
    }
}

/// Encapsulates both a basic FnMut(T) callback and an FnMut() -> bool short-circuiting callback
/// to early-escape potentially expensive callbacks.
#[repr(C)]
pub struct ShortCircuitRibbleWhisperCallback<T, B, CB>
where
    B: FnMut() -> bool + 'static,
    CB: FnMut(T) + 'static,
{
    should_run_callback: B,
    callback: CB,
    _marker: PhantomData<T>,
}

impl<T, B, CB> ShortCircuitRibbleWhisperCallback<T, B, CB>
where
    B: FnMut() -> bool + 'static,
    CB: FnMut(T) + 'static,
{
    pub fn new(should_run_callback: B, callback: CB) -> Self {
        Self {
            should_run_callback,
            callback,
            _marker: PhantomData,
        }
    }
}

impl<T, B, CB> ShortCircuitCallback for ShortCircuitRibbleWhisperCallback<T, B, CB>
where
    B: FnMut() -> bool + 'static,
    CB: FnMut(T) + 'static,
{
    type Argument = T;

    fn should_run_callback(&mut self) -> bool {
        (self.should_run_callback)()
    }

    fn call(&mut self, arg: Self::Argument) {
        (self.callback)(arg)
    }
}

/// To indicate "None" in functions that expect a callback (eg. downloading, loading audio, etc.)
/// Since all optional callbacks are unpacked before a hot loop to cut down on branching,
/// this will get called repeatedly but expect this to be optimized out.
///
/// This can also be used with `Option<impl Callback>` = None to satisfy type annotation requirements.
#[repr(C)]
pub struct Nop<T> {
    _marker: PhantomData<T>,
}

impl<T> Nop<T> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T> Default for Nop<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T> Callback for Nop<T> {
    type Argument = T;
    fn call(&mut self, _arg: T) {}
}

impl<T> ShortCircuitCallback for Nop<T> {
    type Argument = T;
    fn should_run_callback(&mut self) -> bool {
        false
    }
    fn call(&mut self, _arg: T) {}
}
