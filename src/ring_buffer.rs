use crate::traits::Queue;

#[derive()]
pub struct RingBuffer<T: Default> {
    head: usize,
    tail: usize,
    count: usize,
    buffer: Vec<T>,
}

impl<T: Default + Clone> RingBuffer<T> {}

impl<T: Default + Clone> Queue<T> for RingBuffer<T> {
    fn new() -> Self {
        RingBuffer::default()
    }

    fn with_capacity(size: usize) -> Self {
        RingBuffer {
            head: 0,
            tail: 0,
            count: 0,
            buffer: vec![T::default(); size],
        }
    }

    fn resize(&mut self, new_len: usize) {
        let mut new_buffer = vec![T::default(); new_len];
        let len = std::cmp::min(self.len(), new_buffer.len());

        for i in 0..=len {
            new_buffer[i] = self.pop();
        }

        self.head = 0;
        self.tail = std::cmp::max(len - 1, 0);
        self.count = len;
    }

    fn resize_with(&mut self, new_len: usize, mut generator: impl FnMut() -> T) {
        self.resize(new_len);
        let len = self.buffer.len();
        while self.tail < len {
            self.push(&generator());
        }
    }

    fn pop(&mut self) -> T {
        if self.is_empty() {
            panic!("buffer is empty");
        }
        let popped = self.buffer[self.head].clone();
        self.head = (self.head + 1) % self.buffer.len();
        self.count -= 1;
        popped
    }

    fn push(&mut self, item: &T) {
        self.buffer[self.tail] = item.clone();
        self.tail = (self.tail + 1) % self.buffer.len();
        self.count += 1;
    }

    fn peek(&self) -> T {
        self.buffer[self.tail].clone()
    }

    fn len(&self) -> usize {
        self.count
    }

    fn capacity(&self) -> usize {
        self.buffer.len() - self.count
    }

    fn is_empty(&self) -> bool {
        self.head == self.tail && self.count == 0
    }
}

impl<T: Default + Clone> Default for RingBuffer<T> {
    fn default() -> Self {
        RingBuffer {
            head: 0,
            tail: 0,
            count: 0,
            buffer: vec![],
        }
    }
}
