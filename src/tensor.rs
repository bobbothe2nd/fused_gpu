//! Tensors are the core primitive of `fused_gpu`.
//!
//! For more information, see [`Tensor`].

use alloc::vec::Vec;

use crate::dispatch::{GpuBackend, GpuBuffer, backend::MetaId};

#[inline]
pub(crate) fn build_dims(shape: &[MetaId], meta: &[u32]) -> Vec<u32> {
    let mut dims = Vec::with_capacity(shape.len());

    for dim in shape {
        dims.push(meta[*dim]);
    }

    dims
}

#[inline]
pub(crate) fn calc_grid(shape: &[u32], block: [u32; 3]) -> [u32; 3] {
    let out_rank = shape.len();

    if block[0] == block[1] {
        [
            shape[out_rank - 1].div_ceil(block[0]),
            shape[out_rank - 2].div_ceil(block[1]),
            shape[..out_rank - 2]
                .iter()
                .product::<u32>()
                .div_ceil(block[2]),
        ]
    } else if block[0] == 1 {
        [
            1,
            shape[out_rank - 1],
            shape[..out_rank - 2]
                .iter()
                .product::<u32>()
                .div_ceil(block[2]),
        ]
    } else {
        [
            shape[out_rank - 2],
            1,
            shape[..out_rank - 2]
                .iter()
                .product::<u32>()
                .div_ceil(block[2]),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<B: GpuBackend = crate::dispatch::backend::GpuContext> {
    pub(crate) shape: Option<Vec<u32>>,
    pub(crate) data: GpuBuffer<B>,
}

impl<B: GpuBackend> PartialEq for Tensor<B> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
    }
}

impl<B: GpuBackend> Eq for Tensor<B> {}

impl<B: GpuBackend> Tensor<B> {
    #[inline]
    #[must_use]
    pub fn calc_grid(&self, block: [u32; 3]) -> [u32; 3] {
        self.shape
            .as_ref()
            .map_or([0; 3], |shape| calc_grid(shape, block))
    }

    /// Returns the rank (length of shape) of the tensor.
    ///
    /// To get just the shape, use [`Tensor::dims`].
    #[inline]
    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape.as_ref().map_or(2, Vec::len)
    }

    /// Returns the dimensions of the tenosr as raw `u32`.
    ///
    /// This is a borrowed slice into the shape with length `self.rank()`.
    #[inline]
    pub fn dims(&self) -> Result<&Vec<u32>, u32> {
        self.shape.as_ref().ok_or_else(|| self.data.size())
    }

    /// Returns the exact length of the data.
    ///
    /// This will always be equal to `.dims().product()`, only it uses a much faster method by directly checking the length of the buffer.
    #[inline]
    #[must_use]
    pub fn len(&self) -> u32 {
        self.data.size()
    }

    /// Checks if the length is equal to zero, or data has no elements.
    ///
    /// Logically equivalent to `self.len() == 0`.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.size_bytes() == 0
    }
}
