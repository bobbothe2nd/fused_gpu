//! `fused_gpu` implements optimized GPU kernels for almost any operation.
//!
//! It supports CUDA, ROCM, and WGSL. It will compile anything the feature set describes:
//!
//! - `wgsl`: Enable WGSL support
//!
//! You can also use custom backends.

#![forbid(clippy::unimplemented)]
#![forbid(clippy::print_stderr)]
#![forbid(clippy::print_stdout)]
#![forbid(clippy::approx_constant)]
#![deny(clippy::pedantic)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![deny(clippy::nursery)]
#![deny(clippy::all)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![forbid(clippy::unwrap_used)]
#![forbid(clippy::expect_used)]
#![forbid(clippy::panic)]
#![forbid(unsafe_code)]
#![no_std]

extern crate alloc;

extern crate std;

pub mod dispatch;

pub mod errors;

pub mod tensor;
