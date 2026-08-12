//! Safe error handling for all possibilities.

use alloc::vec::Vec;
use core::error::Error as CoreError;
use core::fmt::{Debug, Display, Formatter, Result};

use crate::dispatch::GpuBackend;
use crate::dispatch::backend::{GpuContext, GraphOp, MetaId, NodeId};

/// Generic error type with message, type, and display.
#[derive(Clone)]
pub struct Error<C = ()> {
    pub msg: &'static str,
    pub kind: ErrorKind,
    pub ctx: C,
}

impl core::error::Error for Error {}

impl<C: Debug> Debug for Error<C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        if size_of::<C>() == 0 {
            write!(f, "Error {{ kind: {:?}, msg: {:?} }}", self.kind, self.msg)
        } else {
            write!(
                f,
                "Error {{ kind: {:?}, msg: {:?}, ctx: {:?} }}",
                self.kind, self.msg, self.ctx
            )
        }
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}: {}", self.kind, self.msg)
    }
}

impl<B: GpuBackend + Clone> Display for Error<GraphErrorContext<B>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        debug_assert_eq!(self.kind, ErrorKind::ComputeGraphError);

        write!(f, "{}: {}", self.msg, self.ctx)
    }
}

/// Generic error type for all possiblities within `fused_gpu`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// An environment was not properly setup before use.
    EnvNotSet,

    /// A particular threshold was exceeded.
    LimitsExceeded,

    /// A feature was used that isn't supported.
    UnsupportedFeature,

    /// GPU adapter not found.
    AdapterNotFound,

    /// GPU device not found.
    DeviceNotFound,

    /// Kernel IR could not load input.
    ParamNotMaterialized,

    GraphEmpty,

    InvalidDType,

    UnresolvedInput,

    UnresolvedOutput,

    InvalidArgument,

    ComputeGraphError,

    FailedDownload,
}

impl Display for ErrorKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::EnvNotSet => write!(f, "environment not set"),
            Self::LimitsExceeded => write!(f, "GPU limits exceeded"),
            Self::UnsupportedFeature => write!(f, "unsupported feature"),
            Self::AdapterNotFound => write!(f, "adapter not found"),
            Self::DeviceNotFound => write!(f, "device not found"),
            Self::ParamNotMaterialized => write!(f, "param not materialized"),
            Self::GraphEmpty => write!(f, "graph empty"),
            Self::InvalidDType => write!(f, "invalid data type"),
            Self::UnresolvedInput => write!(f, "unresolved input"),
            Self::UnresolvedOutput => write!(f, "unresolved output"),
            Self::InvalidArgument => write!(f, "invalid argument"),
            Self::ComputeGraphError => write!(f, "compute graph error"),
            Self::FailedDownload => write!(f, "failed GPU buffer download"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum GraphErrorContext<B: GpuBackend = GpuContext> {
    CycleDetected {
        node: NodeId,
        path: Vec<NodeId>,
    },

    InvalidInputs {
        node: NodeId,
        arity: usize,
        args: usize,
    },

    MissingInput {
        node: NodeId,
        input: NodeId,
    },

    ShapeMismatch {
        node: NodeId,
        all_hand_sides: Vec<Vec<MetaId>>,
        op: GraphOp<B>,
    },

    RankMismatch {
        node: NodeId,
        all_hand_sides: Vec<usize>,
    },

    InvalidAxis {
        node: NodeId,
        axis: usize,
        rank: usize,
    },

    MissingMetadata {
        node: NodeId,
        meta: MetaId,
    },

    LowRank {
        node: NodeId,
        rank: usize,
        required: usize,
    },

    CannotInferShape {
        node: NodeId,
        all_hand_sides: Vec<Vec<MetaId>>,
        op: GraphOp<B>,
    },
}

impl<B: GpuBackend + Clone> Display for GraphErrorContext<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::CycleDetected { node, path } => {
                write!(f, "cycle detected at node {node} following {path:?}")
            }

            Self::InvalidAxis { node, axis, rank } => {
                write!(f, "invalid axis {axis} for rank of {rank} at node {node}")
            }

            Self::InvalidInputs { node, arity, args } => {
                write!(
                    f,
                    "invalid input count {args} for arity of {arity} at node {node}"
                )
            }

            Self::MissingInput { node, input } => {
                write!(f, "missing input at node {node} on input node {input}")
            }

            Self::MissingMetadata { node, meta } => {
                write!(f, "missing metadata at node {node} on field f{meta}")
            }

            Self::ShapeMismatch {
                node,
                all_hand_sides,
                op,
            } => write!(
                f,
                "shape mismatch at node {node} {}",
                op.debug(all_hand_sides)
            ),

            Self::RankMismatch {
                node,
                all_hand_sides,
            } => {
                let closure = |f: &mut Formatter, node: &usize, all_hand_sides: &[usize]| {
                    write!(f, "rank mismatch at node {node}")?;

                    write!(f, "{}", all_hand_sides[0])?;

                    for hand_side in all_hand_sides.iter().skip(1) {
                        write!(f, "== {hand_side}")?;
                    }

                    Ok(())
                };

                closure(f, node, all_hand_sides)
            }

            Self::LowRank {
                node,
                rank,
                required,
            } => write!(f, "rank too low at node {node} ({rank} >= {required})"),

            Self::CannotInferShape {
                node,
                all_hand_sides,
                op,
            } => write!(
                f,
                "cannot infer shape of node {node} with op {}",
                op.debug(all_hand_sides)
            ),
        }
    }
}

impl<B: GpuBackend + Clone + Debug> CoreError for GraphErrorContext<B> {}
