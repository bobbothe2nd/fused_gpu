//! Safe error handling for all possibilities.

use alloc::vec::Vec;
use core::error::Error as CoreError;
use core::fmt::{Debug, Display};

use crate::dispatch::backend::{GraphOp, MetaId, NodeId};

/// Generic error type with message, type, and display.
#[derive(Clone)]
pub struct Error<C = ()> {
    pub(crate) msg: &'static str,
    pub(crate) kind: ErrorKind,
    pub(crate) ctx: C,
}

impl core::error::Error for Error {}

impl<C: Debug> Debug for Error<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}: {}", self.kind, self.msg)
    }
}

impl Display for Error<GraphErrorContext> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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

    UnresolvedOutput,

    InvalidArgument,

    ComputeGraphError,

    FailedDownload,
}

impl Display for ErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EnvNotSet => write!(f, "environment not set"),
            Self::LimitsExceeded => write!(f, "GPU limits exceeded"),
            Self::UnsupportedFeature => write!(f, "unsupported feature"),
            Self::AdapterNotFound => write!(f, "adapter not found"),
            Self::DeviceNotFound => write!(f, "device not found"),
            Self::ParamNotMaterialized => write!(f, "param not materialized"),
            Self::GraphEmpty => write!(f, "graph empty"),
            Self::InvalidDType => write!(f, "invalid data type"),
            Self::UnresolvedOutput => write!(f, "unresolved output"),
            Self::InvalidArgument => write!(f, "invalid argument"),
            Self::ComputeGraphError => write!(f, "compute graph error"),
            Self::FailedDownload => write!(f, "failed GPU buffer download"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GraphErrorContext {
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
        lhs: Vec<MetaId>,
        rhs: Vec<MetaId>,
        op: GraphOp,
    },

    RankMismatch {
        node: NodeId,
        lhs: usize,
        rhs: usize,
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
}

impl Display for GraphErrorContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
            Self::ShapeMismatch { node, lhs, rhs, op } => write!(
                f,
                "shape mismatch at node {node} ({lhs:?} {} {rhs:?})",
                op.binary_operator()
            ),
            Self::RankMismatch { node, lhs, rhs } => {
                write!(f, "rank mismatch at node {node} ({lhs} == {rhs})")
            }
            Self::LowRank {
                node,
                rank,
                required,
            } => write!(f, "rank too low at node {node} ({rank} >= {required})"),
        }
    }
}

impl CoreError for GraphErrorContext {}
