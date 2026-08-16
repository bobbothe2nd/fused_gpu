use crate::dispatch::backend::{Axis, MetaId, ParamId, SharedId, ValueId};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    Nop,

    DefineVar {
        id: ValueId,
    },
    OverwriteVar {
        id: ValueId,
        val: ValueId,
    },
    AddAssign {
        id: ValueId,
        val: ValueId,
    },
    MulAssign {
        id: ValueId,
        val: ValueId,
    },
    DivAssign {
        id: ValueId,
        val: ValueId,
    },
    SubAssign {
        id: ValueId,
        val: ValueId,
    },
    ShlAssign {
        id: ValueId,
        val: ValueId,
    },
    ShrAssign {
        id: ValueId,
        val: ValueId,
    },
    CopyVar {
        id: ValueId,
    },

    ConstF32 {
        value: f32,
    },
    ConstU32 {
        value: u32,
    },
    ConstI32 {
        value: i32,
    },

    ReadMeta {
        param: ParamId,
        field: MetaId,
    },

    LocalId {
        axis: Axis,
    },
    BlockId {
        axis: Axis,
    },
    GlobalId {
        axis: Axis,
    },

    Add {
        a: ValueId,
        b: ValueId,
    },
    Sub {
        a: ValueId,
        b: ValueId,
    },
    Mul {
        a: ValueId,
        b: ValueId,
    },
    Div {
        a: ValueId,
        b: ValueId,
    },
    Mod {
        a: ValueId,
        b: ValueId,
    },
    Pow {
        a: ValueId,
        b: ValueId,
    },

    Shl {
        a: ValueId,
        b: ValueId,
    },

    Shr {
        a: ValueId,
        b: ValueId,
    },

    /// (Fused) Operation `a * b + c`
    Fma {
        a: ValueId,
        b: ValueId,
        c: ValueId,
    },

    Exp {
        x: ValueId,
    },
    Abs {
        x: ValueId,
    },
    Neg {
        x: ValueId,
    },
    Log {
        x: ValueId,
    },
    Tanh {
        x: ValueId,
    },
    Sqrt {
        x: ValueId,
    },

    ParamLoad {
        param: ParamId,
        index: ValueId,
    },
    ParamStore {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },
    ParamAccum {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },
    ParamMul {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },
    ParamDiv {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },
    ParamSub {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },
    ParamShl {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },
    ParamShr {
        param: ParamId,
        index: ValueId,
        value: ValueId,
    },

    SharedLoad {
        mem: SharedId,
        index: ValueId,
    },
    SharedStore {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },
    SharedAccum {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },
    SharedMul {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },
    SharedDiv {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },
    SharedSub {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },
    SharedShl {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },
    SharedShr {
        mem: SharedId,
        index: ValueId,
        value: ValueId,
    },

    Eq {
        a: ValueId,
        b: ValueId,
    },

    Ne {
        a: ValueId,
        b: ValueId,
    },

    Lt {
        a: ValueId,
        b: ValueId,
    },

    Gt {
        a: ValueId,
        b: ValueId,
    },

    Le {
        a: ValueId,
        b: ValueId,
    },

    Ge {
        a: ValueId,
        b: ValueId,
    },

    Not {
        cond: ValueId,
    },

    Max {
        a: ValueId,
        b: ValueId,
    },
    Min {
        a: ValueId,
        b: ValueId,
    },

    Select {
        cond: ValueId,
        a: ValueId,
        b: ValueId,
    },

    ForLoopBegin {
        index: ValueId,
        end: ValueId,
        step: ValueId,
    },

    ForeverLoopBegin,

    Continue,

    Break,

    IfBegin {
        cond: ValueId,
    },

    ElseBegin,

    EndScope,

    Barrier,

    Return,
}

impl Op {
    #[must_use]
    pub fn does_read(&self, value_id: ValueId) -> bool {
        match self {
            Self::Barrier
            | Self::BlockId { .. }
            | Self::Break
            | Self::ConstF32 { .. }
            | Self::ConstI32 { .. }
            | Self::ConstU32 { .. }
            | Self::Continue
            | Self::DefineVar { .. }
            | Self::ElseBegin
            | Self::EndScope
            | Self::ForeverLoopBegin
            | Self::GlobalId { .. }
            | Self::LocalId { .. }
            | Self::ReadMeta { .. }
            | Self::Return
            | Self::Nop => false,
            Self::CopyVar { id } => id == &value_id,
            Self::Abs { x }
            | Self::Exp { x }
            | Self::Log { x }
            | Self::Neg { x }
            | Self::Sqrt { x }
            | Self::Tanh { x }
            | Self::Not { cond: x } => x == &value_id,
            Self::Add { a, b }
            | Self::Div { a, b }
            | Self::Eq { a, b }
            | Self::Ge { a, b }
            | Self::Gt { a, b }
            | Self::Le { a, b }
            | Self::Lt { a, b }
            | Self::Ne { a, b }
            | Self::Max { a, b }
            | Self::Min { a, b }
            | Self::Mod { a, b }
            | Self::Mul { a, b }
            | Self::Pow { a, b }
            | Self::Sub { a, b }
            | Self::Shl { a, b }
            | Self::Shr { a, b } => a == &value_id || b == &value_id,
            Self::Fma { a, b, c } => a == &value_id || b == &value_id || c == &value_id,
            Self::AddAssign { val, .. }
            | Self::DivAssign { val, .. }
            | Self::MulAssign { val, .. }
            | Self::ShlAssign { val, .. }
            | Self::ShrAssign { val, .. }
            | Self::SubAssign { val, .. }
            | Self::OverwriteVar { val, .. } => val == &value_id,
            Self::ForLoopBegin { index, end, step } => {
                index == &value_id || end == &value_id || step == &value_id
            }
            Self::IfBegin { cond } => cond == &value_id,
            Self::ParamAccum { index, value, .. }
            | Self::ParamDiv { index, value, .. }
            | Self::ParamMul { index, value, .. }
            | Self::ParamShl { index, value, .. }
            | Self::ParamShr { index, value, .. }
            | Self::ParamStore { index, value, .. }
            | Self::ParamSub { index, value, .. }
            | Self::SharedAccum { index, value, .. }
            | Self::SharedDiv { index, value, .. }
            | Self::SharedMul { index, value, .. }
            | Self::SharedShl { index, value, .. }
            | Self::SharedShr { index, value, .. }
            | Self::SharedStore { index, value, .. }
            | Self::SharedSub { index, value, .. } => index == &value_id || value == &value_id,
            Self::ParamLoad { index, .. } | Self::SharedLoad { index, .. } => index == &value_id,
            Self::Select { cond, a, b } => cond == &value_id || a == &value_id || b == &value_id,
        }
    }

    #[must_use]
    pub fn does_write(&self, value_id: ValueId) -> bool {
        match self {
            Self::DefineVar { id }
            | Self::AddAssign { id, .. }
            | Self::DivAssign { id, .. }
            | Self::MulAssign { id, .. }
            | Self::ShlAssign { id, .. }
            | Self::ShrAssign { id, .. }
            | Self::SubAssign { id, .. }
            | Self::OverwriteVar { id, .. } => id == &value_id,
            _ => false,
        }
    }

    #[must_use]
    pub const fn writes_to(&self) -> Option<ValueId> {
        match self {
            Self::DefineVar { id }
            | Self::AddAssign { id, .. }
            | Self::DivAssign { id, .. }
            | Self::MulAssign { id, .. }
            | Self::ShlAssign { id, .. }
            | Self::ShrAssign { id, .. }
            | Self::SubAssign { id, .. }
            | Self::OverwriteVar { id, .. } => Some(*id),
            _ => None,
        }
    }

    #[must_use]
    pub const fn is_zero(&self) -> bool {
        match self {
            Self::ConstF32 { value } => *value == 0.0,
            Self::ConstI32 { value } => *value == 0,
            Self::ConstU32 { value } => *value == 0,
            _ => false,
        }
    }

    #[must_use]
    pub const fn is_one(&self) -> bool {
        match self {
            Self::ConstF32 { value } => *value == 1.0,
            Self::ConstI32 { value } => *value == 1,
            Self::ConstU32 { value } => *value == 1,
            _ => false,
        }
    }
}
