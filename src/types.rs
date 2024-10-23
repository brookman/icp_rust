use alloc::vec::Vec;
use nalgebra::base::dimension::Const;
use nalgebra::{ArrayStorage, Matrix, U1};

pub type Vector<const D: usize> = Matrix<f32, Const<D>, U1, ArrayStorage<f32, D, 1>>;
pub type MatrixNxN<const N: usize> = Matrix<f32, Const<N>, Const<N>, ArrayStorage<f32, N, N>>;
pub type Rotation<const D: usize> = nalgebra::Rotation<f32, D>;

pub type Matrix2 = MatrixNxN<2>;
pub type Matrix3 = MatrixNxN<3>;
pub type Vector2 = Vector<2>;
pub type Vector3 = Vector<3>;
pub type Rotation2 = Rotation<2>;
pub type DebugInfo2d = Vec<Vec<(Vector2, Vector2)>>;
pub type DebugInfo3d = Vec<Vec<(Vector3, Vector3)>>;
