#![feature(generic_const_exprs)]

//! The detailed jacobian derivation process is at [`doc::jacobian`].

#![cfg_attr(not(feature = "std"), no_std)]
#![feature(stmt_expr_attributes)]

#[allow(unused_imports)]
#[macro_use]
extern crate alloc;

use alloc::vec::Vec;
pub use transform::Transformable;
use transform::Transformer;

pub mod doc;
pub mod se2;
pub mod so2;
pub mod transform;

mod huber;
mod linalg;
mod norm;
mod stats;
mod types;

pub use crate::norm::norm;
pub use crate::transform::Transform;
pub use crate::types::{Rotation2, Vector, Vector2, Vector3};
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;

pub type Param = nalgebra::Vector3<f32>;
pub type Param3d = nalgebra::Vector4<f32>;
type Jacobian = nalgebra::Matrix2x3<f32>;
type Jacobian3d = nalgebra::Matrix3x4<f32>;
type Hessian = nalgebra::Matrix3<f32>;
type Hessian3d = nalgebra::Matrix4<f32>;

const HUBER_K: f32 = 1.345;

pub fn residual<const D: usize>(
    transformer: &impl Transformer<D>,
    src: &Vector<D>,
    dst: &Vector<D>,
) -> Vector<D> {
    transformer.transform(src) - dst
}

pub fn error<const D: usize>(
    transformer: &impl Transformer<D>,
    src: &[Vector<D>],
    dst: &[Vector<D>],
) -> f32 {
    src.iter().zip(dst.iter()).fold(0f32, |sum, (s, d)| {
        let r = residual(transformer, s, d);
        sum + r.dot(&r)
    })
}

pub fn huber_error<const D: usize>(
    transformer: &impl Transformer<D>,
    src: &[Vector<D>],
    dst: &[Vector<D>],
) -> f32 {
    src.iter().zip(dst.iter()).fold(0f32, |sum, (s, d)| {
        let r = residual(transformer, s, d);
        sum + huber::rho(r.dot(&r), HUBER_K)
    })
}

pub fn estimate_transform_2d(src: &[Vector<2>], dst: &[Vector<2>]) -> (Transform<2>, f32) {
    let delta_norm_threshold: f32 = 1e-6;
    let max_iter: usize = 200;

    let mut transform = Transform::identity();
    let mut prev_error = huber_error(&transform, src, dst);

    for _ in 0..max_iter {
        let Some(delta) = weighted_gauss_newton_update_2d(&transform, src, dst) else {
            break;
        };

        if delta.dot(&delta) < delta_norm_threshold {
            break;
        }

        let t = Transform::<2>::new(&delta) * transform;

        let error = huber_error(&t, src, dst);
        if error > prev_error {
            break;
        }
        transform = t;
        prev_error = error;
    }
    (transform, prev_error)
}

pub fn estimate_transform_3d(src: &[Vector<3>], dst: &[Vector<3>]) -> (Transform<3>, f32) {
    let delta_norm_threshold: f32 = 1e-6;
    let max_iter: usize = 200;

    let mut transform = Transform::<3>::identity();
    let mut prev_error = huber_error(&transform, src, dst);

    for _ in 0..max_iter {
        let Some(delta) = weighted_gauss_newton_update_3d(&transform, src, dst) else {
            break;
        };

        if delta.dot(&delta) < delta_norm_threshold {
            break;
        }

        let t = Transform::<3>::new(&delta) * transform;

        let error = huber_error(&t, src, dst);
        if error > prev_error {
            break;
        }
        transform = t;
        prev_error = error;
    }
    (transform, prev_error)
}

pub struct Icp2d {
    pub kdtree: KdTree<f32, usize, [f32; 2]>,
    pub dst: Vec<Vector2>,
}

impl Icp2d {
    pub fn new(dst: Vec<Vector2>) -> Self {
        let mut kdtree = KdTree::new(2);
        for (i, p) in dst.iter().enumerate() {
            kdtree.add([p.x, p.y], i).unwrap();
        }
        Icp2d { kdtree, dst }
    }

    /// Estimates the transform that converts the `src` points to `dst`.
    pub fn estimate(
        &self,
        src: &[Vector<2>],
        initial_transform: &Transform<2>,
        max_iter: usize,
    ) -> (Transform<2>, f32) {
        let mut transform = *initial_transform;
        let mut prev_error = f32::MAX;
        for _ in 0..max_iter {
            let src_tranformed = src.transformed(&transform);
            let nearest_dsts = self.get_nearest_dsts(&src_tranformed);
            let (dtransform, error) = estimate_transform_2d(&src_tranformed, &nearest_dsts);

            transform = dtransform * transform;
            prev_error = error;
        }
        (transform, prev_error)
    }

    pub fn get_nearest_dsts(&self, src: &[Vector2]) -> Vec<Vector2> {
        src.iter()
            .map(|&p| {
                let results = self
                    .kdtree
                    .nearest(&[p.x, p.y], 1, &squared_euclidean)
                    .unwrap();
                let first = results.first().unwrap();
                self.dst[*first.1]
            })
            .collect()
    }
}

pub struct Icp3d {
    pub kdtree: KdTree<f32, usize, [f32; 3]>,
    pub dst: Vec<Vector3>,
}

impl Icp3d {
    pub fn new(dst: Vec<Vector3>) -> Self {
        let mut kdtree = KdTree::new(3);
        for (i, p) in dst.iter().enumerate() {
            kdtree.add([p.x, p.y, p.z], i).unwrap();
        }
        Icp3d { kdtree, dst }
    }

    /// Estimates the transform on the xy-plane that converts the `src` points to `dst`.
    /// This function assumes that the vehicle, LiDAR or other point cloud scanner is moving on the xy-plane.
    pub fn estimate(
        &self,
        src: &[Vector<3>],
        initial_transform: &Transform<3>,
        max_iter: usize,
    ) -> (Transform<3>, f32) {
        let mut transform = *initial_transform;
        let mut prev_error = f32::MAX;
        for _ in 0..max_iter {
            let src_tranformed = src.transformed(&transform);
            let nearest_dsts = self.get_nearest_dsts(&src_tranformed);
            let (dtransform, error) = estimate_transform_3d(&src_tranformed, &nearest_dsts);

            transform = dtransform * transform;
            prev_error = error;
        }
        (transform, prev_error)
    }

    pub fn get_nearest_dsts(&self, src: &[Vector3]) -> Vec<Vector3> {
        src.iter()
            .map(|&p| {
                let results = self
                    .kdtree
                    .nearest(&[p.x, p.y, p.z], 1, &squared_euclidean)
                    .unwrap();
                let first = results.first().unwrap();
                self.dst[*first.1]
            })
            .collect()
    }
}

fn jacobian(rot: &Rotation2, landmark: &Vector2) -> Jacobian {
    let a = Vector2::new(-landmark[1], landmark[0]);
    let r = rot.matrix();
    let b = rot * a;
    #[rustfmt::skip]
    Jacobian::new(
        r[(0, 0)], r[(0, 1)], b[0],
        r[(1, 0)], r[(1, 1)], b[1])
}

// Calculate the Jacobian for 3D points with rotation only around the z-axis
fn jacobian_3d(rot: &Rotation2, landmark: &Vector3) -> Jacobian3d {
    // Rotation around the z-axis affects only the x and y components of the 3D point.
    let a = Vector2::new(-landmark[1], landmark[0]);
    let r = rot.matrix();
    let b = rot * a;

    #[rustfmt::skip]
    Jacobian3d::new(
        r[(0, 0)], r[(0, 1)], 0.0, b[0], // x-translation, rotation affects x and y
        r[(1, 0)], r[(1, 1)], 0.0, b[1], // y-translation, rotation affects x and y
        0.0,             0.0, 1.0,  0.0  // z-translation, no effect from rotation
    )
}

fn check_input_size<const D: usize>(input: &[Vector<D>]) -> bool {
    // Check if the input does not have sufficient samples to estimate the update
    !input.is_empty() && input.len() >= input[0].len()
}

pub fn gauss_newton_update(
    transform: &Transform<2>,
    src: &[Vector<2>],
    dst: &[Vector<2>],
) -> Option<Param> {
    if !check_input_size(src) {
        // The input does not have sufficient samples to estimate the update
        return None;
    }

    let (jtr, jtj) = src.iter().zip(dst.iter()).fold(
        (Param::zeros(), Hessian::zeros()),
        |(jtr, jtj), (s, d)| {
            let j = jacobian(&transform.rot, s);
            let r = transform.transform(s) - d;
            let jtr_: Param = j.transpose() * r;
            let jtj_: Hessian = j.transpose() * j;
            (jtr + jtr_, jtj + jtj_)
        },
    );
    // TODO Check matrix rank before solving linear equation
    linalg::inverse3x3(&jtj).map(|jtj_inv| -jtj_inv * jtr)
}

pub fn weighted_gauss_newton_update_2d(
    transform: &Transform<2>,
    src: &[Vector<2>],
    dst: &[Vector<2>],
) -> Option<Param> {
    debug_assert_eq!(src.len(), dst.len());

    if !check_input_size(src) {
        // The input does not have sufficient samples to estimate the update
        return None;
    }

    let residuals = src
        .iter()
        .zip(dst.iter())
        .map(|(s, d)| residual(transform, s, d))
        .collect::<Vec<_>>();

    let stddevs = stats::calc_stddevs(&residuals)?;

    let mut param = Param::zeros();
    let mut hessian = Hessian::zeros();
    for (source, residual) in src.iter().zip(residuals.iter()) {
        let jacobian = jacobian(&transform.rot, source);
        for (row_index, jacobian_row) in jacobian.row_iter().enumerate() {
            if stddevs[row_index] == 0. {
                continue;
            }
            let g = 1. / stddevs[row_index];
            let row_residual = residual[row_index];
            let d_roh = huber::drho(row_residual * row_residual, HUBER_K);

            param += d_roh * g * jacobian_row.transpose() * row_residual;
            hessian += d_roh * g * jacobian_row.transpose() * jacobian_row;
        }
    }

    linalg::inverse3x3(&hessian).map(|inverse_hessian| -inverse_hessian * param)
}

pub fn weighted_gauss_newton_update_3d(
    transform: &Transform<3>,
    src: &[Vector<3>],
    dst: &[Vector<3>],
) -> Option<Param3d> {
    debug_assert_eq!(src.len(), dst.len());

    if !check_input_size(src) {
        // The input does not have sufficient samples to estimate the update
        return None;
    }

    let residuals = src
        .iter()
        .zip(dst.iter())
        .map(|(s, d)| residual(transform, s, d))
        .collect::<Vec<_>>();

    let stddevs = stats::calc_stddevs(&residuals)?;

    let mut param = Param3d::zeros();
    let mut hessian = Hessian3d::zeros();
    for (source, residual) in src.iter().zip(residuals.iter()) {
        let jacobian = jacobian_3d(&transform.rot, source);
        for (row_index, jacobian_row) in jacobian.row_iter().enumerate() {
            if stddevs[row_index] == 0. {
                continue;
            }
            let g = 1. / stddevs[row_index];
            let row_residual = residual[row_index];
            let d_roh = huber::drho(row_residual * row_residual, HUBER_K);

            param += d_roh * g * jacobian_row.transpose() * row_residual;
            hessian += d_roh * g * jacobian_row.transpose() * jacobian_row;
        }
    }

    linalg::inverse4x4(hessian).map(|inverse_hessian| -inverse_hessian * param)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual() {
        let param: Param = Vector3::new(-10., 20., 0.01);
        let transform = Transform::new(&param);
        let src = Vector2::new(7f32, 8f32);
        let dst = transform.transform(&src);
        assert_eq!(residual(&transform, &src, &dst), Vector2::zeros());
    }

    #[test]
    fn test_error() {
        let src = vec![
            Vector2::new(-6., 9.),
            Vector2::new(-1., 9.),
            Vector2::new(-4., -4.),
        ];

        let dst = vec![
            Vector2::new(-4., 4.),
            Vector2::new(0., 3.),
            Vector2::new(-3., -8.),
        ];

        let param: Param = Vector3::new(10., 20., 0.01);
        let transform = Transform::new(&param);
        let r0 = residual(&transform, &src[0], &dst[0]);
        let r1 = residual(&transform, &src[1], &dst[1]);
        let r2 = residual(&transform, &src[2], &dst[2]);
        let expected = r0.dot(&r0) + r1.dot(&r1) + r2.dot(&r2);
        assert_eq!(error(&transform, &src, &dst), expected);
    }

    #[test]
    fn test_gauss_newton_update_input_size() {
        let param = Param::new(10.0, 30.0, -0.15);
        let transform = Transform::new(&param);

        let src = vec![];
        let dst = vec![];
        assert!(gauss_newton_update(&transform, &src, &dst).is_none());

        let src = vec![Vector2::new(-8.89304516, 0.54202289)];
        let dst = vec![transform.transform(&src[0])];
        assert!(gauss_newton_update(&transform, &src, &dst).is_none());

        let src = vec![
            Vector2::new(-8.89304516, 0.54202289),
            Vector2::new(-4.03198385, -2.81807802),
        ];
        let dst = vec![transform.transform(&src[0]), transform.transform(&src[1])];
        assert!(gauss_newton_update(&transform, &src, &dst).is_some());
    }

    #[test]
    fn test_gauss_newton_update() {
        let true_param = Param::new(10.0, 30.0, -0.15);
        let dparam = Param::new(0.3, -0.5, 0.001);
        let initial_param = true_param + dparam;
        let true_transform = Transform::new(&true_param);
        let initial_transform = Transform::new(&initial_param);

        let src = vec![
            Vector2::new(-8.76116663, 3.50338231),
            Vector2::new(-5.21184804, -1.91561705),
            Vector2::new(6.63141168, 4.8915293),
            Vector2::new(-2.29215281, -4.72658399),
            Vector2::new(6.81352587, -0.81624617),
        ];
        let dst = src
            .iter()
            .map(|p| true_transform.transform(&p))
            .collect::<Vec<_>>();

        let Some(update) = gauss_newton_update(&initial_transform, &src, &dst) else {
            panic!("Return value cannot be None");
        };
        let updated_param = initial_param + update;

        let initial_transform = Transform::new(&initial_param);
        let updated_transform = Transform::new(&updated_param);

        let e0 = error(&initial_transform, &src, &dst);
        let e1 = error(&updated_transform, &src, &dst);
        assert!(e1 < e0 * 0.01);
    }

    #[test]
    fn test_weighted_gauss_newton_update_input_size() {
        let param = Param::new(10.0, 30.0, -0.15);
        let transform = Transform::new(&param);

        // insufficient input size
        let src = vec![];
        let dst = vec![];
        assert!(weighted_gauss_newton_update_2d(&transform, &src, &dst).is_none());

        // insufficient input size
        let src = vec![Vector2::new(-8.89304516, 0.54202289)];
        let dst = vec![transform.transform(&src[0])];
        assert!(weighted_gauss_newton_update_2d(&transform, &src, &dst).is_none());

        // insufficient input size
        let src = vec![
            Vector2::new(-8.89304516, 0.54202289),
            Vector2::new(-4.03198385, -2.81807802),
        ];
        let dst = vec![transform.transform(&src[0]), transform.transform(&src[1])];
        assert!(weighted_gauss_newton_update_2d(&transform, &src, &dst).is_none());

        // sufficient input size but rank is insufficient
        let src = vec![
            Vector2::new(-8.89304516, 0.54202289),
            Vector2::new(-4.03198385, -2.81807802),
            Vector2::new(-4.03198385, -2.81807802),
        ];
        let dst = vec![
            transform.transform(&src[0]),
            transform.transform(&src[1]),
            transform.transform(&src[2]),
        ];
        assert!(weighted_gauss_newton_update_2d(&transform, &src, &dst).is_none());

        // sufficient input size but rank is insufficient
        let src = vec![
            Vector2::new(-8.89304516, 0.54202289),
            Vector2::new(-4.03198385, -2.81807802),
            Vector2::new(4.40356349, -9.43358563),
        ];
        let dst = vec![
            transform.transform(&src[0]),
            transform.transform(&src[1]),
            transform.transform(&src[2]),
        ];
        assert!(weighted_gauss_newton_update_2d(&transform, &src, &dst).is_none());
    }

    #[test]
    fn test_weighted_gauss_newton_update_zero_x_diff() {
        let src = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 0.1),
            Vector2::new(0.0, 0.2),
            Vector2::new(0.0, 0.3),
            Vector2::new(0.0, 0.4),
            Vector2::new(0.0, 0.5),
        ];

        let true_param = Param::new(0.00, 0.01, 0.00);
        let true_transform = Transform::new(&true_param);

        let dst = src
            .iter()
            .map(|p| true_transform.transform(p))
            .collect::<Vec<Vector2>>();

        let initial_param = Param::new(0.00, 0.00, 0.00);
        let initial_transform = Transform::new(&initial_param);
        // TODO Actually there is some error, but Hessian is not invertible so
        // the update cannot be calculated
        assert!(weighted_gauss_newton_update_2d(&initial_transform, &src, &dst).is_none());
    }

    #[test]
    fn test_weighted_gauss_newton_update() {
        let true_param = Param::new(10.0, 30.0, -0.15);
        let dparam = Param::new(0.3, -0.5, 0.001);
        let initial_param = true_param + dparam;

        let true_transform = Transform::new(&true_param);
        let initial_transform = Transform::new(&initial_param);

        let src = vec![
            Vector2::new(-8.89304516, 0.54202289),
            Vector2::new(-4.03198385, -2.81807802),
            Vector2::new(-5.92679530, 9.62339266),
            Vector2::new(-4.04966218, -4.44595403),
            Vector2::new(-2.86369420, -9.13843999),
            Vector2::new(-6.97749644, -8.90180581),
            Vector2::new(-9.66454985, 6.32282424),
            Vector2::new(7.02264007, -0.88684585),
            Vector2::new(4.19700110, -1.42366424),
            // Vector2::new(-1.98903219, -0.96437383),  // corresponds to the large noise
            Vector2::new(-0.68034875, -0.48699014),
            Vector2::new(1.89645382, 1.86119400),
            Vector2::new(7.09550743, 2.18289525),
            Vector2::new(-7.95383118, -5.16650913),
            Vector2::new(-5.40235599, 2.70675665),
            Vector2::new(-5.38909696, -5.48180288),
            Vector2::new(-9.00498232, -5.12191142),
            Vector2::new(-8.54899319, -3.25752055),
            Vector2::new(6.89969814, 3.53276123),
            Vector2::new(5.06875729, -0.28918540),
        ];

        // noise follow the normal distribution with
        // mean 0.0 and standard deviation 0.01
        let noise = [
            Vector2::new(0.01058790, 0.01302535),
            Vector2::new(0.01392508, 0.00835860),
            Vector2::new(0.01113885, -0.00693269),
            Vector2::new(0.01673124, -0.01735564),
            Vector2::new(-0.01219263, 0.00080933),
            Vector2::new(-0.00396817, 0.00111582),
            Vector2::new(-0.00444043, 0.00658505),
            Vector2::new(-0.01576271, -0.00701065),
            Vector2::new(0.00464000, -0.00406790),
            // Vector2::new(-0.32268585,  0.49653010),  // Much larger noise than others
            Vector2::new(0.00269374, -0.00787015),
            Vector2::new(-0.00494243, 0.00350137),
            Vector2::new(0.00343766, -0.00039311),
            Vector2::new(0.00661565, -0.00341112),
            Vector2::new(-0.00936695, -0.00673899),
            Vector2::new(-0.00240039, -0.00314409),
            Vector2::new(-0.01434128, -0.00585390),
            Vector2::new(0.00874225, 0.00295633),
            Vector2::new(0.00736213, -0.00328875),
            Vector2::new(0.00585082, -0.01232619),
        ];

        assert_eq!(src.len(), noise.len());
        let dst = src
            .iter()
            .zip(noise.iter())
            .map(|(p, n)| true_transform.transform(&p) + n)
            .collect::<Vec<_>>();
        let Some(update) = weighted_gauss_newton_update_2d(&initial_transform, &src, &dst) else {
            panic!("Return value cannot be None");
        };
        let updated_param = initial_param + update;
        let updated_transform = Transform::new(&updated_param);

        let e0 = error(&initial_transform, &src, &dst);
        let e1 = error(&updated_transform, &src, &dst);
        assert!(e1 < e0 * 0.1);

        let updated_transform = estimate_transform_2d(&src, &dst);

        let e0 = error(&initial_transform, &src, &dst);
        let e1 = error(&updated_transform, &src, &dst);
        assert!(e1 < e0 * 0.001);
    }

    #[test]
    fn test_icp_3dscan() {
        let src = vec![
            Vector3::new(0.0, 0.0, 2.0),
            Vector3::new(0.0, 0.1, 2.0),
            Vector3::new(0.0, 0.2, 2.0),
            Vector3::new(0.0, 0.3, 2.0),
            Vector3::new(0.0, 0.4, 2.0),
            Vector3::new(0.0, 0.5, 2.0),
            Vector3::new(0.0, 0.6, 2.0),
            Vector3::new(0.0, 0.7, 2.0),
            Vector3::new(0.0, 0.8, 2.0),
            Vector3::new(0.0, 0.9, 2.0),
            Vector3::new(0.0, 1.0, 2.0),
            Vector3::new(0.1, 0.0, 1.0),
            Vector3::new(0.2, 0.0, 1.0),
            Vector3::new(0.3, 0.0, 1.0),
            Vector3::new(0.4, 0.0, 1.0),
            Vector3::new(0.5, 0.0, 1.0),
            Vector3::new(0.6, 0.0, 1.0),
            Vector3::new(0.7, 0.0, 1.0),
            Vector3::new(0.8, 0.0, 1.0),
            Vector3::new(0.9, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 1.0),
        ];

        let true_transform = Transform::new(&Param::new(0.01, 0.01, -0.02));

        let dst = src
            .iter()
            .map(|p| transform_xy(&true_transform, &p))
            .collect::<Vec<Vector3>>();

        let noise = Transform::new(&Param::new(0.05, 0.010, 0.010));
        let initial_transform = noise * true_transform;
        let icp = Icp3d::new(&dst);
        let pred_transform = icp.estimate(&src, &initial_transform, 20);

        for (sp, dp_true) in src.iter().zip(dst.iter()) {
            let dp_pred = transform_xy(&pred_transform, &sp);
            assert!(norm(&(dp_pred - dp_true)) < 1e-3);
        }
    }

    #[test]
    fn test_icp_2dscan() {
        let src = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 0.1),
            Vector2::new(0.0, 0.2),
            Vector2::new(0.0, 0.3),
            Vector2::new(0.0, 0.4),
            Vector2::new(0.0, 0.5),
            Vector2::new(0.0, 0.6),
            Vector2::new(0.0, 0.7),
            Vector2::new(0.0, 0.8),
            Vector2::new(0.0, 0.9),
            Vector2::new(0.0, 1.0),
            Vector2::new(0.1, 0.0),
            Vector2::new(0.2, 0.0),
            Vector2::new(0.3, 0.0),
            Vector2::new(0.4, 0.0),
            Vector2::new(0.5, 0.0),
            Vector2::new(0.6, 0.0),
            Vector2::new(0.7, 0.0),
            Vector2::new(0.8, 0.0),
            Vector2::new(0.9, 0.0),
            Vector2::new(1.0, 0.0),
        ];

        let true_transform = Transform::new(&Param::new(0.01, 0.01, -0.02));

        let dst = src
            .iter()
            .map(|p| true_transform.transform(p))
            .collect::<Vec<Vector2>>();

        let noise = Transform::new(&Param::new(0.05, 0.010, 0.010));
        let initial_transform = noise * true_transform;
        let icp = Icp2d::new(&dst);
        let pred_transform = icp.estimate(&src, &initial_transform, 20);

        for (sp, dp_true) in src.iter().zip(dst.iter()) {
            let dp_pred = pred_transform.transform(&sp);
            assert!(norm(&(dp_pred - dp_true)) < 1e-3);
        }
    }
}
