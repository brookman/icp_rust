use crate::types::{Rotation2, Vector2, Vector3};
use crate::{se2, transform_xy};

use alloc::vec::Vec;
use core::ops::Mul;

#[derive(Copy, Clone, Debug)]
pub struct Transform {
    pub rot: Rotation2,
    pub t: Vector2,
}

impl Transform {
    pub fn new(param: &Vector3) -> Self {
        let (rot, t) = se2::calc_rt(param);
        Transform { rot, t }
    }

    pub fn from_rt(rot: &Rotation2, t: &Vector2) -> Self {
        Transform { rot: *rot, t: *t }
    }

    pub fn transform(&self, landmark: &Vector2) -> Vector2 {
        self.rot * landmark + self.t
    }

    pub fn inverse(&self) -> Self {
        let inv_rot = self.rot.inverse();
        Transform {
            rot: inv_rot,
            t: -(inv_rot * self.t),
        }
    }

    pub fn identity() -> Self {
        Transform {
            rot: Rotation2::identity(),
            t: Vector2::zeros(),
        }
    }
}

impl Mul for Transform {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Transform {
            rot: self.rot * rhs.rot,
            t: self.rot * rhs.t + self.t,
        }
    }
}

pub trait Transformable<T> {
    fn transformed(&self, transform: &Transform) -> Vec<T>;
}

impl Transformable<Vector2> for &[Vector2] {
    fn transformed(&self, transform: &Transform) -> Vec<Vector2> {
        self.iter()
            .map(|sp| transform.transform(&sp))
            .collect::<Vec<Vector2>>()
    }
}

impl Transformable<Vector3> for &[Vector3] {
    fn transformed(&self, transform: &Transform) -> Vec<Vector3> {
        self.iter()
            .map(|sp| transform_xy(&transform, &sp))
            .collect::<Vec<Vector3>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::norm::norm;
    use crate::so2;
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    #[test]
    fn test_transform() {
        let r = so2::new_rotation2(FRAC_PI_2);
        let t = Vector2::new(3., 6.);
        let transform = Transform::from_rt(&r, &t);

        let x = Vector2::new(4., 2.);
        let expected = Vector2::new(-2. + 3., 4. + 6.);
        assert!(norm(&(transform.transform(&x) - expected)) < 1e-8);
    }

    #[test]
    fn test_inverse() {
        let r = so2::new_rotation2(FRAC_PI_2);
        let t = Vector2::new(3., 6.);
        let transform = Transform::from_rt(&r, &t).inverse();
        let x = Vector2::new(-2. + 3., 4. + 6.);
        let expected = Vector2::new(4., 2.);
        assert!(norm(&(transform.transform(&x) - expected)) < 1e-8);
    }

    #[test]
    fn test_mul() {
        let r1 = so2::new_rotation2(FRAC_PI_4);
        let t1 = Vector2::new(2., 1.);
        let r2 = so2::new_rotation2(FRAC_PI_2);
        let t2 = Vector2::new(5., 3.);
        let transform1 = Transform::from_rt(&r1, &t1).inverse();
        let transform2 = Transform::from_rt(&r2, &t2).inverse();

        let x = Vector2::new(-5., 6.);
        let pa = transform1.transform(&transform2.transform(&x));
        let pb = (transform1 * transform2).transform(&x);

        assert!(norm(&(pa - pb)) < 1e-8);
    }
}
