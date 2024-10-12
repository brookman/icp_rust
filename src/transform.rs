use crate::se2;
use crate::types::{Rotation, Rotation2, Vector, Vector2, Vector3};

use alloc::vec::Vec;
use core::ops::Mul;

#[derive(Copy, Clone, Debug)]
pub struct Transform<const D: usize> {
    pub rot: Rotation2,
    pub t: Vector<D>,
}

pub trait Transformer<const D: usize> {
    fn new(param: &Vector<{ D + 1 }>) -> Self;
    fn transform(&self, landmark: &Vector<D>) -> Vector<D>;
    fn from_rt(rot: &Rotation<2>, t: &Vector<D>) -> Self;
    fn inverse(&self) -> Self;
    fn identity() -> Self;
}

impl Transformer<2> for Transform<2> {
    fn new(param: &Vector3) -> Self {
        let (rot, t) = se2::calc_rt(param);
        Self { rot, t }
    }

    fn transform(&self, landmark: &Vector2) -> Vector2 {
        self.rot * landmark + self.t
    }

    fn from_rt(rot: &Rotation2, t: &Vector2) -> Self {
        Self { rot: *rot, t: *t }
    }

    fn inverse(&self) -> Self {
        let inv_rot = self.rot.inverse();
        Self {
            rot: inv_rot,
            t: -(inv_rot * self.t),
        }
    }

    fn identity() -> Self {
        Self {
            rot: Rotation2::identity(),
            t: Vector2::zeros(),
        }
    }
}

impl Mul for Transform<2> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            rot: self.rot * rhs.rot,
            t: self.rot * rhs.t + self.t,
        }
    }
}

impl Transformer<3> for Transform<3> {
    fn new(param: &Vector<4>) -> Self {
        let param_2d = Vector3::new(param[0], param[1], param[3]);
        let (rot, t) = se2::calc_rt(&param_2d);
        Self {
            rot,
            t: Vector3::new(t[0], t[1], param[2]),
        }
    }

    fn transform(&self, landmark: &Vector3) -> Vector3 {
        let r = self.rot * landmark.xy();
        Vector3::new(r.x, r.y, landmark.z) + self.t
    }

    fn from_rt(rot: &Rotation2, t: &Vector3) -> Self {
        Self { rot: *rot, t: *t }
    }

    fn inverse(&self) -> Self {
        let inv_rot = self.rot.inverse();
        let r = inv_rot * self.t.xy();
        Self {
            rot: inv_rot,
            t: -Vector3::new(r.x, r.y, self.t.z),
        }
    }

    fn identity() -> Self {
        Self {
            rot: Rotation2::identity(),
            t: Vector3::zeros(),
        }
    }
}

impl Mul for Transform<3> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let r = self.rot * rhs.t.xy();
        Self {
            rot: self.rot * rhs.rot,
            t: Vector3::new(r.x, r.y, rhs.t.z) + self.t,
        }
    }
}

pub trait Transformable<T, const D: usize> {
    fn transformed(&self, transformer: &impl Transformer<D>) -> Vec<T>;
}

impl<const D: usize> Transformable<Vector<D>, D> for &[Vector<D>] {
    fn transformed(&self, transformer: &impl Transformer<D>) -> Vec<Vector<D>> {
        self.iter().map(|sp| transformer.transform(&sp)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::norm::norm;
    use crate::so2;
    use core::f32::consts::{FRAC_PI_2, FRAC_PI_4};

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
