struct VecN {
    data: Vec<f64>,
}

impl VecN {
    fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    fn filled(n: usize, value: f64) -> Self {
        Self {
            data: vec![value; n],
        }
    }

    fn dim(&self) -> usize {
        self.data.len()
    }

    fn dot(&self, other: &Self) -> f64 {
        assert_eq!(self.dim(), other.dim(), "dimensions mismatch");
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    fn add(&self, other: &Self) -> Self {
        assert_eq!(self.dim(), other.dim(), "dimensions mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self { data }
    }

    fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.dim(), other.dim(), "dimensions mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self { data }
    }

    fn scale(&self, s: f64) -> Self {
        let data = self.data.iter().map(|x| x * s).collect();
        Self { data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_dot() {
        let a = VecN::new(vec![1.0, 2.0, 3.0]);
        let b = VecN::new(vec![4.0, 5.0, 6.0]);
        assert!(approx_eq(a.dot(&b), 32.0, 1e-6));
    }

    #[test]
    fn test_norm() {
        assert!(approx_eq(VecN::new(vec![3.0, 4.0]).norm(), 5.0, 1e-6));
    }
}

fn main() {
    println!("Hello, world!");
}
