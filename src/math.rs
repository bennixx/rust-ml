pub struct VecN {
    data: Vec<f64>,
}

impl VecN {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn filled(n: usize, value: f64) -> Self {
        Self {
            data: vec![value; n],
        }
    }

    pub fn dim(&self) -> usize {
        self.data.len()
    }

    pub fn dot(&self, other: &Self) -> f64 {
        assert_eq!(self.dim(), other.dim(), "dimensions mismatch");
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.dim(), other.dim(), "dimensions mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Self { data }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.dim(), other.dim(), "dimensions mismatch");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self { data }
    }

    pub fn scale(&self, s: f64) -> Self {
        let data = self.data.iter().map(|x| x * s).collect();
        Self { data }
    }
}

pub struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "data length does not match dimensions"
        );
        Self { data, rows, cols }
    }

    pub fn filled(rows: usize, cols: usize, value: f64) -> Self {
        Self {
            data: vec![value; rows * cols],
            rows,
            cols,
        }
    }

    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
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
