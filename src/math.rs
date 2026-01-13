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

    pub fn get(&self, r: usize, c: usize) -> f64 {
        assert!(r < self.rows && c < self.cols);
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        assert!(r < self.rows && c < self.cols);
        self.data[r * self.cols + c] = v;
    }

    pub fn transpose(&self) -> Self {
        let mut transposed = Self {
            data: vec![0.0; self.data.len()],
            rows: self.cols,
            cols: self.rows,
        };

        for r in 0..self.rows {
            for c in 0..self.cols {
                transposed.set(c, r, self.get(r, c));
            }
        }

        transposed
    }

    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(
            self.cols, other.rows,
            "matmul dimensions mismatch: {}x{} · {}x{}",
            self.rows, self.cols, other.rows, other.cols
        );

        let mut product = Self {
            data: vec![0.0; self.rows * other.cols],
            rows: self.rows,
            cols: other.cols,
        };

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                product.set(i, j, sum);
            }
        }

        product
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn assert_approx_eq(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() < eps, "expected {a} ≈ {b} (eps {eps})");
    }

    #[test]
    fn test_dot() {
        let a = VecN::new(vec![1.0, 2.0, 3.0]);
        let b = VecN::new(vec![4.0, 5.0, 6.0]);
        assert_approx_eq(a.dot(&b), 32.0, 1e-6);
    }

    #[test]
    fn test_norm() {
        assert_approx_eq(VecN::new(vec![3.0, 4.0]).norm(), 5.0, 1e-6);
    }

    #[test]
    fn test_transpose() {
        let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

        let t = a.transpose();

        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_approx_eq(t.get(0, 0), 1.0, 1e-6);
        assert_approx_eq(t.get(0, 1), 4.0, 1e-6);
        assert_approx_eq(t.get(1, 0), 2.0, 1e-6);
        assert_approx_eq(t.get(1, 1), 5.0, 1e-6);
        assert_approx_eq(t.get(2, 0), 3.0, 1e-6);
        assert_approx_eq(t.get(2, 1), 6.0, 1e-6);
    }

    #[test]
    fn test_matmul() {
        let a = Matrix::new(vec![4.0, 3.0, 2.0, 1.0, 2.0, 3.0], 2, 3);
        let b = Matrix::new(vec![2.0, 3.0, 4.0], 3, 1);
        let c = a.matmul(&b);

        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 1);
        assert_approx_eq(c.get(0, 0), 25.0, 1e-6);
        assert_approx_eq(c.get(1, 0), 20.0, 1e-6);
    }
}
