use matrix::Matrix;

//basic layer functionality ( without calculating delta )
pub struct SimpleLayer {
    weights: Matrix,
}

impl SimpleLayer {
    pub fn new(input_size: u32, output_size: u32) -> SimpleLayer {
        SimpleLayer {
            weights: Matrix::get_random_mean_zero(input_size, output_size),
        }
    }

    pub fn forward(&self, input: &Matrix) -> Matrix {
        input.dot(&self.weights)
    }

    pub fn update_weights(&mut self, previous_layer_output: &Matrix, delta: &Matrix) {
        let weights_update = previous_layer_output.transpose().dot(delta);
        self.weights = self.weights.add(&weights_update);
    }

    pub fn get_previous_layer_error(&self, delta: &Matrix) -> Matrix {
        delta.dot(&self.weights.transpose())
    }
}