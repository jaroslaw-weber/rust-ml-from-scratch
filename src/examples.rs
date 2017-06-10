extern crate image;
use matrix::Matrix;
use activations::sigmoid;
use layer::SimpleLayer;
//use image;

//3-layer neural network, numpy-style
#[allow(dead_code)]
pub fn three_layers_neural_network() {
    //create train data
    let (x, y) = get_input_and_output_for_examples();
    println!("x: {}", x);
    println!("y: {}", y);

    let mut w0 = Matrix::get_random_mean_zero(3, 4);
    let mut w1 = Matrix::get_random_mean_zero(4, 1);

    let epochs = 1000;
    let print_step = epochs / 10;
    //training
    for i in 0..epochs {
        //feed forward
        let layer0 = &x;
        let layer1 = sigmoid(&layer0.dot(&w0), false);
        let layer2 = sigmoid(&layer1.dot(&w1), false);
        //total error
        let layer2error = y.sub(&layer2);
        //print out about 10 times
        if i % print_step == 0 {
            println!("error {}", layer2error.abs().mean());
        }
        //get gradients and errors
        let layer2delta = layer2error.mul(&sigmoid(&layer2, true));
        let layer1error = layer2delta.dot(&w1.transpose());
        //println!("{:?}", layer1error.get_size());
        let layer1delta = layer1error.mul(&sigmoid(&layer1, true));

        //update weights
        let w1update = layer1.transpose().dot(&layer2delta);
        let w0update = layer0.transpose().dot(&layer1delta);
        w1 = w1.add(&w1update);
        w0 = w0.add(&w0update);
    }
    //test
    let l1 = sigmoid(&x.dot(&w0), false);
    let y: Matrix = sigmoid(&l1.dot(&w1), false);
    println!("{}", y);
}

//3 layer network using SimpleLayer
pub fn modular_model() {
    //get input output
    let (x, y) = get_input_and_output_for_examples();
    println!("x: {}", x);
    println!("y: {}", y);

    //create layers with weights
    let mut l1 = SimpleLayer::new(3, 5);
    let mut l2 = SimpleLayer::new(5, 1);

    //train parameteres
    let epochs = 10000;
    let print_step = epochs / 10;
    //train
    for i in 0..epochs {

        //forward
        let o1 = l1.forward(&x);
        let o1a = sigmoid(&o1, false);
        let o2 = l2.forward(&o1a);
        let o2a = sigmoid(&o2, false);

        //error
        let l2error = y.sub(&o2a);
        let l2delta = l2error.mul(&sigmoid(&o2a, true));
        let l1error = l2.get_previous_layer_error(&l2delta);
        let l1delta = l1error.mul(&sigmoid(&o1a, true));

        //update
        l2.update_weights(&o1a, &l2delta);
        l1.update_weights(&x, &l1delta);

        //show error
        if i % print_step == 0 {
            println!("error {}", &l2error.abs().mean());
        }

    }
    //test

    //forward
    let o1 = l1.forward(&x);
    let o1a = sigmoid(&o1, false);
    let o2 = l2.forward(&o1a);
    let o2a = sigmoid(&o2, false);

    print!("{}", o2a);

}


fn get_input_and_output_for_examples() -> (Matrix, Matrix) {

    let x = Matrix::new(4, 3, vec![0., 0., 1., 0., 1., 1., 1., 0., 1., 1., 1., 1.]);
    let y = Matrix::new(4, 1, vec![0., 0., 1., 1.]);
    (x, y)
}

