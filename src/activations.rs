use matrix::Matrix;

//sigmoid function with derivative option for backprop
pub fn sigmoid(x:&Matrix, derivative:bool)->Matrix
{
	let m;
	//todo static ones to be more efficient?
	let f=|_|1.;
	let ones=x.map(&f);
	//calculating sigmoid derivative
	if derivative
	{
		//return x*(1-x)
		m=x.mul(&ones.sub(x));
	}
	//sigmoid function
	else {
		//1/(1+np.exp(-x));
		let minusx=x.muln(-1.);
		m=ones.div(&ones.add(&minusx.exp()))
	}
	m
}

//softmax function. warning: numerically unstable! use softmax_stable instead
pub fn softmax(x:&Matrix)->Matrix
{
	let exp=x.exp();
	let sum=exp.sum();
	let out=exp.divn(sum);
	out
}

pub fn softmax_stable(x:&Matrix)->Matrix
{
	let submax=x.subn(x.max());
	let ex=submax.exp();
	return ex.divn(ex.sum());
}

//relu is not differentiable, but there is a convention to calculate it
pub fn relu(x:&Matrix, derivative:bool)->Matrix
{
	let out;
	if derivative
	{

		let f=|n:f32|
		{
			let mut nout=1.;
			if n<0. {nout=0.;}
			nout
		};
		out = x.map(&f);
	}
	else {
		let f=|n:f32|
		{
			let mut nout=n;
			if n<=0. {nout=0.;}
			nout
		};
		out = x.map(&f);
	}
	out
}