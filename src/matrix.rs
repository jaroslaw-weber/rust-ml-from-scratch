extern crate rand;
use std::fmt;
use self::rand::distributions::{IndependentSample, Range};
use std::f32;
use std;

#[derive(Debug, Clone)]
pub struct Matrix
{
	cols:u32,
	rows:u32,
	content: Vec<f32>
}

//2d array/matrix
impl Matrix
{
	//new matrix
	pub fn new(rows:u32, cols:u32, content:Vec<f32>)-> Matrix
	{
		Matrix { cols:cols, rows:rows, content: content}
	}
	
	//copy value from matrix
	pub fn copy_value(&self, row:u32, col:u32)-> f32
	{
		let index=self.get_real_index(row,col);
		*(&self.content[index])
	}

	//get index for content. using for accessing matrix by row and column number
	fn get_real_index(&self, row:u32, col:u32)->usize
	{
		if row== 0 || col == 0 || row>self.rows || col>self.cols
		{
			panic!("out of range!");
		}
		let r=row-1;
		let c=col-1;
		let row_length:u32=*(&self.cols);
		let index:u32=r*row_length+c;
		(index as usize)

	}

	//set value at position
	pub fn set_value(&mut self, row:u32, col:u32, new_value:f32)
	{
		let index=self.get_real_index(row,col);
		self.content[index]=new_value;

	}

	//dot product of two matrices
	#[allow(dead_code)]
	pub fn dot(&self, another_matrix:&Matrix)->Matrix
	{
		if self.cols!=another_matrix.rows
		{
			panic!("size not matching: {}x{} and {}x{}", self.rows, self.cols, another_matrix.rows, another_matrix.cols);
		}
		assert!(self.cols == another_matrix.rows);
		let mut m=Matrix::get_uniform_matrix(self.rows, another_matrix.cols,0_f32);

		for i in 0..self.rows
		{
			for j in 0..another_matrix.cols
			{
				let mut temp=0_f32;
				for k in 0..self.cols
				{
					temp+=self.copy_value(i+1, k+1) * another_matrix.copy_value(k+1, j+1);

				}
				m.set_value(i+1,j+1, temp);
			}
		}
		m
	}
	
	//transposing matrix
	#[allow(dead_code)]
	pub fn transpose(&self)-> Matrix
	{
		let mut nc=Vec::new();
		for c in 1..self.cols+1
		{
			for r in 1..self.rows+1
			{
				let val=self.copy_value(r,c);
				nc.push(val);
			}
		}
		Matrix::new(self.cols, self.rows, nc)
	}

	//get matrix size
	pub fn get_size(&self)->(u32,u32)
	{
		(self.rows, self.cols)
	}

	//get matrix with random numbers from -1 to 1
	pub fn get_random_mean_zero(rows:u32,cols:u32)-> Matrix
	{
		let range=Range::new(-1.,1.);
	
        let mut rng = rand::thread_rng();
		let mut mc=Vec::new();
		for _ in 0..rows
		{
			for _ in 0..cols
			{
			    let random:f32=range.ind_sample(&mut rng);
				mc.push(random);
			}
		}
		Matrix::new(rows, cols, mc)

	}

	//get matrix 
	pub fn get_uniform_matrix(rows:u32, cols:u32, value:f32)->Matrix
	{
		let total_size:u32=rows*cols;
		let mc=vec![value; total_size as usize];
		Matrix::new(rows, cols, mc)
	}

	//sum of all values in matrix. warning: numerically unstable! (read more about floating points)
	pub fn sum(&self)->f32
	{
		let mut sum:f32=0.;
		for c in &self.content
		{
			sum+=*c;
		}
		sum
	}

	//max value in matrix
	pub fn max(&self)->f32
	{
		let mut maxn:f32=f32::MIN;
		if self.content.len()<=0
		{
			panic!("empty matrix!");
		}
		for c in &self.content
		{
			let copy=*c;
			if copy>maxn
			{
				maxn=copy;
			}
		}
		maxn
	}

	//numpy's np.exp
	pub fn exp(&self)->Matrix
	{
		let e:f32=std::f32::consts::E;
		let f=|x:f32|e.powf(x);
		self.map(&f)
	}

	//add a number to all elements
	#[allow(dead_code)]
	pub fn addn(&self, number:f32)->Matrix
	{
		let f=|x|x+number;
		self.map(&f)
	}

	//multiply all elements by a number
	pub fn muln(&self, number:f32)->Matrix
	{
		let f=|x|x*number;
		self.map(&f)
	}

	//multiply all elements by a number
	pub fn divn(&self, number:f32)->Matrix
	{
		let f=|x|x/number;
		self.map(&f)
	}

	//substract a number from all elements
	#[allow(dead_code)]
	pub fn subn(&self, number:f32)->Matrix
	{
		let f=|x|x-number;
		self.map(&f)
	}

	pub fn abs(&self)-> Matrix
	{
		let f=|x:f32|x.abs();
		self.map(&f)
	}

	pub fn mean(&self)->f32
	{
		let mut sum=0_f32;
		for n in &self.content
		{
			sum=sum+(*n);
		}
		sum/(*&self.content.len() as f32)

	}

	//add a matrix
	#[allow(dead_code)]
	pub fn add(&self, other_matrix:&Matrix)->Matrix
	{
		let f=|a,b|a+b;
		self.matrix_on_matrix_operation(other_matrix, &f)
	}

	//multiply matrix (element-wise)
	#[allow(dead_code)]
	pub fn mul(&self, other_matrix:&Matrix)->Matrix
	{
		let f=|a,b|a*b;
		self.matrix_on_matrix_operation(other_matrix, &f)
	}
	
	//subtract matrix
	#[allow(dead_code)]
	pub fn sub(&self, other_matrix:&Matrix)->Matrix
	{
		let f=|a,b|a-b;
		self.matrix_on_matrix_operation(other_matrix, &f)
	}

	//divide by matrix
	#[allow(dead_code)]
	pub fn div(&self, other_matrix:&Matrix)->Matrix
	{
		let f=|a,b|a/b;
		self.matrix_on_matrix_operation(other_matrix, &f)
	}

	//operation on two matrices
	pub fn matrix_on_matrix_operation(&self, other_matrix:&Matrix, combine_function:&Fn(f32,f32)->f32)->Matrix
	{
		let mut v=Vec::new();
		for i in 0..self.content.len()
		{
			let new_val=combine_function(self.content[i], other_matrix.content[i]);
			v.push(new_val);
		}
		Matrix::new(self.rows, self.cols, v)

	}
	
	//map a function to all elements in matrix
	pub fn map(&self, function_to_map:&Fn(f32)->f32)->Matrix
	{
		let mut new_content=Vec::new();
		for x in &self.content
		{
			let new_value=function_to_map(*x);
			new_content.push(new_value);
		}
		Matrix::new(self.rows, self.cols, new_content)
	}



}

//display for matrix. show shape and content
impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    	let (size_rows, size_columns)=self.get_size();
        let mut s=String::new();
        let size_text=format!("Matrix of size: {}x{} (rows x columns)", size_rows, size_columns);
        s.push_str(&size_text);
        s.push('\n');
        s.push('\n');
        let mut count=0;
        for i in &self.content
        {

            //copy value from reference
            let number_copy=*i;
            if number_copy>=0.
            {
                s.push(' ');
            }
			let formatted_number = format!("{:.*}", 2, number_copy);
            s.push_str(&formatted_number);
            s.push(' ');
        	count=count+1;
            if count%size_columns==0
            {
            	s.push('\n');
        	}
        }
        write!(f, "{}", s)
    }
}