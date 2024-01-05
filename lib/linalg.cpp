#include <vector>
#include <tuple>

#include <linalg.h>

#include <Eigen/Dense>

Matrix::Matrix() {
	n_rows = 0;
	n_cols = 0;

	matrix = Eigen::MatrixXd::Zero(1,1); 
}

Matrix::Matrix(std::vector<std::vector<double>> input) {
	n_rows = input.size();
	n_cols = input[0].size();

	Eigen::MatrixXd mat(n_rows, n_cols);

	for (size_t i=0; i<n_rows; i++) {
		for(size_t j=0; j<n_cols; j++) {
			mat(i,j) = input[i][j];
		}
	}

	matrix = mat;
}

Matrix::Matrix(std::vector<double> input) {
	n_rows = input.size();
	n_cols = 1;

	Eigen::MatrixXd mat(n_rows, n_cols);

	for (size_t i=0; i<n_rows; i++) {
		for(size_t j=0; j<n_cols; j++) {
			mat(i,j) = input[i];
		}
	}

	matrix = mat;
}

Matrix::Matrix(double value, size_t n_rows, size_t n_cols) {
	this->n_rows = n_rows;
	this->n_cols = n_cols;

	Eigen::MatrixXd mat(n_rows, n_cols);

	for (size_t i=0; i<n_rows; i++) {
		for(size_t j=0; j<n_cols; j++) {
			mat(i,j) = value;
		}
	}

	matrix = mat;
}

Matrix::Matrix(Eigen::MatrixXd input) {
	n_rows = static_cast<size_t>(input.rows());
	n_cols = static_cast<size_t>(input.cols());

	matrix = input;
}

size_t Matrix::get_n_rows() const {
	return this->n_rows;
}

size_t Matrix::get_n_cols() const {
	return this->n_cols;
}

Eigen::MatrixXd Matrix::get_as_eigen() const {
	return this->matrix;
}

Matrix Matrix::transpose() const {
	return Matrix(this->matrix.transpose());
}

Matrix Matrix::inverse() const {
	return Matrix(this->matrix.inverse());
}

Matrix Matrix::matrix_multiply(const Matrix &other) const {
	return Matrix(this->matrix * other.get_as_eigen());
}

Matrix Matrix::dot(const Matrix &other) const {
	return Matrix(this->matrix.transpose() * other.get_as_eigen());
}

bool Matrix::operator==(const Matrix &other) const {
	return this->matrix == other.get_as_eigen();
}

Matrix Matrix::operator-(const Matrix &other) const {
	return Matrix(this->matrix - other.get_as_eigen());
}

Matrix Matrix::operator+(const Matrix &other) const {
	return Matrix(this->matrix + other.get_as_eigen());
}

Matrix operator*(const double &lhs, const Matrix &rhs) {
    	return Matrix(rhs.matrix * lhs);
}

Matrix operator*(const Matrix &lhs, const double &rhs) {
    	return Matrix(lhs.matrix * rhs);
}

Matrix operator*(const Matrix &lhs, const Matrix &rhs) {
	Eigen::MatrixXd result = lhs.matrix.array() * rhs.matrix.array();
    	return Matrix(result);
}

Matrix Matrix::pow_elementwise(const double &exponent) const {
	return Matrix(this->matrix.array().pow(exponent));
}

double Matrix::sum() const {
	return this->matrix.sum();
}

void Matrix::prepend_ones() {
	Eigen::MatrixXd new_matrix(this->n_rows, this->n_cols + 1);

	new_matrix.col(0) = Eigen::VectorXd::Ones(this->n_rows);
	new_matrix.block(0, 1, this->n_rows, this->n_cols) = matrix;

	this->matrix = new_matrix;
}

double Matrix::get_element_at(const int &row, const int &col) const {
	return this->matrix(row, col);
}

Matrix Matrix::get_column(const int &col) const {
	return Matrix(this->matrix.col(col).eval());
}

std::tuple<Matrix, Matrix> Matrix::split_col_at_value(const int &col, const double &value) const {
	std::vector<double> left;
	std::vector<double> right;

	for (int i=0; i<this->n_rows; i++) {
		double val = this->matrix(i, col);

		if (val <= value) {
			left.push_back(val);
		} else {
			right.push_back(val);
		}
	}

	Matrix leftmat = Matrix(left);
	Matrix rightmat = Matrix(right);

	return std::make_tuple(left, right);
}

std::tuple<Matrix, Matrix> Matrix::split_col_by_other_colval(
		const int &thiscol,
		const Matrix &other,
		const int &othercol,
		const double &value) const {
	std::vector<double> left;
	std::vector<double> right;

	for (int i=0; i<this->n_rows; i++) {
		double thisval = this->matrix(i, thiscol);
		double otherval = other.get_element_at(i, othercol);

		if (otherval <= value) {
			left.push_back(thisval);
		} else {
			right.push_back(thisval);
		}
	}

	Matrix leftmat = Matrix(left);
	Matrix rightmat = Matrix(right);

	return std::make_tuple(left, right);
}

Matrix apply_binary(
		Matrix const &left,
		int const &leftcol,
		Matrix const &right,
		int const &rightcol,
		std::function<double(double,double)> const &fun) {
	
	int n_rows = left.get_n_rows();

	std::vector<double> result;

	for (int i=0; i<n_rows; i++) {
		double leftval = left.get_element_at(i, leftcol);
		double rightval = right.get_element_at(i, rightcol);

		result.push_back(fun(leftval, rightval));
	}

	return Matrix(result);
}
