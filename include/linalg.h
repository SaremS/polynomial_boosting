#ifndef LINALG_H
#define LINALG_H

#include <tuple>
#include <vector>
#include <functional>

#include <Eigen/Dense>

class Matrix {
private:
	size_t n_rows;
	size_t n_cols;

	Eigen::MatrixXd matrix;

public:
	Matrix();
	Matrix(std::vector<std::vector<double>> input);
	Matrix(std::vector<double> input);
	Matrix(double value, size_t n_rows, size_t n_cols); //fill matrix with value
	Matrix(Eigen::MatrixXd input);

	size_t get_n_rows() const;
	size_t get_n_cols() const;
	Eigen::MatrixXd get_as_eigen() const;

	Matrix transpose() const;
	Matrix inverse() const;
	Matrix matrix_multiply(const Matrix &other) const;
	Matrix dot(const Matrix &other) const;

	bool operator==(const Matrix &other) const;
	Matrix operator-(const Matrix &other) const;
	Matrix operator+(const Matrix &other) const;
	Matrix operator-() const;
	friend Matrix operator+(const double &lhs, const Matrix &rhs);
	friend Matrix operator*(const double &lhs, const Matrix &rhs);
    	friend Matrix operator*(const Matrix &lhs, const double &rhs);
	friend Matrix operator*(const Matrix &lhs, const Matrix &rhs);
	Matrix pow_elementwise(const double &exponent) const;
	double sum() const;

	void prepend_ones();
	double get_element_at(const int &row, const int &col) const;
	Matrix get_column(const int &col) const;
	
	std::tuple<Matrix, Matrix> split_col_at_value(
			const int &col, 
			const double &value) const;
	std::tuple<Matrix, Matrix> split_col_by_other_colval(
			const int &thiscol, 
			const Matrix &other, 
			const int &othercol, 
			const double &value) const;

	Matrix replicate(int const &rows, int const &cols) const;
	Matrix get_row(int const &row) const;
	Matrix get_row_range(int const &start, int const &end) const;

	//largest to smallest
	Matrix get_rows_by_other_col_rank(
			Matrix const &other,
			int const &col,
			int const &N) const;

	Matrix pop_n_first_rows(int const &N) const;
	Matrix append_rows(Matrix const &other) const;
	Matrix sample_rows(int const &N, int const &seed) const;

};

Matrix apply_binary(
		Matrix const &left,
		int const &leftcol,
		Matrix const &right,
		int const &rightcol,
		std::function<double(double,double)> const &fun);

Matrix apply_triary(
		Matrix const &left,
		int const &leftcol,
		Matrix const &middle,
		int const &middlecol,
		Matrix const &right,
		int const &rightcol,
		std::function<double(double,double,double)> const &fun);
#endif
