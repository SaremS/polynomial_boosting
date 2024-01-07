#include <vector>

#include <Eigen/Dense>

#include "gtest/gtest.h"
#include "linalg.h"


TEST(testlinalg, readmat_from_vec) {
	std::vector<double> vals = {1.0,2.0,3.0};
	Matrix testmat = Matrix(vals);	

	EXPECT_EQ(3, testmat.get_n_rows());
	EXPECT_EQ(1, testmat.get_n_cols());
}

TEST(testlinalg, readmat_from_vec_of_vec) {
	std::vector<double> vals = {1.0,2.0,3.0};
	Matrix testmat = Matrix(vals);	


	std::vector<std::vector<double>> target = {
        	{1.0},
        	{2.0},
        	{3.0}
    	};
	Matrix comparemat = Matrix(target);

	EXPECT_EQ(comparemat, testmat);
}

TEST(testlinalg, readmat_fromvalue) {
	Matrix testmat = Matrix(1.0, 3, 1);	

	std::vector<std::vector<double>> target = {
		{1.0},
		{1.0},
		{1.0}
    	};
	Matrix comparemat = Matrix(target);

	EXPECT_EQ(comparemat, testmat);
}

TEST(testlinalg, transpose_vec) {
	std::vector<double> vals = {1.0,2.0,3.0};
	Matrix testmat = Matrix(vals);	

	std::vector<std::vector<double>> target = {
        	{1.0, 2.0, 3.0}
    	};
	Matrix comparemat = Matrix(target);

	EXPECT_EQ(comparemat, testmat.transpose());
}

TEST(testlinalg, invert_unitmat) {
	std::vector<std::vector<double>> target = {
        	{1.0, 0.0},
		{0.0, 1.0}
    	};
	Matrix comparemat = Matrix(target);

	EXPECT_EQ(comparemat.inverse(), comparemat);
}


TEST(testlinalg, unitmat_multiply) {
	std::vector<std::vector<double>> left = {
        	{1.0, 0.0},
		{0.0, 1.0}
    	};
	Matrix leftmat = Matrix(left);

	std::vector<std::vector<double>> right = {
        	{1.0, 2.0},
		{3.0, 4.0}
    	};
	Matrix rightmat = Matrix(right);

	EXPECT_EQ(leftmat.matrix_multiply(rightmat), rightmat);
}

TEST(testlinalg, dotproduct) {
	std::vector<std::vector<double>> left = {
        	{1.0, 2.0},
		{3.0, 4.0}
    	};
	Matrix leftmat = Matrix(left);

	std::vector<std::vector<double>> right = {
        	{1.0},
		{1.0}
    	};
	Matrix rightmat = Matrix(right);

	std::vector<std::vector<double>> target1 = {
        	{3.0},
		{7.0}
    	};
	Matrix targetmat1 = Matrix(target1);

	EXPECT_EQ(leftmat.matrix_multiply(rightmat), targetmat1);

	std::vector<std::vector<double>> target2 = {
        	{4.0},
		{6.0}
    	};
	Matrix targetmat2 = Matrix(target2);

	EXPECT_EQ(leftmat.dot(rightmat), targetmat2);
}

TEST(testlinalg, prepend_ones) {
	std::vector<std::vector<double>> experiment = {
        	{1.0},
		{0.0}
    	};
	Matrix expmat = Matrix(experiment);

	std::vector<std::vector<double>> target = {
        	{1.0, 1.0},
		{1.0, 0.0}
    	};
	Matrix targmat = Matrix(target);

	expmat.prepend_ones();

	EXPECT_EQ(expmat, targmat);
}

TEST(testlinalg, element_at) {
	std::vector<std::vector<double>> target = {
        	{1.0, 2.0},
		{3.0, 4.0}
    	};
	Matrix comparemat = Matrix(target);

	EXPECT_EQ(comparemat.get_element_at(0,0), 1.0);
	EXPECT_EQ(comparemat.get_element_at(0,1), 2.0);
	EXPECT_EQ(comparemat.get_element_at(1,0), 3.0);
	EXPECT_EQ(comparemat.get_element_at(1,1), 4.0);
}

TEST(testlinalg, get_column) {
	std::vector<std::vector<double>> base = {
        	{1.0, 2.0},
		{3.0, 4.0}
    	};
	Matrix basemat = Matrix(base);

	std::vector<double> target1 = {
		{1.0},
		{3.0}
	};
	Matrix targmat1 = Matrix(target1);

	std::vector<double> target2 = {
		{2.0},
		{4.0}
	};
	Matrix targmat2 = Matrix(target2);

	EXPECT_EQ(basemat.get_column(0), targmat1);
	EXPECT_EQ(basemat.get_column(1), targmat2);
}

TEST(testlinalg, subtract) {
	std::vector<std::vector<double>> left = {
        	{1.0, 2.0},
		{3.0, 4.0}
    	};
	Matrix leftmat = Matrix(left);

	std::vector<std::vector<double>> right = {
        	{2.0, 3.0},
		{4.0, 5.0}
    	};
	Matrix rightmat = Matrix(right);

	std::vector<std::vector<double>> expected = {
        	{-1.0, -1.0},
		{-1.0, -1.0}
    	};
	Matrix expmat = Matrix(expected);

	EXPECT_EQ(leftmat - rightmat, expmat);
}

TEST(testlinalg, addition) {
	std::vector<std::vector<double>> left = {
        	{1.0, 2.0},
		{3.0, 4.0}
    	};
	Matrix leftmat = Matrix(left);

	std::vector<std::vector<double>> right = {
        	{2.0, 3.0},
		{4.0, 5.0}
    	};
	Matrix rightmat = Matrix(right);

	std::vector<std::vector<double>> expected = {
        	{3.0, 5.0},
		{7.0, 9.0}
    	};
	Matrix expmat = Matrix(expected);

	EXPECT_EQ(leftmat + rightmat, expmat);
}

TEST(testlinalg, element_wise_multiplication) {
	std::vector<std::vector<double>> left = {
        	{1.0, 2.0},
		{3.0, 4.0}
    	};
	Matrix leftmat = Matrix(left);

	std::vector<std::vector<double>> right = {
        	{2.0, 3.0},
		{4.0, 5.0}
    	};
	Matrix rightmat = Matrix(right);

	std::vector<std::vector<double>> expected = { 
		{2.0, 6.0},
		{12.0, 20.0}
    	};
	Matrix expmat = Matrix(expected);

	EXPECT_EQ(leftmat * rightmat, expmat);
}

TEST(testlinalg, scalar_multiplication) {
	std::vector<std::vector<double>> left = {
        	{1.0, 2.0},
		{3.0, 4.0}
    	};
	Matrix leftmat = Matrix(left);

	std::vector<std::vector<double>> right = {
        	{2.0, 4.0},
		{6.0, 8.0}
    	};
	Matrix rightmat = Matrix(right);

	EXPECT_EQ(2*leftmat, rightmat);
	EXPECT_EQ(leftmat*2, rightmat);
}

TEST(testlinalg, powing) {
	std::vector<std::vector<double>> target = {
        	{1.0, 2.0},
		{3.0, 4.0}
    	};
	Matrix targmat = Matrix(target);

	std::vector<std::vector<double>> expected = {
        	{1.0, 4.0},
		{9.0, 16.0}
    	};
	Matrix expmat = Matrix(expected);

	EXPECT_EQ(targmat.pow_elementwise(2.0), expmat);
}

TEST(testlinalg, summing) {
	std::vector<std::vector<double>> target = {
        	{1.0, 2.0},
		{3.0, 4.0}
    	};
	Matrix targmat = Matrix(target);

	EXPECT_EQ(targmat.sum(), 10.0);
}

TEST(testlinalg, split) {
	std::vector<std::vector<double>> target = {
        	{1.0, 4.0},
		{2.0, 5.0},
		{3.0, 6.0}
    	};
	Matrix targmat = Matrix(target);

	std::vector<std::vector<double>> left = {
        	{1.0},
		{2.0},
    	};

	Matrix leftmat = Matrix(left);

	std::vector<std::vector<double>> right = {
        	{3.0},
    	};

	Matrix rightmat = Matrix(right);

	auto [leftresult, rightresult] = targmat.split_col_at_value(0, 2.0);

	EXPECT_EQ(rightresult, rightmat);
	EXPECT_EQ(leftresult, leftmat);
}

TEST(testlinalg, split_by_other) {
	std::vector<std::vector<double>> target = {
        	{1.0, 4.0},
		{2.0, 5.0},
		{3.0, 6.0}
    	};
	Matrix targmat = Matrix(target);

	std::vector<std::vector<double>> left = {
        	{1.0},
    	};

	Matrix leftmat = Matrix(left);

	std::vector<std::vector<double>> right = {
		{2.0},
        	{3.0}
    	};

	Matrix rightmat = Matrix(right);

	auto [leftresult, rightresult] = targmat.split_col_by_other_colval(
			0, 
			targmat,
			1,
			4.0);

	EXPECT_EQ(rightresult, rightmat);
	EXPECT_EQ(leftresult, leftmat);
}

TEST(testlinalg, apply_binary) {
	std::vector<std::vector<double>> target = {
        	{1.0, 2.0},
		{3.0, 4.0}
    	};
	Matrix targmat = Matrix(target);

	auto sumup = [](double const &left, double const &right) -> double {
		return left + right;
	};

	std::vector<std::vector<double>> expected = {
        	{3.0},
		{7.0}
    	};
	Matrix expmat = Matrix(expected);

	EXPECT_EQ(apply_binary(targmat, 0, targmat, 1, sumup), expmat);
}

TEST(testlinalg, replicate) {
	std::vector<std::vector<double>> target = {
		{1.0, 2.0},
    	};
	Matrix targmat = Matrix(target);

	std::vector<std::vector<double>> expected = {
		{1.0, 2.0},
		{1.0, 2.0},
		{1.0, 2.0}
    	};
	Matrix expmat = Matrix(expected);

	EXPECT_EQ(targmat.replicate(3,1), expmat);
}

TEST(testlinalg, get_row) {
	std::vector<std::vector<double>> target = {
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0}
    	};
	Matrix targmat = Matrix(target);

	std::vector<std::vector<double>> expected = {
		{3.0, 4.0}
    	};
	Matrix expmat = Matrix(expected);

	EXPECT_EQ(targmat.get_row(1), expmat);
}

TEST(testlinalg, get_rows_by_other_col_rank) {
	std::vector<std::vector<double>> target = {
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0}
    	};
	Matrix targmat = Matrix(target);

	std::vector<std::vector<double>> other = {
		{1.0},
		{2.0},
		{3.0}
    	};
	Matrix othermat = Matrix(other);

	std::vector<std::vector<double>> expected = {
		{5.0, 6.0},
		{3.0, 4.0}
    	};
	Matrix expmat = Matrix(expected);

	EXPECT_EQ(targmat.get_rows_by_other_col_rank(othermat, 0, 2), expmat);
}

TEST(testlinalg, append_rows) {
	std::vector<std::vector<double>> target = {
		{1.0, 2.0},
		{3.0, 4.0},
    	};
	Matrix targmat = Matrix(target);

	std::vector<std::vector<double>> other = {
		{5.0, 6.0}
    	};
	Matrix othermat = Matrix(other);

	std::vector<std::vector<double>> expected = {
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0}
    	};
	Matrix expmat = Matrix(expected);

	EXPECT_EQ(targmat.append_rows(othermat), expmat);
}
