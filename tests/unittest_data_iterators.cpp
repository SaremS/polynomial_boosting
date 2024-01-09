#include "gtest/gtest.h"
#include "tree/data_iterator.h"
#include "linalg.h"

TEST(testdataiterators, simple) {
	std::vector<double> inputs = {
		2.0,
		3.0,
		4.0,
		5.0,
		6.0,
		7.0,
		8.0,
		9.0,
		10.0,
		11.0
	};

	Matrix X = Matrix(inputs);
	Matrix y = Matrix(inputs);
	Matrix weights = Matrix(inputs);

	DataIterator* data_iterator = new SortingDataIterator(X, y, weights, 0, 1);

	DataSplit split = data_iterator->first();

	EXPECT_EQ(split.X_left.get_n_rows(), 1);
	EXPECT_EQ(split.X_right.get_n_rows(), 9);
	EXPECT_EQ(split.y_left.get_n_rows(), 1);
	EXPECT_EQ(split.y_right.get_n_rows(), 9);
	EXPECT_EQ(split.w_left.get_n_rows(), 1);

	for (int i=0; i<9; i++) {
		split = data_iterator->next();

		EXPECT_EQ(split.X_left.get_n_rows(), 1 + i);
		EXPECT_EQ(split.X_right.get_n_rows(), 9 - i);
		EXPECT_EQ(split.y_left.get_n_rows(), 1 + i);
		EXPECT_EQ(split.y_right.get_n_rows(), 9 - i);
		EXPECT_EQ(split.w_left.get_n_rows(), 1 + i);
		EXPECT_EQ(split.w_right.get_n_rows(), 9 - i);
	}

	split = data_iterator->next();

	EXPECT_EQ(split.X_left.get_n_rows(), 1);
	EXPECT_EQ(split.X_right.get_n_rows(), 9);
	EXPECT_EQ(split.y_left.get_n_rows(), 1);
	EXPECT_EQ(split.y_right.get_n_rows(), 9);
	EXPECT_EQ(split.w_left.get_n_rows(), 1);
	EXPECT_EQ(split.w_right.get_n_rows(), 9);
}

