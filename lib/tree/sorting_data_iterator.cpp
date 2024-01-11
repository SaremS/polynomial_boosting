#include <vector>

#include "tree/data_iterator.h"
#include "linalg.h"

DataSplit* SortingDataIterator::next_internal() const {
	DataSplit* split = new DataSplit();
	split->X_left = this->X.get_row_range(0, this->min_samples_split + this->current_index);
	split->y_left = this->y.get_row_range(0, this->min_samples_split + this->current_index);
	split->w_left = this->weights.get_row_range(0, this->min_samples_split + this->current_index);
	split->X_right = this->X.get_row_range(this->min_samples_split + this->current_index, this->n_samples);
	split->y_right = this->y.get_row_range(this->min_samples_split + this->current_index, this->n_samples);
	split->w_right = this->weights.get_row_range(this->min_samples_split + this->current_index, this->n_samples);

	split->X_split = this->X.get_row(this->min_samples_split + this->current_index - 1);
	split->y_split = this->y.get_row(this->min_samples_split + this->current_index - 1);

	split->split_value = this->X_sort.get_element_at(this->min_samples_split + this->current_index - 1, 0);

	return split;
};


SortingDataIterator::SortingDataIterator(Matrix X, Matrix y, Matrix weights, Matrix X_sort, int min_samples_split) : DataIterator(X, y, weights, min_samples_split) {
	std::vector<Matrix> matrices = {this->X, this->y, this->weights, X_sort};
		
	std::vector<Matrix> sorted = sort_matrices_by_other_col(matrices, -X_sort, 0);

	this->X = sorted[0];
	this->y = sorted[1];
	this->weights = sorted[2];
	this->X_sort = sorted[3];
};

DataSplit SortingDataIterator::first() const {
	DataSplit split;
	split.X_left = this->X.get_row_range(0, this->min_samples_split);
	split.y_left = this->y.get_row_range(0, this->min_samples_split);
	split.w_left = this->weights.get_row_range(0, this->min_samples_split);
	split.X_right = this->X.get_row_range(this->min_samples_split, this->n_samples);
	split.y_right = this->y.get_row_range(this->min_samples_split, this->n_samples);
	split.w_right = this->weights.get_row_range(this->min_samples_split, this->n_samples);

	split.X_split = this->X.get_row(this->min_samples_split - 1);
	split.y_split = this->y.get_row(this->min_samples_split - 1);

	split.split_value = this->X_sort.get_element_at(this->min_samples_split - 1, 0);
	return split;
};
	
