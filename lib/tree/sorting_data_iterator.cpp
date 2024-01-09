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

	return split;
};


SortingDataIterator::SortingDataIterator(Matrix X, Matrix y, Matrix weights, int feature_index, int min_samples_split) : DataIterator(X, y, weights, feature_index, min_samples_split) {
	this->weights = this->weights.get_rows_by_other_col_rank(-this->X, 0, this->n_samples);
	this->y = this->y.get_rows_by_other_col_rank(-this->X, 0, this->n_samples);
	this->X = this->X.get_rows_by_other_col_rank(-this->X, 0, this->n_samples);
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
	return split;
};
	
