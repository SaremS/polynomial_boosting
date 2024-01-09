#ifndef DATA_SPLITTER_H
#define DATA_SPLITTER_H

#include "linalg.h"


//helper methods to iterate over candidate splits
//should declutter the tree class fit methods

struct DataSplit {
	Matrix X_left;
	Matrix y_left;
	Matrix w_left;
	Matrix X_right;
	Matrix y_right;
	Matrix w_right;

	Matrix X_split;
	Matrix y_split;
};


class DataIterator {
private:
	int feature_index; //might not be needed

	virtual DataSplit* next_internal() const = 0;

protected:
	Matrix X;
	Matrix y;
	Matrix weights;

	int n_samples;
	int min_samples_split; //minimum samples per left and right split
	int current_index;

public:
	DataIterator(Matrix X, Matrix y, Matrix weights, int feature_index, int min_samples_split):
		X(X.get_column(feature_index)),
		y(y),
		weights(weights),
		feature_index(feature_index),
		n_samples(X.get_n_rows()),
		min_samples_split(min_samples_split),
		current_index(0) {};
	

	DataSplit* next() {
		if (current_index >= this->n_samples - this->min_samples_split) {
			return nullptr;
		}
		
		DataSplit* split = this->next_internal();
		this->current_index++;
		return split;
	};	

	int get_n_samples() const {
		return this->n_samples;
	};

	void reset() {
		this->current_index = 0;	
	};

	virtual DataSplit first() const = 0;

	virtual ~DataIterator() {};
};


class SortingDataIterator : public DataIterator {
private:
	DataSplit* next_internal() const;
public:
	SortingDataIterator(Matrix X, Matrix y, Matrix weights, int feature_index, int min_samples_split);
	DataSplit first() const;
	//destructor inherited from DataIterator
	~SortingDataIterator() {};
};


#endif
