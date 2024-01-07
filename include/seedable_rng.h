#ifndef SEEDABLE_RNG_H
#define SEEDABLE_RNG_H

#include <random>
#include <vector>

class SeedableRNG {
private:
	std::mt19937 rng;
public:
	SeedableRNG(int seed): rng(seed) {};
	std::vector<int> get_ints(int min, int max, int N);
	int get_int(int min, int max);
};

#endif
