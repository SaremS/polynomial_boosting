#include <random>
#include <vector>

#include "seedable_rng.h"

std::vector<int> SeedableRNG::get_ints(int min, int max, int N) {
	std::uniform_int_distribution<int> dist(min, max);
	std::vector<int> result;
	for (int i=0; i<N; i++) {
		result.push_back(dist(rng));
	}
	return result;
}

int SeedableRNG::get_int(int min, int max) {
	std::uniform_int_distribution<int> dist(min, max);
	return dist(rng);
}
