#include <stdalign.h>

#define AT_SIZE 100

double call_tabulate_avx(int n)
{
	alignas(32) static const double weights[1][4] = {
		{1.0, 1.0, 1.0, 1.0}
	};

	static const double* w[1] = { &weights[0][0] };

    alignas(32) static const double coords[4][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };

	alignas(32) double A_T[AT_SIZE];
    for(int i = 0; i < n; ++i) {
        tabulate_tensor_avx(&A_T[0], &w[0], &coords[0][0], 0);
    }

	double result = 0.0;
	for(int i = 0; i < AT_SIZE; ++i) {
		result += fabs(A_T[i]);
	}
    
    return result;
}

double call_tabulate_ffc(int n)
{
	alignas(32) static const double weights[1][4] = {
		{1.0, 1.0, 1.0, 1.0}
	};

	static const double* w[1] = { &weights[0][0] };

    alignas(32) static const double coords[4][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };

	alignas(32) double A_T[AT_SIZE];
    for(int i = 0; i < n; ++i) {
        tabulate_tensor_ffc(&A_T[0], &w[0], &coords[0][0], 0);
    }

	double result = 0.0;
	for(int i = 0; i < AT_SIZE; ++i) {
		result += fabs(A_T[i]);
	}
    
    return result;
}
