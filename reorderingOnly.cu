#include <iostream>
#include <iomanip>      
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <sys/time.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusparse.h>
#include "matrix.h"
#include "preprocessor.h"

using namespace std;

ofstream fout;

#ifndef NCOL
#define NCOL 512
#endif
#define MFACTOR (32)
#define LENGTH (512/SROW_PER_TILE) // SP=128, DP<128

template <class T>
void reorder(SparseMatrixCSR<T> &sm, DenseMatrix<T> &dm, string filename, string outfile) {
	if (sm.ncols != dm.nrows) {
		cerr << "matrix dimensions do not match!" << endl;
		exit(-1);

	} 


	float tot_ms;
	float cublas_time;

	static const int ITER = 8;

	Preprocessor<float> pp(2, filename, outfile);
	struct timeval ts, te;

	gettimeofday(&ts, NULL);
	pp.reordering_and_tiling(sm);
	gettimeofday(&te, NULL);	
	cout << "REORDERING TIME: " << 1e3 * (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1e3 << endl;
	cudaCheckError();
}


int main(int argc, char *argv[]) {
	SparseMatrixCSR<float> sm;
	if (argc > 2)
	sm.loadCOO(argv[1], true);
	else sm.loadCOO(argv[1]);
	std::cout << "Processing Matrix: " << argv[1] << std::endl;
	std::cout << "Rows: " << sm.get_nrows() << " Cols: " << sm.get_ncols() << " NNZ: " << sm.get_nnz() << std::endl;


	if (sm.get_ncols() * 512 > 3.9e9) { exit(-1); }
	DenseMatrix<float> dm;
	dm.initOne(sm.get_ncols(), NCOL);

	reorder(sm, dm, argv[1], argv[2]);

	return 0;
}
