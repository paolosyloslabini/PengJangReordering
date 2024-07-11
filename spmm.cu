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
#include <iterator>

using namespace std;

ofstream fout;

#ifndef NCOL
#define NCOL 512
#endif
#define MFACTOR (32)
#define LENGTH (512/SROW_PER_TILE) // SP=128, DP<128

template <class T>
__global__
void _spmm_sparse2_atomic(int nrows, const int * __restrict__ rowptr, const int * __restrict__ rowidx, const int * __restrict__ colidx, const T * __restrict__ values, int nrows_d, int ncols_d, const T * __restrict__ values_d, T * __restrict__ values_res) {

	__shared__ T values_buf[SROW_PER_TILE][LENGTH];
	__shared__ int colidx_buf[SROW_PER_TILE][LENGTH];

	int r = blockIdx.x * SROW_PER_TILE + threadIdx.y;
	if (r < nrows) {
		int c = blockIdx.y * 64 + threadIdx.x;
		int c2 = c + 32;

		T pares = (T)0;
		T pares2 = (T)0;

		for (int b=rowptr[r]; b<rowptr[r+1]; b+=LENGTH) {
			int length = LENGTH > rowptr[r+1]-b? rowptr[r+1]-b: LENGTH;

			for (int i=threadIdx.x; i < length; i+=32) {
				values_buf[threadIdx.y][i] = values[i+b];	
				colidx_buf[threadIdx.y][i] = colidx[i+b] * ncols_d;	
			} 
			//printf ("%d %f %d\n", length, values[b], b);

			for (int k=0; k<length; k++) {
				pares += values_buf[threadIdx.y][k] * values_d[colidx_buf[threadIdx.y][k] + c];
				pares2 += values_buf[threadIdx.y][k] * values_d[colidx_buf[threadIdx.y][k] + c2];
			}
		} 

	//	if (rowptr[r+1] - rowptr[r] > 0) {
			int oidx = rowidx[r]*ncols_d + c;
			atomicAdd(&values_res[oidx], pares);
			atomicAdd(&values_res[oidx+32], pares2);
	//	}
	}
}




template <class T>
__global__
void _spmm_sparse2_bak(int nrows, const int * __restrict__ rowptr, const int * __restrict__ rowidx, const int * __restrict__ colidx, const T * __restrict__ values, int nrows_d, int ncols_d, const T * __restrict__ values_d, T * __restrict__ values_res) {

	__shared__ T values_buf[SROW_PER_TILE][LENGTH];
	__shared__ int colidx_buf[SROW_PER_TILE][LENGTH];

	int r = blockIdx.x * SROW_PER_TILE + threadIdx.y;
	if (r < nrows) {
		int c = blockIdx.y * 64 + threadIdx.x;
		int c2 = c + 32;

		T pares = (T)0;
		T pares2 = (T)0;

		for (int b=rowptr[r]; b<rowptr[r+1]; b+=LENGTH) {
			int length = LENGTH > rowptr[r+1]-b? rowptr[r+1]-b: LENGTH;

			for (int i=threadIdx.x; i < length; i+=32) {
				values_buf[threadIdx.y][i] = values[i+b];	
				colidx_buf[threadIdx.y][i] = colidx[i+b] * ncols_d;	
			} 
			//printf ("%d %f %d\n", length, values[b], b);

			for (int k=0; k<length; k++) {
				pares += values_buf[threadIdx.y][k] * values_d[colidx_buf[threadIdx.y][k] + c];
				pares2 += values_buf[threadIdx.y][k] * values_d[colidx_buf[threadIdx.y][k] + c2];
			}
		} 

		if (rowptr[r+1] - rowptr[r] > 0) {
			int oidx = rowidx[r]*ncols_d + c;
			values_res[oidx] = pares;
			values_res[oidx+32] = pares2;
		}
	}
}



template <class T>
__global__
void _spmm_sparse3(int nrows, const int * __restrict__ rowptr, const int * __restrict__ rowidx, const int * __restrict__ colidx, const T * __restrict__ values, int nrows_d, int ncols_d, const T * __restrict__ values_d, T * __restrict__ values_res) {

	__shared__ T values_buf[SROW_PER_TILE1][SPARSE3_THRESHOLD];
	__shared__ int colidx_buf[SROW_PER_TILE1][SPARSE3_THRESHOLD];

	int r = blockIdx.x * SROW_PER_TILE1 + threadIdx.y;
	if (r<nrows) {
		int c = blockIdx.y * 64 + threadIdx.x;
		int c2 = c + 32;

		T pares = (T)0;
		T pares2 = (T)0;

		int b = rowptr[r];
		int length = rowptr[r+1] - b; 
		if (length > 0) {
			for (int i=threadIdx.x; i < length; i+=32) {
				values_buf[threadIdx.y][i] = values[i+b];	
				colidx_buf[threadIdx.y][i] = colidx[i+b] * ncols_d;	
			} 

			for (int k=0; k<length; k++) {
				pares += values_buf[threadIdx.y][k] * values_d[colidx_buf[threadIdx.y][k] + c];
				pares2 += values_buf[threadIdx.y][k] * values_d[colidx_buf[threadIdx.y][k] + c2];
			}

			int oidx = rowidx[r]*ncols_d + c;
			values_res[oidx] = pares;
			values_res[oidx+32] = pares2;
		}
	}
}



template <class T>
__global__
void _spmm_dense_tiles(int ntiles, const int * __restrict__ row_tile_ptr, const int *__restrict__ col_tile_ptr, const int *__restrict__ prefetch_colidx, const int *__restrict__ rowptr_tile, const int *__restrict__ rowidx_tile, const int *__restrict__ colidx, const T *__restrict__ values, const T *__restrict__ values_d, T *__restrict__ values_res, int ncols_d) {

	__shared__ T sm_input[DCOL_PER_TILE][64];

	int tile_id = blockIdx.x;
	const int *rowptr = &(rowptr_tile[row_tile_ptr[tile_id]]);
	const int *rowidx = &(rowidx_tile[row_tile_ptr[tile_id]]);

	int base_start = col_tile_ptr[tile_id];
	int base_end = col_tile_ptr[tile_id+1];

	int c = threadIdx.x;
	int c2 = c + 32;

	int d = c + blockIdx.y * 64;
	int d2 = c2 + blockIdx.y * 64;

	for(int i=threadIdx.y; i<base_end - base_start; i+=32) {
		sm_input[i][c] = values_d[prefetch_colidx[i]*ncols_d + d]; // something like threadIdx.x + blockIdx.y*64?
		sm_input[i][c2] = values_d[prefetch_colidx[i]*ncols_d + d2];	
		
	}

	__syncthreads();

	for (int r=threadIdx.y; r<DROW_PER_TILE; r+=blockDim.y) {
	//	if (r < nrows) {

			T pares = (T)0;
			T pares2 = (T)0;
			int loc1 = rowptr[r], loc2 = rowptr[r+1];

			int buf; T buf2;
			int interm = loc1 + (((loc2 - loc1)>>3)<<3);
			int interm2 = loc1 + (((loc2 - loc1)>>2)<<2);
			int interm3 = loc1 + (((loc2 - loc1)>>1)<<1);

			int jj=0, l;
			for(l=loc1; l<interm; l+=8) {
				if(jj == 0) {
					buf = colidx[l+c];
					buf2 = values[l+c];
				}
				T v1 = __shfl_sync(0xFFFFFFFF, buf2, jj,MFACTOR);
				T v2 = __shfl_sync(0xFFFFFFFF, buf2, jj+1,MFACTOR);
				int i1 = __shfl_sync(0xFFFFFFFF, buf, jj,MFACTOR);
				int i2 = __shfl_sync(0xFFFFFFFF, buf, jj+1,MFACTOR);
				pares += v1 * sm_input[i1][c];
				pares2 += v1 * sm_input[i1][c2];
				pares += v2 * sm_input[i2][c];
				pares2 += v2 * sm_input[i2][c2];

				T v3 = __shfl_sync(0xFFFFFFFF, buf2, jj+2,MFACTOR);
				T v4 = __shfl_sync(0xFFFFFFFF, buf2, jj+3,MFACTOR);
				int i3 = __shfl_sync(0xFFFFFFFF, buf, jj+2,MFACTOR);
				int i4 = __shfl_sync(0xFFFFFFFF, buf, jj+3,MFACTOR);
				pares += v3 * sm_input[i3][c];
				pares2 += v3 * sm_input[i3][c2];
				pares += v4 * sm_input[i4][c];
				pares2 += v4 * sm_input[i4][c2];

				T v5 = __shfl_sync(0xFFFFFFFF, buf2, jj+4,MFACTOR);
				T v6 = __shfl_sync(0xFFFFFFFF, buf2, jj+5,MFACTOR);
				int i5 = __shfl_sync(0xFFFFFFFF, buf, jj+4,MFACTOR);
				int i6 = __shfl_sync(0xFFFFFFFF, buf, jj+5,MFACTOR);
				pares += v5 * sm_input[i5][c];
				pares2 += v5 * sm_input[i5][c2];
				pares += v6 * sm_input[i6][c];
				pares2 += v6 * sm_input[i6][c2];

				T v7 = __shfl_sync(0xFFFFFFFF, buf2, jj+6,MFACTOR);
				T v8 = __shfl_sync(0xFFFFFFFF, buf2, jj+7,MFACTOR);
				int i7 = __shfl_sync(0xFFFFFFFF, buf, jj+6,MFACTOR);
				int i8 = __shfl_sync(0xFFFFFFFF, buf, jj+7,MFACTOR);
				pares += v7 * sm_input[i7][c];
				pares2 += v7 * sm_input[i7][c2];
				pares += v8 * sm_input[i8][c];
				pares2 += v8 * sm_input[i8][c2];

				jj = ((jj+8)&(MFACTOR-1));
			}
			if(interm < loc2 && jj == 0) {
				buf = colidx[l+c];
				buf2 = values[l+c];
			}
			if(interm < interm2) {
				T v1 = __shfl_sync(0xFFFFFFFF, buf2, jj,MFACTOR);
				T v2 = __shfl_sync(0xFFFFFFFF, buf2, jj+1,MFACTOR);
				int i1 = __shfl_sync(0xFFFFFFFF, buf, jj,MFACTOR);
				int i2 = __shfl_sync(0xFFFFFFFF, buf, jj+1,MFACTOR);
				pares += v1 * sm_input[i1][c];
				pares2 += v1 * sm_input[i1][c2];
				pares += v2 * sm_input[i2][c];
				pares2 += v2 * sm_input[i2][c2];

				T v3 = __shfl_sync(0xFFFFFFFF, buf2, jj+2,MFACTOR);
				T v4 = __shfl_sync(0xFFFFFFFF, buf2, jj+3,MFACTOR);
				int i3 = __shfl_sync(0xFFFFFFFF, buf, jj+2,MFACTOR);
				int i4 = __shfl_sync(0xFFFFFFFF, buf, jj+3,MFACTOR);
				pares += v3 * sm_input[i3][c];
				pares2 += v3 * sm_input[i3][c2];
				pares += v4 * sm_input[i4][c];
				pares2 += v4 * sm_input[i4][c2];

				jj = (jj+4);
			}
			if(interm2 < interm3) {
				T v1 = __shfl_sync(0xFFFFFFFF, buf2, jj,MFACTOR);
				T v2 = __shfl_sync(0xFFFFFFFF, buf2, jj+1,MFACTOR);
				int i1 = __shfl_sync(0xFFFFFFFF, buf, jj,MFACTOR);
				int i2 = __shfl_sync(0xFFFFFFFF, buf, jj+1,MFACTOR);
				pares += v1 * sm_input[i1][c];
				pares2 += v1 * sm_input[i1][c2];
				pares += v2 * sm_input[i2][c];
				pares2 += v2 * sm_input[i2][c2];

				jj = (jj+2);
			}
			if(interm3 < loc2) {
				pares += __shfl_sync(0xFFFFFFFF, buf2, jj,MFACTOR) * sm_input[__shfl_sync(0xFFFFFFFF, buf, jj,MFACTOR)][c];
				pares2 += __shfl_sync(0xFFFFFFFF, buf2, jj,MFACTOR) * sm_input[__shfl_sync(0xFFFFFFFF, buf, jj,MFACTOR)][c2];
			}
			if (rowptr[r+1] - rowptr[r] > 0) {
				atomicAdd(&(values_res[rowidx[r]*ncols_d + c + blockIdx.y*64]), pares);
				atomicAdd(&(values_res[rowidx[r]*ncols_d + c2 + blockIdx.y*64]), pares2);
			}
	}
}


template <class T>
void spmm(SparseMatrixCSR<T> &sm, DenseMatrix<T> &dm, string filename) {
	if (sm.ncols != dm.nrows) {
		cerr << "matrix dimensions do not match!" << endl;
		exit(-1);

	} 


	float tot_ms;
	float cublas_time;

	static const int ITER = 8;


	cudaEvent_t event1, event2;
	cudaEventCreate(&event1);
	cudaEventCreate(&event2);


	{

		// cublas 


		DenseMatrix<T> res2;
		res2.initEmpty(sm.nrows, dm.ncols);

		float alpha = 1.0f, beta = 0.0f;
		cusparseMatDescr_t descra = 0;
		cusparseHandle_t handle = 0;

		cusparseCreate(&handle);
		cusparseCreateMatDescr(&descra);

		cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO);


		cudaDeviceSynchronize();
		cudaEventRecord(event1, 0);

		for (int i=0; i<ITER; i++)
		cusparseScsrmm2(handle,  CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, sm.nrows, dm.ncols, sm.ncols, sm.nnz, &alpha, descra, sm.values, sm.rowptr, sm.colidx, dm.values, dm.ncols, &beta, res2.values, res2.nrows);

		cudaEventRecord(event2, 0);

		cudaEventSynchronize(event1);
		cudaEventSynchronize(event2);
		cudaEventElapsedTime(&tot_ms, event1, event2);

		cudaDeviceSynchronize();

		res2.display(true);



		cout << "cublas time: " << tot_ms / ITER << endl;
		fout << "cublas time: " << tot_ms / ITER << endl;
		cublas_time = tot_ms;
	}

// tiling + reordering + sparse_reordering
	{


		Preprocessor<float> pp(2, filename);
		struct timeval ts, te;

		gettimeofday(&ts, NULL);
		pp.reordering_and_tiling(sm);
		gettimeofday(&te, NULL);	

		fout << "reordering time: " << 1e3 * (te.tv_sec - ts.tv_sec) + (te.tv_usec - ts.tv_usec) / 1e3 << endl;
		cudaCheckError();
		DenseMatrix<T> res1;
		res1.initEmpty(sm.nrows, dm.ncols);


		cudaStream_t stream1, stream2;
		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);


		dim3 nblocks(pp.dense_data.get_ntiles(), DIV(dm.ncols, 64));
		dim3 thrperblock(32, 32);
		dim3 nblocks_s1(DIV(pp.sparse_csr1.nrows, SROW_PER_TILE), DIV(dm.ncols, 64));
		dim3 nblocks_s2(DIV(pp.sparse_csr2.nrows, SROW_PER_TILE1), DIV(dm.ncols, 64));
		dim3 threadPerBlock1(32, SROW_PER_TILE);
		dim3 threadPerBlock2(32, SROW_PER_TILE1);

		cudaDeviceSynchronize();
		cudaEventRecord(event1, 0);

		for (int i=0; i<ITER; i++) {
		if (pp.sparse_csr1.nnz > 0)
		_spmm_sparse2_bak<<<nblocks_s1, threadPerBlock1, 0, stream1>>>(pp.sparse_csr1.nrows, pp.sparse_csr1.rowptr, pp.sparse_csr1.rowidx, pp.sparse_csr1.colidx, pp.sparse_csr1.values, dm.nrows, dm.ncols, dm.values, res1.values);
		_spmm_sparse3<<<nblocks_s2, threadPerBlock2, 0, stream2>>>(pp.sparse_csr2.nrows, pp.sparse_csr2.rowptr, pp.sparse_csr2.rowidx, pp.sparse_csr2.colidx, pp.sparse_csr2.values, dm.nrows, dm.ncols, dm.values, res1.values);
		if (pp.sparse_csr3.nnz >0)
		_spmm_sparse2_atomic<<<nblocks_s1, threadPerBlock1>>>(pp.sparse_csr3.nrows, pp.sparse_csr3.rowptr, pp.sparse_csr3.rowidx, pp.sparse_csr3.colidx, pp.sparse_csr3.values, dm.nrows, dm.ncols, dm.values, res1.values);
		if (pp.dense_data.get_ntiles() > 0)
		_spmm_dense_tiles<<<nblocks, thrperblock>>>(pp.dense_data.get_ntiles(), pp.dense_data.row_tile_ptr, pp.dense_data.col_tile_ptr, pp.dense_data.prefetch_colidx, pp.dense_data.rowptr, pp.dense_data.rowidx, pp.dense_data.colidx, pp.dense_data.values, dm.values, res1.values, dm.ncols);
		}

		cudaEventRecord(event2, 0);

		cudaEventSynchronize(event1);
		cudaEventSynchronize(event2);
		cudaEventElapsedTime(&tot_ms, event1, event2);

		cudaDeviceSynchronize();

		res1.display();


		cout << "aspt-rr time: " << tot_ms / ITER << endl;
		fout << "our time: " <<  tot_ms / ITER << endl;
		fout << "speedup: " << cublas_time / tot_ms << endl;
	}
	


	//return move(res1);

}

int main(int argc, char *argv[]) {
	SparseMatrixCSR<float> sm;
	if (argc > 2)
	sm.loadCOO(argv[1], true);
	else sm.loadCOO(argv[1]);


	if (sm.get_nrows() < 1e4 || sm.get_nnz() < 1e5 || sm.get_ncols() * 512 > 3.9e9) { exit(-1); }
	fout.open("time_output_" + to_string(NCOL)  + ".txt", ofstream::out | ofstream::app);
	fout << "filename: " << argv[1] << endl;
	cout << "filename: " << argv[1] << endl;

	DenseMatrix<float> dm;
	dm.initOne(sm.get_ncols(), NCOL);

	spmm(sm, dm, argv[1]);

	fout << endl;

	return 0;
}
