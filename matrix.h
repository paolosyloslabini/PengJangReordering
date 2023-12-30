#ifndef GCONV_MATRIX_H
#define GCONV_MATRIX_H

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
#include <set>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusparse.h>
#include <memory>
#include "config.h"
#include "util.h"

template <class T>
__global__
void _init_dense_gpu(int nrows, int ncols, T *values, T val) {
	for (int i=0; i<ncols; i+=blockDim.x) {
		int c = i + threadIdx.x;
		if (c < ncols) {
			int idx = blockIdx.x * blockDim.x + c;
			values[idx] = val;
		}
	}
}



template <class T>
struct SparseMatrixCSR_CPU;

template <class T> 
struct SparseMatrixCOO_CPU {
	int nrows, ncols, nnz = 0;
	vector<int> rowidx_cpu, colidx_cpu;
	vector<T> values_cpu;

	SparseMatrixCOO_CPU(int nr, int nc): nrows(nr), ncols(nc) {}

	void add_nz(int r, int c, T v) {
		rowidx_cpu.push_back(r);
		colidx_cpu.push_back(c);
		values_cpu.push_back(v);
		nnz++;
	}
};

template<class T>
struct SparseMatrixCSC {
	int nrows, ncols, nnz;
	int *colptr = 0x0, *rowidx = 0x0;
	T *values = 0x0;
	int *colidx = 0x0;

	~SparseMatrixCSC() {cudaFree(colptr); cudaFree(rowidx); cudaFree(values); }

	void loadDict(map<int, map<int, T>>& sp) {

		vector<int> colptr_cpu;
		vector<int> colidx_cpu;
		vector<int> rowidx_cpu;
		vector<T> values_cpu;
		nnz = 0;
		ncols = sp.size();

		for (auto &cc: sp) {
			int c = cc.first;
			colidx_cpu.push_back(c);
			colptr_cpu.push_back(nnz);
			for (auto &nz: cc.second) {
				int r = nz.first;
				T v = nz.second;
				rowidx_cpu.push_back(r);
				values_cpu.push_back(v);
				nnz++;
			}
		}

		assert (values_cpu.size() == nnz);

		cudaMalloc(&colptr, sizeof(int)*(ncols+1));
		cudaMalloc(&colidx, sizeof(int)*(ncols+1));
		cudaMalloc(&rowidx, sizeof(int)*nnz);
		cudaMalloc(&values, sizeof(int)*nnz);

		cudaMemcpy(colptr, colptr_cpu.data(), sizeof(int)*(ncols+1), cudaMemcpyHostToDevice);
		cudaMemcpy(colidx, colidx_cpu.data(), sizeof(int)*(ncols+1), cudaMemcpyHostToDevice);
		cudaMemcpy(rowidx, rowidx_cpu.data(), sizeof(int)*(nnz), cudaMemcpyHostToDevice);
		cudaMemcpy(values, values_cpu.data(), sizeof(int)*(nnz), cudaMemcpyHostToDevice);
	} 
};

// sparse matrix with csr storage
template <class T>
struct SparseMatrixCSR {
	int nrows, ncols, nnz;
	int *rowptr = 0x0, *colidx = 0x0;
	T *values = 0x0;
	int *rowidx = 0x0;

	~SparseMatrixCSR() {cudaFree(rowptr); cudaFree(colidx); cudaFree(values); }

	
	void count_dense() {
		vector<int> rowptr_cpu(nrows + 1);
		vector<int> colidx_cpu(nnz);
		vector<T> values_cpu(nnz);
		cudaMemcpy(rowptr_cpu.data(), rowptr, (nrows+1)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(colidx_cpu.data(), colidx, (nnz)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(values_cpu.data(), values, (nnz)*sizeof(T), cudaMemcpyDeviceToHost);

		int dc = 0;
		for (int i=0; i<nrows; i+=4) {
			map<int, vector<int>> col_tmp;
			for (int j=i; j<(i+4>nrows?nrows:i+4); j++) {
				for (int k=rowptr_cpu[j]; k<rowptr_cpu[j+1]; k++) {
					int c = colidx_cpu[k];
					if (col_tmp.find(c) == col_tmp.end()) col_tmp[c] = vector<int>();
					col_tmp[c].push_back(j); 
				}
			}
			for (auto &cc: col_tmp) {
				if (cc.second.size() == 4) dc++;
			}
		}
		cout << "number of dense columns in the sparse part: " << dc << endl;
	}

	void reorder(const vector<int> &reordered_rows, int rowpanel_size=SROW_PER_TILE) {
		vector<int> rowptr_cpu(nrows+1);
		vector<int> colidx_cpu(nnz);
		vector<T> values_cpu(nnz);
		cudaMemcpy(rowptr_cpu.data(), rowptr, (nrows+1)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(colidx_cpu.data(), colidx, (nnz)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(values_cpu.data(), values, (nnz)*sizeof(T), cudaMemcpyDeviceToHost);


		map<int, map<int, T>> rowlist_tmp;
		for (int i=0; i<nrows; i++) {
			for (int j=rowptr_cpu[i]; j<rowptr_cpu[i+1]; j++) {
				if (rowlist_tmp.find(i) == rowlist_tmp.end()) rowlist_tmp[i] = map<int, T>();
				rowlist_tmp[i][colidx_cpu[j]] =  values_cpu[j];
			}
		}

		assert(rowlist_tmp.size() == reordered_rows.size());
		cout << "reordered_row size correct" << endl;

		nrows = rowlist_tmp.size();

		map<int, map<int, T>> rowlist;
		for (int i=0; i<rowlist_tmp.size(); i++) {
			rowlist[i] = rowlist_tmp[reordered_rows[i]];
		}

		rowptr_cpu.clear();
		colidx_cpu.clear();
		values_cpu.clear();

		rowptr_cpu.resize(nrows+1);
		colidx_cpu.resize(nnz);
		values_cpu.resize(nnz);
		rowptr_cpu[0] = 0;
		int k = 0;
		for (int i=1; i<nrows+1; i++) {
			rowptr_cpu[i] = rowptr_cpu[i-1] + rowlist[i-1].size();
			for (auto &tt: rowlist[i-1]) {
				colidx_cpu[k] = tt.first;
				values_cpu[k] = tt.second;	
				k++;
			}
		}
		assert (k == nnz);

		cout << "nnz correct" << endl;

		int dc = 0;
		for (int i=0; i<nrows; i+=rowpanel_size) {
			vector<vector<int>> sorted_cols(rowpanel_size);
			for (auto &nz: rowlist[i]) {
				int c = nz.first;
				int count = 0;
				for (int j=i+1; j<(i+rowpanel_size>nrows? nrows: i+rowpanel_size); j++) {
					if (rowlist[j].find(c) != rowlist[j].end()) {
						count++;
					}
				}
				sorted_cols[count].push_back(c);
			}
			dc += sorted_cols[rowpanel_size-1].size();

			vector<int> flatten_cols;
			for (int j=rowpanel_size-1; j>=0; j--) {
				for (int t: sorted_cols[j]) flatten_cols.push_back(t);
			}
			
			assert(flatten_cols.size() == rowptr_cpu[i+1] - rowptr_cpu[i]);
			for (int j=rowptr_cpu[i]; j<rowptr_cpu[i+1]; j++) {
				colidx_cpu[j] = flatten_cols[j-rowptr_cpu[i]];
				values_cpu[j] = rowlist[i][colidx_cpu[j]];
			}

			for (int j=i+1; j<(i+rowpanel_size>nrows? nrows: i+rowpanel_size); j++) {
				set<int> tmp_colidx(colidx_cpu.begin() + rowptr_cpu[j], colidx_cpu.begin() + rowptr_cpu[j+1]);	
				set<int> inserted;
				int k = 0;
				for (int t: flatten_cols) {
					if (tmp_colidx.find(t) != tmp_colidx.end()) {
						colidx_cpu[rowptr_cpu[j]+k] = t; 
						values_cpu[rowptr_cpu[j]+k] = rowlist[j][t];
						inserted.insert(t);
						k++;
					}
				}
				for (int t: tmp_colidx) {
					if (inserted.find(t) == inserted.end()) {
						colidx_cpu[rowptr_cpu[j]+k] = t;
						values_cpu[rowptr_cpu[j]+k] = rowlist[j][t];
						k++;
					}
				}

				//cout << k << " " << rowptr_cpu[j+1] - rowptr_cpu[j] << endl;
				assert(k == rowptr_cpu[j+1] - rowptr_cpu[j]);
				
			}	
		}
		cout << "sparse reordering num of effective columns: " << dc << endl;

		cudaMemcpy(rowptr, rowptr_cpu.data(), (nrows+1)*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(colidx, colidx_cpu.data(), (nnz)*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(values, values_cpu.data(), (nnz)*sizeof(T), cudaMemcpyHostToDevice);

		cudaMalloc(&rowidx, sizeof(int)*(nrows+1));
		cudaMemcpy(rowidx, reordered_rows.data(), (nrows+1)*sizeof(int), cudaMemcpyHostToDevice);

		

	}

	map<int, map<int, T>> get_rowmap() {
		vector<int> rowptr_cpu(nrows+1);
		vector<int> colidx_cpu(nnz);
		vector<T> values_cpu(nnz);
		cudaMemcpy(rowptr_cpu.data(), rowptr, (nrows+1)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(colidx_cpu.data(), colidx, (nnz)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(values_cpu.data(), values, (nnz)*sizeof(T), cudaMemcpyDeviceToHost);


		map<int, map<int, T>> rowlist;
		for (int i=0; i<nrows; i++) {
			for (int j=rowptr_cpu[i]; j<rowptr_cpu[i+1]; j++) {
				if (rowlist.find(i) == rowlist.end()) rowlist[i] = map<int, T>();
				rowlist[i][colidx_cpu[j]] = values_cpu[j];
			}
		}
		return rowlist;
	}



	map<int, vector<pair<int, T>>> get_rowlist() {
		vector<int> rowptr_cpu(nrows+1);
		vector<int> colidx_cpu(nnz);
		vector<T> values_cpu(nnz);
		cudaMemcpy(rowptr_cpu.data(), rowptr, (nrows+1)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(colidx_cpu.data(), colidx, (nnz)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(values_cpu.data(), values, (nnz)*sizeof(T), cudaMemcpyDeviceToHost);


		map<int, vector<pair<int, T>>> rowlist;
		for (int i=0; i<nrows; i++) {
			for (int j=rowptr_cpu[i]; j<rowptr_cpu[i+1]; j++) {
				if (rowlist.find(i) == rowlist.end()) rowlist[i] = vector<pair<int, T>>();
				rowlist[i].push_back(make_pair(colidx_cpu[j], values_cpu[j]));
			}
		}
		return rowlist;
	}

	map<int, vector<pair<int, T>>> get_collist() {

		vector<int> rowptr_cpu(nrows+1);
		vector<int> colidx_cpu(nnz);
		vector<T> values_cpu(nnz);
		cudaMemcpy(rowptr_cpu.data(), rowptr, (nrows+1)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(colidx_cpu.data(), colidx, (nnz)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(values_cpu.data(), values, (nnz)*sizeof(T), cudaMemcpyDeviceToHost);

		map<int, vector<pair<int, T>>> collist;
		for (int i=0; i<nrows; i++) {
			for (int j=rowptr_cpu[i]; j<rowptr_cpu[i+1]; j++) {
				int c = colidx_cpu[j];
				if (collist.find(c) == collist.end()) collist[c] = vector<pair<int, T>>();
				collist[c].push_back(make_pair(i, values_cpu[j]));
			}
		}
		return collist;
	}


	SparseMatrixCSR<T>& operator=(const SparseMatrixCSR<T>& other) {
		nrows = other.nrows;
		ncols = other.ncols;
		nnz = other.nnz;

		cudaMalloc(&rowptr, sizeof(int)*(nrows+1));
		cudaMalloc(&colidx, sizeof(int)*nnz);
		cudaMalloc(&values, sizeof(T)*nnz);
		{ cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess) { printf("Cuda failure %s:%d: '%s'\n","matrix.h",112,cudaGetErrorString(e)); exit(0); } };

		cudaMemcpy(rowptr, other.rowptr, (nrows+1)*sizeof(int), cudaMemcpyHostToHost);
		cudaMemcpy(colidx, other.colidx, (nnz)*sizeof(int), cudaMemcpyHostToHost);
		cudaMemcpy(values, other.values, (nnz)*sizeof(T), cudaMemcpyHostToHost);
		return *this;
	}

	void loadDict_redundant(map<int, map<int, T>>& sp) {

		vector<int> rowptr_cpu;
		vector<int> rowidx_cpu;
		vector<int> colidx_cpu;
		vector<T> values_cpu;
		nnz = 0;
		nrows = sp.size();

		for (auto &rr: sp) {
			int r = rr.first;
			rowidx_cpu.push_back(r);
			rowptr_cpu.push_back(nnz);
			for (auto &nz: rr.second) {
				int c = nz.first;
				T v = nz.second;
				colidx_cpu.push_back(c);
				values_cpu.push_back(v);
				nnz++;
			}
		}
		rowptr_cpu.push_back(nnz);

		assert (values_cpu.size() == nnz);

		if (nnz > 0) {

			cudaMalloc(&rowptr, sizeof(int)*(nrows+1));
			cudaMalloc(&rowidx, sizeof(int)*(nrows+1));
			cudaMalloc(&colidx, sizeof(int)*nnz);
			cudaMalloc(&values, sizeof(int)*nnz);

			cudaMemcpy(rowptr, rowptr_cpu.data(), sizeof(int)*(nrows+1), cudaMemcpyHostToDevice);
			cudaMemcpy(rowidx, rowidx_cpu.data(), sizeof(int)*(nrows+1), cudaMemcpyHostToDevice);
			cudaMemcpy(colidx, colidx_cpu.data(), sizeof(int)*(nnz), cudaMemcpyHostToDevice);
			cudaMemcpy(values, values_cpu.data(), sizeof(int)*(nnz), cudaMemcpyHostToDevice);
		}
	} 



	void loadDict(map<int, map<int, T>>& sp) {

		vector<int> rowptr_cpu;
		vector<int> rowidx_cpu;
		vector<int> colidx_cpu;
		vector<T> values_cpu;
		nnz = 0;
		nrows = sp.size();

		for (auto &rr: sp) {
			int r = rr.first;
			rowidx_cpu.push_back(r);
			rowptr_cpu.push_back(nnz);
			for (auto &nz: rr.second) {
				int c = nz.first;
				T v = nz.second;
				colidx_cpu.push_back(c);
				values_cpu.push_back(v);
				nnz++;
			}
		}
		rowptr_cpu.push_back(nnz);

		assert (values_cpu.size() == nnz);

		if (nnz > 0) {

			cudaMalloc(&rowptr, sizeof(int)*(nrows+1));
			cudaMalloc(&rowidx, sizeof(int)*(nrows+1));
			cudaMalloc(&colidx, sizeof(int)*nnz);
			cudaMalloc(&values, sizeof(int)*nnz);

			cudaMemcpy(rowptr, rowptr_cpu.data(), sizeof(int)*(nrows+1), cudaMemcpyHostToDevice);
			cudaMemcpy(rowidx, rowidx_cpu.data(), sizeof(int)*(nrows+1), cudaMemcpyHostToDevice);
			cudaMemcpy(colidx, colidx_cpu.data(), sizeof(int)*(nnz), cudaMemcpyHostToDevice);
			cudaMemcpy(values, values_cpu.data(), sizeof(int)*(nnz), cudaMemcpyHostToDevice);
		}
	} 

	void loadSplittedMatrix(vector<pair<int, map<int, T>>> &csr, vector<int> &reordered_rows, int panel_size) {

		assert (csr.size() == reordered_rows.size());
		nrows = csr.size();

		vector<int> rowptr_cpu;
		vector<int> rowidx_cpu;
		vector<int> colidx_cpu;
		vector<T> values_cpu;
		rowptr_cpu.push_back(0);
		nnz = 0;


		for (int i=0; i<nrows; i+=panel_size) {
			map<int, vector<int>> collist;
			for (int j=i; j<(i+panel_size>nrows?nrows:i+panel_size); j++) {
					rowidx_cpu.push_back(csr[reordered_rows[j]].first);
				for (auto &nz: csr[reordered_rows[j]].second) {
					int c = nz.first;
					if (collist.find(c) == collist.end()) collist[c] = vector<int>();
					collist[c].push_back(j-i);
				}
			}
			vector<pair<int, vector<int>>> collist_vec;
			copy(collist.begin(), collist.end(), back_inserter<vector<pair<int, vector<int>>>>(collist_vec));
			sort(collist_vec.begin(), collist_vec.end(), [](pair<int, vector<int>> &a, pair<int, vector<int>> &b) {return a.second.size() > b.second.size(); } );
			vector<vector<int>> colidx_tmp(panel_size);
			for (int j=0; j<collist_vec.size(); j++) {
				for (int rr: collist_vec[j].second) {
					colidx_tmp[rr].push_back(collist_vec[j].first);
				}
			}
			for (int j=0; i+j<(i+panel_size>nrows?nrows:i+panel_size); j++) {
				for (int cc: colidx_tmp[j]) {
					colidx_cpu.push_back(cc);
					T v = (csr[reordered_rows[j+i]].second)[cc];
					//cout << v << endl;
					values_cpu.push_back(v);
				}
				nnz += colidx_tmp[j].size();
				rowptr_cpu.push_back(nnz);
			}
		}

		assert (nnz == colidx_cpu.size());

		if (nnz > 0) {

		cudaMalloc(&rowptr, sizeof(int)*(nrows+1));
		cudaMalloc(&rowidx, sizeof(int)*(nrows));
		cudaMalloc(&colidx, sizeof(int)*(nnz));
		cudaMalloc(&values, sizeof(int)*(nnz));
		cudaCheckError();
		cudaMemcpy(rowptr, rowptr_cpu.data(), (nrows+1)*sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckError();
		cudaMemcpy(colidx, colidx_cpu.data(), (nnz)*sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckError();
		cudaMemcpy(values, values_cpu.data(), (nnz)*sizeof(T), cudaMemcpyHostToDevice);
		cudaCheckError();

		cudaMemcpy(rowidx, rowidx_cpu.data(), (nrows)*sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckError();
		}

	}

	void load_reorderedCSR_v1(map<int, map<int, T>> &csr, vector<int> &reordered_rows, int panel_size) {

		assert (csr.size() == reordered_rows.size());
		nrows = csr.size();

		vector<int> rowptr_cpu;
		vector<int> rowidx_cpu;
		vector<int> colidx_cpu;
		vector<T> values_cpu;
		rowptr_cpu.push_back(0);
		nnz = 0;


		for (int i=0; i<nrows; i+=panel_size) {
			map<int, vector<int>> collist;
			for (int j=i; j<(i+panel_size>nrows?nrows:i+panel_size); j++) {
				for (auto &nz: csr[reordered_rows[j]]) {
					int c = nz.first;
					if (collist.find(c) == collist.end()) collist[c] = vector<int>();
					collist[c].push_back(j-i);
				}
			}
			vector<pair<int, vector<int>>> collist_vec;
			copy(collist.begin(), collist.end(), back_inserter<vector<pair<int, vector<int>>>>(collist_vec));
			sort(collist_vec.begin(), collist_vec.end(), [](pair<int, vector<int>> &a, pair<int, vector<int>> &b) {return a.second.size() > b.second.size(); } );
			vector<vector<int>> colidx_tmp(panel_size);
			for (int j=0; j<collist_vec.size(); j++) {
				for (int rr: collist_vec[j].second) {
					colidx_tmp[rr].push_back(collist_vec[j].first);
				}
			}
			for (int j=0; i+j<(i+panel_size>nrows?nrows:i+panel_size); j++) {
				for (int cc: colidx_tmp[j]) {
					colidx_cpu.push_back(cc);
					T v = csr[reordered_rows[j+i]][cc];
					//cout << v << endl;
					values_cpu.push_back(v);
				}
				nnz += colidx_tmp[j].size();
				rowptr_cpu.push_back(nnz);
			}
		}

		assert (nnz == colidx_cpu.size());

		if (nnz > 0) {


		cudaMalloc(&rowptr, sizeof(int)*(nrows+1));
		cudaMalloc(&rowidx, sizeof(int)*(nrows));
		cudaMalloc(&colidx, sizeof(int)*(nnz));
		cudaMalloc(&values, sizeof(int)*(nnz));
		cudaCheckError();
		cudaMemcpy(rowptr, rowptr_cpu.data(), (nrows+1)*sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckError();
		cudaMemcpy(colidx, colidx_cpu.data(), (nnz)*sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckError();
		cudaMemcpy(values, values_cpu.data(), (nnz)*sizeof(T), cudaMemcpyHostToDevice);
		cudaCheckError();

		cudaMemcpy(rowidx, reordered_rows.data(), (nrows)*sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckError();
		}

	}




	void load_reorderedCSR(map<int, map<int, T>> &csr, vector<int> &reordered_rows, int panel_size, map<int, map<int, T>> &spp3) {

		assert (csr.size() == reordered_rows.size());
		nrows = csr.size();

		vector<int> rowptr_cpu;
		vector<int> rowidx_cpu;
		vector<int> colidx_cpu;
		vector<T> values_cpu;
		rowptr_cpu.push_back(0);
		nnz = 0;


		for (int i=0; i<nrows; i+=panel_size) {
			map<int, vector<int>> collist;
			for (int j=i; j<(i+panel_size>nrows?nrows:i+panel_size); j++) {
				for (auto &nz: csr[reordered_rows[j]]) {
					int c = nz.first;
					if (collist.find(c) == collist.end()) collist[c] = vector<int>();
					collist[c].push_back(j-i);
				}
			}
			vector<pair<int, vector<int>>> collist_vec;
			copy(collist.begin(), collist.end(), back_inserter<vector<pair<int, vector<int>>>>(collist_vec));
			sort(collist_vec.begin(), collist_vec.end(), [](pair<int, vector<int>> &a, pair<int, vector<int>> &b) {return a.second.size() > b.second.size(); } );
			vector<vector<int>> colidx_tmp(panel_size);
			for (int j=0; j<collist_vec.size(); j++) {
				for (int rr: collist_vec[j].second) {
					colidx_tmp[rr].push_back(collist_vec[j].first);
				}
			}
			for (int j=0; i+j<(i+panel_size>nrows?nrows:i+panel_size); j++) {
				while (colidx_tmp[j].size() > 3000) {
					int r = reordered_rows[i+j];
					int c = colidx_tmp[j].back();
					T v = csr[r][c];
					if (spp3.find(r) == spp3.end()) {
						spp3[r] = map<int, T>();
					}
					spp3[r][c] = v;
					colidx_tmp[j].pop_back();
				}
				for (int cc: colidx_tmp[j]) {
					colidx_cpu.push_back(cc);
					T v = csr[reordered_rows[j+i]][cc];
					//cout << v << endl;
					values_cpu.push_back(v);
				}
				nnz += colidx_tmp[j].size();
				rowptr_cpu.push_back(nnz);
			}
		}

		assert (nnz == colidx_cpu.size());

		if (nnz > 0) {


		cudaMalloc(&rowptr, sizeof(int)*(nrows+1));
		cudaMalloc(&rowidx, sizeof(int)*(nrows));
		cudaMalloc(&colidx, sizeof(int)*(nnz));
		cudaMalloc(&values, sizeof(int)*(nnz));
		cudaCheckError();
		cudaMemcpy(rowptr, rowptr_cpu.data(), (nrows+1)*sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckError();
		cudaMemcpy(colidx, colidx_cpu.data(), (nnz)*sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckError();
		cudaMemcpy(values, values_cpu.data(), (nnz)*sizeof(T), cudaMemcpyHostToDevice);
		cudaCheckError();

		cudaMemcpy(rowidx, reordered_rows.data(), (nrows)*sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckError();
		}

	}

	void loadCOO(SparseMatrixCOO_CPU<T> &coos) {
		nrows = coos.nrows;
		ncols = coos.ncols;
		nnz = coos.nnz;
		map<int, vector<pair<int, T>>> rlist;
		for (int i=0; i<coos.nnz; i++) {
			int r = coos.rowidx_cpu[i];
			int c = coos.colidx_cpu[i];
			T v = coos.values_cpu[i];
			if (rlist.find(r) == rlist.end()) rlist[r] = vector<pair<int, T>>();
			rlist[r].push_back(make_pair(c, v));
		}

		//cout << nrows << " " << nnz << endl;

		cudaMalloc(&rowptr, sizeof(int)*(nrows+1));
		cudaMalloc(&colidx, sizeof(int)*nnz);
		cudaMalloc(&values, sizeof(T)*nnz);
		cudaCheckError();

		int *t_rowptr = new int[nrows+1];
		int *t_colidx = new int[nnz];
		T *t_values = new T[nnz];

		t_rowptr[0] = 0;
		int k = 0;
		for (int i=1; i<nrows+1; i++) {
			t_rowptr[i] = t_rowptr[i-1] + rlist[i-1].size();
			for (int j=0; j<rlist[i-1].size(); j++) {
				t_colidx[k] = rlist[i-1][j].first;
				t_values[k] = rlist[i-1][j].second;	
				k++;
			}
		}
		assert (nnz == k);


		cudaMemcpy(rowptr, t_rowptr, (nrows+1)*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(colidx, t_colidx, (nnz)*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(values, t_values, (nnz)*sizeof(T), cudaMemcpyHostToDevice);


		delete []t_rowptr;
		delete []t_colidx;
		delete []t_values;
	}



	void loadCOO(string filename, bool start_one=false) {

		ifstream fin(filename.c_str());
		string line;
		getline(fin, line);
		while (line[0] == '%') getline(fin, line);
		stringstream sin_meta(line);
		sin_meta >> nrows >> ncols >> nnz;

		map<int, vector<pair<int, T>>> dict;
		while (getline(fin, line)) {
			int r, c; 
			T v;
			stringstream sin(line);
			sin >> r >> c >> v;
			if (start_one) {
				r--; c--;
			}
			if (dict.find(r) == dict.end()) {
				dict[r] = vector<pair<int, T>>();
			}
			dict[r].push_back(make_pair(c, v));
		}

		cudaMalloc(&rowptr, sizeof(int)*(nrows+1));
		cudaMalloc(&colidx, sizeof(int)*nnz);
		cudaMalloc(&values, sizeof(T)*nnz);
		cudaCheckError();




		int *t_rowptr = new int[nrows+1];
		int *t_colidx = new int[nnz];
		T *t_values = new T[nnz];

		t_rowptr[0] = 0;
		int k = 0;
		for (int i=1; i<nrows+1; i++) {
			t_rowptr[i] = t_rowptr[i-1] + dict[i-1].size();
			for (int j=0; j<dict[i-1].size(); j++) {
				t_colidx[k] = dict[i-1][j].first;
				t_values[k] = dict[i-1][j].second;	
				k++;
			}
		}
		cout << k << " " << nnz << endl;
		assert(k == nnz);

		cudaMemcpy(rowptr, t_rowptr, (nrows+1)*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(colidx, t_colidx, (nnz)*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(values, t_values, (nnz)*sizeof(T), cudaMemcpyHostToDevice);


		delete []t_rowptr;
		delete []t_colidx;
		delete []t_values;

	}

	void initEmpty(int nr, int nc, int nz) {
		nrows = nr, ncols = nc, nnz = nz;
		cudaMalloc(&rowptr, sizeof(int)*(nrows+1));
		cudaMalloc(&colidx, sizeof(int)*nnz);
		cudaMalloc(&values, sizeof(T)*nnz);
		cudaCheckError();
	}

	
	void display() {
		int *t_rowptr = new int[nrows+1];
		int *t_colidx = new int[nnz];
		T *t_values = new T[nnz];

		cudaMemcpy(t_rowptr, rowptr, (nrows+1)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(t_colidx, colidx, (nnz)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(t_values, values, (nnz)*sizeof(T), cudaMemcpyDeviceToHost);

		for (int i=0; i<nrows; i++) {
			map<int, T> cols;
			for (int j=t_rowptr[i]; j<t_rowptr[i+1]; j++) {
				cols[t_colidx[j]] = t_values[j];
			}
			for (int j=0; j<ncols; j++) {
				if (cols.find(j) == cols.end()) cout << "0 ";
				else cout << cols[j] << " ";
			}
			cout << endl;
		}
		cout << endl;

		delete []t_rowptr;
		delete []t_colidx;
		delete []t_values;

	}

	int get_nrows() {return nrows; }
	int get_ncols() {return ncols; }
	int get_nnz() {return nnz; }

};


template <class T>
struct SparseMatrixCSR_CPU {
	int nrows, ncols, nnz;
	vector<int> rowptr;
	vector<int> colidx;
	vector<T> values;

	SparseMatrixCSR_CPU():nrows(0), ncols(0), nnz(0) {}

	SparseMatrixCSR_CPU(SparseMatrixCSR<T> &csr_gpu) {
		nrows = csr_gpu.nrows;
		ncols = csr_gpu.ncols;
		nnz = csr_gpu.nnz;
		rowptr.resize(nrows+1);
		colidx.resize(nnz);
		values.resize(nnz);

		cudaMemcpy(rowptr.data(), csr_gpu.rowptr, sizeof(int)*(nrows+1), cudaMemcpyDeviceToHost);
		cudaMemcpy(colidx.data(), csr_gpu.colidx, sizeof(int)*(nnz), cudaMemcpyDeviceToHost);
		cudaMemcpy(values.data(), csr_gpu.values, sizeof(T)*(nnz), cudaMemcpyDeviceToHost);
	}

	void loadCOO(string filename, bool start_one=false) {

		ifstream fin(filename.c_str());
		string line;
		getline(fin, line);
		stringstream sin_meta(line);
		sin_meta >> nrows >> ncols >> nnz;

		rowptr.resize(nrows + 1);
		colidx.resize(nnz);
		values.resize(nnz);

		map<int, vector<pair<int, T>>> dict;
		while (getline(fin, line)) {
			int r, c; 
			T v;
			stringstream sin(line);
			sin >> r >> c >> v;
			if (start_one) {
				r--; c--;
			}
			if (dict.find(r) == dict.end()) {
				dict[r] = vector<pair<int, T>>();
			}
			dict[r].push_back(make_pair(c, v));
		}

		rowptr[0] = 0;
		int k = 0;
		for (int i=1; i<nrows+1; i++) {
			rowptr[i] = rowptr[i-1] + dict[i-1].size();
			for (int j=0; j<dict[i-1].size(); j++) {
				colidx[k] = dict[i-1][j].first;
				values[k] = dict[i-1][j].second;	
				k++;
			}
		}
		assert(k == nnz);

	}

};


// dense matrix 
template <class T>
struct DenseMatrix {
	int nrows, ncols;
	T *values;

	DenseMatrix(): nrows(0), ncols(0) {}
	~DenseMatrix() {cudaFree(values);}

	void loadDense(string filename) {
		ifstream fin(filename.c_str());
		string line;
		getline(fin, line);
		stringstream sin_meta(line);
		sin_meta >> nrows >> ncols;

		cudaMalloc(&values, sizeof(T)*nrows*ncols);
		cudaCheckError();
		T *t_values = new T[nrows*ncols];

		for (int i=0; i<nrows; i++) {
			getline(fin, line);
			if (line != "") {
				stringstream sin(line);
				int j = 0;
				while (sin >> t_values[i*ncols + j++]) {}
				if (j < ncols) {
					cerr << "not enough cols in input!" << endl;
					exit(-1);
				}
			} else {
				cerr << "not enough rows in input!" << endl;
				exit(-1);
			}
		}

		cudaMemcpy(values, t_values, sizeof(T)*nrows*ncols, cudaMemcpyHostToDevice);
		delete []t_values;

	}

	void initVal(int nr, int nc, T val) {
		nrows = nr; ncols = nc;
		cudaMalloc(&values, sizeof(T)*nrows*ncols);
		_init_dense_gpu<<<nrows, CLAMP(ncols)>>>(nrows, ncols, values, val);
		cudaCheckError();
		cudaDeviceSynchronize();
	}

	void initOne(int nr, int nc) {
		initVal(nr, nc, (T)1);
	}

	void initZero(int nr, int nc) {
		initVal(nr, nc, (T)0);
	}

	void initRand(int nr, int nc, int seed=12345) {
		nrows = nr; ncols = nc;
		cudaMalloc(&values, sizeof(T)*nrows*ncols);
		curandGenerator_t prng;
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
		curandSetPseudoRandomGeneratorSeed(prng, seed);
		curandGenerateUniform(prng, values, nrows * ncols);
		cudaCheckError();
	}

	void initEmpty(int nr, int nc) {
		nrows = nr; ncols = nc;
		cudaMalloc(&values, sizeof(T)*nrows*ncols);
		cudaMemset(values, 0, sizeof(T)*nrows*ncols);
		cudaCheckError();
	}

	void display(bool colmajor=false) {

		T *t_values = new T[nrows*ncols];

		cudaMemcpy(t_values, values, sizeof(T)*nrows*ncols, cudaMemcpyDeviceToHost);

		if (colmajor) {
			T *tt_values = new T[nrows*ncols];
			for (int i=0; i<nrows; i++) {
				for (int j=0; j<ncols; j++) {
					tt_values[i*ncols+j] = t_values[j*nrows+i];
				}
			}
			delete []t_values;
			t_values = tt_values;
		}

		int count1 = 0;
		for (int i=0; i<nrows; i++) {
			if (i < 5 || i > nrows-5) {
				int count2 = 0;
				for (int j=0; j<ncols; j++) {
					if (j < 5 || j > ncols - 5)
						cout << setprecision(4) << t_values[i*ncols+j] << "\t";
					else if (count2 < 5) {
						cout << ".\t";
						count2++;
					}
				}
				cout << endl;
			} else if (count1 < 5) {
				for (int j=0; j<min(14, ncols); j++) {
					cout << ".\t";
				}
				count1++;
				cout << endl;
			}
		}
		cout << endl;

		delete []t_values;
	}


	int get_nrows() {return nrows; }
	int get_ncols() {return ncols; }


};

#endif
