#ifndef GCONV_PP_H
#define GCONV_PP_H


#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <cuda.h>
#include <cassert>
#include <list>
#include <bitset>
#include <queue>
#include <string>
#include "util.h"
#include "config.h"
#include "matrix.h"
#include "clustering/clustering.h"
using namespace std;

template <class T>
struct Preprocessor {

	struct Tile {
		int nrows, ncols, nnz;
		vector<int> rowptr_cpu;
		vector<int> colidx_cpu;
		vector<T> values_cpu;
		vector<int> rowidx_cpu;

		bool is_dense;
		vector<int> dense_cols;

		vector<tuple<int, int, T>> coos;

		void add_nz(int r, int c, T v) {
			coos.push_back(make_tuple(r, c, v));
		}

		void toCSR() {
			map<int, vector<pair<int, T>>> rlist;
			set<int> cols;
			for (auto &nz: coos) {
				int r = get<0>(nz);
				int c = get<1>(nz);
				//assert (r < 128 && c < 64);
				T v = get<2>(nz);
				if (rlist.find(r) == rlist.end()) rlist[r] = vector<pair<int, T>>();
				rlist[r].push_back(make_pair(c, v));
				cols.insert(c);
			}
			ncols = cols.size();
			rowptr_cpu.push_back(0);	
			nnz = 0;
			//cout << nrows << endl;
			for (int i=0; i<nrows; i++) {
				if (rlist.find(i) != rlist.end()) {
					for (auto &tt: rlist[i]) {
						colidx_cpu.push_back(tt.first);
						values_cpu.push_back(tt.second);
						nnz += 1;
					}
				}
				rowptr_cpu.push_back(nnz);
			}
			while (rowptr_cpu.size() <= DROW_PER_TILE) {
				rowptr_cpu.push_back(nnz);
			}
		}
	};

	struct GPUData {
		int *col_tile_ptr = 0x0;
		int *prefetch_colidx = 0x0;
		int *row_tile_ptr = 0x0;
		int *rowptr = 0x0;
		int *rowidx = 0x0;
		int *colidx = 0x0;
		T *values = 0x0;
		vector<Tile> tiles_cpu; 

		int get_ntiles() {return tiles_cpu.size(); }

		~GPUData() {
			cudaFree(row_tile_ptr);
			cudaFree(col_tile_ptr);
			cudaFree(prefetch_colidx);
			cudaFree(rowptr);
			cudaFree(rowidx);
			cudaFree(colidx);
			cudaFree(values);
		}
	};

	GPUData dense_data;
	SparseMatrixCSR<T> sparse_csr;
	SparseMatrixCSR<T> sparse_csr1;
	SparseMatrixCSR<T> sparse_csr2;
	SparseMatrixCSR<T> sparse_csr3;
	SparseMatrixCSC<T> sparse_csc;

	int reordering_method;

	string filename;

	Preprocessor(int m, string fn):reordering_method(m) {
		filename = fn;

	}


	void generate_tiles(GPUData &gpu_data, bool is_dense=true) {
		int ntiles = gpu_data.tiles_cpu.size();
		int *&col_tile_ptr = gpu_data.col_tile_ptr;
		int *&prefetch_colidx = gpu_data.prefetch_colidx;
		int *&row_tile_ptr = gpu_data.row_tile_ptr;
		int *&rowptr = gpu_data.rowptr;
		int *&rowidx = gpu_data.rowidx;
		int *&colidx = gpu_data.colidx;
		T *&values = gpu_data.values;
		vector<Tile> &tiles_cpu = gpu_data.tiles_cpu;



		cudaMalloc(&row_tile_ptr, sizeof(int)*(ntiles + 1));
		cudaMalloc(&col_tile_ptr, sizeof(int)*(ntiles + 1));


		vector<int> row_tile_ptr_cpu;
		vector<int> rowptr_cpu;
		vector<int> rowidx_cpu;
		vector<int> colidx_cpu;
		vector<T> values_cpu;
		vector<int> col_tile_ptr_cpu;
		vector<int> prefetch_colidx_cpu;

		row_tile_ptr_cpu.push_back(0);
		col_tile_ptr_cpu.push_back(0);

		int cur_rowtile_pos = 0;
		int cur_coltile_pos = 0;


		int cur_nnz = 0;

		for (auto &tl: tiles_cpu) {
			vector<int> tmp_rowptr;
			for (auto &x: tl.rowptr_cpu) tmp_rowptr.push_back(x+cur_nnz);

			rowptr_cpu.insert(rowptr_cpu.end(), tmp_rowptr.begin(), tmp_rowptr.end());

			cur_rowtile_pos += tmp_rowptr.size();

			if (is_dense) {
				prefetch_colidx_cpu.insert(prefetch_colidx_cpu.end(), tl.dense_cols.begin(), tl.dense_cols.end());
				cur_coltile_pos += tl.dense_cols.size();
			} 

			row_tile_ptr_cpu.push_back(cur_rowtile_pos);
			col_tile_ptr_cpu.push_back(cur_coltile_pos);
			//cout << cur_coltile_pos << endl;

			//cout << tmp_rowptr.size() << " " << tl.rowidx_cpu.size() << endl;
			assert (tmp_rowptr.size() == tl.rowidx_cpu.size());

			rowidx_cpu.insert(rowidx_cpu.end(), tl.rowidx_cpu.begin(), tl.rowidx_cpu.end());
			colidx_cpu.insert(colidx_cpu.end(), tl.colidx_cpu.begin(), tl.colidx_cpu.end());
			values_cpu.insert(values_cpu.end(), tl.values_cpu.begin(), tl.values_cpu.end());

			assert (tl.colidx_cpu.size() == tl.values_cpu.size());
			cur_nnz += tl.colidx_cpu.size();
		}

		assert (row_tile_ptr_cpu.size() == ntiles + 1);
		assert (col_tile_ptr_cpu.size() == ntiles + 1);

		cudaMemcpy(row_tile_ptr, row_tile_ptr_cpu.data(), sizeof(int)*row_tile_ptr_cpu.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(col_tile_ptr, col_tile_ptr_cpu.data(), sizeof(int)*col_tile_ptr_cpu.size(), cudaMemcpyHostToDevice);

		//	print(row_tile_ptr_cpu);
		//	print(rowptr_cpu);
		//	print(rowidx_cpu);
		//	print(colidx_cpu);
		//	print(values_cpu);

		if (is_dense == false) assert(prefetch_colidx_cpu.size() == 0);
		if (is_dense) {
			cudaMalloc(&prefetch_colidx, sizeof(int)*prefetch_colidx_cpu.size());
			cudaMemcpy(prefetch_colidx, prefetch_colidx_cpu.data(), sizeof(int)*prefetch_colidx_cpu.size(), cudaMemcpyHostToDevice);
		}

		cudaMalloc(&rowptr, sizeof(int)*rowptr_cpu.size());
		cudaMemcpy(rowptr, rowptr_cpu.data(), sizeof(int)*rowptr_cpu.size(), cudaMemcpyHostToDevice);

		cudaMalloc(&rowidx, sizeof(int)*rowidx_cpu.size());
		cudaMemcpy(rowidx, rowidx_cpu.data(), sizeof(int)*rowidx_cpu.size(), cudaMemcpyHostToDevice);

#define BUFFER (1024)
		cudaMalloc(&colidx, sizeof(int)*colidx_cpu.size()+BUFFER);
		cudaMalloc(&values, sizeof(T)*values_cpu.size()+BUFFER);
		cudaMemset(colidx, 0, sizeof(int)*colidx_cpu.size()+BUFFER);
		cudaMemcpy(colidx, colidx_cpu.data(), sizeof(int)*colidx_cpu.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(values, values_cpu.data(), sizeof(T)*values_cpu.size(), cudaMemcpyHostToDevice);

		cudaCheckError();
	}

	vector<int> clustering(const vector<int> &nodes, const vector<set<int>> &adj, int bsize=2) {
		int length = adj.size() / bsize * bsize;
		vector<vector<int>> neighbors;
		vector<int> reordered(nodes);
		for (int i=0; i<length; i+=bsize) {
			vector<vector<int>> intersect(bsize+1);
			for (int j: reordered) {
				int cnt = 0;
				for (int k=i; k<i+bsize; k++) {
					if (adj[k].find(j) != adj[k].end()) cnt++;
				}
				intersect[cnt].push_back(j);
			}
			reordered.clear();
			for (auto &v: intersect) {
				for (int t: v) {
					reordered.push_back(t);
				}
			}
			//cout << reordered.size() << endl;
		}
		return reordered;
	}

	void write_to_file(vector<int> &vec, string filename) {
		ofstream fout(filename.c_str());
		for (int t: vec) fout << t << endl;
		fout.close();
	}

	void reorder_sparse() {
		auto sp = sparse_csr.get_rowmap();
		map<int, map<int, float>> spp1;
		map<int, map<int, float>> spp2;
		map<int, map<int, float>> spp3;
		for (auto &r: sp) {
			if (r.second.size() > SPARSE3_THRESHOLD) spp1.insert(r);
			else spp2.insert(r);
		}

		//sparse_csr1.loadDict(spp1);
		//sparse_csr2.loadDict(spp2);
		


		vector<int> reordered_rows1;

		float th = LSH::compute_average_similarity(spp1);

		if (th < 0.08) {


			vector<pair<int, map<int, float>>> spp11;
			vector<pair<int, map<int, float>>> spp12;
			for (auto &r: spp1) {
				if (r.second.size() > 24) spp11.push_back(r); 	
				else spp12.push_back(r);
			}
			auto close_pairs1 = LSH::get_close_pairs_v1(spp11, sparse_csr.ncols, 128, 2);
			vector<int> reordered_rows11 = Clustering::hierachical_clustering_v0(spp11, close_pairs1, 256);
		//	cout << reordered_rows11.size() << " " << spp11.size() << endl;
			auto close_pairs2 = LSH::get_close_pairs_v1(spp12, sparse_csr.ncols, 128, 2);
			vector<int> reordered_rows12 = Clustering::hierachical_clustering_v0(spp12, close_pairs2, 256);

		//	cout << reordered_rows12.size() << " " << spp12.size() << endl;
			reordered_rows1 = reordered_rows11;
			for (int i: reordered_rows12) reordered_rows1.push_back(i);
		} else {
			for (auto &r: spp1) reordered_rows1.push_back(r.first);
		}
							
			//write_to_file(reordered_rows1, sn1);
		sparse_csr1.load_reorderedCSR(spp1, reordered_rows1, 256, spp3);

		vector<int> reordered_rows2;
		float th2 = LSH::compute_average_similarity(spp2);

		if (th2 < 0.08) {

			vector<pair<int, map<int, float>>> spp21;
			vector<pair<int, int>> spp22_vec;
			for (auto &r: spp2) {
				if (r.second.size() > 1) spp21.push_back(r); 	
				else {
					spp22_vec.push_back(make_pair(r.first, r.second.begin()->first));
				} 
			}

			auto close_pairs1 = LSH::get_close_pairs_v1(spp21, sparse_csr.ncols, 128, 4);
			vector<int> reordered_rows21 = Clustering::hierachical_clustering_v0(spp21, close_pairs1, 256);
			sort(spp22_vec.begin(), spp22_vec.end(), [](pair<int, int> &a, pair<int, int> &b){return a.second < b.second; });


			reordered_rows2 = reordered_rows21;
			for (auto &t: spp22_vec) reordered_rows2.push_back(t.first);
		} else {
			for (auto &r: spp2) reordered_rows2.push_back(r.first);
		}

	
		sparse_csr2.load_reorderedCSR(spp2, reordered_rows2, 256, spp3);
		vector<pair<int, map<int, T>>> splitted_matrix = split_long_rows(spp3);
		auto close_pairs = LSH::get_close_pairs_v1(splitted_matrix, sparse_csr.ncols, 128, 2);
		vector<int> reordered_rows = Clustering::hierachical_clustering_v1(splitted_matrix, close_pairs, 256);
		sparse_csr3.loadSplittedMatrix(splitted_matrix, reordered_rows, 256);
			

	}

	vector<pair<int, map<int, T>>> split_long_rows(map<int, map<int, T>> &sp) {
		vector<pair<int, map<int, T>>> res;
		for (auto &rr: sp) {
			int r = rr.first;
			int nblocks = rr.second.size() / 1024;
			if (rr.second.size() % 1024) nblocks++;
			int bsize = rr.second.size() / nblocks;
			int l = 0;
			map<int, T> tmp;
			for (auto &nz: rr.second) {
				tmp.insert(nz);
				l++;
				if (l % bsize == 0) {
					res.push_back(make_pair(r, tmp));
					l = 0;
					tmp.clear();
				}
			}
			if (tmp.size()) res.push_back(make_pair(r, tmp));
		}
		return res;
	}

/*
	
	void reorder_sparse() {
	
		auto collist = sparse_csr.get_collist();
		auto rowlist = sparse_csr.get_rowlist();

		vector<int> reordered_rows; 


		vector<pair<int, vector<pair<int, T>>>> collist_vec;
		copy(collist.begin(), collist.end(), back_inserter<vector<pair<int, vector<pair<int, T>>>>>(collist_vec));
		sort(collist_vec.begin(), collist_vec.end(), [](pair<int, vector<pair<int, T>>> &a, pair<int, vector<pair<int, T>>> &b){ return a.second.size() > b.second.size();  });

		vector<int> dense_columns;
		for (int i=0; i<(collist_vec.size()<512?collist_vec.size():512); i++) {
			dense_columns.push_back(collist_vec[i].first);
		}

		vector<set<int>> adjLists;
		for (int c: dense_columns) {
			set<int> tmp_adj;
			for (auto &rr: collist[c]) {
				int r = rr.first;	
				tmp_adj.insert(r);
			}
			adjLists.push_back(tmp_adj);
		}

		vector<int> row_nodes;
		for (auto &r: rowlist) row_nodes.push_back(r.first);

		reordered_rows = clustering(row_nodes, adjLists, 2);

		sparse_csr.reorder(reordered_rows);


	}
	
*/

	void reorder_dense(int method, int level) {
		auto sp = sparse_csr.get_rowmap();

/* 
		int longest_row = 0;
		int shortest_row = sparse_csr.ncols;
		for (auto &r: sp) {
			if (r.second.size() > longest_row) longest_row = r.second.size();
			if (r.second.size() < shortest_row) shortest_row = r.second.size();
		}
		cout << "longest row: " << longest_row << endl;
		cout << "shortest row: " << shortest_row << endl;
*/
		int tot = 0;
		int nr = sparse_csr.nrows;
		for (int i=0; i<nr; i+=128) {
			map<int, vector<int>> collist;
			for (int j=i; j<(nr<i+128?nr:i+128); j++) {
				if (sp.find(j) != sp.end()) {
					for (auto &c: sp[j]) {
						int cc = c.first;
						if (collist.find(cc) == collist.end()) collist[cc] = vector<int>();
						collist[cc].push_back(j);
					}
				}
			}
			vector<vector<int>> collist_vec;
			for (auto &col: collist) collist_vec.push_back(col.second);
			sort(collist_vec.begin(), collist_vec.end(), [](vector<int> &a, vector<int> &b){return a.size() > b.size(); } );
			int endpos = 0;
			for (auto &col: collist_vec) {
				endpos++;
				if (col.size() < 16) break;
			}

			if (endpos > 128 || endpos <=16) {
				endpos = endpos / 128 * 128;
			}
			for (int j=0; j<endpos; j++) {
				tot += collist_vec[j].size();
			}

		}

		float dr = 1.0 * tot / total_nnz;

//		cout << "original dense ratio: " << dr << endl;


		if (dr > 0.1) method = 0;


		vector<vector<int>> row_panels;

		if (method == 0) {
			for (int i=0; i<sparse_csr.nrows; i+=128) {
				vector<int> row_tmp;
				for (int j=i; j<((i+128)>sparse_csr.nrows?sparse_csr.nrows:(i+128)); j++) row_tmp.push_back(j);
				row_panels.push_back(row_tmp);
			}
		} else {
			vector<int> reordered_rows;
				map<int, map<int, float>> spp;
				for (auto &r: sp) {
					if (r.second.size() > 32) {
						spp.insert(r);
					}
				}
				auto close_pairs = LSH::get_close_pairs(spp, sparse_csr.ncols, 128, 2);
				reordered_rows = Clustering::hierachical_clustering(sp, close_pairs, 256);

			vector<int> tmp_rows;	
			for (int i=0; i<reordered_rows.size(); i++) {
				tmp_rows.push_back(reordered_rows[i]);
				if ((i+1) % DROW_PER_TILE == 0) {
					row_panels.push_back(tmp_rows);	
					tmp_rows.clear();
				}	
			}
			if (tmp_rows.size() > 0) row_panels.push_back(tmp_rows); 
		}
		SparseMatrixCOO_CPU<T> sparse_coo(sparse_csr.nrows, sparse_csr.ncols);
		for (auto &rp: row_panels) {
			map<int, vector<pair<int, T>>> cols;
			for (int i=0; i<rp.size(); i++) {
				for (auto &nz: sp[rp[i]]) {
					int c = nz.first;
					T v = nz.second;
					if (cols.find(c) == cols.end()) cols[c] = vector<pair<int, T>>();
					cols[c].push_back(make_pair(i, v));
				}
			}
			vector<pair<int, vector<pair<int, T>>>> dense_cols;
			for (auto &cc: cols) {
				if (cc.second.size() >= DENSE_COL_THRD) {
					dense_cols.push_back(cc);
				} else {
					for (auto &nz: cc.second) {
						sparse_coo.add_nz(rp[nz.first], cc.first, nz.second);
					}
				}
			}

			vector<vector<int>> col_tiles;
			if (dense_cols.size() <= DENSE_ROW_THRD) {
				for (auto &cc: dense_cols) {
					for (auto &nz: cc.second) {
						sparse_coo.add_nz(rp[nz.first], cc.first, nz.second);	
					}
				}
			} else if (dense_cols.size() <= DCOL_PER_TILE ) {
				vector<int> tmp_cols;
				for (auto &cc: dense_cols) tmp_cols.push_back(cc.first);
				col_tiles.push_back(tmp_cols);

			} else if (dense_cols.size() > DCOL_PER_TILE) {
				//cout << dense_cols.size() << " ###" << endl;
				int ncols_per_tile = DCOL_PER_TILE;
				while (dense_cols.size() % DCOL_PER_TILE) {
					auto &cc = dense_cols[dense_cols.size()-1];
					for (auto &nz: cc.second) {
						sparse_coo.add_nz(rp[nz.first], cc.first, nz.second);	
					}

					dense_cols.pop_back();
				}

				if (method == 0) {
					for (int i=0; i<dense_cols.size(); i+=ncols_per_tile) {
						vector<int> tmp_cols;
						for (int j=i; j<((i+ncols_per_tile)>dense_cols.size()?dense_cols.size():(i+ncols_per_tile)); j++) tmp_cols.push_back(dense_cols[j].first);
						col_tiles.push_back(tmp_cols);
					}
				} else {

					map<int, vector<pair<int, T>>> dense_rowlist;
					for (auto &cc: dense_cols) {
						int c = cc.first;
						for (auto &nz: cc.second) {
							int r = nz.first;
							T v = nz.second;
							if (dense_rowlist.find(r) == dense_rowlist.end()) dense_rowlist[r] = vector<pair<int, T>>();
							dense_rowlist[r].push_back(make_pair(c, v));
						}
					}

					vector<pair<int, vector<pair<int, T>>>> dense_rowlist_vec;
					copy(dense_rowlist.begin(), dense_rowlist.end(), back_inserter<vector<pair<int, vector<pair<int, T>>>>>(dense_rowlist_vec));
					sort(dense_rowlist_vec.begin(), dense_rowlist_vec.end(), [](pair<int, vector<pair<int, T>>> &a, pair<int, vector<pair<int, T>>> &b){ return a.second.size() > b.second.size();  });

					vector<int> cols_nodes;

					for (auto &cc: dense_cols) {
						cols_nodes.push_back(cc.first);
					}
					vector<set<int>> adjLists;
					for (auto &rr: dense_rowlist_vec) {
						set<int> tmp_adj;
						for (auto &cc: rr.second) {
							int c = cc.first;	
							tmp_adj.insert(c);
						}
						adjLists.push_back(tmp_adj);
					}

					vector<int> reordered_cols = clustering(cols_nodes, adjLists, 2);

					vector<int> tmp_cols;	
					for (int i=0; i<reordered_cols.size(); i++) {
						tmp_cols.push_back(reordered_cols[i]);
						if ((i+1) % ncols_per_tile == 0) {
							col_tiles.push_back(tmp_cols);	
							tmp_cols.clear();
						}	
					}
					if (tmp_cols.size() > 0) col_tiles.push_back(tmp_cols); 
				}
			}

			for (auto &t: col_tiles) {
				Tile tl;
				tl.is_dense = true;
				tl.rowidx_cpu = rp;
				tl.nrows = rp.size();
				while (tl.rowidx_cpu.size() <= DROW_PER_TILE)
					tl.rowidx_cpu.push_back(-1);
				tl.dense_cols = t;
				map<int, int> indices;
				for (int i=0; i<t.size(); i++) {
					indices[t[i]] = i;
					//cout << t[i] << " ";
				}
				//cout << endl;
				for (int c: t) {
					for (auto &nz: cols[c]) {
						int r = nz.first;
						T v = nz.second;
						tl.add_nz(r, indices[c], v);
					}
				}
				tl.toCSR();
				dense_data.tiles_cpu.push_back(tl);
			}

		}
		sparse_csr.loadCOO(sparse_coo);
		
	}

	int total_nnz;

	void reordering_and_tiling(const SparseMatrixCSR<T> &sm) {

		sparse_csr = sm;
		total_nnz = sm.nnz;
		reorder_dense(reordering_method, 0);
		//reorder_dense(reordering_method, 1);
		generate_tiles(dense_data);
		//sparse_csr.count_dense();
		int tn = 0;
		for (auto &tt: dense_data.tiles_cpu) {
			tn += tt.nnz;
		}

				//assert(tn + sparse_csr.nnz == total_nnz);



		if (reordering_method == 2) {
	//		cout << "start reordering sparse part" << endl;
			reorder_sparse();
		}

//		cout << tn << " " << sparse_csr1.nnz << " " << sparse_csr2.nnz << " " << sparse_csr3.nnz << " " <<  total_nnz << endl;
		if (tn + sparse_csr1.nnz + sparse_csr2.nnz + sparse_csr3.nnz != total_nnz) {
			cerr << "dense and sparse nnz incorrect!" << endl;
			exit(-1);
		}


//		cout << "dense ratio: " << 1 - 1.0 * sparse_csr.nnz / total_nnz << endl;

	}

};

#endif
