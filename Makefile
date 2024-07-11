reorderingOnly: obj_spmm_512 obj_spmm_256

obj_spmm_512: reorderingOnly.cu
	nvcc -std=c++11 -gencode arch=compute_86,code=sm_86 -O3  -DNCOL=512 -DORIG -lcublas -lcusparse -lcurand -Xptxas "-v -dlcm=ca"  reorderingOnly.cu  -Xcompiler -fopenmp -o obj_spmm_512 

obj_spmm_256: reorderingOnly.cu
	nvcc -std=c++11 -gencode arch=compute_86,code=sm_86 -O3  -DNCOL=256 -DORIG -lcublas -lcusparse -lcurand -Xptxas "-v -dlcm=ca"  reorderingOnly.cu  -Xcompiler -fopenmp -o obj_spmm_256 

clean:
	rm -f obj_spmm_512 obj_spmm_256
