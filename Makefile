reorderingOnly: obj_spmm_512 obj_spmm_256

obj_spmm_512: spmm.cu
	nvcc -std=c++11 -gencode arch=compute_60,code=sm_60 -O3  -DNCOL=512 -DORIG -lcublas -lcusparse -lcurand -Xptxas "-v -dlcm=ca"  reorderingOnly.cu  -Xcompiler -fopenmp -o obj_spmm_512 
obj_spmm_256: spmm.cu
	nvcc -std=c++11 -gencode arch=compute_60,code=sm_60 -O3  -DNCOL=256 -DORIG -lcublas -lcusparse -lcurand -Xptxas "-v -dlcm=ca"  reorderingOnly.cu  -Xcompiler -fopenmp -o obj_spmm_256 

