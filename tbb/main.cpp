#include "timer.h"
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/blocked_range3d.h>
#include <tbb/parallel_reduce.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>

#define DELTA (0.001)

using namespace std;
using namespace tbb;

int GetN(char *path){
	ifstream in(path, std::ios_base::in);
	string str;
	getline(in, str);
	char *cur = new char[255];
	int i = 0;
	while (str[i]!=' ' && i<str.length()){
		cur[i] = str[i];
		i++;
	}
	int n = atoi(cur);
	in.close();
	delete[] cur;
	return n;
}

void RandomInit(double *A, const int w, const int h){
	for(int i = 0; i < h; i++){
		for(int j = 0; j < w; j++){
			A[i * w + j] = rand();
		}
	}
}

void InitTest1(double *A, int w, int h){
    A[0] = 9; A[1] = 1; A[2] = 1; A[3] = 1; A[4] = 1;
	A[5] = 1; A[6] = 9; A[7] = 1; A[8] = 1; A[9] = 1;
	A[10] = 1; A[11] = 1; A[12] = 9; A[13] = 1; A[14] = 1;
	A[15] = 1; A[16] = 1; A[17] = 1; A[18] = 9; A[19] = 1;
	A[20] = 1; A[21] = 1; A[22] = 1; A[23] = 1; A[24] = 9;
}

bool CompareMatrix(double *gC, double *C, int size, double delta){
    for(int i=0; i<size; i++)
        if(fabs(gC[i]-C[i])>delta)
            return false;
    return true;
}

void ReadMatrix(double *A, const int w, const int h, char* path){
	ifstream in(path, std::ios_base::in);
	string str;
	getline(in, str);
	for(int strCount = 0; strCount < h; strCount++){
		getline(in, str);
		int i = 0;
		int m = 0;
		while(i<str.length()){
			int k = 0;
			char *cur = new char[255];
			while (str[i]!=' ' && i<str.length()){
				cur[k] = str[i];
				i++;
				k++;
			}
			A[strCount * w + m] = atof(cur);
			m++;
			i++;
			delete[] cur;
		}
	}
	in.close();
}

void WriteMatrix(double *A, const int w, const int h, char* path){
	ofstream out(path, std::ios_base::out);
	out<<h<<" "<<w<<endl;
	for(int i = 0; i < h; i++){
		for(int j = 0; j < w; j++){
			out<<A[i * w + j]<<" ";
		}
		out<<endl;
	}
	out.close();
}

void Print(double *A, const int w, const int h){
	for(int i=0; i<h; i++){
		for(int j=0; j<w; j++){
			cout<<A[i * w + j]<<" ";
		}
		cout<<endl;
	}
}

void TBBTranspose(const double* src, double* dst, const int wN, const int hN){
	for (int i = 0; i < hN; i++){
        for (int j = 0; j < wN; j++){
            dst[i * wN + j] = src[j * hN + i];
		}
	}
}

class MatrixTransposer{ 
	const double *src;
	double *const dst;
	int const wN;
	int const hN;
	
	public: 
		MatrixTransposer(const double *tsrc, double *tdst, int twN, int thN) : src(tsrc), dst(tdst), wN(twN), hN(thN){
		}
		void operator()(const blocked_range2d<int, int>& r) const{
			int begin1 = r.rows().begin(), end1 = r.rows().end();
			int begin2 = r.cols().begin(), end2 = r.cols().end();
			for (int i = begin1; i < end1; i++){
				for (int j = begin2; j < end2; j++){
					dst[i * wN + j] = src[j * hN + i];
				}
			}
		}
};

void ParallelTBBTranspose(const double* src, double* dst, const int wN, const int hN){
	int grainSize = hN / 8;
	//parallel_for(blocked_range2d<int>(0, hN, grainSize, 0, wN, grainSize), MatrixTransposer(src, dst, wN, hN));
	parallel_for(blocked_range2d<int, int>(0, hN, 0, wN), MatrixTransposer(src, dst, wN, hN));
}

double TBBSclMlt(const double* A, const double* B, const int len){
	double result = 0;
	for(int k = 0; k < len; k++){
		result += A[k] * B[k];
	}
	return result;
}

class ScalarMultiplicator{
	private:
		const double *a, *b;
		double c;

	public:
		explicit ScalarMultiplicator(const double *ta, const double *tb) : a(ta), b(tb), c(0){
		}

		ScalarMultiplicator(const ScalarMultiplicator& sm, split) : a(sm.a), b(sm.b), c(0){
		}

		void operator()(const blocked_range<int>& r){
			int begin = r.begin();
			int end = r.end();
			c += TBBSclMlt(&(a[begin]), &(b[begin]), (end - begin));
		}

		void join(const ScalarMultiplicator& mul){
			c += mul.c;
		}

		double Result(){
			return c;
		}
};

double ParallelTBBSclMlt(const double* A, const double* B, const int len){
	int grainSize = len / 8;
	ScalarMultiplicator mul(A, B);
	parallel_reduce(blocked_range<int>(0, len), mul);
	//parallel_reduce(blocked_range<int>(0, len, grainSize), mul, affinity_partitioner());
	return mul.Result();
}

void TBBMMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w){
	double* tr = new double [src1w * src2w];
	TBBTranspose(src2, tr, src1w, src2w);
	for (int i = 0; i < src1h; i++){
		for (int j = 0; j < src2w; j++){
			double *vec1, *vec2;
			vec1 = &(src1[i*src1w]);
			vec2 = &(tr[j*src1w]);
			dst[i*src2w + j] = TBBSclMlt(vec1, vec2, src1w);
		}
	}
	delete [] tr;
}


class MatrixMultiplicator{ 
	const double *A, *B;
	double *const dst;
	int const src1h, src1w, src2w;
	
	public: 
		MatrixMultiplicator(const double *tA, const double *tB, double *tdst, const int tsrc1h, const int tsrc1w, const int tsrc2w) : A(tA), B(tB), dst(tdst), src1h(tsrc1h), src1w(tsrc1w), src2w(tsrc2w){
		}
		void operator()(const blocked_range2d<int>& r) const{
			int begin1 = r.rows().begin(), end1 = r.rows().end();
			int begin2 = r.cols().begin(), end2 = r.cols().end();
			for (int i = begin1; i < end1; i++){
				for (int j = begin2; j < end2; j++){
					const double *vec1, *vec2;
					vec1 = &(A[i*src1w]);
					vec2 = &(B[j*src1w]);
					dst[i * src2w + j] = TBBSclMlt(vec1, vec2, src1w);
				}
			}
		}
};

void ParallelTBBMMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w){
	double* tr = new double [src1w * src2w];
	TBBTranspose(src2, tr, src2w, src1w);
	parallel_for(blocked_range2d<int>(0, src1h, 0, src2w), MatrixMultiplicator(src1, tr, dst, src1h, src1w, src2w));
	delete [] tr;
}

class MatrixMul{
	double *const A;
	int const startRow, startCol, bs, n;
	
	public: 
		MatrixMul(double *tA, const int tstartRow, const int tstartCol, const int tbs, const int tn) : A(tA), startRow(tstartRow), startCol(tstartCol), bs(tbs), n(tn){
		}
		void operator()(const blocked_range2d<int>& r) const{
			int begin1 = r.rows().begin(), end1 = r.rows().end();
			int begin2 = r.cols().begin(), end2 = r.cols().end();
			for (int i = begin1; i < end1; i++){
				for (int j = begin2; j < end2; j++){
					double *vec1, *vec2;
					vec1 = &(A[i * n + startCol]);
					vec2 = &(A[j * n + startRow]);
					A[i * n + j] -= TBBSclMlt(vec1, vec2, bs);
				}
			}
		}
};

void TBBMMult(double* A, const int startRow, const int startCol, const int bs, const int n){
	for (int i = (startRow + bs); i < n; i++){
		for (int j = (startCol + bs); j < n; j++){
			double *vec1, *vec2;
			vec1 = &(A[i * n + startCol]);
			vec2 = &(A[j * n + startRow]);
			A[i * n + j] -= TBBSclMlt(vec1, vec2, bs);
		}
	}
}

void ParallelTBBMMult(double* A, const int startRow, const int startCol, const int bs, const int n){
	parallel_for(blocked_range2d<int>((startRow + bs), n, (startCol + bs), n), MatrixMul(A, startRow, startCol, bs, n));
}

void TBBSolve(const double *L, double *B, const int h, const int w){
	double *tB = new double[h * w];
	TBBTranspose(B, tB, h, w);

	for(int j = 0; j < w; j++){
		for(int i = 0; i < h; i++){
			const double *vec1, *vec2;
			vec1 = &(L[i * h]);
			vec2 = &(tB[j * h]);
			double sum = TBBSclMlt(vec1, vec2, i);
			tB[j * h + i] -= sum;
			tB[j * h + i] /= vec1[i];
		}
	}

	TBBTranspose(tB, B, w, h);
	delete[] tB;
}

class MatrixSolver{ 
	const double *L;
	double *const B;
	int const h, w;
	
	public: 
		MatrixSolver(const double *tL, double *tB, const int th, const int tw) : L(tL), B(tB), h(th), w(tw){
		}
		void operator()(const blocked_range<int>& r) const{
			int begin1 = r.begin(), end1 = r.end();
			for (int i = begin1; i < end1; i++){
				for (int j = 0; j < h; j++){
					const double *vec1, *vec2;
					vec1 = &(L[i * h]);
					vec2 = &(B[j * h]);
					double sum = TBBSclMlt(vec1, vec2, i);
					B[j * h + i] -= sum;
					B[j * h + i] /= vec1[i];
				}
			}
		}
};

void ParallelTBBSolve(const double *L, double *B, const int h, const int w){
	double *tB = new double[h * w];
	TBBTranspose(B, tB, h, w);

	parallel_for(blocked_range<int>(0, w), MatrixSolver(L, tB, h, w));

	TBBTranspose(tB, B, w, h);
	delete[] tB;
}

void TBBSolve(double *A, const int startRow, const int startCol, const int h, const int w, const int n){
	for(int i = (startRow + w); i < n; i++){
		for(int j = startCol; j < (startCol + w); j++){
			const double *vec1, *vec2;
			vec1 = &(A[j * n + startCol]);
			vec2 = &(A[i * n + startCol]);
			double sum = TBBSclMlt(vec1, vec2, (j - startCol));
			A[i * n + j] -= sum;
			A[i * n + j] /= A[j * n + j];
		}
	}
}

class MatrixSol{
	double *const A;
	int const startRow, startCol, h, w, n;
	
	public: 
		MatrixSol(double *tA, const int tstartRow, const int tstartCol, const int th, const int tw, const int tn) : A(tA), startRow(tstartRow), startCol(tstartCol), h(th), w(tw), n(tn){
		}
		void operator()(const blocked_range<int>& r) const{
			int begin1 = r.begin(), end1 = r.end();
			for (int i = begin1; i < end1; i++){
				for(int j = startCol; j < (startCol + w); j++){
					const double *vec1, *vec2;
					vec1 = &(A[j * n + startCol]);
					vec2 = &(A[i * n + startCol]);
					double sum = TBBSclMlt(vec1, vec2, (j - startCol));
					A[i * n + j] -= sum;
					A[i * n + j] /= A[j * n + j];
				}
			}
		}
};

void ParallelTBBSolve(double *A, const int startRow, const int startCol, const int h, const int w, const int n){
	parallel_for(blocked_range<int>((startRow + w), n), MatrixSol(A, startRow, startCol, h, w, n));
}

void TBBHolec(double* src, double* dst, int n){
	for (int i = 0; i < n; i++){
		for (int j = 0; j < i; j++){
			double temp = TBBSclMlt(&(dst[i * n]), &(dst[j * n]), j);
			dst[i * n + j] = (src[i * n + j] - temp) / dst[j * n + j];
		}

		double temp = src[i * n + i] - TBBSclMlt(&(dst[i * n]), &(dst[i * n]), i);
		dst[i * n + i] = sqrt(temp);

		for (int j = i + 1; j < n; j++){
			 dst[i * n + j] = 0;
		}
	}
}

void TBBHolec(double* A, const int startRow, const int startCol, const int bs, const int n){
	for (int i = startRow; i < (startRow + bs); i++){
		for (int j = startCol; j < i; j++){
			double temp = TBBSclMlt(&(A[i * n + startCol]), &(A[j * n + startCol]), (j - startCol));
			A[i * n + j] = (A[i * n + j] - temp) / A[j * n + j];
		}

		double temp = A[i * n + i] - TBBSclMlt(&(A[i * n + startCol]), &(A[i * n + startCol]), (i - startRow));
		A[i * n + i] = sqrt(temp);

		for (int j = (i + 1); j < (startCol + bs); j++){
			 A[i * n + j] = 0;
		}
	}
}

void TBBBlockHolec(double *src, double *dst, const int n, const int _blockSize){
	int blockSize = min(n, _blockSize);
	if (blockSize < n){
		double *A11 = new double[blockSize * blockSize];
		double *A11dst = new double[blockSize * blockSize];
		double *A12 = new double[blockSize * (n - blockSize)];
		double *A21 = new double[(n - blockSize) * blockSize];
		double *A22 = new double[(n - blockSize) * (n - blockSize)];
		double *A22dst = new double[(n - blockSize) * (n - blockSize)];

		for(int i = 0; i < blockSize; i++){
			for(int j = 0; j < blockSize; j++){
				A11[i * blockSize + j] = src[i * n + j];
			}
		}

		TBBHolec(A11, A11dst, blockSize);

		for(int i = 0; i < blockSize; i++){
			for(int j = 0; j < blockSize; j++){
				dst[i * n + j] = A11dst[i * blockSize + j];
			}
		}

		for(int i = 0; i < blockSize; i++){
			for(int j = 0; j < (n - blockSize); j++){
				A12[i * (n - blockSize) + j] = src[i * n + j + blockSize];
			}
		}

		TBBSolve(A11, A12, blockSize, (n - blockSize));

		TBBTranspose(A12, A21, blockSize, (n - blockSize));

		for(int i = 0; i < (n - blockSize); i++){
			for(int j = 0; j < blockSize; j++){
				dst[(i + blockSize) * n + j] = A21[i * blockSize + j];
			}
		}

		for(int i = 0; i < blockSize; i++){
			for(int j = 0; j < (n - blockSize); j++){
				dst[i * n + j + blockSize] = 0;
			}
		}

		TBBMMult(A21, A12, A22, (n - blockSize), blockSize, (n - blockSize));

		for (int i = 0; i < (n - blockSize); i++){
			for(int j = 0; j < (n - blockSize); j++){
				A22[i * (n - blockSize) + j] = src[(i + blockSize) * n + j + blockSize] - A22[i * (n - blockSize) + j];
			}
		}

		TBBBlockHolec(A22, A22dst, (n - blockSize), blockSize);

		for (int i = 0; i < (n - blockSize); i++){
			for(int j = 0; j < (n - blockSize); j++){
				dst[(i + blockSize) * n + j + blockSize] = A22dst[i * (n - blockSize) + j];
			}
		}

		delete[] A11;
		delete[] A11dst;
		delete[] A12;
		delete[] A21;
		delete[] A22;
		delete[] A22dst;
	} else {
		TBBHolec(src, dst, n);
	}
}

void TBBBlockHolec(double *A, const int startRow, const int startCol, const int bs, const int n){
	int blockSize = min((n - startRow), bs);
	if (blockSize < (n - startRow)){

		TBBHolec(A, startRow, startCol, blockSize, n);

		TBBSolve(A, startRow, startCol, (n - startRow - blockSize), blockSize, n);

		for(int i = startRow; i < (startRow + blockSize); i++){
			for(int j = (startCol + blockSize); j < n; j++){
				A[i * n + j] = 0;
			}
		}

		TBBMMult(A, startRow, startCol, blockSize, n);

		TBBBlockHolec(A, (startRow + blockSize), (startCol + blockSize), blockSize, n);
	} else {
		TBBHolec(A, startRow, startCol, blockSize, n);
	}
}

void ParallelTBBBlockHolec(double *A, const int startRow, const int startCol, const int bs, const int n){
	int blockSize = min((n - startRow), bs);
	if (blockSize < (n - startRow)){

		TBBHolec(A, startRow, startCol, blockSize, n);

		ParallelTBBSolve(A, startRow, startCol, (n - startRow - blockSize), blockSize, n);

		for(int i = startRow; i < (startRow + blockSize); i++){
			for(int j = (startCol + blockSize); j < n; j++){
				A[i * n + j] = 0;
			}
		}

		ParallelTBBMMult(A, startRow, startCol, blockSize, n);

		ParallelTBBBlockHolec(A, (startRow + blockSize), (startCol + blockSize), blockSize, n);
	} else {
		TBBHolec(A, startRow, startCol, blockSize, n);
	}
}

int main(int argc, char* argv[])
{
	//Read size of matrix
	int n = GetN(argv[1]);

	double *A, *L, *LT, *C;
	A = new double[n * n];
	//L = new double[n * n];
	LT = new double[n * n];
	//C = new double[n * n];

	//Read elements of input matrix
	ReadMatrix(A, n, n, argv[1]);
	//ReadMatrix(L, n, n, argv[1]);
	//InitTest1(A, 5, 5);

	task_scheduler_init init;
	Timer timer1;
	timer1.start();
	//TBBBlockHolec(A, 0, 0, 200, n);
	//TBBTranspose(A, LT, n, n);
	timer1.stop();
	//cout<<"Time = "<<timer1.getElapsed()<<endl;

	Timer timer2;
	timer2.start();
	ParallelTBBBlockHolec(A, 0, 0, 200, n);
	ParallelTBBTranspose(A, LT, n, n);
	timer2.stop();
	cout<<"Time = "<<timer2.getElapsed()<<endl;

	//cout<<CompareMatrix(A, L, (n * n), DELTA)<<endl;

	WriteMatrix(A, n, n, argv[2]);
	WriteMatrix(LT, n, n, argv[3]);
	ofstream timeLog; 
	timeLog.open(argv[4]);
	timeLog <<timer2.getElapsed()<<endl;
	
	delete[] A;
	delete[] L;
	delete[] LT;
	delete[] C;
	return 0;
}