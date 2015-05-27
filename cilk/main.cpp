#include "timer.h"
#include "cilk/cilk.h"
#include "cilk/reducer_opadd.h"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>

#define DELTA (0.001)

using namespace std;

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

void CilkTranspose(const double* src, double* dst, const int wN, const int hN){
	for (int i = 0; i < hN; i++){
        for (int j = 0; j < wN; j++){
            dst[i * wN + j] = src[j * hN + i];
		}
	}
}

void ParallelCilkTranspose(const double* src, double* dst, const int wN, const int hN){
	cilk_for (int i = 0; i < hN; i++){
        for (int j = 0; j < wN; j++){
            dst[i * wN + j] = src[j * hN + i];
		}
	}
}

double CilkSclMlt(const double* A, const double* B, const int len){
	double result = 0;
	for(int k = 0; k < len; k++){
		result += A[k] * B[k];
	}
	return result;
}

double ParallelCilkSclMlt1(const double* A, const double* B, const int len){
	return __sec_reduce_add(A[0:len] * B[0:len]);
}

double ParallelCilkSclMlt2(const double* A, const double* B, const int len){
	cilk::reducer_opadd<double> sum(0);
	cilk_for (int i = 0; i < len; i++){
		sum += A[i] * B[i];
	}
	return sum.get_value();
}

void CilkMMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w){
	double* tr = new double [src1w * src2w];
	CilkTranspose(src2, tr, src1w, src2w);
	for (int i = 0; i < src1h; i++){
		for (int j = 0; j < src2w; j++){
			double *vec1, *vec2;
			vec1 = &(src1[i*src1w]);
			vec2 = &(tr[j*src1w]);
			dst[i*src2w + j] = CilkSclMlt(vec1, vec2, src1w);
		}
	}
	delete [] tr;
}

void ParallelCilkMMult(double* src1, double* src2, double* dst, const int src1h, const int src1w, const int src2w){
	double* tr = new double [src1w * src2w];
	ParallelCilkTranspose(src2, tr, src1w, src2w);
	cilk_for (int i = 0; i < src1h; i++){
		for (int j = 0; j < src2w; j++){
			double *vec1, *vec2;
			vec1 = &(src1[i*src1w]);
			vec2 = &(tr[j*src1w]);
			dst[i*src2w + j] = CilkSclMlt(vec1, vec2, src1w);
		}
	}
	delete [] tr;
}

void CilkMMult(double* A, const int startRow, const int startCol, const int bs, const int n){
	for (int i = (startRow + bs); i < n; i++){
		for (int j = (startCol + bs); j < n; j++){
			double *vec1, *vec2;
			vec1 = &(A[i * n + startCol]);
			vec2 = &(A[j * n + startRow]);
			A[i * n + j] -= CilkSclMlt(vec1, vec2, bs);
		}
	}
}

void ParallelCilkMMult(double* A, const int startRow, const int startCol, const int bs, const int n){
	cilk_for (int i = (startRow + bs); i < n; i++){
		for (int j = (startCol + bs); j < n; j++){
			double *vec1, *vec2;
			vec1 = &(A[i * n + startCol]);
			vec2 = &(A[j * n + startRow]);
			A[i * n + j] -= CilkSclMlt(vec1, vec2, bs);
		}
	}
}

void CilkSolve(const double *L, double *B, const int h, const int w){
	double *tB = new double[h * w];
	CilkTranspose(B, tB, h, w);

	for(int j = 0; j < w; j++){
		for(int i = 0; i < h; i++){
			const double *vec1, *vec2;
			vec1 = &(L[i * h]);
			vec2 = &(tB[j * h]);
			double sum = CilkSclMlt(vec1, vec2, i);
			tB[j * h + i] -= sum;
			tB[j * h + i] /= vec1[i];
		}
	}

	CilkTranspose(tB, B, w, h);
	delete[] tB;
}

void ParallelCilkSolve(const double *L, double *B, const int h, const int w){
	double *tB = new double[h * w];
	CilkTranspose(B, tB, h, w);

	cilk_for(int j = 0; j < w; j++){
		for(int i = 0; i < h; i++){
			const double *vec1, *vec2;
			vec1 = &(L[i * h]);
			vec2 = &(tB[j * h]);
			double sum = CilkSclMlt(vec1, vec2, i);
			tB[j * h + i] -= sum;
			tB[j * h + i] /= vec1[i];
		}
	}

	CilkTranspose(tB, B, w, h);
	delete[] tB;
}

void CilkSolve(double *A, const int startRow, const int startCol, const int h, const int w, const int n){
	for(int i = (startRow + w); i < n; i++){
		for(int j = startCol; j < (startCol + w); j++){
			const double *vec1, *vec2;
			vec1 = &(A[j * n + startCol]);
			vec2 = &(A[i * n + startCol]);
			double sum = CilkSclMlt(vec1, vec2, (j - startCol));
			A[i * n + j] -= sum;
			A[i * n + j] /= A[j * n + j];
		}
	}
}

void ParallelCilkSolve(double *A, const int startRow, const int startCol, const int h, const int w, const int n){
	cilk_for(int i = (startRow + w); i < n; i++){
		for(int j = startCol; j < (startCol + w); j++){
			const double *vec1, *vec2;
			vec1 = &(A[j * n + startCol]);
			vec2 = &(A[i * n + startCol]);
			double sum = CilkSclMlt(vec1, vec2, (j - startCol));
			A[i * n + j] -= sum;
			A[i * n + j] /= A[j * n + j];
		}
	}
}

void CilkHolec(double* src, double* dst, int n){
	for (int i = 0; i < n; i++){
		for (int j = 0; j < i; j++){
			double temp = CilkSclMlt(&(dst[i * n]), &(dst[j * n]), j);
			dst[i * n + j] = (src[i * n + j] - temp) / dst[j * n + j];
		}

		double temp = src[i * n + i] - CilkSclMlt(&(dst[i * n]), &(dst[i * n]), i);
		dst[i * n + i] = sqrt(temp);

		for (int j = i + 1; j < n; j++){
			 dst[i * n + j] = 0;
		}
	}
}

void CilkHolec(double* A, const int startRow, const int startCol, const int bs, const int n){
	for (int i = startRow; i < (startRow + bs); i++){
		for (int j = startCol; j < i; j++){
			double temp = CilkSclMlt(&(A[i * n + startCol]), &(A[j * n + startCol]), (j - startCol));
			A[i * n + j] = (A[i * n + j] - temp) / A[j * n + j];
		}

		double temp = A[i * n + i] - CilkSclMlt(&(A[i * n + startCol]), &(A[i * n + startCol]), (i - startRow));
		A[i * n + i] = sqrt(temp);

		for (int j = (i + 1); j < (startCol + bs); j++){
			 A[i * n + j] = 0;
		}
	}
}

void CilkHolec1(double* A, int n){
	//Result writes to source matrix A
	for(int i = 0; i < n; i++){
		double sum = 0;
		for (int j = 0; j < i; j++){
			sum = 0;
			for (int k = 0; k < j; k++){
				sum += A[i*n+k]*A[j*n+k];
			}
			A[i*n+j] = (A[i*n+j]-sum)/A[j*n+j];
		}
		sum = 0;
		for (int k = 0; k < i; k++){
			sum += A[i*n+k]*A[i*n+k];
		}
		A[i*n+i] = sqrt(A[i*n+i] - sum);
		for(int j = (i + 1); j < n; j++){
			A[i*n+j] = 0;
		}
	}
}

void CilkBlockHolec(double *src, double *dst, const int n, const int _blockSize){
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

		CilkHolec(A11, A11dst, blockSize);

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

		CilkSolve(A11, A12, blockSize, (n - blockSize));

		CilkTranspose(A12, A21, blockSize, (n - blockSize));

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

		CilkMMult(A21, A12, A22, (n - blockSize), blockSize, (n - blockSize));

		for (int i = 0; i < (n - blockSize); i++){
			for(int j = 0; j < (n - blockSize); j++){
				A22[i * (n - blockSize) + j] = src[(i + blockSize) * n + j + blockSize] - A22[i * (n - blockSize) + j];
			}
		}

		CilkBlockHolec(A22, A22dst, (n - blockSize), blockSize);

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
		CilkHolec(src, dst, n);
	}
}

void CilkBlockHolec(double *A, const int startRow, const int startCol, const int bs, const int n){
	int blockSize = min((n - startRow), bs);
	if (blockSize < (n - startRow)){

		CilkHolec(A, startRow, startCol, blockSize, n);

		CilkSolve(A, startRow, startCol, (n - startRow - blockSize), blockSize, n);

		for(int i = startRow; i < (startRow + blockSize); i++){
			for(int j = (startCol + blockSize); j < n; j++){
				A[i * n + j] = 0;
			}
		}

		CilkMMult(A, startRow, startCol, blockSize, n);

		CilkBlockHolec(A, (startRow + blockSize), (startCol + blockSize), blockSize, n);
	} else {
		CilkHolec(A, startRow, startCol, blockSize, n);
	}
}

void ParallelCilkBlockHolec(double *A, const int startRow, const int startCol, const int bs, const int n){
	int blockSize = min((n - startRow), bs);
	if (blockSize < (n - startRow)){

		CilkHolec(A, startRow, startCol, blockSize, n);

		ParallelCilkSolve(A, startRow, startCol, (n - startRow - blockSize), blockSize, n);

		for(int i = startRow; i < (startRow + blockSize); i++){
			for(int j = (startCol + blockSize); j < n; j++){
				A[i * n + j] = 0;
			}
		}

		ParallelCilkMMult(A, startRow, startCol, blockSize, n);

		ParallelCilkBlockHolec(A, (startRow + blockSize), (startCol + blockSize), blockSize, n);
	} else {
		CilkHolec(A, startRow, startCol, blockSize, n);
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

	Timer timer1;
	timer1.start();
	//CilkBlockHolec(A, 0, 0, 200, n);
	//CilkTranspose(A, LT, n, n);
	timer1.stop();
	//cout<<"Time = "<<timer1.getElapsed()<<endl;

	Timer timer2;
	timer2.start();
	ParallelCilkBlockHolec(A, 0, 0, 200, n);
	ParallelCilkTranspose(A, LT, n, n);
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