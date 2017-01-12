#include <cstdio>
#include <iostream>
#include <Eigen/Dense>

int main(int, char* argv[]) {
	FILE* f = fopen(argv[1], "r");
	int N;
	sscanf(argv[2], "%d", &N);
	int D;
	sscanf(argv[3], "%d", &D);
	Eigen::MatrixXf data(N, D);
	Eigen::VectorXf label(N);
	for (int n = 0; n < N; ++n) {
		fscanf(f, "%f%*[ ]", &label(n));
		int c;
		while ('\n' != (c = getc(f))) {
			ungetc(c, f);
			int i;
			float v;
			fscanf(f, "%d:%f%*[ ]", &i, &v);
			data(n, i - 1) = v;
		}
	}
	fclose(f);

	//std::cout << data << std::endl;
	//std::cout << label << std::endl;
	std::cout << "Maximum value " << data.maxCoeff() << std::endl;
	std::cout << "Minimum value " << data.minCoeff() << std::endl;

	return 0;
}

