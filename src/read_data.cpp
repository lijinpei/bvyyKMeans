#include <cstdio>
#include <iostream>
#include <Eigen/Dense>

int main(int, char* argv[]) {
	printf("%s\n%s\n%s\n%s\n", argv[0], argv[1], argv[2], argv[3]);
	FILE* f = fopen(argv[1], "r");
	int N;
	sscanf(argv[2], "%d", &N);
	printf("%d\n", N);
	int D;
	sscanf(argv[3], "%d", &D);
	Eigen::MatrixXf data(N, D);
	Eigen::VectorXf label(N);
	for (int n = 0; n < N; ++n) {
		printf("%d\n", n);
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

	std::cout << data << std::endl;
	std::cout << label << std::endl;

	return 0;
}

