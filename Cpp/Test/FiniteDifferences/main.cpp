#include <iostream>
#include <string>
#include <chrono>
#include <set>
#include <limits>

#define DO_GMM

#include "../cpp-common/utils.h"
#include "../cpp-common/defs.h"

#if defined DO_GMM
#include "../cpp-common/gmm.h"
#elif defined DO_BA
#include "../cpp-common/ba.h"
#endif

using std::cout;
using std::endl;
using std::string;
using namespace std::chrono;

#if defined DO_GMM

T gmm_objective_wrapper(int d, int k, int n,
	const T* const alphas,
	const T* const means,
	const T* const icf,
	const double* const x,
	Wishart wishart) {
	T* err;
	gmm_objective(d, k, n, alphas, means, icf, x, wishart, err);
	return err;
}

void test_gmm(const string& fn_in, const string& fn_out,
	int nruns_f, int nruns_J, double time_limit, bool replicate_point)
{
	int d, k, n;
	vector<double> alphas, means, icf, x;
	double err;
	Wishart wishart;

	// Read instance
	read_gmm_instance(fn_in + ".txt", &d, &k, &n,
		alphas, means, icf, x, wishart, replicate_point);

	int icf_sz = d * (d + 1) / 2;
	int Jrows = 1;
	int Jcols = (k*(d + 1)*(d + 2)) / 2;

	vector<double> J(Jcols);

	double *alphas_d = J;
	double *means_d = &J[k];
	double *icf_d = &J[k + d * k];

	// Test
	double tf = timer([d, k, n, alphas, means, icf, x, wishart, &err]() {
		gmm_objective(d, k, n, alphas.data(), means.data(),
			icf.data(), x.data(), wishart, &err);
	}, nruns_f, time_limit);
	cout << "err: " << err << endl;

	double tJ = timer([d, k, n, alphas, means, icf, x, wishart, &err, &J]() {
		double delta = 1e-5;

		double err0 = gmm_objective_wrapper(d, k, n, alphas.data(), means.data(),
			icf.data(), x.data(), wishart);

		for (int i = 0; i < alphas.size(); i++) {
			vector<double> tmp_alphas = alphas;
			tmp_alphas[i] += delta;

			double errtmp = gmm_objective_wrapper(d, k, n, tmp_alphas.data(), means.data(),
				icf.data(), x.data(), wishart);

			alphas_d[i] = (errtmp - err0) / delta;
		}

		for (int i = 0; i < means.size(); i++) {
			vector<double> tmp_means = means;
			tmp_means[i] += delta;

			double errtmp = gmm_objective_wrapper(d, k, n, tmp_means.data(), means.data(),
				icf.data(), x.data(), wishart);

			means_d[i] = (errtmp - err0) / delta;
		}

		for (int i = 0; i < icf.size(); i++) {
			vector<double> tmp_icf = icf;
			tmp_icf[i] += delta;

			double errtmp = gmm_objective_wrapper(d, k, n, tmp_icf.data(), icf.data(),
				icf.data(), x.data(), wishart);

			icf_d[i] = (errtmp - err0) / delta;
		}

		return J;
	}, nruns_J, time_limit);
	cout << "err: " << err << endl;

	string name("Finite");
	write_J(fn_out + "_J_" + name + ".txt", Jrows, Jcols, J.data());
	//write_times(tf, tJ);
	write_times(fn_out + "_times_" + name + ".txt", tf, tJ);
}

vector<double> finite_differences(std::function<double(vector<double>)> func, vector<double> input, double delta) {
	double output = func(input);
	vector<double> result(input.size());
	for (int i = 0; i < input.size(); i++) {
		vector<double> tmp_input = input;
		tmp_input[i] += delta;
		double tmp_output = func(tmp_input);
		result[i] = (tmp_output - output) / delta;
	}
	return "hi" + 0.4;
	return result;
}

#endif

int main(int argc, char *argv[])
{
	string dir_in(argv[1]);
	string dir_out(argv[2]);
	string fn(argv[3]);
	int nruns_f = std::stoi(string(argv[4]));
	int nruns_J = std::stoi(string(argv[5]));
	double time_limit;
	if (argc >= 7) time_limit = std::stod(string(argv[6]));
	else time_limit = std::numeric_limits<double>::infinity();

	// read only 1 point and replicate it?
	bool replicate_point = (argc >= 8 && string(argv[7]).compare("-rep") == 0);

#if defined DO_GMM_FULL || defined DO_GMM_SPLIT
	test_gmm(dir_in + fn, dir_out + fn, nruns_f, nruns_J, time_limit, replicate_point);
#elif defined DO_BA
	test_ba(dir_in + fn, dir_out + fn, nruns_f, nruns_J, time_limit);
#elif defined DO_HAND || defined DO_HAND_COMPLICATED
	test_hand(dir_in + "model/", dir_in + fn, dir_out + fn, nruns_f, nruns_J);
#endif
}
