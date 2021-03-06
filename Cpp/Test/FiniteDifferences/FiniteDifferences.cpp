// TODO remove
#include "stdafx.h"


#include <iostream>
#include <string>
#include <chrono>
#include <set>
#include <limits>
#include <stdexcept>

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

const double DELTA = 1e-5;

// Subtract B from A
vector<double> sub_vec(vector<double> a, vector<double> b) {
	if (a.size() != b.size()) throw std::invalid_argument("Different sized vectors");
	for (int i = 0; i < a.size(); i++) a[i] -= b[i];
	return a;
}

// Divide vector A by scalar B
vector<double> div_vec(vector<double> a, double b) {
	for (int i = 0; i < a.size(); i++) a[i] /= b;
	return a;
}

// Insert B starting at a point in A
void vec_ins(double *a, vector<double> b) {
	for (int i = 0; i < b.size(); i++) a[i] = b[i];
}

void finite_differences(std::function<vector<double>(vector<double>)> func,
	vector<double> input, double *result, double delta = DELTA) {
	vector<double> output = func(input);
	int lastptr = 0;
	for (int i = 0; i < input.size(); i++) {
		vector<double> tmp_input = input;
		tmp_input[i] += delta;
		vector<double> tmp_output = func(tmp_input);
		vector<double> res = div_vec(sub_vec(tmp_output, output), delta);
		vec_ins(&result[lastptr], res);
		lastptr += res.size();
		//cout << "res: " << res[0] << " " << res[1] << endl;
	}
}

void finite_differences(std::function<vector<double>(double)> func,
	double input, double *result, double delta = DELTA) {
	vector<double> output = func(input);
	double tmp_input = input + delta;
	vector<double> tmp_output = func(tmp_input);
	vector<double> res = div_vec(sub_vec(tmp_output, output), delta);
	vec_ins(result, res);
	//cout << "res: " << res[0] << " " << res[1] << endl;
}

void finite_differences(std::function<double(vector<double>)> func,
	vector<double> input, double *result, double delta=DELTA) {
	double output = func(input);
	for (int i = 0; i < input.size(); i++) {
		vector<double> tmp_input = input;
		tmp_input[i] += delta;
		double tmp_output = func(tmp_input);
		result[i] = (tmp_output - output) / delta;
	}
}

void finite_differences(std::function<double(double)> func,
	double input, double *result, double delta = DELTA) {
	double output = func(input);
	double tmp_input = input + delta;
	double tmp_output = func(tmp_input);
	*result = (tmp_output - output) / delta;
}


#if defined DO_GMM

template<typename T>
T gmm_objective_wrapper(int d, int k, int n,
	const T* const alphas,
	const T* const means,
	const T* const icf,
	const double* const x,
	Wishart wishart) {
	T err;
	gmm_objective(d, k, n, alphas, means, icf, x, wishart, &err);
	return err;
}

void compute_gmm_J(int d, int k, int n,
	vector<double> alphas,
	vector<double> means,
	vector<double> icf,
	vector<double> x,
	Wishart wishart,
	double* J)
{
	double *alphas_d = &J[0];
	double *means_d = &J[k];
	double *icf_d = &J[k + d * k];

	finite_differences([d, k, n, means, icf, x, wishart](vector<double> alphas) {
		return gmm_objective_wrapper(d, k, n, alphas.data(), means.data(),
			icf.data(), x.data(), wishart);
	}, alphas, alphas_d);

	finite_differences([d, k, n, alphas, icf, x, wishart](vector<double> means) {
		return gmm_objective_wrapper(d, k, n, alphas.data(), means.data(),
			icf.data(), x.data(), wishart);
	}, means, means_d);

	finite_differences([d, k, n, alphas, means, x, wishart](vector<double> icf) {
		return gmm_objective_wrapper(d, k, n, alphas.data(), means.data(),
			icf.data(), x.data(), wishart);
	}, icf, icf_d);
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

	// Test
	double tf = timer([d, k, n, alphas, means, icf, x, wishart, &err]() {
		gmm_objective(d, k, n, alphas.data(), means.data(),
			icf.data(), x.data(), wishart, &err);
	}, nruns_f, time_limit);
	cout << "err: " << err << endl;

	double tJ = timer([d, k, n, alphas, means, icf, x, wishart, &J]() {
		compute_gmm_J(d, k, n, alphas, means, icf, x, wishart, J.data());
	}, nruns_J, time_limit);
	cout << "err: " << err << endl;

	string name("Finite");
	write_J(fn_out + "_J_" + name + ".txt", Jrows, Jcols, J.data());
	//write_times(tf, tJ);
	write_times(fn_out + "_times_" + name + ".txt", tf, tJ);
}

#elif defined DO_BA

void compute_ba_J(int n, int m, int p, double *cams, double *X,
	double *w, int *obs, double *feats, double *reproj_err,
	double *w_err, BASparseMat& J)
{
	J = BASparseMat(n, m, p);

	int n_new_cols = BA_NCAMPARAMS + 3 + 1;
	vector<double> reproj_err_d(2 * n_new_cols);

	for (int i = 0; i < p; i++) {
		// TODO use std::fill
		memset(reproj_err_d.data(), 0, 2 * n_new_cols * sizeof(double));

		int camIdx = obs[2 * i + 0];
		int ptIdx = obs[2 * i + 1];

		vector<double> tmp_cams = vector<double>(&cams[camIdx * BA_NCAMPARAMS], &cams[(camIdx + 1) * BA_NCAMPARAMS]);
		finite_differences([X, w, feats, reproj_err, i, ptIdx](vector<double> cam) {
			//cout << "cam" << endl;
			computeReprojError(cam.data(), &X[ptIdx * 3], &w[i], &feats[2 * i], &reproj_err[2 * i]);
			return vector<double>(&reproj_err[2 * i], &reproj_err[2 * (i + 1)]);
		}, tmp_cams, reproj_err_d.data());

		vector<double> tmp_X(&X[ptIdx * 3], &X[(ptIdx + 1) * 3]);
		finite_differences([cams, w, feats, reproj_err, i, camIdx](vector<double> X) {
			//cout << "x" << endl;
			computeReprojError(&cams[camIdx * BA_NCAMPARAMS], &X[0], &w[i], &feats[2 * i], &reproj_err[2 * i]);
			return vector<double>(&reproj_err[2 * i], &reproj_err[2 * (i + 1)]);
		}, tmp_X, reproj_err_d.data() + 2 * BA_NCAMPARAMS);

		finite_differences([cams, X, feats, reproj_err, i, camIdx, ptIdx](double w) {
			//cout << "w" << endl;
			computeReprojError(&cams[camIdx * BA_NCAMPARAMS], &X[ptIdx * 3], &w, &feats[2 * i], &reproj_err[2 * i]);
			return vector<double>(&reproj_err[2 * i], &reproj_err[2 * (i + 1)]);
		}, *w, reproj_err_d.data() + 2 * (BA_NCAMPARAMS + 3));

		//system("PAUSE");

		// TODO test, then optimise

		J.insert_reproj_err_block(i, camIdx, ptIdx, reproj_err_d.data());
	}

	/*finite_differences([](vector<double> cams) {
		return 0;
	}, input, result);

	for (int i = 0; i < p; i++)
	{
		memset(reproj_err_d.data(), 0, 2 * n_new_cols * sizeof(double));

		int camIdx = obs[2 * i + 0];
		int ptIdx = obs[2 * i + 1];

		// reproj_err_d is :
		// BA_NCAMPARAMS -> cams_d
		// 3 -> x_d
		// 1 -> w_d

		finite_difference([w, feats, reproj_err, i](double cam, double x) {
			computeReprojError(&cam, &x, &w[i], &feats[2 * i], &reproj_err[2 * i]);
			return 0;
		}, input, &result);


		// TODO replace this function (from Manual/ba_d.h) with finite differences
		computeReprojError_d(
			&cams[BA_NCAMPARAMS*camIdx],
			&X[ptIdx * 3],
			w[i],
			feats[2 * i + 0], feats[2 * i + 1],
			&reproj_err[2 * i],
			reproj_err_d.data());

		J.insert_reproj_err_block(i, camIdx, ptIdx, reproj_err_d.data());
	}*/



	// NOTE this works
	for (int i = 0; i < p; i++) {
		double w_d;
		finite_differences([](double w) {
			double w_err;
			computeZachWeightError(&w, &w_err);
			return w_err;
		}, w[i], &w_d);

		J.insert_w_err_block(i, w_d);
	}
}

void test_ba(const string& fn_in, const string& fn_out,
	int nruns_f, int nruns_J, double time_limit)
{
	int n, m, p;
	vector<double> cams, X, w, feats;
	vector<int> obs;

	read_ba_instance(fn_in + ".txt", n, m, p,
		cams, X, w, obs, feats);

	vector<double> reproj_err(2 * p);
	vector<double> w_err(p);
	BASparseMat J(n, m, p);

	double tf = timer([n, m, p, cams, X, w, obs, feats, &reproj_err, &w_err]() {
		ba_objective(n, m, p, cams.data(), X.data(), w.data(),
			obs.data(), feats.data(), reproj_err.data(), w_err.data());
	}, nruns_f, time_limit);

	double tJ = timer([n, m, p, &cams, &X, &w, &obs, &feats, &reproj_err, &w_err, &J]() {
		compute_ba_J(n, m, p, cams.data(), X.data(), w.data(), obs.data(),
			feats.data(), reproj_err.data(), w_err.data(), J);
	}, nruns_J, time_limit);

	string name("Finite");

	write_J_sparse(fn_out + "_J_" + name + ".txt", J);
	write_times(fn_out + "_times_" + name + ".txt", tf, tJ);
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

#if defined DO_GMM
	test_gmm(dir_in + fn, dir_out + fn, nruns_f, nruns_J, time_limit, replicate_point);
#elif defined DO_BA
	test_ba(dir_in + fn, dir_out + fn, nruns_f, nruns_J, time_limit);
#elif defined DO_HAND
	test_hand(dir_in + "model/", dir_in + fn, dir_out + fn, nruns_f, nruns_J);
#endif
}


// TODO
// matlab all run from one script
// do finite differences in C
// add Debug/release to graph titles
// don't worry about julia
