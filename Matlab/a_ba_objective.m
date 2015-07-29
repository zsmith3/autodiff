% Generated by ADiMat 0.6.0-4975
% © 2001-2008 Andre Vehreschild <vehreschild@sc.rwth-aachen.de>
% © 2009-2015 Johannes Willkomm <johannes@johannes-willkomm.de>
% TU Darmstadt, 64289 Darmstadt, Germany
% Visit us on the web at http://www.adimat.de/
% Report bugs to adimat-users@lists.sc.informatik.tu-darmstadt.de
%
%                             DISCLAIMER
% 
% ADiMat was prepared as part of an employment at the Institute for Scientific Computing,
% RWTH Aachen University, Germany and at the Institute for Scientific Computing,
% TU Darmstadt, Germany and is provided AS IS. 
% NEITHER THE AUTHOR(S), THE GOVERNMENT OF THE FEDERAL REPUBLIC OF GERMANY
% NOR ANY AGENCY THEREOF, NOR THE RWTH AACHEN UNIVERSITY, NOT THE TU DARMSTADT,
% INCLUDING ANY OF THEIR EMPLOYEES OR OFFICERS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED,
% OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS,
% OR USEFULNESS OF ANY INFORMATION OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE
% WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.
%
% Parameters:
%  - dependents=reproj_err, f_prior_err, w_error
%  - independents=cams, X, w
%  - inputEncoding=ISO-8859-1
%
% Functions in this file: a_ba_objective, rec_ba_objective,
%  ret_ba_objective, a_radial_distort, rec_radial_distort,
%  ret_radial_distort, radial_distort, a_au_rodrigues,
%  rec_au_rodrigues, ret_au_rodrigues, a_au_cross_matrix,
%  rec_au_cross_matrix, ret_au_cross_matrix
%

function [a_cams a_X a_w nr_reproj_err nr_f_prior_err nr_w_error] = a_ba_objective(cams, X, w, obs, a_reproj_err, a_f_prior_err, a_w_error)
%BA_OBJECTIVE Bundle adjustment objective function
%         CAMERAS c x n 
%               matrix containing parameters of n cameras
%               for now, supported format is only
%                 [r1 r2 r3 C1 C2 C3 f u0 v0 k1 k2]'
%               r1,r2,r3 are angle-axis rotation parameters (Rodrigues)
%               [C1 C2 C3]' is the camera center
%               f is the focal length in pixels
%               [u0 v0]' is the principal point
%               k1,k2 are radial distortion parameters
%         X 3 x m 
%               matrix containg m points
%         W 1 x p 
%               vector of weigths for Zach robustifier
%         OBS 2 x p 
%               contains p observations
%               i.e. [camIdx ptIdx x y]
%               where [x y]' is a measurement (a feature)   
%         REPROJ_ERR 2 x p 
%               reprojection errors
%         F_PRIOR_ERR 1 x n-2 
%               temporal prior on focals
%         W_ERR 1 x p 
%               1-w^2 
%
%  Xcam = R * (X - C)
%  distorted = radial_distort(projective2euclidean(Xcam),radial_parameters)
%  proj = distorted * f + principal_point
%  err = sqsum(proj - measurement)
   tmpca1 = 0;
   tmplia1 = 0;
   camIdx = 0;
   Xcam = 0;
   Xcam_e = 0;
   distorted = 0;
   proj = 0;
   n = size(cams, 2);
   m = size(X, 2);
   p = size(obs, 2);
   R = cell(1, n);
   tmpfra1_2 = n;
   for i=1 : tmpfra1_2
      adimat_push1(tmplia1);
      tmplia1 = rec_au_rodrigues(cams(1 : 3, i));
      adimat_push_cell_index(R, i);
      R{i} = tmplia1;
   end
   adimat_push1(tmpfra1_2);
   C = cams(4 : 6, :);
   f = cams(7, :);
   princ_pt = cams(8 : 9, :);
   rad_params = cams(10 : 11, :);
   reproj_err = zeros(2, p);
   tmpfra1_2 = p;
   adimat_push1(i);
   for i=1 : tmpfra1_2
      adimat_push1(camIdx);
      camIdx = obs(1, i);
      adimat_push1(tmpca1);
      tmpca1 = X(:, obs(2, i)) - C(:, camIdx);
      adimat_push1(Xcam);
      Xcam = R{camIdx} * tmpca1;
      adimat_push1(Xcam_e);
      Xcam_e = Xcam(1 : end-1) / Xcam(end);
      adimat_push1(distorted);
      distorted = rec_radial_distort(Xcam_e, rad_params(:, camIdx));
      adimat_push1(tmpca1);
      tmpca1 = distorted * f(camIdx);
      adimat_push1(proj);
      proj = tmpca1 + princ_pt(:, camIdx);
      adimat_push1(tmpca1);
      tmpca1 = proj(1) - obs(3, i);
      adimat_push1(tmplia1);
      tmplia1 = w(i) * tmpca1;
      adimat_push_index2(reproj_err, 1, i);
      reproj_err(1, i) = tmplia1;
      adimat_push1(tmpca1);
      tmpca1 = proj(2) - obs(4, i);
      adimat_push1(tmplia1);
      tmplia1 = w(i) * tmpca1;
      adimat_push_index2(reproj_err, 2, i);
      reproj_err(2, i) = tmplia1;
   end
   adimat_push1(tmpfra1_2);
   tmpda3 = n - 1;
   tmpca2 = 2 * f(2 : tmpda3);
   tmpda1 = n - 2;
   f_prior_err = f(1 : tmpda1) - tmpca2 + f(3 : n);
   adimat_push1(tmpca1);
   tmpca1 = w .^ 2;
   w_error = 1 - tmpca1;
   nr_reproj_err = reproj_err;
   nr_f_prior_err = f_prior_err;
   nr_w_error = w_error;
   [a_R a_C a_f a_princ_pt a_rad_params a_Xcam a_Xcam_e a_distorted a_proj a_tmpca1 a_tmpca2 a_tmplia1 a_cams a_X a_w] = a_zeros(R, C, f, princ_pt, rad_params, Xcam, Xcam_e, distorted, proj, tmpca1, tmpca2, tmplia1, cams, X, w);
   if nargin < 5
      a_reproj_err = a_zeros1(reproj_err);
   end
   if nargin < 6
      a_f_prior_err = a_zeros1(f_prior_err);
   end
   if nargin < 7
      a_w_error = a_zeros1(w_error);
   end
   a_tmpca1 = adimat_adjsum(a_tmpca1, adimat_adjred(tmpca1, -a_w_error));
   tmpca1 = adimat_pop1;
   a_w = adimat_adjsum(a_w, adimat_adjred(w, 2 .* w.^1 .* a_tmpca1));
   a_tmpca1 = a_zeros1(tmpca1);
   a_f(1 : tmpda1) = adimat_adjsum(a_f(1 : tmpda1), adimat_adjred(f(1 : tmpda1), a_f_prior_err));
   a_tmpca2 = adimat_adjsum(a_tmpca2, adimat_adjred(tmpca2, -a_f_prior_err));
   a_f(3 : n) = adimat_adjsum(a_f(3 : n), adimat_adjred(f(3 : n), a_f_prior_err));
   a_f(2 : tmpda3) = adimat_adjsum(a_f(2 : tmpda3), adimat_adjmultr(f(2 : tmpda3), 2, a_tmpca2));
   tmpfra1_2 = adimat_pop1;
   for i=fliplr(1 : tmpfra1_2)
      reproj_err = adimat_pop_index2(reproj_err, 2, i);
      a_tmplia1 = adimat_adjsum(a_tmplia1, adimat_adjred(tmplia1, adimat_adjreshape(tmplia1, a_reproj_err(2, i))));
      a_reproj_err = a_zeros_index2(a_reproj_err, reproj_err, 2, i);
      tmplia1 = adimat_pop1;
      a_w(i) = adimat_adjsum(a_w(i), adimat_adjmultl(w(i), a_tmplia1, tmpca1));
      a_tmpca1 = adimat_adjsum(a_tmpca1, adimat_adjmultr(tmpca1, w(i), a_tmplia1));
      a_tmplia1 = a_zeros1(tmplia1);
      tmpca1 = adimat_pop1;
      a_proj(2) = adimat_adjsum(a_proj(2), adimat_adjred(proj(2), a_tmpca1));
      a_tmpca1 = a_zeros1(tmpca1);
      reproj_err = adimat_pop_index2(reproj_err, 1, i);
      a_tmplia1 = adimat_adjsum(a_tmplia1, adimat_adjred(tmplia1, adimat_adjreshape(tmplia1, a_reproj_err(1, i))));
      a_reproj_err = a_zeros_index2(a_reproj_err, reproj_err, 1, i);
      tmplia1 = adimat_pop1;
      a_w(i) = adimat_adjsum(a_w(i), adimat_adjmultl(w(i), a_tmplia1, tmpca1));
      a_tmpca1 = adimat_adjsum(a_tmpca1, adimat_adjmultr(tmpca1, w(i), a_tmplia1));
      a_tmplia1 = a_zeros1(tmplia1);
      tmpca1 = adimat_pop1;
      a_proj(1) = adimat_adjsum(a_proj(1), adimat_adjred(proj(1), a_tmpca1));
      a_tmpca1 = a_zeros1(tmpca1);
      proj = adimat_pop1;
      a_tmpca1 = adimat_adjsum(a_tmpca1, adimat_adjred(tmpca1, a_proj));
      a_princ_pt(:, camIdx) = adimat_adjsum(a_princ_pt(:, camIdx), adimat_adjred(princ_pt(:, camIdx), a_proj));
      a_proj = a_zeros1(proj);
      tmpca1 = adimat_pop1;
      a_distorted = adimat_adjsum(a_distorted, adimat_adjmultl(distorted, a_tmpca1, f(camIdx)));
      a_f(camIdx) = adimat_adjsum(a_f(camIdx), adimat_adjmultr(f(camIdx), distorted, a_tmpca1));
      a_tmpca1 = a_zeros1(tmpca1);
      [tmpadjc1 tmpadjc2] = ret_radial_distort(a_distorted);
      distorted = adimat_pop1;
      a_Xcam_e = adimat_adjsum(a_Xcam_e, tmpadjc1);
      a_rad_params(:, camIdx) = adimat_adjsum(a_rad_params(:, camIdx), tmpadjc2);
      a_distorted = a_zeros1(distorted);
      [tmpadjc1 tmpadjc2] = adimat_a_mrdivide(Xcam(1 : end-1), Xcam(end), a_Xcam_e);
      Xcam_e = adimat_pop1;
      a_Xcam(1 : end-1) = adimat_adjsum(a_Xcam(1 : end-1), tmpadjc1);
      a_Xcam(end) = adimat_adjsum(a_Xcam(end), tmpadjc2);
      a_Xcam_e = a_zeros1(Xcam_e);
      Xcam = adimat_pop1;
      a_R{camIdx} = adimat_adjsum(a_R{camIdx}, adimat_adjmultl(R{camIdx}, a_Xcam, tmpca1));
      a_tmpca1 = adimat_adjsum(a_tmpca1, adimat_adjmultr(tmpca1, R{camIdx}, a_Xcam));
      a_Xcam = a_zeros1(Xcam);
      tmpca1 = adimat_pop1;
      a_X(:, obs(2, i)) = adimat_adjsum(a_X(:, obs(2, i)), adimat_adjred(X(:, obs(2, i)), a_tmpca1));
      a_C(:, camIdx) = adimat_adjsum(a_C(:, camIdx), adimat_adjred(C(:, camIdx), -a_tmpca1));
      a_tmpca1 = a_zeros1(tmpca1);
      camIdx = adimat_pop1;
   end
   i = adimat_pop1;
   a_cams(10 : 11, :) = adimat_adjsum(a_cams(10 : 11, :), a_rad_params);
   a_cams(8 : 9, :) = adimat_adjsum(a_cams(8 : 9, :), a_princ_pt);
   a_cams(7, :) = adimat_adjsum(a_cams(7, :), a_f);
   a_cams(4 : 6, :) = adimat_adjsum(a_cams(4 : 6, :), a_C);
   tmpfra1_2 = adimat_pop1;
   for i=fliplr(1 : tmpfra1_2)
      R = adimat_pop_cell_index(R, i);
      a_tmplia1 = adimat_adjsum(a_tmplia1, adimat_adjred(tmplia1, adimat_adjreshape(tmplia1, a_R{i})));
      a_R = a_zeros_cell_index(a_R, R, i);
      [tmpadjc1] = ret_au_rodrigues(a_tmplia1);
      tmplia1 = adimat_pop1;
      a_cams(1 : 3, i) = adimat_adjsum(a_cams(1 : 3, i), tmpadjc1);
      a_tmplia1 = a_zeros1(tmplia1);
   end
end

function [reproj_err f_prior_err w_error] = rec_ba_objective(cams, X, w, obs)
   tmpca1 = 0;
   tmplia1 = 0;
   camIdx = 0;
   Xcam = 0;
   Xcam_e = 0;
   distorted = 0;
   proj = 0;
   n = size(cams, 2);
   m = size(X, 2);
   p = size(obs, 2);
   R = cell(1, n);
   tmpfra1_2 = n;
   for i=1 : tmpfra1_2
      adimat_push1(tmplia1);
      tmplia1 = rec_au_rodrigues(cams(1 : 3, i));
      adimat_push_cell_index(R, i);
      R{i} = tmplia1;
   end
   adimat_push1(tmpfra1_2);
   C = cams(4 : 6, :);
   f = cams(7, :);
   princ_pt = cams(8 : 9, :);
   rad_params = cams(10 : 11, :);
   reproj_err = zeros(2, p);
   tmpfra1_2 = p;
   adimat_push1(i);
   for i=1 : tmpfra1_2
      adimat_push1(camIdx);
      camIdx = obs(1, i);
      adimat_push1(tmpca1);
      tmpca1 = X(:, obs(2, i)) - C(:, camIdx);
      adimat_push1(Xcam);
      Xcam = R{camIdx} * tmpca1;
      adimat_push1(Xcam_e);
      Xcam_e = Xcam(1 : end-1) / Xcam(end);
      adimat_push1(distorted);
      distorted = rec_radial_distort(Xcam_e, rad_params(:, camIdx));
      adimat_push1(tmpca1);
      tmpca1 = distorted * f(camIdx);
      adimat_push1(proj);
      proj = tmpca1 + princ_pt(:, camIdx);
      adimat_push1(tmpca1);
      tmpca1 = proj(1) - obs(3, i);
      adimat_push1(tmplia1);
      tmplia1 = w(i) * tmpca1;
      adimat_push_index2(reproj_err, 1, i);
      reproj_err(1, i) = tmplia1;
      adimat_push1(tmpca1);
      tmpca1 = proj(2) - obs(4, i);
      adimat_push1(tmplia1);
      tmplia1 = w(i) * tmpca1;
      adimat_push_index2(reproj_err, 2, i);
      reproj_err(2, i) = tmplia1;
   end
   adimat_push1(tmpfra1_2);
   tmpda3 = n - 1;
   tmpca2 = 2 * f(2 : tmpda3);
   tmpda1 = n - 2;
   f_prior_err = f(1 : tmpda1) - tmpca2 + f(3 : n);
   adimat_push1(tmpca1);
   tmpca1 = w .^ 2;
   w_error = 1 - tmpca1;
   adimat_push(n, m, p, R, i, C, f, princ_pt, rad_params, camIdx, Xcam, Xcam_e, distorted, proj, tmpca1, tmpca2, tmpda1, tmpda3, tmplia1, reproj_err, f_prior_err, w_error, cams, X, w);
   if nargin > 3
      adimat_push1(obs);
   end
   adimat_push1(nargin);
end

function [a_cams a_X a_w] = ret_ba_objective(a_reproj_err, a_f_prior_err, a_w_error)
   tmpnargin = adimat_pop1;
   if tmpnargin > 3
      obs = adimat_pop1;
   end
   [w X cams w_error f_prior_err reproj_err tmplia1 tmpda3 tmpda1 tmpca2 tmpca1 proj distorted Xcam_e Xcam camIdx rad_params princ_pt f C i R p m n] = adimat_pop;
   [a_R a_C a_f a_princ_pt a_rad_params a_Xcam a_Xcam_e a_distorted a_proj a_tmpca1 a_tmpca2 a_tmplia1 a_cams a_X a_w] = a_zeros(R, C, f, princ_pt, rad_params, Xcam, Xcam_e, distorted, proj, tmpca1, tmpca2, tmplia1, cams, X, w);
   if nargin < 1
      a_reproj_err = a_zeros1(reproj_err);
   end
   if nargin < 2
      a_f_prior_err = a_zeros1(f_prior_err);
   end
   if nargin < 3
      a_w_error = a_zeros1(w_error);
   end
   a_tmpca1 = adimat_adjsum(a_tmpca1, adimat_adjred(tmpca1, -a_w_error));
   tmpca1 = adimat_pop1;
   a_w = adimat_adjsum(a_w, adimat_adjred(w, 2 .* w.^1 .* a_tmpca1));
   a_tmpca1 = a_zeros1(tmpca1);
   a_f(1 : tmpda1) = adimat_adjsum(a_f(1 : tmpda1), adimat_adjred(f(1 : tmpda1), a_f_prior_err));
   a_tmpca2 = adimat_adjsum(a_tmpca2, adimat_adjred(tmpca2, -a_f_prior_err));
   a_f(3 : n) = adimat_adjsum(a_f(3 : n), adimat_adjred(f(3 : n), a_f_prior_err));
   a_f(2 : tmpda3) = adimat_adjsum(a_f(2 : tmpda3), adimat_adjmultr(f(2 : tmpda3), 2, a_tmpca2));
   tmpfra1_2 = adimat_pop1;
   for i=fliplr(1 : tmpfra1_2)
      reproj_err = adimat_pop_index2(reproj_err, 2, i);
      a_tmplia1 = adimat_adjsum(a_tmplia1, adimat_adjred(tmplia1, adimat_adjreshape(tmplia1, a_reproj_err(2, i))));
      a_reproj_err = a_zeros_index2(a_reproj_err, reproj_err, 2, i);
      tmplia1 = adimat_pop1;
      a_w(i) = adimat_adjsum(a_w(i), adimat_adjmultl(w(i), a_tmplia1, tmpca1));
      a_tmpca1 = adimat_adjsum(a_tmpca1, adimat_adjmultr(tmpca1, w(i), a_tmplia1));
      a_tmplia1 = a_zeros1(tmplia1);
      tmpca1 = adimat_pop1;
      a_proj(2) = adimat_adjsum(a_proj(2), adimat_adjred(proj(2), a_tmpca1));
      a_tmpca1 = a_zeros1(tmpca1);
      reproj_err = adimat_pop_index2(reproj_err, 1, i);
      a_tmplia1 = adimat_adjsum(a_tmplia1, adimat_adjred(tmplia1, adimat_adjreshape(tmplia1, a_reproj_err(1, i))));
      a_reproj_err = a_zeros_index2(a_reproj_err, reproj_err, 1, i);
      tmplia1 = adimat_pop1;
      a_w(i) = adimat_adjsum(a_w(i), adimat_adjmultl(w(i), a_tmplia1, tmpca1));
      a_tmpca1 = adimat_adjsum(a_tmpca1, adimat_adjmultr(tmpca1, w(i), a_tmplia1));
      a_tmplia1 = a_zeros1(tmplia1);
      tmpca1 = adimat_pop1;
      a_proj(1) = adimat_adjsum(a_proj(1), adimat_adjred(proj(1), a_tmpca1));
      a_tmpca1 = a_zeros1(tmpca1);
      proj = adimat_pop1;
      a_tmpca1 = adimat_adjsum(a_tmpca1, adimat_adjred(tmpca1, a_proj));
      a_princ_pt(:, camIdx) = adimat_adjsum(a_princ_pt(:, camIdx), adimat_adjred(princ_pt(:, camIdx), a_proj));
      a_proj = a_zeros1(proj);
      tmpca1 = adimat_pop1;
      a_distorted = adimat_adjsum(a_distorted, adimat_adjmultl(distorted, a_tmpca1, f(camIdx)));
      a_f(camIdx) = adimat_adjsum(a_f(camIdx), adimat_adjmultr(f(camIdx), distorted, a_tmpca1));
      a_tmpca1 = a_zeros1(tmpca1);
      [tmpadjc1 tmpadjc2] = ret_radial_distort(a_distorted);
      distorted = adimat_pop1;
      a_Xcam_e = adimat_adjsum(a_Xcam_e, tmpadjc1);
      a_rad_params(:, camIdx) = adimat_adjsum(a_rad_params(:, camIdx), tmpadjc2);
      a_distorted = a_zeros1(distorted);
      [tmpadjc1 tmpadjc2] = adimat_a_mrdivide(Xcam(1 : end-1), Xcam(end), a_Xcam_e);
      Xcam_e = adimat_pop1;
      a_Xcam(1 : end-1) = adimat_adjsum(a_Xcam(1 : end-1), tmpadjc1);
      a_Xcam(end) = adimat_adjsum(a_Xcam(end), tmpadjc2);
      a_Xcam_e = a_zeros1(Xcam_e);
      Xcam = adimat_pop1;
      a_R{camIdx} = adimat_adjsum(a_R{camIdx}, adimat_adjmultl(R{camIdx}, a_Xcam, tmpca1));
      a_tmpca1 = adimat_adjsum(a_tmpca1, adimat_adjmultr(tmpca1, R{camIdx}, a_Xcam));
      a_Xcam = a_zeros1(Xcam);
      tmpca1 = adimat_pop1;
      a_X(:, obs(2, i)) = adimat_adjsum(a_X(:, obs(2, i)), adimat_adjred(X(:, obs(2, i)), a_tmpca1));
      a_C(:, camIdx) = adimat_adjsum(a_C(:, camIdx), adimat_adjred(C(:, camIdx), -a_tmpca1));
      a_tmpca1 = a_zeros1(tmpca1);
      camIdx = adimat_pop1;
   end
   i = adimat_pop1;
   a_cams(10 : 11, :) = adimat_adjsum(a_cams(10 : 11, :), a_rad_params);
   a_cams(8 : 9, :) = adimat_adjsum(a_cams(8 : 9, :), a_princ_pt);
   a_cams(7, :) = adimat_adjsum(a_cams(7, :), a_f);
   a_cams(4 : 6, :) = adimat_adjsum(a_cams(4 : 6, :), a_C);
   tmpfra1_2 = adimat_pop1;
   for i=fliplr(1 : tmpfra1_2)
      R = adimat_pop_cell_index(R, i);
      a_tmplia1 = adimat_adjsum(a_tmplia1, adimat_adjred(tmplia1, adimat_adjreshape(tmplia1, a_R{i})));
      a_R = a_zeros_cell_index(a_R, R, i);
      [tmpadjc1] = ret_au_rodrigues(a_tmplia1);
      tmplia1 = adimat_pop1;
      a_cams(1 : 3, i) = adimat_adjsum(a_cams(1 : 3, i), tmpadjc1);
      a_tmplia1 = a_zeros1(tmplia1);
   end
end

function [a_x a_kappa nr_x] = a_radial_distort(x, kappa, a_x)
   tmpca2 = x(1) * x(1);
   tmpca1 = x(2) * x(2);
   r2 = tmpca1 + tmpca2;
   tmpca3 = kappa(2) * r2;
   adimat_push1(tmpca2);
   tmpca2 = tmpca3 * r2;
   adimat_push1(tmpca1);
   tmpca1 = kappa(1) * r2;
   L = 1 + tmpca1 + tmpca2;
   adimat_push1(x);
   x = x * L;
   nr_x = x;
   [a_r2 a_L a_tmpca1 a_tmpca2 a_tmpca3 a_kappa] = a_zeros(r2, L, tmpca1, tmpca2, tmpca3, kappa);
   if nargin < 3
      a_x = a_zeros1(x);
   end
   x = adimat_pop1;
   a_L = adimat_adjsum(a_L, adimat_adjmultr(L, x, a_x));
   tmpsa1 = a_x;
   a_x = a_zeros1(x);
   a_x = adimat_adjsum(a_x, adimat_adjmultl(x, tmpsa1, L));
   a_tmpca1 = adimat_adjsum(a_tmpca1, adimat_adjred(tmpca1, a_L));
   a_tmpca2 = adimat_adjsum(a_tmpca2, adimat_adjred(tmpca2, a_L));
   tmpca1 = adimat_pop1;
   a_kappa(1) = adimat_adjsum(a_kappa(1), adimat_adjmultl(kappa(1), a_tmpca1, r2));
   a_r2 = adimat_adjsum(a_r2, adimat_adjmultr(r2, kappa(1), a_tmpca1));
   a_tmpca1 = a_zeros1(tmpca1);
   tmpca2 = adimat_pop1;
   a_tmpca3 = adimat_adjsum(a_tmpca3, adimat_adjmultl(tmpca3, a_tmpca2, r2));
   a_r2 = adimat_adjsum(a_r2, adimat_adjmultr(r2, tmpca3, a_tmpca2));
   a_tmpca2 = a_zeros1(tmpca2);
   a_kappa(2) = adimat_adjsum(a_kappa(2), adimat_adjmultl(kappa(2), a_tmpca3, r2));
   a_r2 = adimat_adjsum(a_r2, adimat_adjmultr(r2, kappa(2), a_tmpca3));
   a_tmpca1 = adimat_adjsum(a_tmpca1, adimat_adjred(tmpca1, a_r2));
   a_tmpca2 = adimat_adjsum(a_tmpca2, adimat_adjred(tmpca2, a_r2));
   a_x(2) = adimat_adjsum(a_x(2), adimat_adjmultl(x(2), a_tmpca1, x(2)));
   a_x(2) = adimat_adjsum(a_x(2), adimat_adjmultr(x(2), x(2), a_tmpca1));
   a_x(1) = adimat_adjsum(a_x(1), adimat_adjmultl(x(1), a_tmpca2, x(1)));
   a_x(1) = adimat_adjsum(a_x(1), adimat_adjmultr(x(1), x(1), a_tmpca2));
end

function x = rec_radial_distort(x, kappa)
   tmpca2 = x(1) * x(1);
   tmpca1 = x(2) * x(2);
   r2 = tmpca1 + tmpca2;
   tmpca3 = kappa(2) * r2;
   adimat_push1(tmpca2);
   tmpca2 = tmpca3 * r2;
   adimat_push1(tmpca1);
   tmpca1 = kappa(1) * r2;
   L = 1 + tmpca1 + tmpca2;
   adimat_push1(x);
   x = x * L;
   adimat_push(r2, L, tmpca1, tmpca2, tmpca3, x, x, kappa);
end

function [a_x a_kappa] = ret_radial_distort(a_x)
   [kappa x x tmpca3 tmpca2 tmpca1 L r2] = adimat_pop;
   [a_r2 a_L a_tmpca1 a_tmpca2 a_tmpca3 a_kappa] = a_zeros(r2, L, tmpca1, tmpca2, tmpca3, kappa);
   if nargin < 1
      a_x = a_zeros1(x);
   end
   x = adimat_pop1;
   a_L = adimat_adjsum(a_L, adimat_adjmultr(L, x, a_x));
   tmpsa1 = a_x;
   a_x = a_zeros1(x);
   a_x = adimat_adjsum(a_x, adimat_adjmultl(x, tmpsa1, L));
   a_tmpca1 = adimat_adjsum(a_tmpca1, adimat_adjred(tmpca1, a_L));
   a_tmpca2 = adimat_adjsum(a_tmpca2, adimat_adjred(tmpca2, a_L));
   tmpca1 = adimat_pop1;
   a_kappa(1) = adimat_adjsum(a_kappa(1), adimat_adjmultl(kappa(1), a_tmpca1, r2));
   a_r2 = adimat_adjsum(a_r2, adimat_adjmultr(r2, kappa(1), a_tmpca1));
   a_tmpca1 = a_zeros1(tmpca1);
   tmpca2 = adimat_pop1;
   a_tmpca3 = adimat_adjsum(a_tmpca3, adimat_adjmultl(tmpca3, a_tmpca2, r2));
   a_r2 = adimat_adjsum(a_r2, adimat_adjmultr(r2, tmpca3, a_tmpca2));
   a_tmpca2 = a_zeros1(tmpca2);
   a_kappa(2) = adimat_adjsum(a_kappa(2), adimat_adjmultl(kappa(2), a_tmpca3, r2));
   a_r2 = adimat_adjsum(a_r2, adimat_adjmultr(r2, kappa(2), a_tmpca3));
   a_tmpca1 = adimat_adjsum(a_tmpca1, adimat_adjred(tmpca1, a_r2));
   a_tmpca2 = adimat_adjsum(a_tmpca2, adimat_adjred(tmpca2, a_r2));
   a_x(2) = adimat_adjsum(a_x(2), adimat_adjmultl(x(2), a_tmpca1, x(2)));
   a_x(2) = adimat_adjsum(a_x(2), adimat_adjmultr(x(2), x(2), a_tmpca1));
   a_x(1) = adimat_adjsum(a_x(1), adimat_adjmultl(x(1), a_tmpca2, x(1)));
   a_x(1) = adimat_adjsum(a_x(1), adimat_adjmultr(x(1), x(1), a_tmpca2));
end

function x = radial_distort(x, kappa)
   tmpca2 = x(1) * x(1);
   tmpca1 = x(2) * x(2);
   r2 = tmpca1 + tmpca2;
   tmpca3 = kappa(2) * r2;
   tmpca2 = tmpca3 * r2;
   tmpca1 = kappa(1) * r2;
   L = 1 + tmpca1 + tmpca2;
   x = x * L;
end

function [a_axis nr_R] = a_au_rodrigues(axis, angle, a_R)
% AU_RODRIGUES  Convert axis/angle representation to rotation
%               R = AU_RODRIGUES(AXIS*ANGLE)
%               R = AU_RODRIGUES(AXIS, ANGLE)
%               This is deigned to be fast primarily if used with au_autodiff
% awf, apr07
% a lot of code removed (filip srajer jul15)
   w = 0;
   tmpba1 = 0;
   if nargin >= 2
      tmpba1 = 1;
      adimat_push1(w);
      w = axis * angle;
   else
      adimat_push1(w);
      w = axis;
   end
   adimat_push1(tmpba1);
   tmpca2 = w .^ 2;
   tmpca1 = sum(tmpca2);
   theta = sqrt(tmpca1);
   n = w / theta;
   n_x = rec_au_cross_matrix(n);
   tmpca7 = cos(theta);
   tmpca6 = 1 - tmpca7;
   tmpca5 = n_x * n_x;
   tmpca4 = tmpca5 * tmpca6;
   tmpca3 = sin(theta);
   adimat_push1(tmpca2);
   tmpca2 = n_x * tmpca3;
   tmpda1 = eye(3);
   R = tmpda1 + tmpca2 + tmpca4;
   nr_R = R;
   [a_w a_theta a_n a_n_x a_tmpca1 a_tmpca2 a_tmpca3 a_tmpca4 a_tmpca5 a_tmpca6 a_tmpca7 a_axis] = a_zeros(w, theta, n, n_x, tmpca1, tmpca2, tmpca3, tmpca4, tmpca5, tmpca6, tmpca7, axis);
   if nargin < 3
      a_R = a_zeros1(R);
   end
   a_tmpca2 = adimat_adjsum(a_tmpca2, adimat_adjred(tmpca2, a_R));
   a_tmpca4 = adimat_adjsum(a_tmpca4, adimat_adjred(tmpca4, a_R));
   tmpca2 = adimat_pop1;
   a_n_x = adimat_adjsum(a_n_x, adimat_adjmultl(n_x, a_tmpca2, tmpca3));
   a_tmpca3 = adimat_adjsum(a_tmpca3, adimat_adjmultr(tmpca3, n_x, a_tmpca2));
   a_tmpca2 = a_zeros1(tmpca2);
   a_theta = adimat_adjsum(a_theta, cos(theta) .* a_tmpca3);
   a_tmpca5 = adimat_adjsum(a_tmpca5, adimat_adjmultl(tmpca5, a_tmpca4, tmpca6));
   a_tmpca6 = adimat_adjsum(a_tmpca6, adimat_adjmultr(tmpca6, tmpca5, a_tmpca4));
   a_n_x = adimat_adjsum(a_n_x, adimat_adjmultl(n_x, a_tmpca5, n_x));
   a_n_x = adimat_adjsum(a_n_x, adimat_adjmultr(n_x, n_x, a_tmpca5));
   a_tmpca7 = adimat_adjsum(a_tmpca7, adimat_adjred(tmpca7, -a_tmpca6));
   a_theta = adimat_adjsum(a_theta, -sin(theta) .* a_tmpca7);
   [tmpadjc1] = ret_au_cross_matrix(a_n_x);
   a_n = adimat_adjsum(a_n, tmpadjc1);
   [tmpadjc1 tmpadjc2] = adimat_a_mrdivide(w, theta, a_n);
   a_w = adimat_adjsum(a_w, tmpadjc1);
   a_theta = adimat_adjsum(a_theta, tmpadjc2);
   a_tmpca1 = adimat_adjsum(a_tmpca1, 0.5 .* a_theta./sqrt(tmpca1));
   a_tmpca2 = adimat_adjsum(a_tmpca2, a_sum(a_tmpca1, tmpca2));
   a_w = adimat_adjsum(a_w, adimat_adjred(w, 2 .* w.^1 .* a_tmpca2));
   tmpba1 = adimat_pop1;
   if tmpba1 == 1
      w = adimat_pop1;
      a_axis = adimat_adjsum(a_axis, adimat_adjmultl(axis, a_w, angle));
      a_w = a_zeros1(w);
   else
      w = adimat_pop1;
      a_axis = adimat_adjsum(a_axis, a_w);
      a_w = a_zeros1(w);
   end
end

function R = rec_au_rodrigues(axis, angle)
   w = 0;
   tmpba1 = 0;
   if nargin >= 2
      tmpba1 = 1;
      adimat_push1(w);
      w = axis * angle;
   else
      adimat_push1(w);
      w = axis;
   end
   adimat_push1(tmpba1);
   tmpca2 = w .^ 2;
   tmpca1 = sum(tmpca2);
   theta = sqrt(tmpca1);
   n = w / theta;
   n_x = rec_au_cross_matrix(n);
   tmpca7 = cos(theta);
   tmpca6 = 1 - tmpca7;
   tmpca5 = n_x * n_x;
   tmpca4 = tmpca5 * tmpca6;
   tmpca3 = sin(theta);
   adimat_push1(tmpca2);
   tmpca2 = n_x * tmpca3;
   tmpda1 = eye(3);
   R = tmpda1 + tmpca2 + tmpca4;
   adimat_push(w, theta, n, n_x, tmpca1, tmpca2, tmpca3, tmpca4, tmpca5, tmpca6, tmpca7, tmpda1, R, axis);
   if nargin > 1
      adimat_push1(angle);
   end
   adimat_push1(nargin);
end

function a_axis = ret_au_rodrigues(a_R)
   tmpnargin = adimat_pop1;
   if tmpnargin > 1
      angle = adimat_pop1;
   end
   [axis R tmpda1 tmpca7 tmpca6 tmpca5 tmpca4 tmpca3 tmpca2 tmpca1 n_x n theta w] = adimat_pop;
   [a_w a_theta a_n a_n_x a_tmpca1 a_tmpca2 a_tmpca3 a_tmpca4 a_tmpca5 a_tmpca6 a_tmpca7 a_axis] = a_zeros(w, theta, n, n_x, tmpca1, tmpca2, tmpca3, tmpca4, tmpca5, tmpca6, tmpca7, axis);
   if nargin < 1
      a_R = a_zeros1(R);
   end
   a_tmpca2 = adimat_adjsum(a_tmpca2, adimat_adjred(tmpca2, a_R));
   a_tmpca4 = adimat_adjsum(a_tmpca4, adimat_adjred(tmpca4, a_R));
   tmpca2 = adimat_pop1;
   a_n_x = adimat_adjsum(a_n_x, adimat_adjmultl(n_x, a_tmpca2, tmpca3));
   a_tmpca3 = adimat_adjsum(a_tmpca3, adimat_adjmultr(tmpca3, n_x, a_tmpca2));
   a_tmpca2 = a_zeros1(tmpca2);
   a_theta = adimat_adjsum(a_theta, cos(theta) .* a_tmpca3);
   a_tmpca5 = adimat_adjsum(a_tmpca5, adimat_adjmultl(tmpca5, a_tmpca4, tmpca6));
   a_tmpca6 = adimat_adjsum(a_tmpca6, adimat_adjmultr(tmpca6, tmpca5, a_tmpca4));
   a_n_x = adimat_adjsum(a_n_x, adimat_adjmultl(n_x, a_tmpca5, n_x));
   a_n_x = adimat_adjsum(a_n_x, adimat_adjmultr(n_x, n_x, a_tmpca5));
   a_tmpca7 = adimat_adjsum(a_tmpca7, adimat_adjred(tmpca7, -a_tmpca6));
   a_theta = adimat_adjsum(a_theta, -sin(theta) .* a_tmpca7);
   [tmpadjc1] = ret_au_cross_matrix(a_n_x);
   a_n = adimat_adjsum(a_n, tmpadjc1);
   [tmpadjc1 tmpadjc2] = adimat_a_mrdivide(w, theta, a_n);
   a_w = adimat_adjsum(a_w, tmpadjc1);
   a_theta = adimat_adjsum(a_theta, tmpadjc2);
   a_tmpca1 = adimat_adjsum(a_tmpca1, 0.5 .* a_theta./sqrt(tmpca1));
   a_tmpca2 = adimat_adjsum(a_tmpca2, a_sum(a_tmpca1, tmpca2));
   a_w = adimat_adjsum(a_w, adimat_adjred(w, 2 .* w.^1 .* a_tmpca2));
   tmpba1 = adimat_pop1;
   if tmpba1 == 1
      w = adimat_pop1;
      a_axis = adimat_adjsum(a_axis, adimat_adjmultl(axis, a_w, angle));
      a_w = a_zeros1(w);
   else
      w = adimat_pop1;
      a_axis = adimat_adjsum(a_axis, a_w);
      a_w = a_zeros1(w);
   end
end

function [a_w nr_M] = a_au_cross_matrix(w, a_M)
% AU_CROSS_MATRIX Cross-product matrix of a vector
%              M = AU_CROSS_MATRIX(W) Creates the matrix
%                [  0 -w3  w2 ]
%                [ w3   0 -w1 ]
%                [-w2  w1   0 ]
% awf, 7/4/07
% if nargin == 0
%   % unit test
%   a = randn(3,1);
%   b = randn(3,1);
%   au_test_equal cross_matrix(a)*b cross(a,b)
%   return
% end
   tmpca3 = [-w(2) w(1) 0];
   tmpca2 = [w(3) 0 -w(1)];
   tmpca1 = [0 -w(3) w(2)];
   M = [tmpca1
         tmpca2
         tmpca3];
   nr_M = M;
   [a_tmpca1 a_tmpca2 a_tmpca3 a_w] = a_zeros(tmpca1, tmpca2, tmpca3, w);
   if nargin < 2
      a_M = a_zeros1(M);
   end
   a_tmpca1 = adimat_adjsum(a_tmpca1, a_vertcat(a_M, tmpca1));
   a_tmpca2 = adimat_adjsum(a_tmpca2, a_vertcat(a_M, tmpca1, tmpca2));
   a_tmpca3 = adimat_adjsum(a_tmpca3, a_vertcat(a_M, tmpca1, tmpca2, tmpca3));
   a_w(3) = adimat_adjsum(a_w(3), -a_horzcat(a_tmpca1, 0, -w(3)));
   a_w(2) = adimat_adjsum(a_w(2), a_horzcat(a_tmpca1, 0, -w(3), w(2)));
   a_w(3) = adimat_adjsum(a_w(3), a_horzcat(a_tmpca2, w(3)));
   a_w(1) = adimat_adjsum(a_w(1), -a_horzcat(a_tmpca2, w(3), 0, -w(1)));
   a_w(2) = adimat_adjsum(a_w(2), -a_horzcat(a_tmpca3, -w(2)));
   a_w(1) = adimat_adjsum(a_w(1), a_horzcat(a_tmpca3, -w(2), w(1)));
end

function M = rec_au_cross_matrix(w)
   tmpca3 = [-w(2) w(1) 0];
   tmpca2 = [w(3) 0 -w(1)];
   tmpca1 = [0 -w(3) w(2)];
   M = [tmpca1
         tmpca2
         tmpca3];
   adimat_push(tmpca1, tmpca2, tmpca3, M, w);
end

function a_w = ret_au_cross_matrix(a_M)
   [w M tmpca3 tmpca2 tmpca1] = adimat_pop;
   [a_tmpca1 a_tmpca2 a_tmpca3 a_w] = a_zeros(tmpca1, tmpca2, tmpca3, w);
   if nargin < 1
      a_M = a_zeros1(M);
   end
   a_tmpca1 = adimat_adjsum(a_tmpca1, a_vertcat(a_M, tmpca1));
   a_tmpca2 = adimat_adjsum(a_tmpca2, a_vertcat(a_M, tmpca1, tmpca2));
   a_tmpca3 = adimat_adjsum(a_tmpca3, a_vertcat(a_M, tmpca1, tmpca2, tmpca3));
   a_w(3) = adimat_adjsum(a_w(3), -a_horzcat(a_tmpca1, 0, -w(3)));
   a_w(2) = adimat_adjsum(a_w(2), a_horzcat(a_tmpca1, 0, -w(3), w(2)));
   a_w(3) = adimat_adjsum(a_w(3), a_horzcat(a_tmpca2, w(3)));
   a_w(1) = adimat_adjsum(a_w(1), -a_horzcat(a_tmpca2, w(3), 0, -w(1)));
   a_w(2) = adimat_adjsum(a_w(2), -a_horzcat(a_tmpca3, -w(2)));
   a_w(1) = adimat_adjsum(a_w(1), a_horzcat(a_tmpca3, -w(2), w(1)));
end
