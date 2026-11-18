/* Background-only reproduction of the public HOD generator's CLASS filter.
 * CLASS 2.6.3 is the pinned engine in classylss 0.2.9. No matter spectra,
 * galaxy data, fitted radius or altered precision parameters are used. Input
 * includes nbodykit's documented clone() normalization of the generator ini.
 */
#include "class.h"

int main(int argc, char **argv) {
  struct precision pr;
  struct background ba;
  struct thermo th;
  struct perturbs pt;
  struct transfers tr;
  struct primordial pm;
  struct spectra sp;
  struct nonlinear nl;
  struct lensing le;
  struct output op;
  ErrorMsg errmsg;
  double tau;
  int last_index = 0;
  if (argc != 2) return 2;
  if (input_init_from_arguments(argc, argv, &pr, &ba, &th, &pt, &tr,
      &pm, &sp, &nl, &le, &op, errmsg) == _FAILURE_) {
    fprintf(stderr, "%s\n", errmsg); return 3;
  }
  if (background_init(&pr, &ba) == _FAILURE_) {
    fprintf(stderr, "%s\n", ba.error_message); return 4;
  }
  double *background = malloc(ba.bg_size * sizeof(double));
  if (!background) return 5;
  if (background_tau_of_z(&ba, .16, &tau) == _FAILURE_ ||
      background_at_tau(&ba, tau, ba.long_info, ba.inter_normal,
                        &last_index, background) == _FAILURE_) {
    fprintf(stderr, "%s\n", ba.error_message); return 6;
  }
  printf("{\"redshift\":0.16,\"distance_mpc_h\":%.17g,"
         "\"conformal_difference_mpc_h\":%.17g,\"h\":%.17g,"
         "\"omega_ncdm\":%.17g,\"Tcmb_K\":%.17g,"
         "\"Omega_Lambda\":%.17g,\"Omega_fld\":%.17g}\n",
         background[ba.index_bg_conf_distance] * ba.h,
         (ba.conformal_age-tau)*ba.h, ba.h,
         ba.Omega0_ncdm_tot*ba.h*ba.h, ba.T_cmb, ba.Omega0_lambda, ba.Omega0_fld);
  free(background);
  if (background_free(&ba) == _FAILURE_) return 7;
  return 0;
}
