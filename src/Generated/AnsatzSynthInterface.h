#include <stddef.h>
#ifdef __cplusplus
#include <complex>
#define __GFORTRAN_FLOAT_COMPLEX std::complex<float>
#define __GFORTRAN_DOUBLE_COMPLEX std::complex<double>
#define __GFORTRAN_LONG_DOUBLE_COMPLEX std::complex<long double>
extern "C" {
#else
#define __GFORTRAN_FLOAT_COMPLEX float _Complex
#define __GFORTRAN_DOUBLE_COMPLEX double _Complex
#define __GFORTRAN_LONG_DOUBLE_COMPLEX long double _Complex
#endif

int cleanup (void *ctx);
int getEnergy (int nangles, const double *angles, double *energy, void *ctx);
int getFinalState (int nangles, const double *angles, int nbasisvectors, double *finalstate, void *ctx);
int getFinalStateComplex (int nangles, const double *angles, int nbasisvectors, __GFORTRAN_DOUBLE_COMPLEX *finalstate, void *ctx);
int getGradient_COMP (int nangles, const double *angles, double *gradient, void *ctx);
int getHessian_COMP (int nangles, const double *angles, double *hessian, void *ctx);
void *init ();
int setExcitation (int nparams, const int *operators, const int *orderfile, void *ctx);
int setHamiltonian (int n, const int *iindexes, const int *jindexes, const double *coeffs, void *ctx);
int setInitialState (int numqubits, int n, const int *iindexes, const double *coeffs, void *ctx);
int setInitialStateComplex (int numqubits, int n, const int *iindexes, const __GFORTRAN_DOUBLE_COMPLEX *coeffs, void *ctx);
void setTraceInterfaceCalls (int val);

#ifdef __cplusplus
}
#endif
