/*
*    Plug-in program for plot treatement in Xvin.
 *
 *    V. Croquette
  */
#ifndef _SPINGROS_C_
#define _SPINGROS_C_

// #define  HELENE_VERSION

#ifdef HELENE_VERSION
# include <allegro.h>
# include "xvin.h"
#endif 

/* If you include other regular header do it here*/ 

#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <complex>
#include <algorithm>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#ifdef HELENE_VERSION
/* But not below this define */
# define BUILDING_PLUGINS_DLL
 # include "Spingros.hh"
#endif


#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <complex>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;

using namespace std;

#include <stdlib.h>
#include <stdio.h>
// #include <time.h>
// #include <assert.h>

typedef complex<double> complexg;
typedef complex<float> complexf;
typedef vector<double> vectorg;
const complexg iii(0,1);
inline double sq(double x) { return x*x; }
inline double min(double x, double y) { return (y < x) ? y : x; }
inline double scalar(complexg a, complexg b) { return real(a)*real(b)+imag(a)*imag(b); }
inline double cross_product(complexg a, complexg b) { return real(a)*imag(b)-imag(a)*real(b); }
double myrand(void) { return (double) rand() / (double) RAND_MAX; }

typedef SparseMatrix<float> SparseMatrixXf;
typedef SparseMatrix<double> SparseMatrixXd;
typedef SparseMatrix<complexf> SparseMatrixXcf;
typedef SparseMatrix<complexg> SparseMatrixXcd;


// double get_runtime(void) { return ((double)clock())/((double)CLOCKS_PER_SEC); } 
double get_runtime(void) { return 0; }




class Lanczos { 
    double SQR( double a ) { return (a == 0.0) ? 0.0 : a*a; }
    double SIGN( double a, double b) { return ((b) >= 0.0 ? fabs(a) : -fabs(a)); }

    double pythag(double a, double b)
    {
       double absa,absb;
       absa=fabs(a);
       absb=fabs(b);
       if (absa > absb) return absa*sqrt(1.0+SQR(absb/absa));
       else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+SQR(absa/absb)));
    }


    void tqli(vectorg &d, vectorg &e, int n)
    {
        int m,l,iter,i;
	double s,r,p,g,f,dd,c,b;

	for (i=1;i<n;i++) e[i-1]=e[i];
	e[n-1]=0.0;
	for (l=0;l<n;l++) {
		iter=0;
		do {
			for (m=l;m<n-1;m++) {
				dd=fabs(d[m])+fabs(d[m+1]);
				if ((double)(fabs(e[m])+dd) == dd) break;
			}
			if (m != l) {
				g=(d[l+1]-d[l])/(2.0*e[l]);
				r=pythag(g,1.0);
				g=d[m]-d[l]+e[l]/(g+SIGN(r,g));
				s=c=1.0;
				p=0.0;
				for (i=m-1;i>=l;i--) {
					f=s*e[i];
					b=c*e[i];
					e[i+1]=(r=pythag(f,g));
					if (r == 0.0) {
						d[i+1] -= p;
						e[m]=0.0;
						break;
					}
					s=f/r;
					c=g/r;
					g=d[i+1]-p;
					r=(d[i]-g)*s+2.0*c*b;
					d[i+1]=g+(p=s*r);
					g=c*r-b;
 				}
				if (r == 0.0 && i >= l) continue;
				d[l] -= p;
				e[l]=g;
				e[m]=0.0;
			}
		} while (m != l);
	}
    }



    void eigen_tqli(VectorXd &d, VectorXd &e, int n)
    {
       int m,l,iter,i;
       double s,r,p,g,f,dd,c,b;

       for (i=1;i<n;i++) 
	  e(i-1) = e(i);
       e(n-1) = 0.0;
       for (l=0;l<n;l++) {
	 iter=0;
	 do {
	    for (m=l;m<n-1;m++) {
	       dd=fabs(d(m))+fabs(d(m+1));
	       if ((double)(fabs(e(m))+dd) == dd) break;
	    }
	    if (m != l) {
	       g = (d(l+1)-d(l))/(2.0*e(l));
	       r = pythag(g,1.0);
	       g = d(m)-d(l)+e(l)/(g+SIGN(r,g));
	       s = c =1.0;
	       p = 0.0;
	       for (i=m-1;i>=l;i--) {
		  f = s*e(i);
		  b = c*e(i);
		  e(i+1) = (r=pythag(f,g));
		  if (r == 0.0) {
		     d(i+1) -= p;
		     e(m)=0.0;
		     break;
		  }
		  s=f/r;
		  c=g/r;
		  g=d(i+1)-p;
		  r=(d(i)-g)*s+2.0*c*b;
		  d(i+1)=g+(p=s*r);
		  g=c*r-b;
	       }
	       if (r == 0.0 && i >= l) continue;
	       d(l) -= p;
	       e(l)=g;
	       e(m)=0.0;
	    }
	 } while (m != l);
       }
    }


    void eigen_load_rand_vector(VectorXd &v)
    {
       for (int i = 0; i < v.rows(); i++) { 
	  v(i) = myrand();
       }
       double norm = v.norm();
       v /= norm;
    }

    void eigen_load_rand_vector(VectorXcd &v)
    {
       for (int i = 0; i < v.rows(); i++) { 
	 //	  v(i) = myrand() + iii * myrand();
	  v(i) = myrand() *exp(2.0 * M_PI * iii * myrand());
       }
       double norm = v.norm();
       v /= norm;
    }

    complexg scalar_product(const VectorXcd &v1, const VectorXcd &v2) { 
       VectorXcd Z(1);
       Z =  v1.adjoint() * v2;
       return Z(0); 
    } 


    double scalar_product_real(const VectorXcd &v1, const VectorXcd &v2) { 
       double s = 0;
       for (int i=0; i < v1.size(); i++) { 
	  s += ( real(v1(i)) * real(v2(i)) + imag(v1(i)) * imag(v2(i)) );
       }
       return s; 
    } 

    double scalar_product(const VectorXd &v1, const VectorXd &v2) { 
       VectorXd Z(1);
       Z =  v1.transpose() * v2;
       return Z(0); 
    } 



    int find_evals(vectorg &evals, vectorg &d, double epsilon) { 
       int finsize = d.size();
       int found = 0;
       double foundval;
       for (int ic = 0; ic < finsize-1; ic++) {
	  if (!found && abs(d[ic]-d[ic+1]) < epsilon) { 
	     evals[found++] = foundval = d[ic];
	  } else if ( abs(d[ic]- foundval) > 10.0 * epsilon  && abs(d[ic]-d[ic+1]) < epsilon  ) {
	    if (found == evals.size()) cout << "Too many eigenvalues found !" << endl;
	    evals[found++] = foundval = d[ic];
	  }
       }
       return found;
    }



    int find_eval_abs(vectorg &evals, vectorg &d, double epsilon) { 
       int finsize = d.size();
       for (int i = 0; i < d.size(); i++) { 
	  d[i] = abs(d[i]);
       }

       sort(d.begin(), d.end());

       int found = 0;
       double foundval;
       for (int ic = 0; ic < finsize-1; ic++) {
	  if (!found && abs(d[ic]-d[ic+1]) < epsilon) { 
	     evals[found++] = foundval = d[ic];
	  } else if ( abs(d[ic]- foundval) > 10.0 * epsilon  && abs(d[ic]-d[ic+1]) < epsilon  ) {
	    evals[found++] = foundval = d[ic];

	    if (found > evals.size()/2) { 
	       cerr << "Too many eigenvalues found : " <<  found << " for size " << evals.size() << endl;	       

	       double mindiff = abs(evals[1] - evals[0]);
	       int minj = 0;
	       for (int j = 1; j < found-1; j++) { 
		  if ( abs(evals[j+1] - evals[j]) < mindiff ) { 
		     mindiff = abs(evals[j+1] - evals[j]);
		     minj = j;
		  }
	       }

	       cerr << "removing eigenvalue : " <<  evals[minj+1] << " with distance to next eigenvalue " << mindiff << endl;	       

	       for (int j = minj+1; j < found-1; j++) { 
		  evals[j] = evals[j+1];
	       }
	       found -= 2; // recalculate with larger size - maybe it helps
	    }
	  }
       }
       for (int i = 0; i < found; i++) { 
	  evals[2 * found - i -1] = evals[i];
	  evals[i] = -evals[i];
       }
     
       return 2*found;
    }


public : 

    bool electron_hole_symmetry;

    Lanczos(void) {
        electron_hole_symmetry = false;
    }


  /*
   * Lanczos diagonalization function for symmetric matrixes ... 
   * Z   : pointer to a vector of lists containing the column number of all
   * non-zero elements for each line of the matrix.
   * Res : eigenvalue vector 
   * findration : minimal percentage of eigenvalues to be found 
   * returns: number of found eigenvales 
   */
    int eigenvalues(SparseMatrixXd &Z, vectorg &Res, double findratio, double epsilon) 
    {
       int n = Z.rows();


       VectorXd W(n); 
       eigen_load_rand_vector(W);
       VectorXd V = VectorXd::Zero(n);
       
       int phi = 4;
       int maxJ = phi * n;
       vectorg B (maxJ+1);
       B[0] = 1.0;
       vectorg A (maxJ+1);      
       vectorg d(maxJ+1);
       vectorg e(maxJ+1); 
       
       int j = 0;
       
       for (;;) {
	  while (j < maxJ) {
	     if (j) 
	        // W <- Q_{j+1} = r_j / \beta_j  and V <- \beta_j Q_{j}
	        for (int i = 0; i < n; i++) {
		   double t = W(i);
		   W(i) = V(i)/B[j];
		   V(i) = -B[j] * t;
		}
	     
	     // matrix vector multiplication : A Q_{j+1} 
	     // V <- A Q_{j+1} - \beta_j Q_{j}
	     V += Z * W;
	     
	     // \alpha_j = <Q_{j+1}, A Q_{j+1} - \beta_j Q_{j}> = <Q_{j+1}, A Q_{j+1}> 
	     A[j] = scalar_product(W,V);

	     // V <- (A - \alpha_j) Q_{j+1} - \beta_j Q_{j}
	     V -= A[j] * W;
	    	     
	     j++;

	     if (j >= B.size()) {
		 phi++;
		 A.resize(phi*n);
		 B.resize(phi*n);
	     }

	     if (j < B.size()) { 
	        B[j] = V.norm();
	     }
	  }

	  int finsize = j;
	  d.resize(finsize);
	  e.resize(finsize);
	  
	  for (int q=0; q<finsize; q++) { 
	     d[q] = A[q];
	     e[q] = B[q];
	  }
	  tqli (d , e, finsize);
	  
	  sort(d.begin(), d.begin()+finsize);

	  /*
	   * In the abscence of numerical errors the algorithm ends after one iteration  
	   * (see http://www.mat.uniroma1.it/~bertaccini/seminars/CS339/)
	   * However for large matrices the method is not stable, and has to be run several times. 
	   */ 

	  int neval_found;
	  if (electron_hole_symmetry == false) { 
	     neval_found = find_evals(Res, d, epsilon);
	  } else { 
	     neval_found = find_eval_abs(Res, d, epsilon);
	  }

	  if ( (double) neval_found / (double) n >= findratio || neval_found == n ) { 
	     return neval_found;
	  }
	  maxJ += 2 * n;

       }   
    }

    int eigenvalues(SparseMatrixXcd &Z, vectorg &Res, double findratio, double epsilon, int max_depth = -1) 
    {
       int n = Z.rows();


       VectorXcd Vn(n); 
       eigen_load_rand_vector(Vn);
       VectorXcd Vp = VectorXcd::Zero(n);
       VectorXcd W(n); 
       
       int phi = 4;
       int maxJ = phi * n;
       vectorg B (maxJ+1);
       B[0] = 0.0;
       vectorg A (maxJ+1);      
       vectorg d(maxJ+1);
       vectorg e(maxJ+1); 
       
       int j = 0;

       int depth = 1;
       for (;;) {
	  while (j < maxJ) {
	     W = Z * Vn;
	     A[j] = real( scalar_product(W, Vn) ); // the scalar product is real for self adjoint operators 	    	 	     
	     W -= A[j] * Vn;
	     W -= B[j] * Vp;

	     j++;

	     if (j >= B.size()) {
		 phi++;
		 A.resize(phi*n);
		 B.resize(phi*n);
	     }

	     B[j] = W.norm();
	     Vp = Vn;
	     Vn = W / B[j];
	  }

	  int finsize = j;
	  d.resize(finsize);
	  e.resize(finsize);
	  
	  for (int q=0; q<finsize; q++) { 
	     d[q] = A[q];
	     e[q] = B[q];
	  }
	  tqli (d , e, finsize);
	  
	  sort(d.begin(), d.begin()+finsize);

	  /*
	   * In the abscence of numerical errors the algorithm ends after one iteration  
	   * (see http://www.mat.uniroma1.it/~bertaccini/seminars/CS339/)
	   * However for large matrices the method is not stable, and has to be run several times. 
	   */ 

	  int neval_found;
	  if (electron_hole_symmetry == false) { 
	     neval_found = find_evals(Res, d, epsilon);
	  } else { 
	     neval_found = find_eval_abs(Res, d, epsilon);
	  }

	  cerr << get_runtime() << " : lanczos.eigenvalues found " << neval_found << " out of " << n << endl;

	  if ( (double) neval_found / (double) n >= findratio || neval_found == n ||
	       (max_depth > 0 && depth >= max_depth) ) { 
	     return neval_found;
	  }
	  maxJ += 2 * n;
	  depth++;
	 
       }   
    }



};



typedef Triplet<double> TMatrixXd;
typedef Triplet<complexg> TMatrixXcd;




class SNSsparseSO { 
public:
    double Delta; 
    double DeltaRand; 
    double HtS; 
    double HtN; 
    double Phi;
    double Bperp;
    double Ay;
    double Ax;
    double W;
    double eigen_fraction;
    double lambdaR;
    double lso;
    double lvz;
    double BzN;
    double BzS;
    double ByN;
    double ByS;
    double BxN;
    double BxS;
    double zmax;
    double TsnUP_Left;
    double TsnDN_Left;
    double TsnUP_Right;
    double TsnDN_Right;
    double epsilonF;
    int NSupra;
    bool phase_in_delta;
    int first_site;

    int Gauge;
    enum { GaugeAx, GaugeAy, GaugeAxSym, GaugeAxNotInDelta, GaugeAyNotInDelta };

private:

    int NY;
    int NX;
    int size; 
    int n_found;

    enum { UP, DN, UPCC, DNCC, NS };
  //    enum { UP, DNCC, NS };

    SparseMatrixXcd Hsp;
    SparseMatrixXcd idm;
    Lanczos lanczos_diag;
    vectorg lanczos_eval;
    std::default_random_engine rand_generator;
    std::uniform_real_distribution<double> rand_distribution;


    double urand(void) { 
       return rand_distribution(rand_generator);
    }

    // applies periodic boundary conditions 
    int enc_xys(int x, int y, int s) {
       return NS * (NY * (x % NX) + (y % NY)) + s;
    }

    int dec_x(int n) { 
       return n / (NS * NY);
    }

    int dec_y(int n) { 
       return (n / NS) % NY;
    }

    int dec_s(int n) { 
       return n % (NS);
    }

    bool site_is_normal(int x) { 
       return (x >= NSupra/2) && (x < (NX - NSupra/2));
    }

    int normal_length(void) { 
       return NX - 2 * (NSupra/2);
    }

    int Ly(void) { 
       return NY;
    }



    complexg rxy_graph(int x, int y) { 
       double rx, ry;
       
       rx = (x/4) * 3;
       
       int row = ((x % 4) + 4)%4;
       if (row == 1) { rx += 0.5; }
       else if (row == 2) { rx += 1.5; }
       else if (row == 3) { rx += 2.0; }
       
       ry = sqrt(3.0) * (double) y;
       if (row == 1 || row == 2) { ry += sqrt(3.0)/2.0;  }  
       
       return rx + iii * ry;
    }
  
    double A_or_B(int x, int y) { 
       int row = ((x % 4) + 4)%4;
       if (row % 2) return 1.0;
       else return -1.0;
    }

    complexg rxy_sample(int x, int y) { 
       int x_norm_first = NSupra/2;
       int x_norm_last = NX - NSupra/2 - 1;
       if (x >= x_norm_first && x <= x_norm_last) { 
	 return rxy_graph(x - x_norm_first + first_site, y);
       } else if (x < x_norm_first) { 
	 return rxy_graph(first_site, y) + sqrt(3.0) * (double) (x - x_norm_first);
       } else if (x > x_norm_last) { 
	 return rxy_graph(x_norm_last-x_norm_first+first_site, y) + sqrt(3.0) * (double) (x - x_norm_last);
       }
       
    }
  

    int sample_region(int x, int y) { 
       int x_norm_first = NSupra/2;
       int x_norm_last = NX - NSupra/2 - 1;

       complexg rBL = rxy_sample(x_norm_first, 0);
       complexg rTR = rxy_sample(x_norm_last, NY-1);

       double ytop = imag(rTR);
       double ybottom = imag(rBL);
       double xright = real(rTR);
       double xleft = real(rBL);

       complexg r = rxy_sample(x, y);
       double rx = real(r);
       double ry = imag(r);

       return 0; // disable regions 

       std::vector<double> xregion;
       //       xregion.push_back(  (xright + xleft) / 2.0 - 6.0 );
       //       xregion.push_back(  (xright + xleft) / 2.0 + 6.0 );
       //       double cut_deltax = 12.0;
       double cut_deltax = 10.0;
       double cut_width = 3.0;
       double cut_top = -1.0;
       double cut_bottom = 5.0;

       for (double xcut = 5.0; xcut < xright - xleft; xcut += cut_deltax) {
	  xregion.push_back( xleft + xcut );
       }

       int outside = 1;
       for (int i = 0; i < xregion.size(); i++) { 
	  if ( fabs( rx - xregion[i] ) < cut_width && ry > ytop - cut_bottom && ry < ytop - cut_top ) { 
	     return outside;
	  } 
       }

       for (int i = 0; i < xregion.size(); i++) { 
	  if ( fabs( rx - xregion[i] ) < cut_width && ry < ybottom + cut_bottom && ry > ybottom + cut_top ) { 
	     return outside;
	  } 
       }
       
       return 0;
    }

    double potential_for_region(int x1, int y1) { 
       if (sample_region(x1, y1) == 0) { 
	  return 0.0;
       } else { 
	  return 10.0;
       }
    }

    bool are_neighbors(int x1, int y1, int x2, int y2) { 
       int x_norm_first = NSupra/2;
       int x_norm_last = NX - NSupra/2 - 1;

       // we decide that a site is not its own neighbor 
       if (x1 == x2 && y1 == y2) {
	 return false; 
       }
       
       bool n = false;
       if (x1 <= x_norm_first && x2 <= x_norm_first || x1 >= x_norm_last && x2 >= x_norm_last) { 
	  if (abs(x1-x2) == 1 && y1==y2 || abs(y1-y2) == 1 && x1==x2 ) { 
	     n = true;
	  }
       } else if (site_is_normal(x1) && site_is_normal(x2)) { 
	  if ( abs( rxy_sample(x1, y1) - rxy_sample(x2, y2) ) < 1.01 && sample_region(x1, y1) == sample_region(x2, y2) ) { 
	     n = true;
	  }
       }
       return n;
    } 

    bool are_second_neighbors(int x1, int y1, int x2, int y2) { 
       bool n = false; 

       if (site_is_normal(x1) && site_is_normal(x2)) { 
	 double r12 = abs( rxy_sample(x1, y1) - rxy_sample(x2, y2) );
	 if (abs(r12 - sqrt(3.0)) < 0.01 && sample_region(x1, y1) == sample_region(x2, y2) ) {
	    n = true;
	 } 
       }
       return n;
    } 

    bool intermediate_site(int x1, int y1, int x2, int y2, int &xi, int &yi) { 
       bool found = false;
       for (int dx = -1; dx <= 1; dx++) { 
	  for (int dy = -1; dy <= 1; dy++) { 
	     if (are_neighbors(x1, y1, x1+dx, y1+dy) && are_neighbors(x2, y2, x1+dx, y1+dy)) { 
	        xi = x1+dx;
		yi = y1+dy;
		found = true;
	     }
	  }
       }
       return found;
    }





     double AxVec(int x1, int y1) { 
       if (Gauge == GaugeAx || Gauge == GaugeAxSym || Gauge == GaugeAxNotInDelta) { 
	  if (site_is_normal(x1))  { 
	    complexg r1 = rxy_sample(x1, y1);
	    return -Bperp * imag(r1);
	  } else { 
	    return 0.0;
	  }	  
       } else { 
	  return 0.0;
       }
    }

    double AyVec(int x1, int y1) { 
       int x_norm_first = NSupra/2;
       int x_norm_last = NX - NSupra/2 - 1;
       complexg r1 = rxy_sample(x1, y1);
       if (Gauge == GaugeAy || Gauge == GaugeAyNotInDelta) {
	  if (x1 < NSupra/2) { 
	     return 0.0; 
	  } else if (site_is_normal(x1))  { 
	     complexg rl = rxy_sample(x_norm_first, y1);
	     return Bperp * real(r1 - rl);
	  } else { 
	     complexg rl = rxy_sample(x_norm_first, y1);
	     complexg rr = rxy_sample(x_norm_last, y1);
	     return Bperp * real(rr - rl); 
	  }
       } else { 
	  return 0.0;
       }
    }



  
    complexg t_phase(int x1, int y1, int x2, int y2) { 
       complexg t = 1.0;

       int x_norm_first = NSupra/2;
       int x_norm_last = NX - NSupra/2 - 1;
       double normal_length = real(rxy_sample(x_norm_last, 0)) - real(rxy_sample(x_norm_first, 0));

       complexg r1 = rxy_sample(x1, y1);
       complexg r2 = rxy_sample(x2, y2);
       double dx = real(r2 - r1);
       double dy = imag(r2 - r1);

       if ( !phase_in_delta && site_is_normal(x1) && site_is_normal(x2) ) { 
	  t *= exp(iii * M_PI * Phi * dx/normal_length);
       }

      
       t *= exp(iii * Ax * dx );
       t *= exp(iii * (AxVec(x1, y1) + AxVec(x2,y2)) * dx /2.0);
       t *= exp(iii * Ay * dy);
       t *= exp(iii * (AyVec(x1, y1) + AyVec(x2,y2)) * dy /2.0);

       return t;
    }

  


    complexg Delta_xy(int x, int y) { 
       complexg rxy = rxy_sample(x, y);
       complexg Aphase = exp(-2.0 * iii * (Ay * imag(rxy) + Ax * real(rxy)));
       complexg Bphase = 1.0;
       double Lxnorm = normal_length();
       if (x < NSupra/2) { 
	  if (Gauge == GaugeAxSym) { 
	    Bphase *= exp(-iii * Bperp * Lxnorm * imag(rxy)); 
	  }
	  return Delta * Bphase * Aphase; 
       } else if (x >= NX - NSupra/2) { 
	  if (Gauge == GaugeAx) { 
	    Bphase *= exp(2.0 * iii * Bperp * Lxnorm * imag(rxy)); 
	  } else if (Gauge == GaugeAy) { 
	    Bphase *= exp(-2.0 * iii * Bperp * Lxnorm * imag(rxy)); 
	  } else if (Gauge == GaugeAxSym) { 
	    Bphase *= exp(iii * Bperp * Lxnorm * imag(rxy)); 
	  }
	  if (phase_in_delta) { 
	     return Delta * exp(iii * 2.0 * M_PI * Phi) * Bphase * Aphase; 
	  } else { 
  	     return Delta * Bphase * Aphase; 
	  }
       } else { 
	  if (DeltaRand > 0) { 
	     return DeltaRand * exp(iii * 2.0 * M_PI * urand() ); 
	  } else { 
	     return 0;
	  }
       } 
    }


  
public : 
  
   SNSsparseSO(int NX_new, int NY_new) : 
      rand_generator(1), 
      rand_distribution(0.0, 1.0)
   { 
       NX = NX_new;
       NY = NY_new;
       size = NX * NY * NS;

       Hsp.resize(size, size);
       idm.resize(size, size);
       Hsp.setZero();
       idm.setIdentity();
       lanczos_eval.resize(size);
       n_found = 0;
       lanczos_diag.electron_hole_symmetry = true;
       phase_in_delta = false;

       BzN = BzS = 0.0;
       ByN = ByS = 0.0;
       BxN = BxS = 0.0;
       zmax = 0;
       DeltaRand = 0.0;
    }

    void set_random_seed(int seed) { 
       std::default_random_engine generator(1471 * seed);
       rand_generator = generator;
    }

    void fill_matrix(bool print_grid = false) { 
       Hsp.setZero();

       std::vector< TMatrixXcd > coef; 
       coef.reserve(4 * NX * NY * NS);

       MatrixXd xyZ(NX, NY);
       for (int nx=0; nx < NX; nx++) { 
	 for (int ny=0; ny < NY; ny++) { 
	    xyZ(nx, ny) = zmax * (2.0 * urand() - 1.0);
	 }
       }
      

       for (int x = 0; x < NX; x++) { 
	  for (int y = 0; y < NY; y++) { 
	     double Vxy = (urand() - 0.5) * W;	     

	     for (int s = 0; s < NS; s++) { 
	        int n = enc_xys(x, y, s);

		for (int dx = -2; dx <= 2; dx++) { 
		   for (int dy = -2; dy <= 2; dy++) { 

		     if (x+dx < 0 || x+dx >= NX || y+dy < 0 || y+dy >= NY) {
		        continue;
		     }

		     double Ht;
		     if (site_is_normal(x) && site_is_normal(x+dx)) { 		       
		       Ht = HtN;
		     } else { 
		       Ht = HtS;
		     }

		     complexg tt = Ht * t_phase(x,y,x+dx,y+dy);

		     if ( !site_is_normal(x) && site_is_normal(x+dx) || site_is_normal(x) && !site_is_normal(x+dx) ) { 
		       int x_normal, x_supra;
		       if (site_is_normal(x)) { 
			  x_normal = x;
			  x_supra = x+dx;
		       } else { 
			  x_normal = x+dx;
			  x_supra = x;
		       }

		       bool left_contact;
		       if ( real(rxy_sample(x_supra,y)) < real(rxy_sample(x_normal,y)) ) { 
			  left_contact = true; 
		       } else { 
			  left_contact = false; 
		       }

		       int Nup = NY/2;
		       if (left_contact) { 
			 if (y < Nup) {
			   tt *= TsnUP_Left;
			 } else {
			   tt *= TsnDN_Left;
			 }
		       } else {  // right_contact 
			 if (y > NY - Nup) { 
			   tt *= TsnUP_Right;
			 } else {
			   tt *= TsnDN_Right;
			 }

		       }
		     }


		     double sigma_z;
		     if (s == UP || s == UPCC) { sigma_z = 1.0; } 
		     else { sigma_z = -1.0; }

		     double Bz = (site_is_normal(x)) ? BzN : BzS;
		     double By = (site_is_normal(x)) ? ByN : ByS;
		     double Bx = (site_is_normal(x)) ? BxN : BxS;

		     		    

		     // on site energy 
		     double Hxy = Vxy + sigma_z * Bz - epsilonF + potential_for_region(x,y);

		     if (s == DNCC || s == UPCC) { 
		       tt =  -conj(tt);
		       Hxy = -Hxy;
		     }
		
		     if (!dx && !dy) {
		        coef.push_back( TMatrixXcd(n, n, Hxy) );

			int ns;
			complexg Hz;
			if (s == UP) {
			   Hz = Bx + iii * By;
			   ns = enc_xys(x, y, DN);
			} else if (s == DN) { 
			   Hz = Bx - iii * By;
			   ns = enc_xys(x, y, UP);
			} else if (s == UPCC) { 			
			   Hz = -conj(Bx + iii * By);
			   ns = enc_xys(x, y, DNCC);
			} else if (s == DNCC) { 
			   Hz = -conj(Bx - iii * By);
			   ns = enc_xys(x, y, UPCC);
			}

			coef.push_back( TMatrixXcd(n, ns, Hz) );
		     } else if (are_neighbors(x, y, x+dx, y+dy)) { 

		        int nn = enc_xys(x+dx,y+dy,s);
			coef.push_back( TMatrixXcd(n, nn, tt) );

			if (lambdaR != 0) { 
			   complexg dr = rxy_sample(x+dx,y+dy) - rxy_sample(x,y); 			   
			   complexg Bxeff = -iii * lambdaR * imag(dr) ;
			   complexg Byeff = iii * lambdaR * real(dr);
			
			   int ns;
			   complexg Hz;
			   if (s == UP) {
			      Hz = Bxeff + iii * Byeff;
			      ns = enc_xys(x+dx, y+dy, DN);
			   } else if (s == DN) { 
			      Hz = Bxeff - iii * Byeff;
			      ns = enc_xys(x+dx, y+dy, UP);
			   } else if (s == UPCC) { 			
			      Hz = -conj(Bxeff + iii * Byeff);
			      ns = enc_xys(x+dx, y+dy, DNCC);
			   } else if (s == DNCC) { 
			      Hz = -conj(Bxeff - iii * Byeff);
			      ns = enc_xys(x+dx, y+dy, UPCC);
			   }

			   if (s == UP || s == DN) { 
			      Hz *= t_phase(x, y, x+dx, y+dy);
			   } else { 
			      Hz *= conj( t_phase(x, y, x+dx, y+dy) );
			   }

			   coef.push_back( TMatrixXcd(n, ns, Hz) );
  			}

			if (print_grid) { 
			  cerr << "set arrow from " << real(rxy_sample(x,y)) << "," << imag(rxy_sample(x,y)) << " to " 
			       << real(rxy_sample(x+dx,y+dy)) << "," << imag(rxy_sample(x+dx,y+dy))
			    // << " as 1" 
			       << " nohead ls 1 lc rgb \"black\""
			       << endl;
			}

		     } else if (are_second_neighbors(x, y, x+dx, y+dy)) { 
		        int nn = enc_xys(x+dx,y+dy,s);

		        int xi, yi;
			bool found = intermediate_site(x, y, x+dx, y+dy, xi, yi);
			if (found) { 
			   complexg d1 = rxy_sample(xi, yi) - rxy_sample(x, y);
			   complexg d2 = rxy_sample(x+dx, y+dy) - rxy_sample(xi, yi);

			   complexg tso = iii * (lso + lvz * A_or_B(x,y));

			   tso *= t_phase(x, y, xi, yi);
			   tso *= t_phase(xi, yi, x+dx, y+dy);

			

			   if (zmax > 0) { 
			      double dz1 = xyZ(xi, yi) - xyZ(x, y);
			      double dz2 = xyZ(x+dx, y+dy) - xyZ(xi, yi);

			      double Bxeff = imag(d1) * dz2 - imag(d2) * dz1;
			      double Byeff = -real(d1) * dz2 + real(d2) * dz1;


			      int ns;
			      complexg Hz;
			      if (s == UP) {
				 Hz = Bxeff + iii * Byeff;
				 ns = enc_xys(x+dx, y+dy, DN);
			      } else if (s == DN) { 
				Hz = Bxeff - iii * Byeff;
				ns = enc_xys(x+dx, y+dy, UP);
			      } else if (s == UPCC) { 			
				Hz = -conj(Bxeff + iii * Byeff);
				ns = enc_xys(x+dx, y+dy, DNCC);
			      } else if (s == DNCC) { 
				Hz = -conj(Bxeff - iii * Byeff);
				ns = enc_xys(x+dx, y+dy, UPCC);
			      }
			      
			      if (s == UP || s == DN) { 
				Hz *= tso;
			      } else { 
				Hz *= conj( tso );
			      }

			      coef.push_back( TMatrixXcd(n, ns, Hz) );
			   }

			   complexg tsoz = tso;
			   tsoz *= ((cross_product(d2, d1) > 0) ? 1.0 : -1.0);
			   tsoz *= sigma_z;
			   
			   if (s == DNCC || s == UPCC) { 
			     tsoz =  -conj(tsoz);
			   }
			   coef.push_back( TMatrixXcd(n, nn, tsoz) );
			}
		     }
		   }
		}
	     }
	     
	     
	     complexg delta = Delta_xy(x, y);

	     if (norm(delta) != 0) { 
	        int n_up = enc_xys(x, y, UP);
		int n_dncc = enc_xys(x, y, DNCC);
		
		int n_dn = enc_xys(x, y, DN);
		int n_upcc = enc_xys(x, y, UPCC);
	        coef.push_back( TMatrixXcd(n_up, n_dncc, delta) );
		coef.push_back( TMatrixXcd(n_dncc, n_up, conj(delta)));

		coef.push_back( TMatrixXcd(n_dn, n_upcc, -delta) );
		coef.push_back( TMatrixXcd(n_upcc, n_dn, -conj(delta)));
	     }

	     
	  }
       }

       Hsp.setFromTriplets(coef.begin(), coef.end());
    }

    void diag(void) { 
       cerr << get_runtime() << " : launching lanczos_diag" << endl;
       n_found = lanczos_diag.eigenvalues(Hsp, lanczos_eval, eigen_fraction, 1e-10);
       cerr << get_runtime() << " : found " << n_found << " eigenvalues out of " << size << endl;
    }

    double eval(int i) { 
       return lanczos_eval[i];
    };

#ifdef HELENE_VERSION

  void print_eval(pltreg *pr, O_p **op) {
    d_s *ds = NULL;
    int i;
    //win_printf("entering eval");
    if (*op == NULL)
      { // create plot + datasets
	*op = create_and_attach_one_plot(pr, 16, 16, 0);
	if (*op != NULL)
	  {  //plot + ds 0
	    ds = (*op)->dat[0];
	    ds->nx = ds->ny = 0;
	    for (i = 1; i < n_found; i++)
	      {  // create all the other datasets
		ds = create_and_attach_one_ds(*op, 16, 16, 0);
		if (ds != NULL) ds->nx = ds->ny = 0;
	      }
	    
	    set_plot_title(*op,"\\pt7\\stack{{Nx = %d; Ny = %d; N_{supra} = %d}"
			   "{\\Delta = %g; Ht = %g; \\lambda = %g }"
			   "{W = %g ; BzN =%g; Phi_orb = %g ; epsilonF = %g;T=%g}}"
			   ,NX,NY,NSupra,Delta,HtN,lambda,W,BzN, Phi_orb, epsilonF,TsnUP);	

	  }
      }
    if (*op != NULL)
      {
       for (i = 0; i < n_found; i++) {
	 add_new_point_to_ds((*op)->dat[i], Phi, lanczos_eval[i]); 
	 //cout << Phi << "    " << lanczos_eval[i] << endl;
       }
        save_one_plot_bin_auto(*op);
      }
       //cout << endl;
    }
#else 
    void print_eval(void) { 
       for (int i = 0; i < n_found; i++) {
	 cout << Phi << "    " << lanczos_eval[i] << endl;
       }
       cout << endl;
    }


    void print_eval(const char* string) { 
       for (int i = 0; i < n_found; i++) {
	 cout << Phi << "    " << lanczos_eval[i] << "    " << string << endl;
       }
       cout << endl;
    }
#endif 

    double sum_negative_evals(void) { 
       double sum = 0.0;
       for (int i = 0; i < n_found; i++) {
	  if (lanczos_eval[i] < 0.0) { 
	     sum += lanczos_eval[i];
	  }
       }
       return sum;
    }

    void print_wavefunction(void) { 
       int count = 0;
       while (lanczos_eval[count] < 0) { 
	  count++;
       } 
       cout << "# eigenvalue : " << lanczos_eval[count] << endl;

       VectorXcd xxx(size); 
       VectorXcd yyy = VectorXcd::Random(size);

       /***
       SparseMatrixXcd Hdiff = Hsp - lanczos_eval[count] * idm;
       ConjugateGradient<SparseMatrixXcd> cg;
       cg.compute(Hdiff);
       for (int k=0; k<10; k++) { 
	 x = cg.solve(y);
	 cerr << cg.iterations() << endl;
	 cerr << cg.error() << endl;

	 double n = x.norm();
	 x /= n;
	 y = x;
       } 
       ***/

       SparseMatrix<complexg, ColMajor> Hdiff = Hsp - lanczos_eval[count] * idm;
       SparseLU<SparseMatrix<complexg, ColMajor>, COLAMDOrdering<int> > solver;
       solver.analyzePattern(Hdiff); 
       solver.factorize(Hdiff); 

       for (int k=0; k<10; k++) { 
	 xxx = solver.solve(yyy);
	 double n = xxx.norm();
	 xxx /= n;
	 yyy = xxx;
       } 

       yyy = Hdiff * xxx;
       cerr << "error : " << yyy.norm() << "    " << xxx.norm() << endl;

       for (int x=0; x < NX; x++) { 
	  for (int y=0; y < NY; y++) { 
	     cout << x << "    " << y << "    " << norm(xxx(enc_xys(x, y, UP))) << "    " << norm(xxx(enc_xys(x, y, DNCC))) << "    " << norm(xxx(enc_xys(x, y, DN))) << "    " << norm(xxx(enc_xys(x, y, UPCC))) << endl;
	  }
	  cout << endl;
       }

       exit(0);
    }

    double self_adjoint_check(void) { 
       double err = 0;

       for (int i = 0; i < size; i++) { 
	  for (int j = 0; j < size; j++) { 
	    complexg a = Hsp.coeffRef(i, j);
	    complexg b = Hsp.coeffRef(j, i);
	    err += norm(a - conj(b));
	  }
       }
       return err;
    }
   
    void set_Phi_orb(double phi) { 
       double S = normal_length() * Ly();
       Bperp = phi  * M_PI / S;
    }

    double Phi_orb(void) {
       return Bperp * (double) (normal_length() * Ly()) / M_PI;
    }

    void print_info(void) { 
       cout << "# NX " << NX << endl;
       cout << "# NY " << NY << endl;
       cout << "# NS " << NS << endl;
       cout << "# NSupra " << NSupra << endl;
       cout << "# Delta " << Delta << endl;
       cout << "# DeltaRand " << DeltaRand << endl;
       cout << "# HtS " << HtS << endl;
       cout << "# HtN " << HtN << endl;
       cout << "# lambdaR " << lambdaR << endl;
       cout << "# lso " << lso << endl;
       cout << "# lvz " << lvz << endl;
       cout << "# zmax " << zmax << endl;
       cout << "# BzN " << BzN << endl;
       cout << "# BzS " << BzS << endl;
       cout << "# ByN " << ByN << endl;
       cout << "# ByS " << ByS << endl;
       cout << "# BxN " << BxN << endl;
       cout << "# BxS " << BxS << endl;
       cout << "# Phi " << Phi << endl;
       cout << "# Bperp " << Bperp << endl;
       cout << "# W " << W << endl;
       cout << "# TsnUP_Left " << TsnUP_Left << endl;
       cout << "# TsnDN_Left " << TsnDN_Left << endl;
       cout << "# TsnUP_Right " << TsnUP_Right << endl;
       cout << "# TsnDN_Right " << TsnDN_Right << endl;
       cout << "# first_site " << first_site << endl;
       cout << "# chemical potential " << epsilonF << endl;
       cout << "# eigen_fraction " << eigen_fraction << endl;
       cout << "# phase_in_delta " << phase_in_delta << endl;
       cout << "# Ax " << Ax << endl;
       cout << "# Ay " << Ay << endl;
       cout << "#";
       if (Gauge == GaugeAx) { 
	  cout << " GaugeAx ";
       } else if (Gauge == GaugeAy) { 
	  cout << " GaugeAy ";
       } else if (Gauge == GaugeAxSym) { 
	  cout << " GaugeAxSym ";
       } else if (Gauge == GaugeAxNotInDelta) { 
	  cout << " GaugeAxNotInDelta ";
       } else if (Gauge == GaugeAyNotInDelta) { 
	  cout << " GaugeAyNotInDelta ";
       }
       cout << "Gauge " << Gauge << endl;
    }
};




#ifdef HELENE_VERSION
int do_Spingros_hello(void)
{
  int i;

# else
// main version 
int  main(int argc, char **argv)
{
  bool p_in_d = false;
  int i;
# endif

  static int nph = 22;
  double Dphi = 1.0/(double)nph;
  VectorXd Svec(nph+1);
  VectorXd dSav(Svec.size());
  VectorXi dSavcount(Svec.size());
  double Ilimit = 20.0;

  dSav.setZero();
  dSavcount.setZero();

  int rcount = atoi(argv[3]);
  int nav = 1;


  static int Nx = 20;
  static int Ny = 100;
  static int Ns = 10;

  static double Ht = 4.5;
  static double Delta = 1.0;
  static double BzS = 1e-6;
  static double BzN = 1e-6;
  static double ByS = 1e-6;
  static double ByN = 1e-6;
  static double BxS = 1e-6;
  static double BxN = 1e-6;
  static double lvz = atof(argv[2]);
  static double lambda = 0.0;
  static double lambdaR = 0.0;
  static double drh = 10.0;  //not used
  static double phi0 = 0.01;
  static double Phi_orb = 0.0;
  static double W = 2.0;
  static double eigen_fraction = 0.99999;
  static double epsilonF = 1.0;
  static double Tcontacts = 1.0;   //not used
  static double Tdn_left = 0.5;
  static double Tup_left = 0.5;
  static double Tdn_right = 0.5;
  static double Tup_right = 0.5;

#ifdef HELENE_VERSION
  static int p_in_d = 0;

  O_p *op = NULL;
  d_s *ds = NULL;
  pltreg *pr = NULL;

  if(updating_menu_state != 0)	return D_O_K;

  if (ac_grep(cur_ac_reg,"%pr",&pr) != 1)
    return win_printf_OK("cannot find data");


  i = win_scanf("Define the size Nx=%4d; Ny=%4d Ns=%4d\n"
		"Ht = %6lf, Delta = %6lf, W = %6lf\n"
		"eigen_fraction = %6lf, lambda = %6lf\n"
		"BzS = %6lf, BzN = %6lf\n"
			"BxS = %6lf, BxN = %6lf\n"
			"ByS = %6lf, ByN = %6lf\n"
		"phase_in_delta %b\n"
		"phi_value %6lf Nb. of points to compute phase %4d\n"
		"drh = %6lf, Tup_left= %6lf, Tdn_left= %6lf, Tup_right= %6lf, Tdn_right= %6lf\n"
		"\\Phi_orb = %6lf \\epsilon_f = %6lf\n"
		,&Nx, &Ny, &Ns, &Ht, &Delta, &W, &eigen_fraction, &lambda, &BzS, &BzN,  &BxS, &BxN,  &ByS, &ByN, 
		&p_in_d, &phi0, &nph, &drh, &Tup_left,&Tdn_left, &Tup_right,&Tdn_right , &Phi_orb, &epsilonF);
  if (i == CANCEL) return 0;

# endif

  double phi_start = atof(argv[1]);
  for (Phi_orb = phi_start; Phi_orb < phi_start + 2.0; Phi_orb += 0.1) { 

    
  Svec.setZero();

  /**
  std::vector< SNSsparseSO > stackHr;
  for (i = 0; i < nph; i++) {
     stackHr.push_back( SNSsparseSO(Nx, Ny));
  }
  **/
#pragma omp parallel for
  for (i = 0; i < Svec.size(); i++) { 

  SNSsparseSO Hr(Nx, Ny);

  Hr.NSupra = Ns;
  Hr.HtS = Ht;
  Hr.HtN = Ht;
  Hr.Delta = Delta;
  Hr.DeltaRand = 0.0 * Delta;
  Hr.W = W;
  Hr.eigen_fraction = eigen_fraction;
  Hr.lambdaR = lambdaR;  // Rashba 
  Hr.lso = lambda;
  Hr.lvz = lvz;
  Hr.BzS = BzS;
  Hr.BzN = BzN;
  Hr.ByS = ByS;
  Hr.ByN = ByN;
  Hr.BxS = BxS;
  Hr.BxN = BxN;
  Hr.TsnUP_Left = Tup_left;
  Hr.TsnDN_Left = Tdn_left;
  Hr.TsnUP_Right = Tup_right;
  Hr.TsnDN_Right = Tdn_right;
  Hr.Gauge = Hr.GaugeAy;
  Hr.zmax = -0.01;
  Hr.Ax = 0.0;
  Hr.Ay = 0.0;
  Hr.first_site = 1;
  Hr.phase_in_delta = (p_in_d) ? true : false;
  Hr.epsilonF = epsilonF;
  if (i == 0) Hr.print_info();

  double Phi0 = phi0;
  double Phi = Phi0;

  Hr.Phi = 1e-3;

  double S = 1.0;

  //  for (Phi_orb = 1e-4; Phi_orb <= 5.0; Phi_orb += 0.1) { 
  Hr.set_Phi_orb(Phi_orb);
     Phi = ((double)i + 0.5)/nph;
     Phi += Phi0;
     Hr.Phi = Phi;
     Hr.set_random_seed(rcount);
     Hr.fill_matrix();
     Hr.diag();
     std::cerr << "# " << i << " done " << std::endl;
     //     Hr.print_eval();

#ifdef HELENE_VERSION
     Hr.print_eval(pr, &op);
     op->need_to_refresh = 1;
     refresh_plot(pr, pr->n_op-1);
#else
     //	Hr.print_eval();
     Svec(i) = Hr.sum_negative_evals();
#endif
	//    Hr.print_wavefunction()
	
  }
     //  }  

  for (i = 0; i < Svec.size(); i++) {
     double dSdPhi = (Svec(i) - Svec( (i-1+nph) % nph ))/Dphi;
     cout << Phi_orb << "    " << (double)i/(double)nph << "     " << dSdPhi << "    " << Svec(i) << "    " << Svec( (i-1+nph) % nph ) << "   " << rcount <<  "%%% " << std::endl;
     if (fabs(dSdPhi) < Ilimit) { 
        dSav(i) += dSdPhi;
	dSavcount(i) += 1.0;
     }
  }
  cout << std::endl;

  }

  //  for (i = 0; i < Svec.size(); i++) {
  //     if (dSavcount(i) > 0) { 
  //        cout << (double)i/(double)nph << "     " << dSav(i) / (double)dSavcount(i) << "   " << dSavcount(i) << " AV%%% " << std::endl;
  //     }
  //  }

  return 0;
}

#ifdef HELENE_VERSION

MENU *Spingros_plot_menu(void)
{
  static MENU mn[32];

  if (mn[0].text != NULL)	return mn;
  add_item_to_menu(mn,"Diagonalization", do_Spingros_hello,NULL,0,NULL);
  return mn;
}

int	SpinGrOs_main(int argc, char **argv)
{
  add_plot_treat_menu_item ( "Spingros", NULL, Spingros_plot_menu(), 0, NULL);
  return D_O_K;
}

int	SpinGrOs_unload(int argc, char **argv)
{
  remove_item_to_menu(plot_treat_menu, "Spingros", NULL, NULL);
  return D_O_K;
}
#endif
 

#endif

