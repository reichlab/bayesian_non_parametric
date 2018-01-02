




/**
 * @file
 *
 * Automatically generated by LibBi, do not edit.
 */
#ifndef LIBBI_ACTION22_HPP
#define LIBBI_ACTION22_HPP

#include "ActionCoord22.hpp"

#include "bi/state/State.hpp"
#include "bi/state/Mask.hpp"
#include "bi/cuda/cuda.hpp"
#include "bi/math/scalar.hpp"
#include "bi/math/constant.hpp"
#include "bi/math/function.hpp"
#ifdef ENABLE_SSE
#include "bi/sse/math/scalar.hpp"
#endif

class ModelSIR;

/**
 * Action: wiener.
 */
class Action22 {
public:
  
  /**
   * Target type.
   */
  typedef Var4 target_type;

  /**
   * Coordinate type.
   */
  typedef ActionCoord22 coord_type;

  /**
   * Size of the action.
   */
  static const int SIZE = 1;

  /**
   * Is this a matrix action?
   */
  static const bool IS_MATRIX = 0;

  
  
  template <class T1, bi::Location L, class CX, class PX, class OX>
  static CUDA_FUNC_BOTH void simulates(const T1 t1, const T1 t2, bi::State<ModelSIR,L>& s, const int p, const int ix, const CX& cox, const PX& pax, OX& x);
  

  
  
  template <class R1, class T1, bi::Location L, class CX, class PX, class OX>
  static CUDA_FUNC_BOTH void samples(R1& rng, const T1 t1, const T1 t2, bi::State<ModelSIR,L>& s, const int p, const int ix, const CX& cox, const PX& pax, OX& x);
  

  
  
  template <class T1, bi::Location L, class CX, class PX, class OX, class T2>
  static CUDA_FUNC_BOTH void logDensities(const T1 t1, const T1 t2, bi::State<ModelSIR,L>& s, const int p, const int ix, const CX& cox, const PX& pax, OX& x, T2& lp);
  

  
  
  template <class T1, bi::Location L, class CX, class PX, class OX, class T2>
  static CUDA_FUNC_BOTH void maxLogDensities(const T1 t1, const T1 t2, bi::State<ModelSIR,L>& s, const int p, const int ix, const CX& cox, const PX& pax, OX& x, T2& lp);
  

};

#include "bi/math/constant.hpp"


  
  
  template <class T1, bi::Location L, class CX, class PX, class OX>
  void Action22::simulates(const T1 t1, const T1 t2, bi::State<ModelSIR,L>& s, const int p, const int ix, const CX& cox, const PX& pax, OX& x)
  
 {
    
    simulates(s);
    
  }



  
  template <class R1, class T1, bi::Location L, class CX, class PX, class OX>
  void Action22::samples(R1& rng, const T1 t1, const T1 t2, bi::State<ModelSIR,L>& s, const int p, const int ix, const CX& cox, const PX& pax, OX& x)
  
 {
  
  
  
  /* variables */
  
  
  const BOOST_AUTO(n_recovery__, pax.template fetch<Var4>(s, p, 0));
  /* inlines */
  
  const CX& cox_ = cox;

  real mu = 0.0;
  real sigma = bi::sqrt(bi::abs(t2 - t1));
  real u = rng.gaussian(mu, sigma);
    
  x.template fetch<target_type>(s, p, cox_.index()) = u;
}


  
  template <class T1, bi::Location L, class CX, class PX, class OX, class T2>
  void Action22::logDensities(const T1 t1, const T1 t2, bi::State<ModelSIR,L>& s, const int p, const int ix, const CX& cox, const PX& pax, OX& x, T2& lp)
  
 {
  
  
  
  /* variables */
  
  
  const BOOST_AUTO(n_recovery__, pax.template fetch<Var4>(s, p, 0));
  /* inlines */
  
  const CX& cox_ = cox;

  real sigma = bi::sqrt(bi::abs(t2 - t1));  
  real xy = pax.template fetch_alt<target_type>(s, p, cox_.index());

  lp += BI_REAL(-0.5)*bi::pow(xy/sigma, BI_REAL(2.0)) - BI_REAL(BI_HALF_LOG_TWO_PI) - bi::log(sigma);

  x.template fetch<target_type>(s, p, cox_.index()) = xy;
}


  
  template <class T1, bi::Location L, class CX, class PX, class OX, class T2>
  void Action22::maxLogDensities(const T1 t1, const T1 t2, bi::State<ModelSIR,L>& s, const int p, const int ix, const CX& cox, const PX& pax, OX& x, T2& lp)
  
 {
  
  
  
  /* variables */
  
  
  const BOOST_AUTO(n_recovery__, pax.template fetch<Var4>(s, p, 0));
  /* inlines */
  
  const CX& cox_ = cox;

  real sigma = bi::sqrt(bi::abs(t2 - t1));
  real xy = pax.template fetch_alt<target_type>(s, p, cox_.index());

  lp += -BI_REAL(BI_HALF_LOG_TWO_PI) - bi::log(sigma);

  x.template fetch<target_type>(s, p, cox_.index()) = xy;
}


#endif
