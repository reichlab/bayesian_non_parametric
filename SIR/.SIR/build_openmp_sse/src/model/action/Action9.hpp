




/**
 * @file
 *
 * Automatically generated by LibBi, do not edit.
 */
#ifndef LIBBI_ACTION9_HPP
#define LIBBI_ACTION9_HPP

#include "ActionCoord9.hpp"

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
 * Action: ode_.
 */
class Action9 {
public:
  
  /**
   * Target type.
   */
  typedef Var5 target_type;

  /**
   * Coordinate type.
   */
  typedef ActionCoord9 coord_type;

  /**
   * Size of the action.
   */
  static const int SIZE = 1;

  /**
   * Is this a matrix action?
   */
  static const bool IS_MATRIX = 0;

  /**
   * Compute time derivative of variable.
   */
  template <class T1, bi::Location L, class CX, class PX, class T2>
  static CUDA_FUNC_BOTH void dfdt(const T1 t,
      const bi::State<ModelSIR,L>& s, const int p,
      const CX& cox, const PX& pax, T2& dfdt);
};

template <class T1, bi::Location L, class CX, class PX, class T2>
inline void Action9::dfdt(const T1 t,
      const bi::State<ModelSIR,L>& s, const int p,
      const CX& cox, const PX& pax, T2& dfdt) {
  
  
  /* constants */
  
  const BOOST_AUTO(N_, BI_REAL(1000));
  
  /* variables */
  
  
  const BOOST_AUTO(S__, pax.template fetch<Var5>(s, p, 0));
  
  const BOOST_AUTO(I__, pax.template fetch<Var6>(s, p, 0));
  /* inlines */
  
  const BOOST_AUTO(i_beta_, BI_REAL(2));
  const CX& cox_ = cox;
  dfdt = (((-i_beta_*S__)*I__)/N_);
}


#endif