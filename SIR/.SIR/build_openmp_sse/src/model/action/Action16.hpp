




/**
 * @file
 *
 * Automatically generated by LibBi, do not edit.
 */
#ifndef LIBBI_ACTION16_HPP
#define LIBBI_ACTION16_HPP

#include "ActionCoord16.hpp"

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
 * Action: eval_.
 */
struct Action16 {
  
  /**
   * Target type.
   */
  typedef Var7 target_type;

  /**
   * Coordinate type.
   */
  typedef ActionCoord16 coord_type;

  /**
   * Size of the action.
   */
  static const int SIZE = 1;

  /**
   * Is this a matrix action?
   */
  static const bool IS_MATRIX = 0;
  
  
  
  template <bi::Location L, class CX, class PX, class OX>
  static CUDA_FUNC_BOTH void simulates(bi::State<ModelSIR,L>& s, const int p, const int ix, const CX& cox, const PX& pax, OX& x);
  

  
  
  template <class T1, bi::Location L, class CX, class PX, class OX>
  static CUDA_FUNC_BOTH void simulates(const T1 t1, const T1 t2, bi::State<ModelSIR,L>& s, const int p, const int ix, const CX& cox, const PX& pax, OX& x);
  

};



  
  template <bi::Location L, class CX, class PX, class OX>
  void Action16::simulates(bi::State<ModelSIR,L>& s, const int p, const int ix, const CX& cox, const PX& pax, OX& x)
  
 {
  
  
  
  /* variables */
  
  
  const BOOST_AUTO(R__, pax.template fetch<Var7>(s, p, 0));
  /* inlines */
  
  const CX& cox_ = cox;
  x.template fetch<target_type>(s, p, cox_.index()) = BI_REAL(0);
}


  
  template <class T1, bi::Location L, class CX, class PX, class OX>
  void Action16::simulates(const T1 t1, const T1 t2, bi::State<ModelSIR,L>& s, const int p, const int ix, const CX& cox, const PX& pax, OX& x)
  
 {  
  
  
  
  /* variables */
  
  
  const BOOST_AUTO(R__, pax.template fetch<Var7>(s, p, 0));
  /* inlines */
  
  const CX& cox_ = cox;
  x.template fetch<target_type>(s, p, cox_.index()) = BI_REAL(0);
}


#endif
