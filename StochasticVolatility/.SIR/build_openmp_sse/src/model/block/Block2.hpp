





/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_BLOCK2_HPP
#define LIBBI_BLOCK2_HPP

#include "Block3.hpp"

#include "bi/typelist/macro_typelist.hpp"
#include "bi/traits/block_traits.hpp"

#include "boost/typeof/typeof.hpp"


/**
 * Type list of sub-blocks.
 */
BEGIN_TYPELIST(Block2BlockTypeList)
SINGLE_TYPE(1, Block3)
END_TYPELIST()

/**
 * Block: transition.
 */
class Block2 {
public:
  /**
   * Type list of sub-blocks.
   */
  typedef GET_TYPETREE(Block2BlockTypeList) block_typelist;

  
  
  template<class T1, bi::Location L>
  static void simulates(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s);
  

  
  
  template<class T1, bi::Location L>
  static void samples(bi::Random& rng, const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s);
  

  
  
  template<class T1, bi::Location L, class V1>
  static void logDensities(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s, V1 lp);
  

  
  
  template<class T1, bi::Location L, class V1>
  static void maxLogDensities(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s, V1 lp);
  

  
  /**
   * Time step.
   */
  static real getDelta();
};


inline real Block2::getDelta() {
  return 1;
}


  
  template<class T1, bi::Location L>
  void Block2::simulates(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s)
  
 {
  Block3::simulates(t1, t2, onDelta, s);
}


  
  template<class T1, bi::Location L>
  void Block2::samples(bi::Random& rng, const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s)
  
 {
  Block3::samples(rng, t1, t2, onDelta, s);
}


  
  template<class T1, bi::Location L, class V1>
  void Block2::logDensities(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s, V1 lp)
  
 {
  Block3::logDensities(t1, t2, onDelta, s);
}


  
  template<class T1, bi::Location L, class V1>
  void Block2::maxLogDensities(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s, V1 lp)
  
 {
  Block3::maxLogDensities(t1, t2, onDelta, s);
}
 


#endif

