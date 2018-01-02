




/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_BLOCK4_HPP
#define LIBBI_BLOCK4_HPP

#include "Block13.hpp"

#include "bi/typelist/macro_typelist.hpp"
#include "bi/traits/block_traits.hpp"

#include "boost/typeof/typeof.hpp"

/**
 * Type list of sub-blocks.
 */
BEGIN_TYPELIST(Block4BlockTypeList)
SINGLE_TYPE(1, Block13)
END_TYPELIST()

#include "bi/state/Mask.hpp"

/**
 * Block: observation.
 */
class Block4 {
public:
  /**
   * Type list of sub-blocks.
   */
  typedef GET_TYPETREE(Block4BlockTypeList) block_typelist;

  
  
  template<bi::Location L>
  static void simulates(bi::State<ModelSIR,L>& s);
  

  
  
  template<bi::Location L>
  static void samples(bi::Random& rng, bi::State<ModelSIR,L>& s);
  

  
  
  template<bi::Location L, class V1>
  static void logDensities(bi::State<ModelSIR,L>& s, V1 lp);
  

  
  
  template<bi::Location L, class V1>
  static void maxLogDensities(bi::State<ModelSIR,L>& s, V1 lp);
  

  
  
  
  template<bi::Location L>
  static void simulates(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask);
  

  
  
  template<bi::Location L>
  static void samples(bi::Random& rng, bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask);
  

  
  
  template<bi::Location L, class V1>
  static void logDensities(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask, V1 lp);
  

  
  
  template<bi::Location L, class V1>
  static void maxLogDensities(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask, V1 lp);
  

};


  
  template<bi::Location L>
  void Block4::simulates(bi::State<ModelSIR,L>& s)
  
 {
  Block13::simulates(s);
}


  
  template<bi::Location L>
  void Block4::samples(bi::Random& rng, bi::State<ModelSIR,L>& s)
  
 {
  Block13::samples(rng, s);
}


  
  template<bi::Location L, class V1>
  void Block4::logDensities(bi::State<ModelSIR,L>& s, V1 lp)
  
 {
  Block13::logDensities(s, lp);
}


  
  template<bi::Location L, class V1>
  void Block4::maxLogDensities(bi::State<ModelSIR,L>& s, V1 lp)
  
 {
  Block13::maxLogDensities(s, lp);
}


  
  template<bi::Location L>
  void Block4::simulates(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask)
  
 {
  Block13::simulates(s, mask);
}


  
  template<bi::Location L>
  void Block4::samples(bi::Random& rng, bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask)
  
 {
  Block13::samples(rng, s, mask);
}


  
  template<bi::Location L, class V1>
  void Block4::logDensities(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask, V1 lp)
  
 {
  Block13::logDensities(s, mask, lp);
}


  
  template<bi::Location L, class V1>
  void Block4::maxLogDensities(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask, V1 lp)
  
 {
  Block13::maxLogDensities(s, mask, lp);
}


#endif
