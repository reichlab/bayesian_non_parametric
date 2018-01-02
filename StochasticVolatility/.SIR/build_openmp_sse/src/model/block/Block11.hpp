






/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_BLOCK11_HPP
#define LIBBI_BLOCK11_HPP


#include "bi/typelist/macro_typelist.hpp"
#include "bi/traits/block_traits.hpp"

#include "boost/typeof/typeof.hpp"

/**
 * Type list of sub-blocks.
 */
BEGIN_TYPELIST(Block11BlockTypeList)
END_TYPELIST()

/**
 * Block: proposal_parameter.
 */
class Block11 {
public:
  /**
   * Type list of sub-blocks.
   */
  typedef GET_TYPETREE(Block11BlockTypeList) block_typelist;
    
  
  
  template<bi::Location L>
  static void simulates(bi::State<ModelSIR,L>& s);
  

  
  
  template<bi::Location L>
  static void samples(bi::Random& rng, bi::State<ModelSIR,L>& s);
  

  
  
  template<bi::Location L, class V1>
  static void logDensities(bi::State<ModelSIR,L>& s, V1 lp);
  

  
  
  template<bi::Location L, class V1>
  static void maxLogDensities(bi::State<ModelSIR,L>& s, V1 lp);
  

};


  
  template<bi::Location L>
  void Block11::simulates(bi::State<ModelSIR,L>& s)
  
 {
}


  
  template<bi::Location L>
  void Block11::samples(bi::Random& rng, bi::State<ModelSIR,L>& s)
  
 {
}


  
  template<bi::Location L, class V1>
  void Block11::logDensities(bi::State<ModelSIR,L>& s, V1 lp)
  
 {
}


  
  template<bi::Location L, class V1>
  void Block11::maxLogDensities(bi::State<ModelSIR,L>& s, V1 lp)
  
 {
}


#endif

