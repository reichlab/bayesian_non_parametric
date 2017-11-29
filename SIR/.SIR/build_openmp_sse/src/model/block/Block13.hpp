




/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_BLOCK13_HPP
#define LIBBI_BLOCK13_HPP

#include "../action/Action28.hpp"

#include "bi/typelist/macro_typelist.hpp"
#include "bi/traits/block_traits.hpp"

#include "boost/typeof/typeof.hpp"

/**
 * Type tree for actions.
 */
BEGIN_TYPETREE(Block13ActionTypeList)
LEAF_NODE(1, Action28)

END_TYPETREE()


/**
 * Block: eval_.
 */
class Block13 {
public:
  
  /**
   * Type list for actions.
   */
  typedef GET_TYPETREE(Block13ActionTypeList) action_typelist;


  
  
  template<bi::Location L>
  static void simulates(bi::State<ModelSIR,L>& s);
  

  
  
  template<bi::Location L>
  static void samples(bi::Random& rng, bi::State<ModelSIR,L>& s);
  

  
  
  template<bi::Location L, class V1>
  static void logDensities(bi::State<ModelSIR,L>& s, V1 lp);
  

  
  
  template<bi::Location L, class V1>
  static void maxLogDensities(bi::State<ModelSIR,L>& s, V1 lp);
  


  
  
  template<class T1, bi::Location L>
  static void simulates(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s);
  

  
  
  template<class T1, bi::Location L>
  static void samples(bi::Random& rng, const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s);
  

  
  
  template<class T1, bi::Location L, class V1>
  static void logDensities(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s, V1 lp);
  

  
  
  template<class T1, bi::Location L, class V1>
  static void maxLogDensities(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s, V1 lp);
  


  
  
  template<bi::Location L>
  static void simulates(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask);
  

  
  
  template<bi::Location L>
  static void samples(bi::Random& rng, bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask);
  

  
  
  template<bi::Location L, class V1>
  static void logDensities(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask, V1 lp);
  
  
  
  
  template<bi::Location L, class V1>
  static void maxLogDensities(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask, V1 lp);
  
  
};

#include "bi/updater/DynamicUpdater.hpp"
#include "bi/updater/StaticUpdater.hpp"
#include "bi/updater/SparseStaticUpdater.hpp"


  
  template<bi::Location L>
  void Block13::simulates(bi::State<ModelSIR,L>& s)
  
 {
  
  bi::StaticUpdater<ModelSIR,action_typelist>::update(s);
  
  
}


  
  template<bi::Location L>
  void Block13::samples(bi::Random& rng, bi::State<ModelSIR,L>& s)
  
 {
  
  bi::StaticUpdater<ModelSIR,action_typelist>::update(s);
  
  
}


  
  template<bi::Location L, class V1>
  void Block13::logDensities(bi::State<ModelSIR,L>& s, V1 lp)
  
 {
  
  bi::StaticUpdater<ModelSIR,action_typelist>::update(s);
  
  
}


  
  template<bi::Location L, class V1>
  void Block13::maxLogDensities(bi::State<ModelSIR,L>& s, V1 lp)
  
 {
  
  bi::StaticUpdater<ModelSIR,action_typelist>::update(s);
  
  
}


  
  template<class T1, bi::Location L>
  void Block13::simulates(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s)
  
 {
  if (onDelta) {
    
    bi::DynamicUpdater<ModelSIR,action_typelist>::update(t1, t2, s);
    
  }
  
}


  
  template<class T1, bi::Location L>
  void Block13::samples(bi::Random& rng, const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s)
  
 {
  if (onDelta) {
    
    bi::DynamicUpdater<ModelSIR,action_typelist>::update(t1, t2, s);
    
  }
  
}


  
  template<class T1, bi::Location L, class V1>
  void Block13::logDensities(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s, V1 lp)
  
 {
  if (onDelta) {
    
    bi::DynamicUpdater<ModelSIR,action_typelist>::update(t1, t2, s);
    
  }
  
}


  
  template<class T1, bi::Location L, class V1>
  void Block13::maxLogDensities(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s, V1 lp)
  
 {
  if (onDelta) {
    
    bi::DynamicUpdater<ModelSIR,action_typelist>::update(t1, t2, s);
    
  }
  
}


  
  template<bi::Location L>
  void Block13::simulates(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask)
  
 {
  
  bi::SparseStaticUpdater<ModelSIR,action_typelist>::update(s, mask);
  

}


  
  template<bi::Location L>
  void Block13::samples(bi::Random& rng, bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask)
  
 {
  
  bi::SparseStaticUpdater<ModelSIR,action_typelist>::update(s, mask);
  

}


  
  template<bi::Location L, class V1>
  void Block13::logDensities(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask, V1 lp)
  
 {
  
  bi::SparseStaticUpdater<ModelSIR,action_typelist>::update(s, mask);
  

}


  
  template<bi::Location L, class V1>
  void Block13::maxLogDensities(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask, V1 lp)
  
 {
  
  bi::SparseStaticUpdater<ModelSIR,action_typelist>::update(s, mask);
  

}



#endif
