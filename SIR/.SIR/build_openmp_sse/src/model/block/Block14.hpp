




/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_BLOCK14_HPP
#define LIBBI_BLOCK14_HPP

#include "../action/Action2.hpp"
#include "../action/Action3.hpp"
#include "../action/Action4.hpp"
#include "../action/Action5.hpp"

#include "bi/typelist/macro_typelist.hpp"
#include "bi/traits/block_traits.hpp"

#include "boost/typeof/typeof.hpp"

/**
 * Type tree for actions.
 */
BEGIN_TYPETREE(Block14ActionTypeList)
BEGIN_NODE(1)
BEGIN_NODE(1)
LEAF_NODE(1, Action2)
JOIN_NODE
LEAF_NODE(1, Action3)
END_NODE
JOIN_NODE
BEGIN_NODE(1)
LEAF_NODE(1, Action4)
JOIN_NODE
LEAF_NODE(1, Action5)
END_NODE
END_NODE

END_TYPETREE()


/**
 * Block: eval_.
 */
class Block14 {
public:
  
  /**
   * Type list for actions.
   */
  typedef GET_TYPETREE(Block14ActionTypeList) action_typelist;


  
  
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
  void Block14::simulates(bi::State<ModelSIR,L>& s)
  
 {
  
  bi::StaticUpdater<ModelSIR,action_typelist>::update(s);
  
  
}


  
  template<bi::Location L>
  void Block14::samples(bi::Random& rng, bi::State<ModelSIR,L>& s)
  
 {
  
  bi::StaticUpdater<ModelSIR,action_typelist>::update(s);
  
  
}


  
  template<bi::Location L, class V1>
  void Block14::logDensities(bi::State<ModelSIR,L>& s, V1 lp)
  
 {
  
  bi::StaticUpdater<ModelSIR,action_typelist>::update(s);
  
  
}


  
  template<bi::Location L, class V1>
  void Block14::maxLogDensities(bi::State<ModelSIR,L>& s, V1 lp)
  
 {
  
  bi::StaticUpdater<ModelSIR,action_typelist>::update(s);
  
  
}


  
  template<class T1, bi::Location L>
  void Block14::simulates(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s)
  
 {
  if (onDelta) {
    
    bi::DynamicUpdater<ModelSIR,action_typelist>::update(t1, t2, s);
    
  }
  
}


  
  template<class T1, bi::Location L>
  void Block14::samples(bi::Random& rng, const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s)
  
 {
  if (onDelta) {
    
    bi::DynamicUpdater<ModelSIR,action_typelist>::update(t1, t2, s);
    
  }
  
}


  
  template<class T1, bi::Location L, class V1>
  void Block14::logDensities(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s, V1 lp)
  
 {
  if (onDelta) {
    
    bi::DynamicUpdater<ModelSIR,action_typelist>::update(t1, t2, s);
    
  }
  
}


  
  template<class T1, bi::Location L, class V1>
  void Block14::maxLogDensities(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s, V1 lp)
  
 {
  if (onDelta) {
    
    bi::DynamicUpdater<ModelSIR,action_typelist>::update(t1, t2, s);
    
  }
  
}


  
  template<bi::Location L>
  void Block14::simulates(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask)
  
 {
  
  bi::SparseStaticUpdater<ModelSIR,action_typelist>::update(s, mask);
  

}


  
  template<bi::Location L>
  void Block14::samples(bi::Random& rng, bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask)
  
 {
  
  bi::SparseStaticUpdater<ModelSIR,action_typelist>::update(s, mask);
  

}


  
  template<bi::Location L, class V1>
  void Block14::logDensities(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask, V1 lp)
  
 {
  
  bi::SparseStaticUpdater<ModelSIR,action_typelist>::update(s, mask);
  

}


  
  template<bi::Location L, class V1>
  void Block14::maxLogDensities(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask, V1 lp)
  
 {
  
  bi::SparseStaticUpdater<ModelSIR,action_typelist>::update(s, mask);
  

}



#endif

