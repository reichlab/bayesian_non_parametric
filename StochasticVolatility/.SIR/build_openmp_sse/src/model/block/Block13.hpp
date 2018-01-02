




/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_BLOCK13_HPP
#define LIBBI_BLOCK13_HPP

#include "../action/Action6.hpp"

#include "bi/typelist/macro_typelist.hpp"
#include "bi/traits/block_traits.hpp"

#include "boost/typeof/typeof.hpp"

/**
 * Type tree for actions.
 */
BEGIN_TYPETREE(Block13ActionTypeList)
LEAF_NODE(1, Action6)

END_TYPETREE()


/**
 * Block: pdf_.
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

#include "bi/updater/StaticUpdater.hpp"
#include "bi/updater/StaticSampler.hpp"
#include "bi/updater/StaticLogDensity.hpp"
#include "bi/updater/StaticMaxLogDensity.hpp"
#include "bi/updater/SparseStaticSampler.hpp"
#include "bi/updater/SparseStaticLogDensity.hpp"
#include "bi/updater/SparseStaticMaxLogDensity.hpp"


  
  template<bi::Location L>
  void Block13::simulates(bi::State<ModelSIR,L>& s)
  
 {
  bi::StaticUpdater<ModelSIR,action_typelist>::update(s);
}


  
  template<bi::Location L>
  void Block13::samples(bi::Random& rng, bi::State<ModelSIR,L>& s)
  
 {
  bi::StaticSampler<ModelSIR,action_typelist>::samples(rng, s);
}


  
  template<bi::Location L, class V1>
  void Block13::logDensities(bi::State<ModelSIR,L>& s, V1 lp)
  
 {
  bi::StaticLogDensity<ModelSIR,action_typelist>::logDensities(s, lp);
}


  
  template<bi::Location L, class V1>
  void Block13::maxLogDensities(bi::State<ModelSIR,L>& s, V1 lp)
  
 {
  bi::StaticMaxLogDensity<ModelSIR,action_typelist>::maxLogDensities(s, lp);
}


  
  
  template<class T1, bi::Location L>
  void Block13::simulates(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s)
  
 {
    if (onDelta) {
      
      simulates(s);
      
    }
  }


  
  
  template<class T1, bi::Location L>
  void Block13::samples(bi::Random& rng, const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s)
  
 {
    if (onDelta) {
      
      samples(rng, s);
      
    }
  }


  
  
  template<class T1, bi::Location L, class V1>
  void Block13::logDensities(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s, V1 lp)
  
 {
    if (onDelta) {
      
      logDensities(s, lp);
      
    }
  }


  
  
  template<class T1, bi::Location L, class V1>
  void Block13::maxLogDensities(const T1 t1, const T1 t2, const bool onDelta, bi::State<ModelSIR,L>& s, V1 lp)
  
 {
    if (onDelta) {
      
      maxLogDensities(s, lp);
      
    }
  }



  
  
  template<bi::Location L>
  void Block13::simulates(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask)
  
 {
    
    BI_ASSERT(false);
    
  }



  
  template<bi::Location L>
  void Block13::samples(bi::Random& rng, bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask)
  
 {
  bi::SparseStaticSampler<ModelSIR,action_typelist>::samples(rng, mask, s);
}


  
  template<bi::Location L, class V1>
  void Block13::logDensities(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask, V1 lp)
  
 {
  bi::SparseStaticLogDensity<ModelSIR,action_typelist>::logDensities(s, mask, lp);
}


  
  template<bi::Location L, class V1>
  void Block13::maxLogDensities(bi::State<ModelSIR,L>& s, const bi::Mask<L>& mask, V1 lp)
  
 {
  bi::SparseStaticMaxLogDensity<ModelSIR,action_typelist>::maxLogDensities(s, mask, lp);
}


#endif
