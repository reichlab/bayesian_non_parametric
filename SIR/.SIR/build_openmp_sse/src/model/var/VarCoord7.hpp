





/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_VARCOORD7_HPP
#define LIBBI_VARCOORD7_HPP

/**
 * Coordinate: 7.
 */
class VarCoord7 {
public:
  /**
   * Default constructor.
   */
  CUDA_FUNC_BOTH VarCoord7();
  
  

  
  /**
   * Construct from serial index.
   *
   * @param ix Serial index.
   */
  CUDA_FUNC_BOTH VarCoord7(const int ix);
  
  
  /**
   * Increment to next coordinate in serial ordering.
   */
  CUDA_FUNC_BOTH void inc();
   
  /**
   * Decrement to previous coordinate in serial ordering.
   */
  CUDA_FUNC_BOTH void dec();

  /**
   * Recover serial index.
   * 
   * @return Serial index for coordinate.
   */
  CUDA_FUNC_BOTH int index() const;
  
  /**
   * Set serial index.
   *
   * @param ix Serial index for coordinate.
   *
   * Sets the coordinate to be equivalent to the given serial index.
   */
  CUDA_FUNC_BOTH void setIndex(const int ix);

  /**
   * Prefix increment operator.
   */
  CUDA_FUNC_BOTH VarCoord7& operator++() {
    inc();
    return *this;
  }
  
  /**
   * Postfix increment operator.
   */
  CUDA_FUNC_BOTH VarCoord7 operator++(int) {
    VarCoord7 tmp(*this);
    inc();
    return tmp;
  }
  
  /**
   * Prefix decrement operator.
   */
  CUDA_FUNC_BOTH VarCoord7& operator--() {
    dec();
    return *this;
  }
  
  /**
   * Postfix decrement operator.
   */
  CUDA_FUNC_BOTH VarCoord7 operator--(int) {
    VarCoord7 tmp(*this);
    dec();
    return tmp;
  }

  
  
  
};

inline VarCoord7::VarCoord7() {
}




inline VarCoord7::VarCoord7(const int ix) {
  setIndex(ix);
}


inline void VarCoord7::inc() {
  
  
}

inline void VarCoord7::dec() {
  
  
}

inline int VarCoord7::index() const {
  
  return 0;
  
}

inline void VarCoord7::setIndex(const int ix) {  
}

#endif