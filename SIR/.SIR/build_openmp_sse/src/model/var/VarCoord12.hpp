





/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_VARCOORD12_HPP
#define LIBBI_VARCOORD12_HPP

/**
 * Coordinate: 12.
 */
class VarCoord12 {
public:
  /**
   * Default constructor.
   */
  CUDA_FUNC_BOTH VarCoord12();
  
  

  
  /**
   * Construct from serial index.
   *
   * @param ix Serial index.
   */
  CUDA_FUNC_BOTH VarCoord12(const int ix);
  
  
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
  CUDA_FUNC_BOTH VarCoord12& operator++() {
    inc();
    return *this;
  }
  
  /**
   * Postfix increment operator.
   */
  CUDA_FUNC_BOTH VarCoord12 operator++(int) {
    VarCoord12 tmp(*this);
    inc();
    return tmp;
  }
  
  /**
   * Prefix decrement operator.
   */
  CUDA_FUNC_BOTH VarCoord12& operator--() {
    dec();
    return *this;
  }
  
  /**
   * Postfix decrement operator.
   */
  CUDA_FUNC_BOTH VarCoord12 operator--(int) {
    VarCoord12 tmp(*this);
    dec();
    return tmp;
  }

  
  
  
};

inline VarCoord12::VarCoord12() {
}




inline VarCoord12::VarCoord12(const int ix) {
  setIndex(ix);
}


inline void VarCoord12::inc() {
  
  
}

inline void VarCoord12::dec() {
  
  
}

inline int VarCoord12::index() const {
  
  return 0;
  
}

inline void VarCoord12::setIndex(const int ix) {  
}

#endif
