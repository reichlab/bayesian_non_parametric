





/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_VARCOORD2_HPP
#define LIBBI_VARCOORD2_HPP

/**
 * Coordinate: 2.
 */
class VarCoord2 {
public:
  /**
   * Default constructor.
   */
  CUDA_FUNC_BOTH VarCoord2();
  
  

  
  /**
   * Construct from serial index.
   *
   * @param ix Serial index.
   */
  CUDA_FUNC_BOTH VarCoord2(const int ix);
  
  
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
  CUDA_FUNC_BOTH VarCoord2& operator++() {
    inc();
    return *this;
  }
  
  /**
   * Postfix increment operator.
   */
  CUDA_FUNC_BOTH VarCoord2 operator++(int) {
    VarCoord2 tmp(*this);
    inc();
    return tmp;
  }
  
  /**
   * Prefix decrement operator.
   */
  CUDA_FUNC_BOTH VarCoord2& operator--() {
    dec();
    return *this;
  }
  
  /**
   * Postfix decrement operator.
   */
  CUDA_FUNC_BOTH VarCoord2 operator--(int) {
    VarCoord2 tmp(*this);
    dec();
    return tmp;
  }

  
  
  
};

inline VarCoord2::VarCoord2() {
}




inline VarCoord2::VarCoord2(const int ix) {
  setIndex(ix);
}


inline void VarCoord2::inc() {
  
  
}

inline void VarCoord2::dec() {
  
  
}

inline int VarCoord2::index() const {
  
  return 0;
  
}

inline void VarCoord2::setIndex(const int ix) {  
}

#endif
