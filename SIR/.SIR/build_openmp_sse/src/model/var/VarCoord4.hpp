





/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_VARCOORD4_HPP
#define LIBBI_VARCOORD4_HPP

/**
 * Coordinate: 4.
 */
class VarCoord4 {
public:
  /**
   * Default constructor.
   */
  CUDA_FUNC_BOTH VarCoord4();
  
  

  
  /**
   * Construct from serial index.
   *
   * @param ix Serial index.
   */
  CUDA_FUNC_BOTH VarCoord4(const int ix);
  
  
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
  CUDA_FUNC_BOTH VarCoord4& operator++() {
    inc();
    return *this;
  }
  
  /**
   * Postfix increment operator.
   */
  CUDA_FUNC_BOTH VarCoord4 operator++(int) {
    VarCoord4 tmp(*this);
    inc();
    return tmp;
  }
  
  /**
   * Prefix decrement operator.
   */
  CUDA_FUNC_BOTH VarCoord4& operator--() {
    dec();
    return *this;
  }
  
  /**
   * Postfix decrement operator.
   */
  CUDA_FUNC_BOTH VarCoord4 operator--(int) {
    VarCoord4 tmp(*this);
    dec();
    return tmp;
  }

  
  
  
};

inline VarCoord4::VarCoord4() {
}




inline VarCoord4::VarCoord4(const int ix) {
  setIndex(ix);
}


inline void VarCoord4::inc() {
  
  
}

inline void VarCoord4::dec() {
  
  
}

inline int VarCoord4::index() const {
  
  return 0;
  
}

inline void VarCoord4::setIndex(const int ix) {  
}

#endif
