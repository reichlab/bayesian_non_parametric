





/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev: 3738 $
 * $Date: 2013-04-16 23:24:15 +1000 (Tue, 16 Apr 2013) $
 */
#ifndef LIBBI_ACTIONCOORD19_HPP
#define LIBBI_ACTIONCOORD19_HPP

/**
 * Action coordinate: 19.
 *
 * An action coordinate behaves like a variable coordinate, but over only a
 * subrange of the target variable's dimensions. When setting the serial
 * index, a serial index within the subrange is given. When getting the
 * serial index, a serial index within the whole range of the variable is
 * returned.
 */
class ActionCoord19 {
public:
  /**
   * Default constructor.
   */
  CUDA_FUNC_BOTH ActionCoord19();

  /**
   * Construct from serial index.
   *
   * @param ix Serial index within the subrange.
   */
  CUDA_FUNC_BOTH ActionCoord19(const int ix);

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
   * @return Serial index within the whole range.
   */
  CUDA_FUNC_BOTH int index() const;
  
  /**
   * Set serial index.
   *
   * @param ix Serial index within the subrange.
   *
   * Sets the coordinate to be equivalent to the given serial index.
   */
  CUDA_FUNC_BOTH void setIndex(const int ix);

  /**
   * Prefix increment operator.
   */
  CUDA_FUNC_BOTH ActionCoord19& operator++() {
    inc();
    return *this;
  }
  
  /**
   * Postfix increment operator.
   */
  CUDA_FUNC_BOTH ActionCoord19 operator++(int) {
    ActionCoord19 tmp(*this);
    inc();
    return tmp;
  }
  
  /**
   * Prefix decrement operator.
   */
  CUDA_FUNC_BOTH ActionCoord19& operator--() {
    dec();
    return *this;
  }
  
  /**
   * Postfix decrement operator.
   */
  CUDA_FUNC_BOTH ActionCoord19 operator--(int) {
    ActionCoord19 tmp(*this);
    dec();
    return tmp;
  }

  

  

  
};

inline ActionCoord19::ActionCoord19() {
  
}

inline ActionCoord19::ActionCoord19(const int ix) {
  setIndex(ix);
}

inline void ActionCoord19::inc() {
  
  
}

inline void ActionCoord19::dec() {
  
  
}

inline int ActionCoord19::index() const {
  
  return 0;
  
}

inline void ActionCoord19::setIndex(const int ix) {  
}

#endif
