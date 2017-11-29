





/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev: 3738 $
 * $Date: 2013-04-16 23:24:15 +1000 (Tue, 16 Apr 2013) $
 */
#ifndef LIBBI_ACTIONCOORD29_HPP
#define LIBBI_ACTIONCOORD29_HPP

/**
 * Action coordinate: 29.
 *
 * An action coordinate behaves like a variable coordinate, but over only a
 * subrange of the target variable's dimensions. When setting the serial
 * index, a serial index within the subrange is given. When getting the
 * serial index, a serial index within the whole range of the variable is
 * returned.
 */
class ActionCoord29 {
public:
  /**
   * Default constructor.
   */
  CUDA_FUNC_BOTH ActionCoord29();

  /**
   * Construct from serial index.
   *
   * @param ix Serial index within the subrange.
   */
  CUDA_FUNC_BOTH ActionCoord29(const int ix);

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
  CUDA_FUNC_BOTH ActionCoord29& operator++() {
    inc();
    return *this;
  }
  
  /**
   * Postfix increment operator.
   */
  CUDA_FUNC_BOTH ActionCoord29 operator++(int) {
    ActionCoord29 tmp(*this);
    inc();
    return tmp;
  }
  
  /**
   * Prefix decrement operator.
   */
  CUDA_FUNC_BOTH ActionCoord29& operator--() {
    dec();
    return *this;
  }
  
  /**
   * Postfix decrement operator.
   */
  CUDA_FUNC_BOTH ActionCoord29 operator--(int) {
    ActionCoord29 tmp(*this);
    dec();
    return tmp;
  }

  

  

  
};

inline ActionCoord29::ActionCoord29() {
  
}

inline ActionCoord29::ActionCoord29(const int ix) {
  setIndex(ix);
}

inline void ActionCoord29::inc() {
  
  
}

inline void ActionCoord29::dec() {
  
  
}

inline int ActionCoord29::index() const {
  
  return 0;
  
}

inline void ActionCoord29::setIndex(const int ix) {  
}

#endif
