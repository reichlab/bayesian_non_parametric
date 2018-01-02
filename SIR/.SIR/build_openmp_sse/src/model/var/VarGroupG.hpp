





/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_VARGROUPG_HPP
#define LIBBI_VARGROUPG_HPP

#include "bi/cuda/cuda.hpp"
#include "bi/math/scalar.hpp"

/**
 * Variable group: G.
 */
class VarGroupG {
public:
  /**
   * Get name of the variable group.
   *
   * @return Name of the variable group.
   */
  static const char* getName();
      
  /**
   * Size.
   */
  static const int START = 0;
  
  /**
   * Size.
   */
  static const int SIZE = 0;
  
  /**
   * Type.
   */
  static const bi::VarType TYPE = bi::  DX_VAR;
};

inline const char* VarGroupG::getName() {
  return "G";
}

#endif
