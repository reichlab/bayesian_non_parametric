





/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_VAR6_HPP
#define LIBBI_VAR6_HPP

#include "bi/model/Var.hpp"
#include "bi/cuda/cuda.hpp"
#include "bi/math/scalar.hpp"


#include "VarCoord6.hpp"

/**
 * Variable: 6.
 */
class Var6 : public bi::Var {
public:
  /**
   * Coordinate type.
   */
  typedef VarCoord6 coord_type;

  /**
   * Constructor.
   */
  Var6();

  /**
   * Get name of the variable.
   *
   * @return Name of the variable.
   */
  static const char* getName();

  /**
   * Get the name used for the variable in input files.
   *
   * @return Input name of the variable.
   */
  static const char* getInputName();

  /**
   * Get the name used for the variable in output files.
   *
   * @return Output name of the variable.
   */
  static const char* getOutputName();

  /**
   * Should the variable be included in input files?
   */
  static bool hasInput();

  /**
   * Should the variable be included in output files?
   */
  static bool hasOutput();
  
  /**
   * Should the variable be output only once, not at every time?
   */
  static bool getOutputOnce();

  /**
   * Initialise dimensions. Called by Model::addVar() after construction.
   *
   * @tparam B Model type.
   *
   * @param m Model.
   */
  template<class B>
  void initDims(const B& m);
    
  /**
   * Id.
   */
  static const int ID = 0;

  /**
   * Size.
   */
  static const int START = 0;
  
  /**
   * Size.
   */
  static const int SIZE = 1;

  /**
   * Number of dimensions.
   */
  static const int NUM_DIMS = 0;
  
  
  
  /**
   * Type.
   */
  static const bi::VarType TYPE = bi::  O_VAR;
};

inline Var6::Var6() : bi::Var(*this) {
  //
}

inline const char* Var6::getName() {
  return "y";
}

inline const char* Var6::getInputName() {
  return "y";
}

inline const char* Var6::getOutputName() {
  return "y";
}

inline bool Var6::hasInput() {
  return BI_REAL(1);
}

inline bool Var6::hasOutput() {
  return BI_REAL(1);
}

inline bool Var6::getOutputOnce() {
  return 0;
}

template<class B>
inline void Var6::initDims(const B& m) {
}

#endif
