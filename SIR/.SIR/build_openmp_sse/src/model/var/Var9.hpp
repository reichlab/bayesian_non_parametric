





/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_VAR9_HPP
#define LIBBI_VAR9_HPP

#include "bi/model/Var.hpp"
#include "bi/cuda/cuda.hpp"
#include "bi/math/scalar.hpp"


#include "VarCoord9.hpp"

/**
 * Variable: 9.
 */
class Var9 : public bi::Var {
public:
  /**
   * Coordinate type.
   */
  typedef VarCoord9 coord_type;

  /**
   * Constructor.
   */
  Var9();

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

inline Var9::Var9() : bi::Var(*this) {
  //
}

inline const char* Var9::getName() {
  return "Incidence";
}

inline const char* Var9::getInputName() {
  return "Incidence";
}

inline const char* Var9::getOutputName() {
  return "Incidence";
}

inline bool Var9::hasInput() {
  return BI_REAL(1);
}

inline bool Var9::hasOutput() {
  return BI_REAL(1);
}

inline bool Var9::getOutputOnce() {
  return 0;
}

template<class B>
inline void Var9::initDims(const B& m) {
}

#endif