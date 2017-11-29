





/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_VAR13_HPP
#define LIBBI_VAR13_HPP

#include "bi/model/Var.hpp"
#include "bi/cuda/cuda.hpp"
#include "bi/math/scalar.hpp"


#include "VarCoord13.hpp"

/**
 * Variable: 13.
 */
class Var13 : public bi::Var {
public:
  /**
   * Coordinate type.
   */
  typedef VarCoord13 coord_type;

  /**
   * Constructor.
   */
  Var13();

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
  static const int ID = 1;

  /**
   * Size.
   */
  static const int START = 1;
  
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
  static const bi::VarType TYPE = bi::  PX_VAR;
};

inline Var13::Var13() : bi::Var(*this) {
  //
}

inline const char* Var13::getName() {
  return "param_aux__13_";
}

inline const char* Var13::getInputName() {
  return "param_aux__13_";
}

inline const char* Var13::getOutputName() {
  return "param_aux__13_";
}

inline bool Var13::hasInput() {
  return BI_REAL(0);
}

inline bool Var13::hasOutput() {
  return BI_REAL(0);
}

inline bool Var13::getOutputOnce() {
  return 1;
}

template<class B>
inline void Var13::initDims(const B& m) {
}

#endif