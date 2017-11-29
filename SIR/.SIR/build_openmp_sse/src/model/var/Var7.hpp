





/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_VAR7_HPP
#define LIBBI_VAR7_HPP

#include "bi/model/Var.hpp"
#include "bi/cuda/cuda.hpp"
#include "bi/math/scalar.hpp"


#include "VarCoord7.hpp"

/**
 * Variable: 7.
 */
class Var7 : public bi::Var {
public:
  /**
   * Coordinate type.
   */
  typedef VarCoord7 coord_type;

  /**
   * Constructor.
   */
  Var7();

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
  static const int ID = 2;

  /**
   * Size.
   */
  static const int START = 2;
  
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
  static const bi::VarType TYPE = bi::  D_VAR;
};

inline Var7::Var7() : bi::Var(*this) {
  //
}

inline const char* Var7::getName() {
  return "R";
}

inline const char* Var7::getInputName() {
  return "R";
}

inline const char* Var7::getOutputName() {
  return "R";
}

inline bool Var7::hasInput() {
  return BI_REAL(1);
}

inline bool Var7::hasOutput() {
  return BI_REAL(1);
}

inline bool Var7::getOutputOnce() {
  return 0;
}

template<class B>
inline void Var7::initDims(const B& m) {
}

#endif