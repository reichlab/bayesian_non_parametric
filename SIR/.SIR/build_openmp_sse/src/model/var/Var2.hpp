





/**
 * @file
 *
 * @author Generated by LibBi
 * $Rev$
 * $Date$
 */
#ifndef LIBBI_VAR2_HPP
#define LIBBI_VAR2_HPP

#include "bi/model/Var.hpp"
#include "bi/cuda/cuda.hpp"
#include "bi/math/scalar.hpp"


#include "VarCoord2.hpp"

/**
 * Variable: 2.
 */
class Var2 : public bi::Var {
public:
  /**
   * Coordinate type.
   */
  typedef VarCoord2 coord_type;

  /**
   * Constructor.
   */
  Var2();

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
  static const bi::VarType TYPE = bi::  B_VAR;
};

inline Var2::Var2() : bi::Var(*this) {
  //
}

inline const char* Var2::getName() {
  return "t_next_obs";
}

inline const char* Var2::getInputName() {
  return "t_next_obs";
}

inline const char* Var2::getOutputName() {
  return "t_next_obs";
}

inline bool Var2::hasInput() {
  return BI_REAL(1);
}

inline bool Var2::hasOutput() {
  return BI_REAL(1);
}

inline bool Var2::getOutputOnce() {
  return 0;
}

template<class B>
inline void Var2::initDims(const B& m) {
}

#endif
