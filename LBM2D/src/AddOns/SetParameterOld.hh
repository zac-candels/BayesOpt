#pragma once
#include <math.h>

#include <iostream>

#include "../Parameters.hh"
#include "AddOnBase.hh"

template <class TParameter, class TParameterOld, int TNumDir = 1>
class SetParameterOld : public AddOnBase {
   public:
    template <class TTraits>
    inline void compute(int k);

    template <class TTraits>
    inline void communicate();
};

template <class TParameter, class TParameterOld, int TNumDir>
template <class TTraits>
inline void SetParameterOld<TParameter, TParameterOld, TNumDir>::compute(int k) {
    using Lattice = typename TTraits::Lattice;
    for (int i=0; i<TNumDir; i++) TParameterOld::template get<Lattice,TNumDir>(k,i) = TParameter::template get<Lattice>(k,i);
}

template <class TParameter, class TParameterOld, int TNumDir>
template <class TTraits>
inline void SetParameterOld<TParameter, TParameterOld,TNumDir>::communicate() {
    using Lattice = typename TTraits::Lattice;
    Lattice::communicate(TParameterOld::template getInstance<Lattice>());
}
