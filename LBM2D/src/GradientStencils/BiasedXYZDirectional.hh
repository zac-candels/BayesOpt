#pragma once
#include "../Service.hh"
#include "GradientBase.hh"

struct BiasedXYZDirectional : GradientBase<Gradient, Cartesian> {
    template <class TTraits, class TParameter, int TNdir>
    inline double compute(int direction, int k, int dir);
};

template <class TTraits, class TParameter, int TNdir>
inline double BiasedXYZDirectional::compute(int graddir, int k, int dir) {
    using Lattice = typename TTraits::Lattice;
    using Stencil = typename TTraits::Stencil;

    using DataType = Data_Base<Lattice, Stencil>;

    DataType& data = DataType::getInstance();

    if (this->isBoundary<Lattice>(k)) return 0;

    double gradientsum = 0;

    for (int idx = 1; idx < Stencil::Q; idx++) {
        
            gradientsum += Stencil::Weights[idx] * Stencil::Ci_xyz(graddir)[idx] *
                           (TParameter::template get<Lattice,TNdir>(data.getNeighbor(data.getNeighbor(k, idx), idx), dir)+4*TParameter::template get<Lattice,TNdir>(data.getNeighbor(k, idx), dir)-3*TParameter::template get<Lattice,TNdir>(k, dir));
        
    }
    return 1.0 / (Stencil::Cs2 * Lattice::DT) * gradientsum;
}
