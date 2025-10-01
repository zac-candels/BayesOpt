#pragma once
#include "../Service.hh"
#include "GradientBase.hh"

struct BiasedXYZ : GradientBase<GradientBiased, Cartesian> {
    template <class TTraits, class TParameter>
    inline double compute(const int direction, const int k);
};

template <class TTraits, class TParameter>
inline double BiasedXYZ::compute(const int direction, const int k) {
    using Lattice = typename TTraits::Lattice;
    using Stencil = typename TTraits::Stencil;

    using DataType = Data_Base<Lattice, Stencil>;

    DataType& data = DataType::getInstance();

    double gradientsum = 0;

    const static auto& param = TParameter::template get<Lattice>();
    const static auto& neighbors = data.getNeighbors();

    for (int idx = 1; idx < Stencil::Q; idx++) {

        gradientsum +=
            Stencil::Weights[idx] * Stencil::Ci_xyz(direction)[idx] *
            (-param[neighbors[neighbors[k * Stencil::Q + idx] * Stencil::Q + idx]] +
             4 * param[neighbors[k * Stencil::Q + idx]] -
             3 * param[k]);
             
    }

    return 0.25 / (Stencil::Cs2 * Lattice::DT) * gradientsum;
}
