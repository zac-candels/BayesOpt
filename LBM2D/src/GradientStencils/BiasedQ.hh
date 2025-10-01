#pragma once
#include "../Service.hh"
#include "GradientBase.hh"

struct BiasedQ : GradientBase<GradientBiased, AllDirections> {
    template <class TTraits, class TParameter>
    inline double compute(const int direction, const int k);
};

template <class TTraits, class TParameter>
inline double BiasedQ::compute(const int direction, const int k) {
    using Lattice = typename TTraits::Lattice;
    using Stencil = typename TTraits::Stencil;

    using DataType = Data_Base<Lattice, Stencil>;
    
    DataType& data = DataType::getInstance();

    const static auto& param = TParameter::template get<Lattice>();
    const static auto& neighbors = data.getNeighbors();

    return 0.5 *
           (-param[neighbors[neighbors[k * Stencil::Q + direction] * Stencil::Q + direction]] +
            4 * param[neighbors[k * Stencil::Q + direction]] -
            3 * param[k]);
}
