#pragma once
#include "../Service.hh"
#include "GradientBase.hh"

struct CentralQFourthOrder : GradientBase<Gradient, AllDirections> {
    template <class TTraits, class TParameter>
    inline double compute(const int direction, const int k);
};

template <class TTraits, class TParameter>
inline double CentralQFourthOrder::compute(const int direction, const int k) {
    using Lattice = typename TTraits::Lattice;
    using Stencil = typename TTraits::Stencil;

    using DataType = Data_Base<Lattice, Stencil>;

    DataType& data = DataType::getInstance();

    const double& param1 = TParameter::template get<Lattice>(data.getNeighbor(k, direction));
    const double& param2 = TParameter::template get<Lattice>(data.getNeighbor(k, Stencil::Opposites[direction]));
    const double& param1_neighbor =
        TParameter::template get<Lattice>(data.getNeighbor(data.getNeighbor(k, direction), direction));
    const double& param2_neighbor = TParameter::template get<Lattice>(
        data.getNeighbor(data.getNeighbor(k, Stencil::Opposites[direction]), Stencil::Opposites[direction]));

    
    return(-param1_neighbor + 8 * param1 - 8 * param2 + param2_neighbor)/12.0;
}
