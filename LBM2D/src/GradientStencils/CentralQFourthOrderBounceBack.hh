#pragma once
#include "../Service.hh"
#include "GradientBase.hh"

struct CentralQFourthOrderBounceBack : GradientBase<Gradient, AllDirections> {
    template <class TTraits, class TParameter>
    inline double compute(const int direction, const int k);
};

template <class TTraits, class TParameter>
inline double CentralQFourthOrderBounceBack::compute(const int direction, const int k) {
    using Lattice = typename TTraits::Lattice;
    using Stencil = typename TTraits::Stencil;

    using DataType = Data_Base<Lattice, Stencil>;

    DataType& data = DataType::getInstance();


    if ((this->isBoundary<Lattice>(data.getNeighbors()[k * Stencil::Q + direction]))) {
        if ((this->isBoundary<Lattice>(data.getNeighbors()[k * Stencil::Q + Stencil::Opposites[direction]]))) return 0;
        if ((this->isBoundary<Lattice>(data.getNeighbors()[data.getNeighbors()[k * Stencil::Q + Stencil::Opposites[direction]]* Stencil::Q + Stencil::Opposites[direction]]))) {
            const double& param1 = TParameter::template get<Lattice>(k);
            const double& param2 = TParameter::template get<Lattice>(data.getNeighbor(k, Stencil::Opposites[direction]));
            const double& param1_neighbor = param2;
            const double& param2_neighbor = param2;
            return(-param1_neighbor + 8 * param1 - 8 * param2 + param2_neighbor)/12.0;
        }
        const double& param1 = TParameter::template get<Lattice>(k);
        const double& param2 = TParameter::template get<Lattice>(data.getNeighbor(k, Stencil::Opposites[direction]));
        const double& param1_neighbor = param2;
        const double& param2_neighbor = TParameter::template get<Lattice>(
            data.getNeighbor(data.getNeighbor(k, Stencil::Opposites[direction]), Stencil::Opposites[direction]));
        return(-param1_neighbor + 8 * param1 - 8 * param2 + param2_neighbor)/12.0;

    } else if ((this->isBoundary<Lattice>(data.getNeighbors()[k * Stencil::Q + Stencil::Opposites[direction]]))) {
        const double& param1 = TParameter::template get<Lattice>(data.getNeighbor(k, direction));
        const double& param2 = TParameter::template get<Lattice>(k);
        const double& param1_neighbor =
            TParameter::template get<Lattice>(data.getNeighbor(data.getNeighbor(k, direction), direction));
        const double& param2_neighbor = param1;
        return(-param1_neighbor + 8 * param1 - 8 * param2 + param2_neighbor)/12.0;

    } 
    else if ((this->isBoundary<Lattice>(data.getNeighbors()[data.getNeighbors()[k * Stencil::Q + Stencil::Opposites[direction]]* Stencil::Q + Stencil::Opposites[direction]]))) {
        const double& param1 = TParameter::template get<Lattice>(data.getNeighbor(k, direction));
        const double& param2 = TParameter::template get<Lattice>(k);
        const double& param1_neighbor =
            TParameter::template get<Lattice>(data.getNeighbor(data.getNeighbor(k, direction), direction));
        const double& param2_neighbor = param1;
        return(-param1_neighbor + 8 * param1 - 8 * param2 + param2_neighbor)/12.0;

    } else {
        const double& param1 = TParameter::template get<Lattice>(data.getNeighbor(k, direction));
        const double& param2 = TParameter::template get<Lattice>(data.getNeighbor(k, Stencil::Opposites[direction]));
        const double& param1_neighbor =
            TParameter::template get<Lattice>(data.getNeighbor(data.getNeighbor(k, direction), direction));
        const double& param2_neighbor = TParameter::template get<Lattice>(
            data.getNeighbor(data.getNeighbor(k, Stencil::Opposites[direction]), Stencil::Opposites[direction]));
        return(-param1_neighbor + 8 * param1 - 8 * param2 + param2_neighbor)/12.0;
    }
    
}
