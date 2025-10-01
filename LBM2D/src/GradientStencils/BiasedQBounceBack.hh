#pragma once
#include "../Service.hh"
#include "GradientBase.hh"

struct BiasedQBounceBack : GradientBase<GradientBiased, AllDirections> {
    template <class TTraits, class TParameter>
    inline double compute(const int direction, const int k);
};

template <class TTraits, class TParameter>
inline double BiasedQBounceBack::compute(const int direction, const int k) {
    using Lattice = typename TTraits::Lattice;
    using Stencil = typename TTraits::Stencil;

    using DataType = Data_Base<Lattice, Stencil>;

    if (this->isBoundary<Lattice>(k)) return 0;

    DataType& data = DataType::getInstance();

    const static auto& param = TParameter::template get<Lattice>();
    const static auto& neighbors = data.getNeighbors();

    if ((!this->isBoundary<Lattice>(neighbors[k * Stencil::Q + direction])) &&
        (!this->isBoundary<Lattice>(
            neighbors[neighbors[k * Stencil::Q + direction] * Stencil::Q + direction])) &&
        (!this->isBoundary<Lattice>(neighbors[k * Stencil::Q + Stencil::Opposites[direction]]))) {

        return 0.5 *
               (-param[neighbors[neighbors[k * Stencil::Q + direction] * Stencil::Q + direction]] +
                4 * param[neighbors[k * Stencil::Q + direction]] - 3 * param[k]);

    } else if ((this->isBoundary<Lattice>(neighbors[k * Stencil::Q + direction]))) {

        if ((this->isBoundary<Lattice>(neighbors[k * Stencil::Q + Stencil::Opposites[direction]]))) {

            return 0;
        
        }

        return 0.5 * (param[k] - param[neighbors[k * Stencil::Q + Stencil::Opposites[direction]]]);

    } else if ((this->isBoundary<Lattice>(
                   neighbors[neighbors[k * Stencil::Q + direction] * Stencil::Q + direction]))) {

        return 0.5 * (3 * param[neighbors[k * Stencil::Q + direction]] - 3 * param[k]);

    } else

        return 0;
        
}
