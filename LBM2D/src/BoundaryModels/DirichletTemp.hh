#pragma once
#include <iostream>
#include "../LBModels/ModelBase.hh"
#include "../Parameters.hh"
#include "BoundaryBase.hh"

class DirichletTemp : public BoundaryBase {
   public:
    template <class TTraits, class TDistributionType>
    inline void compute(TDistributionType& mDistribution, int k);

    template <class TTraits>
    inline void communicate(){};

    template <class TTraits, class TDistributionType>
    inline void communicate(TDistributionType& mDistribution);

    inline void setInterfaceVal(double val) { mInterfaceVal = val; };

   private:
    double mInterfaceVal;
};

template <class TTraits, class TDistributionType>
inline void DirichletTemp::compute(TDistributionType& distribution, int k) {
    using Lattice = typename TTraits::Lattice;

    if (!this->apply<Lattice>(k)) return;
    auto model = static_cast<ModelBase<Lattice, TTraits>*>(mModel);

    
    for (int idx = 1; idx < TTraits::Stencil::Q; idx++) {
        if (this->apply<Lattice>(distribution.streamIndex(k, idx))) continue;
        Velocity<>::template get<Lattice,Lattice::NDIM>(k,0) =  Velocity<>::template get<Lattice,Lattice::NDIM>(distribution.streamIndex(k, idx),0);
        if constexpr (Lattice::NDIM >= 2)
            Velocity<>::template get<Lattice,Lattice::NDIM>(k,1) =  Velocity<>::template get<Lattice,Lattice::NDIM>(distribution.streamIndex(k, idx),1);
        Humidity<>::template get<Lattice>(k) = mInterfaceVal;  // Calculate density
        distribution.getDistributionPointer(distribution.streamIndex(k, idx))[idx] =
            -distribution.getPostCollisionDistribution(distribution.streamIndex(k, idx),
                                                       distribution.getOpposite(idx)) +
            2 * model->computeEquilibrium(k, idx);//TTraits::Stencil::Weights[idx] * mInterfaceVal;
    }
}

template <class TTraits, class TDistributionType>
inline void DirichletTemp::communicate(TDistributionType& distribution) {
    using Lattice = typename TTraits::Lattice;
    Lattice::communicateDistributionAll(distribution);
}
