#pragma once
#include <iostream>

#include "../Geometry.hh"
#include "../LBModels/Humidity.hh"
#include "BoundaryBase.hh"

class PressureOutflow : public BoundaryBase {
   public:
    template <class TTraits, class TDistributionType>
    inline void compute(TDistributionType& mDistribution, int k);

    inline void setNumberIterations(int num) { mNumIterations = num; }

    template <class TTraits>
    inline void communicate(){};

    template <class TTraits, class TDistributionType>
    inline void communicate(TDistributionType& mDistribution);

   private:
    double mInterfaceVal;

    int mNumIterations = 5;
};

template <class TTraits, class TDistributionType>
inline void PressureOutflow::compute(TDistributionType& distribution, int k) {
    static_assert(TDistributionType::SaveEquilibrium,
                  "Cannot use this boundary type unless a data type that saves equilibrium information is used.");

    using Lattice = typename TTraits::Lattice;

    if (!this->apply<Lattice>(k)) return;

    const std::array<int8_t, TTraits::Lattice::NDIM>& normal =
        BoundaryLabels<TTraits::Lattice::NDIM>::template get<typename TTraits::Lattice>(k).NormalDirection;
    const int& normalq =
        TTraits::Stencil::QMap
            .find(BoundaryLabels<TTraits::Lattice::NDIM>::template get<typename TTraits::Lattice>(k).NormalDirection)
            ->second;

    const double& pressure0 = Pressure<>::get<typename TTraits::Lattice>(distribution.streamIndex(k, normalq));
    double pressure = pressure0;

    for (int iter = 0; iter < mNumIterations; iter++) {
        std::vector<double> pressurevec(TTraits::Stencil::Q, 0);
        double prefactorsum = 0;

        for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
            if (TTraits::Stencil::Ci_x[idx] * normal[0] +
                    TTraits::Stencil::Ci_y[idx] * normal[1] * (TTraits::Lattice::NDIM > 1) +
                    TTraits::Stencil::Ci_z[idx] * normal[2] * (TTraits::Lattice::NDIM > 2) >
                0) {
                double gamma = TTraits::Stencil::Weights[idx];
                distribution.getDistributionPointer(distribution.streamIndex(k, normalq))[idx] =
                    distribution.getEquilibriumPointer(distribution.streamIndex(k, normalq))[idx] - pressure0 * gamma;
                prefactorsum += gamma;

            } else if (idx > 0 && Geometry<typename TTraits::Lattice>::isCorner(k) &&
                       TTraits::Stencil::Ci_x[idx] * normal[0] +
                               TTraits::Stencil::Ci_y[idx] * normal[1] * (TTraits::Lattice::NDIM > 1) +
                               TTraits::Stencil::Ci_z[idx] * normal[2] * (TTraits::Lattice::NDIM > 2) ==
                           0) {
                distribution.getDistributionPointer(distribution.streamIndex(k, normalq))[idx] =
                    distribution.getEquilibriumPointer(distribution.streamIndex(k, normalq))[idx];
            }
        }

        auto pressureModel = static_cast<PressureLeeHumidity<typename TTraits::Lattice, TTraits>*>(mModel);
        double ptmp = pressureModel->computePressure(distribution.streamIndex(k, normalq));
        pressure =
            ptmp + prefactorsum * pressure;  //(1 - TTraits::Stencil::Weights[0]) / prefactorsum * (pressure0 - ptmp);
    }

    for (int idx = 1; idx < TTraits::Stencil::Q; idx++) {
        if (TTraits::Stencil::Ci_x[idx] * normal[0] +
                TTraits::Stencil::Ci_y[idx] * normal[1] * (TTraits::Lattice::NDIM > 1) +
                TTraits::Stencil::Ci_z[idx] * normal[2] * (TTraits::Lattice::NDIM > 2) >
            0) {
            double gamma = TTraits::Stencil::Weights[idx];
            distribution.getDistributionPointer(distribution.streamIndex(k, normalq))[idx] =
                distribution.getEquilibriumPointer(distribution.streamIndex(k, normalq))[idx] - pressure0 * gamma +
                pressure * gamma;

        } else if (Geometry<typename TTraits::Lattice>::isCorner(k) &&
                   TTraits::Stencil::Ci_x[idx] * normal[0] +
                           TTraits::Stencil::Ci_y[idx] * normal[1] * (TTraits::Lattice::NDIM > 1) +
                           TTraits::Stencil::Ci_z[idx] * normal[2] * (TTraits::Lattice::NDIM > 2) ==
                       0) {
            distribution.getDistributionPointer(distribution.streamIndex(k, normalq))[idx] =
                distribution.getEquilibriumPointer(
                    distribution.streamIndex(k, normalq))[idx];  // - pressure0 * gamma + pressure * gamma;
        }
    }
}

template <class TTraits, class TDistributionType>
inline void PressureOutflow::communicate(TDistributionType& distribution) {
    using Lattice = typename TTraits::Lattice;
    Lattice::communicateDistributionAll(distribution);
    Lattice::communicateDistributionAllEquilibrium(distribution);
}
