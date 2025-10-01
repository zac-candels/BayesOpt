#pragma once
#include "../Service.hh"
#include "GradientBase.hh"

struct CentralXYZWetting : GradientBase<Gradient, Cartesian> {
    template <class TTraits, class TParameter>
    inline double compute(int direction, int k);
};

template <class TTraits, class TParameter>
inline double CentralXYZWetting::compute(int direction, int k) {
    using Lattice = typename TTraits::Lattice;
    using Stencil = typename TTraits::Stencil;

    using DataType = Data_Base<Lattice, Stencil>;

    DataType& data = DataType::getInstance();

    double gradientsum = 0;

    for (int idx = 1; idx < Stencil::Q; idx++) {
        if ((this->isBoundary<Lattice>(data.getNeighbor(k, idx)))) {
            const int& normalq =
                TTraits::Stencil::QMap
                    .find(BoundaryLabels<TTraits::Lattice::NDIM>::template get<typename TTraits::Lattice>(
                              data.getNeighbor(k, idx))
                              .NormalDirection)
                    ->second;

            double factor = 0.5;
            
            if constexpr (TTraits::NumberOfComponents<=2){
                double csolid = TParameter::template get<Lattice>(data.getNeighbor(data.getNeighbor(k, idx), normalq));
                gradientsum += Stencil::Weights[idx] * Stencil::Ci_xyz(direction)[idx] *
                           (csolid - factor * this->mPrefactor * (csolid - pow(csolid, 2)));
            }
            else{
                double csolidsum = 0;
                double csum = 0;
                auto& opcurrent = OrderParameter<TParameter::instance>::template get<Lattice>();
                const double& csolidcurrent = opcurrent[data.getNeighbor(data.getNeighbor(k, idx), normalq)];
                for (int component = 0; component < TTraits::NumberOfComponents-1; component++) {
                    auto& op = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, Lattice>(component);
                    const double& csolid = op[k];//[data.getNeighbor(data.getNeighbor(k, idx), normalq)];
                    const double& c = op[k];
                    csolidsum += csolid;
                    csum += c;
                    gradientsum += Stencil::Weights[idx] * Stencil::Ci_xyz(direction)[idx] *
                           (- factor * this->mvPrefactor[component] * (csolid * opcurrent[k]));
                }
                double csolidN = 1.0 - csolidsum;
                    gradientsum += Stencil::Weights[idx] * Stencil::Ci_xyz(direction)[idx] *
                           (- factor * this->mvPrefactor[TTraits::NumberOfComponents-1] * (csolidN * opcurrent[k]));
                //laplaciansum += Stencil::Weights[idx] * 2 *
                //                ((-factor * this->mvPrefactor[TParameter::instance] * (opcurrent[k]-opcurrent[k]*opcurrent[k])));
                gradientsum += Stencil::Weights[idx] * Stencil::Ci_xyz(direction)[idx] *
                           (csolidcurrent);
            }

        } else {
            gradientsum += Stencil::Weights[idx] * Stencil::Ci_xyz(direction)[idx] *
                           (TParameter::template get<Lattice>(data.getNeighbor(k, idx)));
        }
    }

    return 1.0 / (Stencil::Cs2 * Lattice::DT) * gradientsum;
}
