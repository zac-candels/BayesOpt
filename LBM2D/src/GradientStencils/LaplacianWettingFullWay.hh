#pragma once
#include <cmath>

#include "../Service.hh"
#include "GradientBase.hh"

struct LaplacianCentralWettingFullWay : GradientBase<Laplacian, One> {
    template <class TTraits, class TParameter>
    inline double compute(int direction, int k);
};

template <class TTraits, class TParameter>
inline double LaplacianCentralWettingFullWay::compute(int direction, int k) {
    using Lattice = typename TTraits::Lattice;
    using Stencil = typename TTraits::Stencil;
    constexpr int N = TTraits::NumberOfComponents;

    using DataType = Data_Base<Lattice, Stencil>;

    DataType& data = DataType::getInstance();

    double laplaciansum = 0;
    const static auto& param = TParameter::template get<Lattice>();

    for (int idx = 1; idx < 5/*Stencil::Q*/; idx++) {
        if (!this->isBoundary<Lattice>(data.getNeighbor(k, idx))) {
            laplaciansum += Stencil::Weights[idx] * 2 * (param[data.getNeighbor(k, idx)] - param[k]);

        } else {
            const int& normalq =
                TTraits::Stencil::QMap
                    .find(BoundaryLabels<TTraits::Lattice::NDIM>::template get<typename TTraits::Lattice>(
                              data.getNeighbor(k, idx))
                              .NormalDirection)
                    ->second;
            double factor = 1.0;
            
            if constexpr (N<=2){
                double csolid = param[data.getNeighbor(data.getNeighbor(data.getNeighbor(k, idx), normalq), normalq)];
                laplaciansum += /*Stencil::Weights[idx]*/1./6. * 2 *
                                ((csolid - factor * this->mPrefactor * (csolid - pow(csolid, 2))) - param[k]);
            }
            else{
                double csolidsum = 0;
                double csum = 0;
                auto& opcurrent = OrderParameter<TParameter::instance>::template get<Lattice>();
                const double& csolidcurrent = opcurrent[data.getNeighbor(data.getNeighbor(data.getNeighbor(k, idx), normalq), normalq)];
                for (int component = 0; component < N-1; component++) {
                    auto& op = getInstance<OrderParameter, N - 1, Lattice>(component);
                    const double& csolid = op[data.getNeighbor(data.getNeighbor(data.getNeighbor(k, idx), normalq), normalq)];
                    const double& c = op[k];
                    csolidsum += csolid;
                    csum += c;
                    laplaciansum += /*Stencil::Weights[idx]*/1./6. * 2 *
                                ((-factor * this->mvPrefactor[component] * (csolid * csolidcurrent)));
                }
                double csolidN = 1.0 - csolidsum;
                laplaciansum += /*Stencil::Weights[idx]*/1./6. * 2 *
                                ((csolidcurrent - factor * this->mvPrefactor[N-1] * (csolidN * csolidcurrent)) - opcurrent[k]);
            }

            
            
        }
    }
    return 1.0 / (Stencil::Cs2 * Lattice::DT * Lattice::DT) * laplaciansum;
}
