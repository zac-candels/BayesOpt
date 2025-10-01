#pragma once
#include <cmath>

#include "../Service.hh"
#include "GradientBase.hh"

struct LaplacianCentralFourthOrderWetting : GradientBase<Laplacian, One> {
    template <class TTraits, class TParameter>
    inline double compute(int direction, int k);
};

template <class TTraits, class TParameter>
inline double LaplacianCentralFourthOrderWetting::compute(int direction, int k) {
    using Lattice = typename TTraits::Lattice;
    using Stencil = typename TTraits::Stencil;
    constexpr int N = TTraits::NumberOfComponents;
    using DataType = Data_Base<Lattice, Stencil>;

    DataType& data = DataType::getInstance();

    double laplaciansum = 0;
    const static auto& param = TParameter::template get<Lattice>();
    for (int idx = 1; idx < Stencil::Q; idx++) {
        if (!this->isBoundary<Lattice>(data.getNeighbor(k, idx))) {
            if (!this->isBoundary<Lattice>(data.getNeighbor(data.getNeighbor(k, idx), idx))){
                laplaciansum += Stencil::Weights[idx] * 2 *
                            (-TParameter::template get<Lattice>(data.getNeighbor(data.getNeighbor(k, idx), idx)) +
                            16 * TParameter::template get<Lattice>(data.getNeighbor(k, idx)) -
                            15 * TParameter::template get<Lattice>(k));
            }
            else{
                const int& normalq =
                TTraits::Stencil::QMap
                    .find(BoundaryLabels<TTraits::Lattice::NDIM>::template get<typename TTraits::Lattice>(
                              data.getNeighbor(data.getNeighbor(k, idx), idx))
                              .NormalDirection)
                    ->second;
                double factor = 0.5;

                if constexpr (N<=2){
                    double csolid = param[data.getNeighbor(data.getNeighbor(data.getNeighbor(k, idx), idx), normalq)];
                    // Geometry<Lattice>::isCorner(data.getNeighbor(k, idx)) ? 1.0-sqrt(2)/2.0 : 0.5;
                    laplaciansum += Stencil::Weights[idx] * 2 *
                            (-(csolid - factor * this->mPrefactor * (csolid - pow(csolid, 2))) +
                            16 * TParameter::template get<Lattice>(data.getNeighbor(k, idx)) -
                            15 * TParameter::template get<Lattice>(k));
                }
                else{
                    double csolidsum = 0;
                    double csum = 0;
                    auto& opcurrent = OrderParameter<TParameter::instance>::template get<Lattice>();
                    const double& csolidcurrent = opcurrent[data.getNeighbor(data.getNeighbor(data.getNeighbor(k, idx), idx), normalq)];
                    for (int component = 0; component < N-1; component++) {
                        auto& op = getInstance<OrderParameter, N - 1, Lattice>(component);
                        const double& csolid = op[data.getNeighbor(data.getNeighbor(data.getNeighbor(k, idx), idx), normalq)];
                        const double& c = op[k];
                        csolidsum += csolid;
                        csum += c;
                        laplaciansum += Stencil::Weights[idx] * 2 *(((factor) * this->mvPrefactor[component] * (csolid * csolidcurrent)));
                    }
                    double csolidN = 1.0 - csolidsum;
                    laplaciansum += Stencil::Weights[idx] * 2 *(((factor) * this->mvPrefactor[N-1] * (csolidN * csolidcurrent)));
                                
                    laplaciansum += Stencil::Weights[idx] * 2 *
                                    (-csolidcurrent+16*TParameter::template get<Lattice>(data.getNeighbor(k, idx)) - 15 *opcurrent[k]);
                }
            }
            
        } else {
            const int& normalq =
                TTraits::Stencil::QMap
                    .find(BoundaryLabels<TTraits::Lattice::NDIM>::template get<typename TTraits::Lattice>(
                              data.getNeighbor(k, idx))
                              .NormalDirection)
                    ->second;
            double factor = 0.5;
            
            if constexpr (N<=2){
                double csolid = param[data.getNeighbor(data.getNeighbor(k, idx), normalq)];
                //double csolid2 = param[data.getNeighbor(data.getNeighbor(data.getNeighbor(k, idx), normalq), normalq)];
                // Geometry<Lattice>::isCorner(data.getNeighbor(k, idx)) ? 1.0-sqrt(2)/2.0 : 0.5;
                laplaciansum += Stencil::Weights[idx] * 2 *
                            (-(csolid - (1+factor) * this->mPrefactor * (csolid - pow(csolid, 2))) +
                            16 * (csolid - factor * this->mPrefactor * (csolid - pow(csolid, 2))) -
                            15 * TParameter::template get<Lattice>(k));
            }
            else{
                double csolidsum = 0;
                double csolidsum2 = 0;
                double csum = 0;
                auto& opcurrent = OrderParameter<TParameter::instance>::template get<Lattice>();
                const double& csolidcurrent = opcurrent[data.getNeighbor(data.getNeighbor(k, idx), normalq)];
                const double& csolidcurrent2 = opcurrent[data.getNeighbor(data.getNeighbor(data.getNeighbor(data.getNeighbor(k, idx), idx), normalq), normalq)];
                for (int component = 0; component < N-1; component++) {
                    auto& op = getInstance<OrderParameter, N - 1, Lattice>(component);
                    const double& csolid = op[data.getNeighbor(data.getNeighbor(k, idx), normalq)];
                    const double& csolid2 = op[data.getNeighbor(data.getNeighbor(data.getNeighbor(data.getNeighbor(k, idx), idx), normalq), normalq)];
                    const double& c = op[k];
                    csolidsum += csolid;
                    csolidsum2 += csolid2;
                    csum += c;
                    laplaciansum += Stencil::Weights[idx] * 2 *(+((1+factor) * this->mvPrefactor[component] * (csolid2 * csolidcurrent2)) +
                            16 * (-factor * this->mvPrefactor[component] * (csolid * csolidcurrent)));
                }
                double csolidN = 1.0 - csolidsum;
                double csolidN2 = 1.0 - csolidsum2;
                laplaciansum += Stencil::Weights[idx] * 2 *(+((1+factor) * this->mvPrefactor[N-1] * (csolidN2 * csolidcurrent2)) +
                            16 * (-factor * this->mvPrefactor[N-1] * (csolidN * csolidcurrent)));
                            
                laplaciansum += Stencil::Weights[idx] * 2 *
                                (-csolidcurrent2 + 16*(csolidcurrent) - 15 *opcurrent[k]);
            }
        }
    }
    return 1.0 / (12.0 * Stencil::Cs2 * Lattice::DT * Lattice::DT) * laplaciansum;
}
