#pragma once
#include <cmath>

#include "../Service.hh"
#include "GradientBase.hh"

struct LaplacianCentralWetting : GradientBase<Laplacian, One> {
    template <class TTraits, class TParameter>
    inline double compute(int direction, int k);
};

template <class TTraits, class TParameter>
inline double LaplacianCentralWetting::compute(int direction, int k) {
    using Lattice = typename TTraits::Lattice;
    using Stencil = typename TTraits::Stencil;
    constexpr int N = TTraits::NumberOfComponents;

    using DataType = Data_Base<Lattice, Stencil>;

    DataType& data = DataType::getInstance();


    double laplaciansum = 0;
    const static auto& param = TParameter::template get<Lattice>();

    for (int idx = 1; idx < Stencil::Q; idx++) {
        if (!this->isBoundary<Lattice>(data.getNeighbor(k, idx))) {
            laplaciansum += Stencil::Weights[idx] * 2 * (param[data.getNeighbor(k, idx)] - param[k]);

        } else {
            const int& normalq =
                TTraits::Stencil::QMap
                    .find(BoundaryLabels<TTraits::Lattice::NDIM>::template get<typename TTraits::Lattice>(
                              data.getNeighbor(k, idx))
                              .NormalDirection)
                    ->second;
            double factor = 0.5;
            //double cutoff=1e-1;
            
            if constexpr (N<=2){
                double csolid = param[data.getNeighbor(data.getNeighbor(k, idx), normalq)];
                laplaciansum += Stencil::Weights[idx] * 2 *
                                ((csolid - factor * this->mPrefactor * (csolid - pow(csolid, 2))) - param[k]);
            }/*
            else if constexpr (N==3){
                double csolidsum = 0;
                double csum = 0;
                auto& opcurrent = OrderParameter<TParameter::instance>::template get<Lattice>();
                const double& csolidcurrent = opcurrent[data.getNeighbor(data.getNeighbor(k, idx), normalq)];
                auto& op1 = OrderParameter<0>::template get<Lattice>();
                const double& csolid1 = op1[data.getNeighbor(data.getNeighbor(k, idx), normalq)];
                auto& op2 = OrderParameter<1>::template get<Lattice>();
                const double& csolid2 = op2[data.getNeighbor(data.getNeighbor(k, idx), normalq)];
                const double& csolid3 = 1.0 - csolid1-csolid2;

                double csolidN = 1.0 - csolidsum;
                if(csolid1>csolid3&&csolid2>csolid3&&(TParameter::instance==0)) laplaciansum += Stencil::Weights[idx] * 2 *
                                ((-factor * this->mvPrefactor[1] * (csolid1 * csolid2)/pow(csolid1 + csolid2,2)));
                if(csolid1>csolid3&&csolid2>csolid3&&(TParameter::instance==1)) laplaciansum += Stencil::Weights[idx] * 2 *
                                ((-factor * this->mvPrefactor[0] * (csolid1 * csolid2)/pow(csolid1 + csolid2,2)));
                if(csolid1>csolid2&&csolid3>csolid2&&TParameter::instance==0) laplaciansum += Stencil::Weights[idx] * 2 *
                                ((-factor * this->mvPrefactor[2] * (csolid1 * csolid3)/pow(csolid1 + csolid3,2)));
                if(csolid3>csolid1&&csolid2>csolid1&&TParameter::instance==1) laplaciansum += Stencil::Weights[idx] * 2 *
                                ((-factor * this->mvPrefactor[2] * (csolid2 * csolid3)/pow(csolid2 + csolid3,2)));
                laplaciansum += Stencil::Weights[idx] * 2 *
                                ((csolidcurrent) - opcurrent[k]);
            }*/
            else{
                double csolidsum = 0;
                double csum = 0;
                auto& opcurrent = OrderParameter<TParameter::instance>::template get<Lattice>();
                //const double& csolidcurrent = opcurrent[data.getNeighbor(data.getNeighbor(k, idx), normalq)];
                for (int component = 0; component < N-1; component++) {
                    auto& op = getInstance<OrderParameter, N - 1, Lattice>(component);
                    const double& csolid = op[k];//[data.getNeighbor(data.getNeighbor(k, idx), normalq)];
                    const double& c = op[k];
                    csolidsum += csolid;
                    csum += c;
                    if (csolid>0 && opcurrent[k]>0) laplaciansum += Stencil::Weights[idx] * 2 *
                                ((-factor * this->mvPrefactor[component] * (csolid * opcurrent[k])));//csolidcurrent)));
                }
                double csolidN = 1.0 - csolidsum;
                if (csolidN>0 && opcurrent[k]>0) laplaciansum += Stencil::Weights[idx] * 2 *
                                ((-factor * this->mvPrefactor[N-1] * (csolidN * opcurrent[k])));//csolidcurrent)));
                laplaciansum += Stencil::Weights[idx] * 2 *
                                ((-factor * this->mvPrefactor[TParameter::instance] * (opcurrent[k]-opcurrent[k]*opcurrent[k])));
                //laplaciansum += Stencil::Weights[idx] * 2 *
                //                ((csolidcurrent) - opcurrent[k]);
            }

            
            
        }
    }
    return 1.0 / (Stencil::Cs2 * Lattice::DT * Lattice::DT) * laplaciansum;
}
