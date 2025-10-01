#pragma once
#include <iostream>

#include "../Geometry.hh"
#include "BoundaryBase.hh"

template <class TParam0thMoment>
class Neumann : public BoundaryBase {
   public:
    template <class TTraits, class TDistributionType>
    inline void compute(TDistributionType& mDistribution, int k);

    inline void setNormalGradient(double normalgradient) { mNormalGradient = normalgradient; }

    inline void cutoffNegative(bool cutoffnegative) { mCutoffNegative = cutoffnegative; }

    template <class TTraits>
    inline void communicate(){};

    template <class TTraits, class TDistributionType>
    inline void communicate(TDistributionType& mDistribution);

   private:
    double mNormalGradient = 0;
    bool mCutoffNegative = false;
};

template <class TParam0thMoment>
template <class TTraits, class TDistributionType>
inline void Neumann<TParam0thMoment>::compute(TDistributionType& distribution, int k) {
    using Lattice = typename TTraits::Lattice;

    if (!this->apply<Lattice>(k)) return;

    const std::array<int8_t, TTraits::Lattice::NDIM>& normal =
        BoundaryLabels<TTraits::Lattice::NDIM>::template get<typename TTraits::Lattice>(k).NormalDirection;
    const int& normalq =
        TTraits::Stencil::QMap
            .find(BoundaryLabels<TTraits::Lattice::NDIM>::template get<typename TTraits::Lattice>(k).NormalDirection)
            ->second;

    const double oldval = TParam0thMoment::template get<typename TTraits::Lattice>(k);

    double distsum = 0;
    double weightsum = 0;
    //double prefactorsum = 0;
    if (Geometry<typename TTraits::Lattice>::isCorner(k)){
        for (int qq = 0; qq < TTraits::Stencil::Q; qq++) {
            double cidotnormal0 = TTraits::Stencil::Ci_x[qq] * normal[0] +
                                    TTraits::Stencil::Ci_y[qq] * normal[1] * (TTraits::Lattice::NDIM > 1) +
                                    TTraits::Stencil::Ci_z[qq] * normal[2] * (TTraits::Lattice::NDIM > 2);
            distsum = 0;
            weightsum = 0;
            double magqq = TTraits::Stencil::Ci_x[qq] * TTraits::Stencil::Ci_x[qq] +
                                    TTraits::Stencil::Ci_y[qq] * TTraits::Stencil::Ci_y[qq] * (TTraits::Lattice::NDIM > 1) +
                                    TTraits::Stencil::Ci_z[qq] * TTraits::Stencil::Ci_z[qq] * (TTraits::Lattice::NDIM > 2);
            if (cidotnormal0>0){
                for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
                    double cidotqq = TTraits::Stencil::Ci_x[idx] * TTraits::Stencil::Ci_x[qq] +
                                        TTraits::Stencil::Ci_y[idx] * TTraits::Stencil::Ci_y[qq] * (TTraits::Lattice::NDIM > 1) +
                                        TTraits::Stencil::Ci_z[idx] * TTraits::Stencil::Ci_z[qq] * (TTraits::Lattice::NDIM > 2);
                    double cidotnormal = TTraits::Stencil::Ci_x[idx] * normal[0] +
                                        TTraits::Stencil::Ci_y[idx] * normal[1] * (TTraits::Lattice::NDIM > 1) +
                                        TTraits::Stencil::Ci_z[idx] * normal[2] * (TTraits::Lattice::NDIM > 2);
                    
                    if (cidotqq >= magqq && cidotnormal > 0) {
                        weightsum += TTraits::Stencil::Weights[idx] * (cidotqq);
                    } 
                    else {
                        distsum += distribution.getDistributionPointer(distribution.streamIndex(k, qq))[idx] * (cidotqq);
                    }
                            
                }
                static auto model = static_cast<ModelBase<Lattice, TTraits>*>(mModel);

                TParam0thMoment::template get<typename TTraits::Lattice>(k) = (mNormalGradient - distsum) / (weightsum);
                if (mCutoffNegative) {
                    TParam0thMoment::template get<typename TTraits::Lattice>(k) = std::max(0.0, TParam0thMoment::template get<typename TTraits::Lattice>(k));
                }

                for (int idx = 1; idx < TTraits::Stencil::Q; idx++) {
                    if (TTraits::Stencil::Ci_x[idx] * normal[0] +
                            TTraits::Stencil::Ci_y[idx] * normal[1] * (TTraits::Lattice::NDIM > 1) +
                            TTraits::Stencil::Ci_z[idx] * normal[2] * (TTraits::Lattice::NDIM > 2) >=
                        magqq) {
                        distribution.getDistributionPointer(distribution.streamIndex(k, qq))[idx] =
                            model->computeEquilibrium(k, idx);

                    }
                }

                TParam0thMoment::template get<typename TTraits::Lattice>(k) = oldval;

            }
        }

    }
    else {
        for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
            double cidotnormal = TTraits::Stencil::Ci_x[idx] * normal[0] +
                                TTraits::Stencil::Ci_y[idx] * normal[1] * (TTraits::Lattice::NDIM > 1) +
                                TTraits::Stencil::Ci_z[idx] * normal[2] * (TTraits::Lattice::NDIM > 2);

            
            if (cidotnormal > 0) {
                weightsum += TTraits::Stencil::Weights[idx] * (cidotnormal);
    
            } else {
                distsum += distribution.getDistributionPointer(distribution.streamIndex(k, normalq))[idx] * (cidotnormal);
            }
                      
        }

        static auto model = static_cast<ModelBase<Lattice, TTraits>*>(mModel);

        TParam0thMoment::template get<typename TTraits::Lattice>(k) = (mNormalGradient - distsum) / weightsum;
        if (mCutoffNegative) {
            TParam0thMoment::template get<typename TTraits::Lattice>(k) = std::max(0.0, TParam0thMoment::template get<typename TTraits::Lattice>(k));
        }

        for (int idx = 1; idx < TTraits::Stencil::Q; idx++) {
            if (TTraits::Stencil::Ci_x[idx] * normal[0] +
                    TTraits::Stencil::Ci_y[idx] * normal[1] * (TTraits::Lattice::NDIM > 1) +
                    TTraits::Stencil::Ci_z[idx] * normal[2] * (TTraits::Lattice::NDIM > 2) >
                0) {
                distribution.getDistributionPointer(distribution.streamIndex(k, normalq))[idx] =
                    model->computeEquilibrium(k, idx);

            }
        }

        TParam0thMoment::template get<typename TTraits::Lattice>(k) = oldval;

    }

}

template <class TParam0thMoment>
template <class TTraits, class TDistributionType>
inline void Neumann<TParam0thMoment>::communicate(TDistributionType& distribution) {
    using Lattice = typename TTraits::Lattice;
    Lattice::communicateDistributionAll(distribution);
}
