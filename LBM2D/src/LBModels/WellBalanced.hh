#pragma once
#include <utility>
#include <iomanip>
#include "../AddOns/AddOns.hh"
#include "../BoundaryModels/Boundaries.hh"
#include "../Collide.hh"
#include "../Data.hh"
#include "../Forces/Forces.hh"
#include "../GradientStencils/GradientStencils.hh"
#include "../Parallel.hh"
#include "../Parameters.hh"
#include "ModelBase.hh"

// WellBalanced.hh: Contains the details of the LBM model to solve the Navier-Stokes and continuity equation. Each
// Model is given a "TTraits" class that contains stencil, data, force and boundary information

template <class TLattice>
using DefaultTraitWellBalanced =
    typename DefaultTrait<TLattice, 2>::template SetBoundary<BounceBack>::template SetProcessor<
        GradientsMultiStencil<Density<>, CentralXYZ, LaplacianCentral>,
        ChemicalPotentialCalculatorRho>::template AddProcessor<std::tuple<Gradients<ChemicalPotential<>, CentralXYZ>>>::
        template SetForce<ChemicalForceRho<WellBalancedForce<>, Gradient>>;

template <class TLattice, class TTraits = DefaultTraitWellBalanced<TLattice>>
class WellBalanced : public CollisionBase<TLattice, typename TTraits::Stencil>,
                     public ModelBase<TLattice, TTraits> {  // Inherit from base class to avoid repetition of common
    // calculations

    using Stencil = typename TTraits::Stencil;
    static constexpr int mNDIM = TLattice::NDIM;

   public:
    inline void collide() override;  // Collision step

    inline void initialise() override;  // Initialisation step

    inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    inline double computeEquilibrium(int k, int idx) override;  // Calculate equilibrium in direction idx

   private:
    static constexpr double mTau = 1.;                 // TEMPORARY relaxation time
    static constexpr double mInverseTau = 1.0 / mTau;  // TEMPORARY inverse relaxation time

    double mComponentDensity = 1.;

    std::vector<double>& density = Density<>::get<TLattice>();                    // Reference to vector of TDensities
    std::vector<double>& velocity = Velocity<>::get<TLattice, TLattice::NDIM>();  // Reference to vector of velocities

    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions
};

template <class TLattice, class TTraits>
inline void WellBalanced<TLattice, TTraits>::collide() {  // Collision step

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        if (this->isCollisionNode(k)) {
            double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

            double equilibriums[TTraits::Stencil::Q];

            for (int idx = 1; idx < TTraits::Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
            }
            equilibriums[0] = computeEquilibrium(k, 0);

            this->collisionQ(equilibriums, old_distribution, mInverseTau, k);
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicateDistribution();
}

template <class TLattice, class TTraits>
inline void WellBalanced<TLattice, TTraits>::initialise() {  // Initialise model
    this->initialiseProcessors();

    ModelBase<TLattice, TTraits>::mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<typename TTraits::Stencil>::template initialise<TLattice>(this->mt_Forces, mTau,
                                                                                               mTau);

#pragma omp parallel for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
        double* old_distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionOldPointer(k);

        Density<>::initialise<TLattice>(1.0, k);  // Set density to 1 initially (This will change)
        Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, x);
        Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, y);
        if constexpr (TLattice::NDIM == 3) Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, z);
        // mInvTau.initialise(0.85,k);

        for (int idx = 1; idx < TTraits::Stencil::Q; idx++) {
            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }

        double equilibrium = computeEquilibrium(k, 0);
        distribution[0] = equilibrium;  // Set distributions to equillibrium
        old_distribution[0] = equilibrium;
    }
}

template <class TLattice, class TTraits>
inline void WellBalanced<TLattice, TTraits>::computeMomenta() {  // Calculate Density<> and Velocity

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
            velocity[k * TTraits::Stencil::D + x] =
                this->computeVelocity(distribution, this->mt_Forces, density[k], x, k);  // Calculate velocities
            velocity[k * TTraits::Stencil::D + y] =
                this->computeVelocity(distribution, this->mt_Forces, density[k], y, k);
            if constexpr (TLattice::NDIM == 3)
                velocity[k * TTraits::Stencil::D + z] =
                    this->computeVelocity(distribution, this->mt_Forces, density[k], z, k);
            density[k] = this->computeDensity(distribution, k);  // Calculate density
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits>
inline double WellBalanced<TLattice, TTraits>::computeEquilibrium(int k, int idx) {
    double velocityFactor = CollisionBase<TLattice, Stencil>::computeVelocityFactor(&velocity[k * mNDIM], idx);
    if (idx == 0) {
        return density[k] * (Stencil::Weights[idx] * velocityFactor + 1);
    } else {
        return density[k] * Stencil::Weights[idx] * velocityFactor;
    }
}

template <class TMethod, template<class, template<class> class> class TChemicalForce = ChemicalForce>
class PressureForceWellBalanced : public TChemicalForce<TMethod,Gradient> {
   public:
    template <class TTraits>
    inline double computeXYZ(const int xyz, const int k) {
        return TChemicalForce<TMethod,Gradient>::template computeXYZ<TTraits>(xyz, k);
    }
    template <class TTraits>
    inline double computeQ(const int idx, const int k) {
        return TChemicalForce<TMethod,Gradient>::template computeQ<TTraits>(idx, k);
    }
    template <class TTraits>
    inline double computeVelocitySource(const int xyz, const int k) {  // Need to correct velocity
        return mChemicalForce.template computeVelocitySource<TTraits>(xyz, k);
    }

   private:

        TChemicalForce<TMethod, Gradient> mChemicalForce;

};

template <class TMethod>
class GradPressureForce : public ForceBase<TMethod> {
   public:
    template <class TTraits>
    inline double computeXYZ(const int xyz, const int k) {
        return -(Gradient<Pressure<>>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k,xyz));//+0.5*((GradientBiased<Pressure<>>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k,xyz))-(Gradient<Pressure<>>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k,xyz)));//-(GradientMixed<Pressure<>>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k,xyz));
    }
    template <class TTraits>
    inline double computeQ(const int idx, const int k) {
        return -(Gradient<Pressure<>>::get<typename TTraits::Lattice, TTraits::Stencil::Q>(k,idx));//+0.5*((GradientBiased<Pressure<>>::get<typename TTraits::Lattice, TTraits::Stencil::Q>(k,idx))-(Gradient<Pressure<>>::get<typename TTraits::Lattice, TTraits::Stencil::Q>(k,idx)));//-(GradientMixed<Pressure<>>::get<typename TTraits::Lattice, TTraits::Stencil::Q>(k,idx));
    }
    template <class TTraits>
    inline double computeVelocitySource(const int xyz, const int k) {  // Need to correct velocity
        return -0.5*(Gradient<Pressure<>>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k,xyz));
    }

};

template <class TMethod>
class GradPressureForce2 : public ForceBase<TMethod> {
   public:
    template <class TTraits>
    inline double computeXYZ(const int xyz, const int k) {
        return +0.5*((GradientBiased<Pressure<>>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k,xyz))-(Gradient<Pressure<>>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k,xyz)));//-(GradientMixed<Pressure<>>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k,xyz));
    }
    template <class TTraits>
    inline double computeQ(const int idx, const int k) {
        return +0.5*((GradientBiased<Pressure<>>::get<typename TTraits::Lattice, TTraits::Stencil::Q>(k,idx))-(Gradient<Pressure<>>::get<typename TTraits::Lattice, TTraits::Stencil::Q>(k,idx)));//-(GradientMixed<Pressure<>>::get<typename TTraits::Lattice, TTraits::Stencil::Q>(k,idx));
    }
    template <class TTraits>
    inline double computeVelocitySource(const int xyz, const int k) {  // Need to correct velocity
        return 0;//-0.5*(Gradient<Pressure<>>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k,xyz));
    }

};

template <class TMethod>
class GradPressureForceEq : public ForceBase<TMethod> {
   public:
    template <class TTraits>
    inline double computeXYZ(const int xyz, const int k) {
        return -(Gradient<Pressure<>>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k,xyz));//-(GradientMixed<Pressure<>>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k,xyz));
    }
    template <class TTraits>
    inline double computeQ(const int idx, const int k) {
        return -(Gradient<Pressure<>>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k,idx));//-(GradientMixed<Pressure<>>::get<typename TTraits::Lattice, TTraits::Stencil::Q>(k,idx));
    }
    template <class TTraits>
    inline double computeVelocitySource(const int xyz, const int k) {  // Need to correct velocity
        return 0;
    }

};

template <class TMethod, template<class, template<class> class> class TChemicalForce = ChemicalForce>
class PressureForceWellBalancedTau : public TChemicalForce<TMethod, Gradient> {
   public:
    template <class TTraits>
    inline double computeXYZ(const int xyz, const int k) {
        return TChemicalForce<TMethod, Gradient>::template computeXYZ<TTraits>(xyz, k);
    }
    template <class TTraits>
    inline double computeQ(const int idx, const int k) {
        return (TChemicalForce<TMethod, Gradient>::template computeQ<TTraits>(idx, k));
    }
    template <class TTraits>
    inline double computeVelocitySource(const int xyz, const int k) {  // Need to correct velocity

        return 0;
    }

   private:
};

template<int N=0>
class NoSource : public ChemicalForceBinaryMu<WellBalancedCHForce<N>, Gradient> { //ForceBase?
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return 0;
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return 0;
    }
};

template<int N>
class WellBalancedCHSource : public ChemicalForceBinaryMu<WellBalancedCHForce<N>, Gradient> {
   public:
    inline void setAij(std::vector<double> mij) {
        mMij = mij;
    }
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        
        double sum = 0;
        double opsum = 0;
        double summij = 0;
        double& op =OrderParameter<N>::template get<typename TTraitsF::Lattice>(k);
        for (int j=0; j<TTraitsF::NumberOfComponents; j++) {
            opsum += getInstance<OrderParameter, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice>(j)[k];
            if (j != N) {
                summij += 1;
                sum += 1 *  getGradientInstance<Gradient, ChemicalPotential, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(
                j)[k * TTraitsF::Lattice::NDIM + xyz];
            }
        }
        
        if (TTraitsF::NumberOfComponents-1 != N) {
            summij += 1;
            sum += 1 * getGradientInstance<Gradient, ChemicalPotential, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>>(
                TTraitsF::NumberOfComponents-1)[k * TTraitsF::Stencil::Q + xyz];
        }
        
        sum += -1 * (summij) * getGradientInstance<Gradient, ChemicalPotential, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(
                N)[k * TTraitsF::Lattice::NDIM + xyz];
        return sum;
        
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        double sum = 0;
        double opsum = 0;
        double summij = 0;
        double& op =OrderParameter<N>::template get<typename TTraitsF::Lattice>(k);
        if(TTraitsF::NumberOfComponents>2) {
            for (int j=0; j<TTraitsF::NumberOfComponents-1; j++) {
                opsum += getInstance<OrderParameter, TTraitsF::NumberOfComponents-1, typename TTraitsF::Lattice>(j)[k];
                if (j != N) {
                    summij +=1;
                    double mij = mMij[j];
                    sum += 1 * mij * getGradientInstance<Gradient, ChemicalPotential, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(
                    j)[k * TTraitsF::Stencil::Q + idx];
                }
            }
            if (TTraitsF::NumberOfComponents-1 != N) {
                summij +=1;
                double mij = 1;
                mij=mMij[TTraitsF::NumberOfComponents-1];
                sum += 1 * mij * getGradientInstance<Gradient, ChemicalPotential, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(
                    TTraitsF::NumberOfComponents-1)[k * TTraitsF::Stencil::Q + idx];
            }
        }
        else{
            summij=1;
        }
        
        sum += mMij[N]*getGradientInstance<Gradient, ChemicalPotential, TTraitsF::NumberOfComponents-(TTraitsF::NumberOfComponents==2), typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(
                N)[k * TTraitsF::Stencil::Q + idx];

        return sum;
    }
    std::vector<double> mMij;
};

template<int N>
class LinNCompCHSource : public ChemicalForceBinaryMu<WellBalancedCHForce<N>, Gradient> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        
        double sum = 0;
        double opsum = 0;
        double summij = 0;
        double& op =OrderParameter<N>::template get<typename TTraitsF::Lattice>(k);
        for (int j=0; j<TTraitsF::NumberOfComponents; j++) {
            opsum += getInstance<OrderParameter, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice>(j)[k];
            if (j != N) {
                summij += (op>0)*(getInstance<OrderParameter, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice>(j)[k]>0)*op*getInstance<OrderParameter, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice>(j)[k];
                sum += 1 * (op>0)*(getInstance<OrderParameter, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice>(j)[k]>0)*op*getInstance<OrderParameter, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice>(j)[k]* getGradientInstance<Gradient, ChemicalPotential, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(
                j)[k * TTraitsF::Lattice::NDIM + xyz];
            }
        }
        
        if (TTraitsF::NumberOfComponents-1 != N) {
            summij += (op>0)*((1-opsum)>0)*op*(1-opsum);
            sum += 1 * (op>0)*((1-opsum)>0)*op*(1-opsum)* getGradientInstance<Gradient, ChemicalPotential, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>>(
                TTraitsF::NumberOfComponents-1)[k * TTraitsF::Stencil::Q + xyz];
        }
        
        sum += -1 * (summij) * getGradientInstance<Gradient, ChemicalPotential, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(
                N)[k * TTraitsF::Lattice::NDIM + xyz];
        return 5*sum;
        
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        double sum = 0;
        double opsum = 0;
        double summij = 0;
        double& op =OrderParameter<N>::template get<typename TTraitsF::Lattice>(k);

        for (int j=0; j<TTraitsF::NumberOfComponents-1; j++) {
            opsum += getInstance<OrderParameter, TTraitsF::NumberOfComponents-1, typename TTraitsF::Lattice>(j)[k];
            if (j != N) {
                summij += op*getInstance<OrderParameter, TTraitsF::NumberOfComponents-1, typename TTraitsF::Lattice>(j)[k]* op*getInstance<OrderParameter, TTraitsF::NumberOfComponents-1, typename TTraitsF::Lattice>(j)[k];
                double mij = op*getInstance<OrderParameter, TTraitsF::NumberOfComponents-1, typename TTraitsF::Lattice>(j)[k]* op*getInstance<OrderParameter, TTraitsF::NumberOfComponents-1, typename TTraitsF::Lattice>(j)[k];
                sum += 1 * mij * getGradientInstance<Gradient, ChemicalPotential, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(
                j)[k * TTraitsF::Stencil::Q + idx];
            }
        }

        if (TTraitsF::NumberOfComponents-1 != N) {
            summij += op*(1-opsum)* op*(1-opsum);
            double mij = op*(1-opsum)* op*(1-opsum);
            sum += 1 * mij * getGradientInstance<Gradient, ChemicalPotential, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(
                TTraitsF::NumberOfComponents-1)[k * TTraitsF::Stencil::Q + idx];
        }
        
        sum += - 1* (summij) * getGradientInstance<Gradient, ChemicalPotential, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(
                N)[k * TTraitsF::Stencil::Q + idx];

        return 5*sum;
    }
};

class Lin2CompCHSource : public ChemicalForceBinaryMu<WellBalancedCHForce<0>, Gradient> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return 0;//
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {

        double& op =OrderParameter<0>::template get<typename TTraitsF::Lattice>(k);

        return -op*(1-op)* op*(1-op)*getGradientInstance<Gradient, ChemicalPotential2, TTraitsF::NumberOfComponents, typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(
                0)[k * TTraitsF::Stencil::Q + idx];
    }
};

template<class TLattice>
using DefaultTraitPressureWellBalanced = typename DefaultTrait<TLattice,2> :: template SetBoundary<BounceBack>
                                                               :: template SetProcessor<GradientsMultiStencil<Density<>,CentralXYZBounceBack,CentralQBounceBack>,GradientsMultiStencil<Pressure<>,CentralQBounceBack,CentralXYZBounceBack>,GradientsDirectional<Velocity<>, CentralXYZBounceBackDirectional>>
                                                               :: template SetForce< PressureForceWellBalanced<WellBalancedForce<GuoPrefactor>,ChemicalForceBinaryMu> >;

template<class TLattice>
using DefaultTraitPressureWellBalanced2 = typename DefaultTrait<TLattice,2> :: template SetBoundary<BounceBack>
                                                               :: template SetProcessor<GradientsMultiStencil<Density<>,CentralXYZBounceBack,CentralQBounceBack>,GradientsMultiStencil<Pressure<>,CentralQBounceBack,CentralXYZBounceBack>,GradientsDirectional<Velocity<>, CentralXYZBounceBackDirectional>>
                                                               :: template SetForce< PressureForceWellBalanced<WellBalancedForce<NoTauDependence>,ChemicalForceBinary>, PressureForceWellBalancedTau<SimpleForcingQ<GuoPrefactor>,ChemicalForceBinary>, GradPressureForce<SimpleForcingQ<GuoPrefactor>> >;


template<int Nmax,class TLattice>
using DefaultTraitPressureWellBalancedN = typename DefaultTrait<TLattice,Nmax> :: template SetBoundary<BounceBack>
                                                                :: template SetProcessor<GradientsMultiStencil<Density<>,CentralXYZBounceBack>,GradientsMultiStencil<Pressure<>,CentralQBounceBack,BiasedQBounceBack,CentralXYZBounceBack>,GradientsDirectional<Velocity<>, CentralXYZBounceBackDirectional>>
                                                               :: template SetForce< PressureForceWellBalanced<WellBalancedForce<NoTauDependence>,ChemicalForce>, PressureForceWellBalancedTau<SimpleForcingQ<SplitGuoPrefactor>,ChemicalForce> >;//, PressureForceWellBalancedTau<SimpleForcingQ<SplitGuoPrefactor>,ChemicalForce>


template<int Nmax,class TLattice>
using _DefaultTraitPressureWellBalancedN = typename DefaultTrait<TLattice,Nmax> :: template SetBoundary<BounceBack>
                                                                :: template SetProcessor<GradientsMultiStencil<Density<>,CentralXYZBounceBack>,GradientsMultiStencil<Pressure<>,CentralQBounceBack,BiasedQBounceBack,CentralXYZBounceBack>,GradientsDirectional<Velocity<>, CentralXYZBounceBackDirectional>>
                                                               :: template SetForce< PressureForceWellBalanced<WellBalancedForce<GuoPrefactor>,ChemicalForceMu> >;//, PressureForceWellBalancedTau<SimpleForcingQ<SplitGuoPrefactor>,ChemicalForce>


template<int Nmax,class TLattice>
using DefaultTraitPressureWellBalancedN2 = typename DefaultTrait<TLattice,Nmax> :: template SetBoundary<BounceBack>
                                                                :: template SetProcessor<GradientsMultiStencil<Density<>,CentralXYZBounceBack>,GradientsMultiStencil<Pressure<>,CentralQBounceBack,/*BiasedQ,*/CentralXYZBounceBack>,GradientsDirectional<Velocity<>, CentralXYZBounceBackDirectional>>
                                                               :: template SetForce< PressureForceWellBalanced<WellBalancedForce<GuoPrefactor>,ChemicalForce> , GradPressureForce<GuoCorrect<GuoPrefactor>> >;//, PressureForceWellBalancedTau<SimpleForcingQ<SplitGuoPrefactor>,ChemicalForce>


template<class TLattice>
using DefaultTraitPressureQuaWellBalancedN = typename DefaultTrait<TLattice,4> :: template SetBoundary<BounceBack>
                                                               :: template SetForce< PressureForceWellBalanced<WellBalancedForce<NoTauDependence>,ChemicalForce>, PressureForceWellBalancedTau<SimpleForcingQ<SplitGuoPrefactor>,ChemicalForce> >;

template<int N, int Nmax, class TLattice>
using DefaultTraitWellBalancedCH = typename DefaultTrait<TLattice,Nmax> :: template SetBoundary<BounceBack>
/* Change this so only done for N=0 except for order param*/   :: template SetProcessor<GradientsMultiStencil<OrderParameter<N>,CentralXYZBounceBack,LaplacianCentralWetting/*,BiasedQBounceBack,BiasedXYZBounceBack*/,CentralQBounceBack>>
                                                               :: template SetForce< NoSource<N> >;


template<int Nmax,class TLattice>
using DefaultTraitWellBalancedCH0 = typename DefaultTraitWellBalancedCH<0,Nmax,TLattice>::template AddProcessorIdx<0, ChemicalPotentialCalculatorNComp>;


template<int N>
struct WellBalancedEQ : ForcingBase<One> {

    inline void setAij(std::vector<double> mij) {
        mMij = mij;
    }

    using Prefactor = NoTauDependence;

    double m_CPSum=0;

    inline void reset() {
        m_CPSum=0;
    }

    template <class TTraits>
    inline void precompute(int k) {
        constexpr int Num = TTraits::NumberOfComponents;
        if (Num<=2) {
            m_CPSum=(mMij[N])*ChemicalPotential<N>::template get<typename TTraits::Lattice>(k);
            return;
        }
        using Lattice = typename TTraits::Lattice;
        m_CPSum=0;
        for(int i=0; i<TTraits::NumberOfComponents; i++)
            m_CPSum += mMij[i]*getInstance<ChemicalPotential, Num, Lattice>(i)[k];
    }

    template <class TTraits>
    inline double compute(int idx, int k) {  // Guo forcing

        return -m_CPSum;
    }

    std::vector<double> mMij;
};

template<int N, int Nmax, class TLattice, class TTraits = DefaultTraitWellBalancedCH<N, Nmax, TLattice>>
class WellBalancedCH : public CollisionBase<TLattice,typename TTraits::Stencil>, public ModelBase<TLattice, TTraits> { //Inherit from base class to avoid repetition of common
                                                         //calculations

    using Stencil = typename TTraits::Stencil;  
    static constexpr int mNDIM = TLattice::NDIM; 

    public:
        std::vector<double>& density; //Reference to vector of TDensities
        std::vector<double>& orderparameter;
        std::vector<double>& velocity; //Reference to vector of velocities
        std::vector<double>& itau; 

        WellBalancedCH() : CollisionBase<TLattice,typename TTraits::Stencil>(), ModelBase<TLattice,TTraits>(),
                           density(Density<>::get<TLattice>()), orderparameter(OrderParameter<N>::template get<TLattice>()),
                           velocity(Velocity<>::get<TLattice,TLattice::NDIM>()), itau(InverseTau<>::get<TLattice>()) {}

        WellBalancedCH(WellBalancedCH<N,Nmax,TLattice, TTraits>& other) : CollisionBase<TLattice,typename TTraits::Stencil>(), ModelBase<TLattice,TTraits>(other),
                    density(Density<>::get<TLattice>()), orderparameter(OrderParameter<N>::template get<TLattice>()),
                           velocity(Velocity<>::get<TLattice,TLattice::NDIM>()), itau(InverseTau<>::get<TLattice>()) {}

        WellBalancedCH<N,Nmax,TLattice, TTraits>& operator=(const WellBalancedCH<N,Nmax,TLattice, TTraits>& other) {
            density = Density<>::get<TLattice>();
            orderparameter = OrderParameter<N>::template get<TLattice>();
            velocity = Velocity<>::get<TLattice,TLattice::NDIM>();
            itau = InverseTau<>::get<TLattice>();
            ModelBase<TLattice,TTraits>::operator=(other);
            return *this;
        }

        inline void collide() override; //Collision step

        inline void initialise() override; //Initialisation step

        inline void computeMomenta() override; //Momenta (density, velocity) calculation

        inline double computeEquilibrium(int k, int idx) override; //Calculate equilibrium in direction idx

        void setMobility(double m) { mobility = m; gamma = mobility/TTraits::Stencil::Cs2/(mTau-0.5); }

        inline void setAij(std::vector<double> mij) {
            #pragma omp parallel
            {
            mWeightedMuSum.setAij(mij);
            }
        }

        void setTau(double tau) { //Set relaxation time
            mTau = tau;
            mInverseTau = 1.0 / mTau;}

    private:

        double mTau = 1.0; //TEMPORARY relaxation time
        double mInverseTau = 1.0 / mTau; //TEMPORARY inverse relaxation time

        double mComponentDensity=1.;

        enum{ x = 0, y = 1, z = 2 }; //Indices corresponding to x, y, z directions

        double mobility = 1;
        double gamma = 1;

        thread_local static WellBalancedEQ<N> mWeightedMuSum;
        
};

template<int N, int Nmax, class TLattice, class TTraits>
thread_local WellBalancedEQ<N> WellBalancedCH<N, Nmax, TLattice, TTraits>::mWeightedMuSum;

template<int N, int Nmax, class TLattice, class TTraits>
inline void WellBalancedCH<N, Nmax, TLattice, TTraits>::collide() { //Collision step

    #pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) { //loop over k

        if(this->isCollisionNode(k)){

            double* old_distribution = this -> mDistribution.getDistributionOldPointer(k);

            double equilibriums[TTraits::Stencil::Q];
            mWeightedMuSum.reset();
            mWeightedMuSum.template precompute<TTraits>(k);
            double sum = 0;
            for (int idx = 1; idx < TTraits::Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
                sum += equilibriums[idx];
            }
            equilibriums[0] = computeEquilibrium(k, 0);
            
            this -> collisionQ(equilibriums, old_distribution, mInverseTau,k); // CHANGE NEEDED If no forces, don't require them to be passed

        }
        
    }

    ModelBase<TLattice, TTraits>::mData.communicateDistribution();

}

template<int N, int Nmax, class TLattice, class TTraits>
inline void WellBalancedCH<N, Nmax, TLattice, TTraits>::initialise() { //Initialise model
    this->initialiseProcessors();
    ModelBase<TLattice, TTraits>::mData.generateNeighbors(); //Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<typename TTraits::Stencil>::template initialise<TLattice>(this -> mt_Forces,mTau,mTau);

    #pragma omp parallel for schedule(guided)
    for (int k = TLattice::HaloSize; k <TLattice::N - TLattice::HaloSize; k++) { //loop over k

        double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
        double* old_distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionOldPointer(k);
        ChemicalPotential<N>::template initialise<TLattice>(0,k);
        ChemicalPotential<TTraits::NumberOfComponents-1>::template initialise<TLattice>(0,k);
        OrderParameter<N>::template initialise<TLattice>(1.0,k);
        mWeightedMuSum.reset();
        mWeightedMuSum.template precompute<TTraits>(k);
        for (int idx = 1; idx <TTraits::Stencil::Q; idx++) {

            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium; //Set distributions to equillibrium
            old_distribution[idx] = equilibrium;        

        }

        double equilibrium = computeEquilibrium(k, 0);
        distribution[0] = equilibrium; //Set distributions to equillibrium
        old_distribution[0] = equilibrium;    
    }
    
    ModelBase<TLattice, TTraits>::mData.communicate(BoundaryLabels<TTraits::Lattice::NDIM>::template getInstance<TLattice>());
    this -> mData.communicate(ChemicalPotential<N>::template getInstance<TLattice>());
    this -> mData.communicate(ChemicalPotential<TTraits::NumberOfComponents>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<N>::template getInstance<TLattice>());

    //this->initialiseBoundaries();
}


template<int N, int Nmax, class TLattice, class TTraits>
inline void WellBalancedCH<N, Nmax, TLattice, TTraits>::computeMomenta() { //Calculate Density<> and Velocity

    #pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k <TLattice::N - TLattice::HaloSize; k++) { //Loop over k

        if(this->isCollisionNode(k)){

            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
            orderparameter[k] = this -> computeDensity(distribution, k);
            //if (fabs(orderparameter[k]) < 1e-14) orderparameter[k] = 0;
        }

    }

    this -> mData.communicate(ChemicalPotential<N>::template getInstance<TLattice>());
    this -> mData.communicate(ChemicalPotential<TTraits::NumberOfComponents>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<N>::template getInstance<TLattice>());

}


template<int N, int Nmax, class TLattice, class TTraits>
inline double WellBalancedCH<N, Nmax, TLattice, TTraits>::computeEquilibrium(int k, int idx) {
    double velocityFactor = 0;//CollisionBase<TLattice, Stencil>::computeVelocityFactor(&(this->velocity[k * mNDIM]), idx);
    if (idx == 0) {
        return ((orderparameter[k] - (1 - Stencil::Weights[0]) * (gamma * (mWeightedMuSum.template compute<TTraits>(idx,k)))))+Stencil::Weights[idx] *orderparameter[k]*velocityFactor;
    } else {
        return Stencil::Weights[idx] * gamma * (mWeightedMuSum.template compute<TTraits>(idx,k))+Stencil::Weights[idx] *orderparameter[k]*velocityFactor;
    }

}


template<int N, int Nmax, class TLattice, class TTraits = DefaultTraitWellBalancedCH<N, Nmax, TLattice>>
class WellBalancedCHFrozen : public CollisionBase<TLattice,typename TTraits::Stencil>, public ModelBase<TLattice, TTraits> { //Inherit from base class to avoid repetition of common
                                                         //calculations

    using Stencil = typename TTraits::Stencil;  
    static constexpr int mNDIM = TLattice::NDIM; 

    public:
        std::vector<double>& density; //Reference to vector of TDensities
        std::vector<double>& orderparameter;
        std::vector<double>& velocity; //Reference to vector of velocities
        std::vector<double>& itau; 

        WellBalancedCHFrozen() : CollisionBase<TLattice,typename TTraits::Stencil>(), ModelBase<TLattice,TTraits>(),
                           density(Density<>::get<TLattice>()), orderparameter(OrderParameter<N>::template get<TLattice>()),
                           velocity(Velocity<>::get<TLattice,TLattice::NDIM>()), itau(InverseTau<>::get<TLattice>()) {}

        WellBalancedCHFrozen(WellBalancedCHFrozen<N,Nmax,TLattice, TTraits>& other) : CollisionBase<TLattice,typename TTraits::Stencil>(), ModelBase<TLattice,TTraits>(other),
                    density(Density<>::get<TLattice>()), orderparameter(OrderParameter<N>::template get<TLattice>()),
                           velocity(Velocity<>::get<TLattice,TLattice::NDIM>()), itau(InverseTau<>::get<TLattice>()) {}

        WellBalancedCHFrozen<N,Nmax,TLattice, TTraits>& operator=(const WellBalancedCHFrozen<N,Nmax,TLattice, TTraits>& other) {
            density = Density<>::get<TLattice>();
            orderparameter = OrderParameter<N>::template get<TLattice>();
            velocity = Velocity<>::get<TLattice,TLattice::NDIM>();
            itau = InverseTau<>::get<TLattice>();
            ModelBase<TLattice,TTraits>::operator=(other);
            return *this;
        }

        inline void collide() override; //Collision step

        inline void initialise() override; //Initialisation step

        inline void computeMomenta() override; //Momenta (density, velocity) calculation

        inline double computeEquilibrium(int k, int idx) override; //Calculate equilibrium in direction idx

        void setMobility(double m) { mobility = m; gamma = mobility/TTraits::Stencil::Cs2/(mTau-0.5); }

        inline void setAij(std::vector<double> mij) {
            #pragma omp parallel
            {
            mWeightedMuSum.setAij(mij);
            }
        }

    private:

        static constexpr double mTau = 1.0; //TEMPORARY relaxation time
        static constexpr double mInverseTau = 1.0 / mTau; //TEMPORARY inverse relaxation time

        double mComponentDensity=1.;

        enum{ x = 0, y = 1, z = 2 }; //Indices corresponding to x, y, z directions

        double mobility = 1;
        double gamma = 1;

        thread_local static WellBalancedEQ<N> mWeightedMuSum;
        
};

template<int N, int Nmax, class TLattice, class TTraits>
thread_local WellBalancedEQ<N> WellBalancedCHFrozen<N, Nmax, TLattice, TTraits>::mWeightedMuSum;

template<int N, int Nmax, class TLattice, class TTraits>
inline void WellBalancedCHFrozen<N, Nmax, TLattice, TTraits>::collide() { //Collision step

    #pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) { //loop over k

        if(this->isCollisionNode(k)){

            double* old_distribution = this -> mDistribution.getDistributionOldPointer(k);

            double equilibriums[TTraits::Stencil::Q];
            mWeightedMuSum.reset();
            mWeightedMuSum.template precompute<TTraits>(k);
            double sum = 0;
            for (int idx = 1; idx < TTraits::Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
                sum += equilibriums[idx];
            }
            equilibriums[0] = computeEquilibrium(k, 0);
            
            this -> collisionQ(equilibriums, old_distribution, mInverseTau,k); // CHANGE NEEDED If no forces, don't require them to be passed

        }
        
    }

    ModelBase<TLattice, TTraits>::mData.communicateDistribution();

}

template<int N, int Nmax, class TLattice, class TTraits>
inline void WellBalancedCHFrozen<N, Nmax, TLattice, TTraits>::initialise() { //Initialise model
    this->initialiseProcessors();
    ModelBase<TLattice, TTraits>::mData.generateNeighbors(); //Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<typename TTraits::Stencil>::template initialise<TLattice>(this -> mt_Forces,mTau,mTau);
    
    #pragma omp parallel for schedule(guided)
    for (int k = TLattice::HaloSize; k <TLattice::N - TLattice::HaloSize; k++) { //loop over k

        double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
        double* old_distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionOldPointer(k);
        ChemicalPotential<N>::template initialise<TLattice>(0,k);
        ChemicalPotential<TTraits::NumberOfComponents-1>::template initialise<TLattice>(0,k);
        OrderParameter<N>::template initialise<TLattice>(1.0,k);
        mWeightedMuSum.reset();
        mWeightedMuSum.template precompute<TTraits>(k);
        for (int idx = 1; idx <TTraits::Stencil::Q; idx++) {

            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium; //Set distributions to equillibrium
            old_distribution[idx] = equilibrium;        

        }

        double equilibrium = computeEquilibrium(k, 0);
        distribution[0] = equilibrium; //Set distributions to equillibrium
        old_distribution[0] = equilibrium;    
    }
    
    ModelBase<TLattice, TTraits>::mData.communicate(BoundaryLabels<TTraits::Lattice::NDIM>::template getInstance<TLattice>());
    this -> mData.communicate(ChemicalPotential<N>::template getInstance<TLattice>());
    this -> mData.communicate(ChemicalPotential<TTraits::NumberOfComponents>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<N>::template getInstance<TLattice>());

    //this->initialiseBoundaries();
}


template<int N, int Nmax, class TLattice, class TTraits>
inline void WellBalancedCHFrozen<N, Nmax, TLattice, TTraits>::computeMomenta() { //Calculate Density<> and Velocity

    #pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k <TLattice::N - TLattice::HaloSize; k++) { //Loop over k

        if(this->isCollisionNode(k)){

            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
            orderparameter[k] = this -> computeDensity(distribution, k);
            if (fabs(orderparameter[k]) < 1e-14) orderparameter[k] = 0;
        }

    }

    this -> mData.communicate(ChemicalPotential<N>::template getInstance<TLattice>());
    this -> mData.communicate(ChemicalPotential<TTraits::NumberOfComponents>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<N>::template getInstance<TLattice>());

}


template<int N, int Nmax, class TLattice, class TTraits>
inline double WellBalancedCHFrozen<N, Nmax, TLattice, TTraits>::computeEquilibrium(int k, int idx) {
    double velocityFactor = CollisionBase<TLattice, Stencil>::computeVelocityFactor(&(this->velocity[k * mNDIM]), idx);
    if (idx == 0) {
        return ((orderparameter[k] - (1 - Stencil::Weights[0]) * (gamma * (mWeightedMuSum.template compute<TTraits>(idx,k)))))+Stencil::Weights[idx] *orderparameter[k]*velocityFactor;
    } else {
        return Stencil::Weights[idx] * gamma * (mWeightedMuSum.template compute<TTraits>(idx,k))+Stencil::Weights[idx] *orderparameter[k]*velocityFactor;
    }

}

template <class TLattice, class TTraits = DefaultTraitPressureWellBalanced<TLattice>>
class FlowFieldPressureWellBalanced
    : public FlowFieldPressure<TLattice, TTraits> {  // Inherit from base class to avoid repetition of common
    // calculations

    using Stencil = typename TTraits::Stencil;
    static constexpr int mNDIM = TLattice::NDIM;

   public:
    virtual inline void initialise() override;  // Initialisation step

    inline void collide() override;  // Collision step

    virtual inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    virtual inline double computeEquilibrium(int k, int idx) override;  // Calculate equilibrium in direction idx

    virtual inline double computeConcentration(int k, int iFluid) {
        double sum = 0;
        for (int i = 0; i < TTraits::NumberOfComponents; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            if (i==iFluid) return orderParameter;
            sum += orderParameter;
        }
        if (iFluid == TTraits::NumberOfComponents-1) return 1-sum;
        else {
            throw std::invalid_argument("Invalid fluid number, should be less than the number of components.");
        }
    }
    
    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions

    template <typename... TTaus>
    inline void setTaus(TTaus... tau) {
        static_assert(sizeof...(TTaus) == TTraits::NumberOfComponents,
                      "Number of relaxation times must correspond to the number of components.");

        std::vector<double> t{{tau...}};
        mv_Tau = t;

        this->mTauMax = *std::max_element(std::begin(t), std::end(t));
        this->mTauMin = *std::min_element(std::begin(t), std::end(t));
    }

    inline void setTaus(std::vector<double> tau) {
        if ((int)tau.size() != TTraits::NumberOfComponents)
            throw std::runtime_error("Number of relaxation times must correspond to the number of components.");
        mv_Tau = tau;

        this->mTauMax = *std::max_element(std::begin(mv_Tau), std::end(mv_Tau));
        this->mTauMin = *std::min_element(std::begin(mv_Tau), std::end(mv_Tau));
    }

    template <typename... TDensities>
    inline void setDensities(TDensities... density) {
        static_assert(sizeof...(TDensities) == TTraits::NumberOfComponents,
                      "Number of densities times must correspond to the number of components.");

        std::vector<double> d{{density...}};
        mv_Density = d;
    }

    inline void setDensities(std::vector<double> densities) {
        if ((int)densities.size() != TTraits::NumberOfComponents)
            throw std::runtime_error("Number of densities times must correspond to the number of components.");
        mv_Density = densities;
    }

   private:
    std::vector<double> mv_Tau;
    std::vector<double> mv_Density;


};

template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced<TLattice, TTraits>::collide() {  // Collision step

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        if (this->isCollisionNode(k)) {
            double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

            double equilibriums[Stencil::Q];
            
            for (int idx = 0; idx < Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
            }

            this->collisionQ(equilibriums, old_distribution, InverseTau<>::get<TLattice>(k),
                             k);  // CHANGE NEEDED If no forces, don't require them to be passed
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicateDistribution();
}


template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced<TLattice, TTraits>::initialise() {  // Initialise model
    this->initialiseProcessors();

    ModelBase<TLattice, TTraits>::mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<Stencil>::template initialise<TLattice>(this->mt_Forces, this->mTauMin, this->mTauMax);

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
        double* old_distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionOldPointer(k);

        Pressure<>::initialise<TLattice>(1.0, k);  // Set density to 1 initially (This will change)
        double dens = 0;
        double sumorderparameter = 0;
        for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            sumorderparameter += orderParameter;
            dens += orderParameter * mv_Density[i];
        }
        dens += (1.0 - sumorderparameter) * mv_Density.back();

        Density<>::template initialise<TLattice>(dens, k);

        double invtau = 0;
        sumorderparameter = 0;
        for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            sumorderparameter += orderParameter;
            invtau += orderParameter / mv_Tau[i];
        }
        invtau += (1.0 - sumorderparameter) / mv_Tau.back();
        InverseTau<>::template initialise<TLattice>(invtau, k);
        Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, x);
        Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, y);
        if constexpr (TLattice::NDIM == 3) Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, z);
       
        for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }
    }
    ModelBase<TLattice, TTraits>::mData.communicate(BoundaryLabels<TTraits::Lattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced<TLattice, TTraits>::computeMomenta() {  // Calculate Density<> and Velocity

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);

            
            double dens = 0;
            double sumorderparameter = 0;
            for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
                double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
                sumorderparameter += orderParameter;
                dens += orderParameter * mv_Density[i];
            }
            dens += (1.0 - sumorderparameter) * mv_Density.back();

            this->density[k] = dens;

            double invtau = 0;
            sumorderparameter = 0;
            double maxorderparameter = 0;
            int maxindex = 0;
            for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
                double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
                sumorderparameter += orderParameter;
                if (orderParameter>maxorderparameter&&i!=1) {
                    maxorderparameter = orderParameter;
                    maxindex=i;
                }
                invtau += orderParameter / mv_Tau[i];
            }
            invtau += (1.0 - sumorderparameter) / mv_Tau.back();
            if ((1.0 - sumorderparameter)>maxorderparameter) maxindex=TTraits::NumberOfComponents-1;
            invtau = 1.0/mv_Tau[maxindex];
            this->itau[k] = invtau;
            // + source; 
            this->velocity[k * TTraits::Stencil::D + x] =
                //1. / (TTraits::Stencil::Cs2) *
                this->computeVelocity(distribution, this->mt_Forces, this->density[k], x, k);  // Calculate velocities
            this->velocity[k * TTraits::Stencil::D + y] =
                //1. / (TTraits::Stencil::Cs2) * 
                this->computeVelocity(distribution, this->mt_Forces, this->density[k], y, k);
            if constexpr (TLattice::NDIM == 3)
                this->velocity[k * TTraits::Stencil::D + z] =
                    //1. / (TTraits::Stencil::Cs2) *
                    this->computeVelocity(distribution, this->mt_Forces, this->density[k], z, k);
            double velfactor = CollisionBase<TLattice, Stencil>::computeVelocityFactor(&this->velocity[k * mNDIM], 0);
            double source = 0;
            for (int xyz = 0; xyz < TTraits::Lattice::NDIM; xyz++)
                source += TTraits::Lattice::DT * 0.5 * 
                        GradientDensity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz) *
                        Velocity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz);
            this->pressure[k] = Stencil::Cs2/(1.0-Stencil::Weights[0])*(this->computeDensity(distribution, k)-distribution[0]+source+Stencil::Weights[0]*this->density[k]*velfactor);
            //this->pressure[k] = Stencil::Cs2*(this->computeDensity(distribution, k)+source);

            
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits>
inline double FlowFieldPressureWellBalanced<TLattice, TTraits>::computeEquilibrium(int k, int idx) {
    double force = 0;//-0.5*mForcingScheme1.compute<TTraits>(idx, k);
    double velocityFactor = CollisionBase<TLattice, Stencil>::computeVelocityFactor(&(this->velocity[k * mNDIM]), idx);
    return -(idx==0)*(this->pressure[k])/Stencil::Cs2+Stencil::Weights[idx] * (this->pressure[k]/Stencil::Cs2 + this->density[k] * velocityFactor)+force;
}



template <class TLattice, class TTraits = DefaultTraitPressureWellBalanced<TLattice>>
class FlowFieldPressureWellBalanced2
    : public FlowFieldPressure<TLattice, TTraits> {  // Inherit from base class to avoid repetition of common
    // calculations

    using Stencil = typename TTraits::Stencil;
    static constexpr int mNDIM = TLattice::NDIM;

   public:
    virtual inline void initialise() override;  // Initialisation step

    inline void collide() override;  // Collision step

    virtual inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    virtual inline double computeEquilibrium(int k, int idx) override;  // Calculate equilibrium in direction idx

    virtual inline double computeConcentration(int k, int iFluid) {
        double sum = 0;
        for (int i = 0; i < TTraits::NumberOfComponents; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            if (i==iFluid) return orderParameter;
            sum += orderParameter;
        }
        if (iFluid == TTraits::NumberOfComponents-1) return 1-sum;
        else {
            throw std::invalid_argument("Invalid fluid number, should be less than the number of components.");
        }
    }

    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions

    template <typename... TTaus>
    inline void setTaus(TTaus... tau) {
        static_assert(sizeof...(TTaus) == TTraits::NumberOfComponents,
                      "Number of relaxation times must correspond to the number of components.");

        std::vector<double> t{{tau...}};
        mv_Tau = t;

        this->mTauMax = *std::max_element(std::begin(t), std::end(t));
        this->mTauMin = *std::min_element(std::begin(t), std::end(t));
    }

    inline void setTaus(std::vector<double> tau) {
        if ((int)tau.size() != TTraits::NumberOfComponents)
            throw std::runtime_error("Number of relaxation times must correspond to the number of components.");
        mv_Tau = tau;

        this->mTauMax = *std::max_element(std::begin(mv_Tau), std::end(mv_Tau));
        this->mTauMin = *std::min_element(std::begin(mv_Tau), std::end(mv_Tau));
    }

    template <typename... TDensities>
    inline void setDensities(TDensities... density) {
        static_assert(sizeof...(TDensities) == TTraits::NumberOfComponents,
                      "Number of densities times must correspond to the number of components.");

        std::vector<double> d{{density...}};
        mv_Density = d;
    }

    inline void setDensities(std::vector<double> densities) {
        if ((int)densities.size() != TTraits::NumberOfComponents)
            throw std::runtime_error("Number of densities times must correspond to the number of components.");
        mv_Density = densities;
    }

   private:
    std::vector<double> mv_Tau;
    std::vector<double> mv_Density;

    double mA=0.0;
};

template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced2<TLattice, TTraits>::collide() {  // Collision step

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        if (this->isCollisionNode(k)) {
            double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

            double equilibriums[Stencil::Q];
            
            for (int idx = 0; idx < Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
            }

            this->collisionQ(equilibriums, old_distribution, 1.0,
                             k);  // CHANGE NEEDED If no forces, don't require them to be passed
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicateDistribution();
}


template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced2<TLattice, TTraits>::initialise() {  // Initialise model
    this->initialiseProcessors();

    ModelBase<TLattice, TTraits>::mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<Stencil>::template initialise<TLattice>(this->mt_Forces, this->mTauMin, this->mTauMax);

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
        double* old_distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionOldPointer(k);

        Pressure<>::initialise<TLattice>(1.0, k);  // Set density to 1 initially (This will change)
        double dens = 0;
        double sumorderparameter = 0;
        for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            sumorderparameter += orderParameter;
            dens += orderParameter * mv_Density[i];
        }
        dens += (1.0 - sumorderparameter) * mv_Density.back();

        Density<>::template initialise<TLattice>(dens, k);

        double invtau = 0;
        sumorderparameter = 0;
        for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            sumorderparameter += orderParameter;
            invtau += orderParameter / mv_Tau[i];
        }
        invtau += (1.0 - sumorderparameter) / mv_Tau.back();
        InverseTau<>::template initialise<TLattice>(invtau, k);
        Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, x);
        Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, y);
        if constexpr (TLattice::NDIM == 3) Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, z);

        for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(BoundaryLabels<TTraits::Lattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced2<TLattice, TTraits>::computeMomenta() {  // Calculate Density<> and Velocity

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);

            //pressure[k] = this->computeDensity(distribution, k);  // Calculate density
            
            
            double dens = 0;
            double sumorderparameter = 0;
            for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
                double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
                sumorderparameter += orderParameter;
                dens += orderParameter * mv_Density[i];
            }
            dens += (1.0 - sumorderparameter) * mv_Density.back();

            this->density[k] = dens;

            double invtau = 0;
            sumorderparameter = 0;
            double maxorderparameter = 0;
            int maxindex = 0;
            for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
                double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
                sumorderparameter += orderParameter;
                if (orderParameter>maxorderparameter) {
                    maxorderparameter = orderParameter;
                    maxindex=i;
                }
                invtau += orderParameter / mv_Tau[i];
            }
            invtau += (1.0 - sumorderparameter) / mv_Tau.back();
            if ((1.0 - sumorderparameter)>maxorderparameter) maxindex=TTraits::NumberOfComponents-1;
            invtau = 1.0/mv_Tau[maxindex];
            this->itau[k] = invtau;
            // + source; 
            this->velocity[k * TTraits::Stencil::D + x] =
                //1. / (TTraits::Stencil::Cs2) *
                this->computeVelocity(distribution, this->mt_Forces, this->density[k], x, k);  // Calculate velocities
            this->velocity[k * TTraits::Stencil::D + y] =
                //1. / (TTraits::Stencil::Cs2) * 
                this->computeVelocity(distribution, this->mt_Forces, this->density[k], y, k);
            if constexpr (TLattice::NDIM == 3)
                this->velocity[k * TTraits::Stencil::D + z] =
                    //1. / (TTraits::Stencil::Cs2) *
                    this->computeVelocity(distribution, this->mt_Forces, this->density[k], z, k);
            //double velfactor = CollisionBase<TLattice, Stencil>::computeVelocityFactor(&this->velocity[k * mNDIM], 0);
            double source = 0;
            for (int xyz = 0; xyz < TTraits::Lattice::NDIM; xyz++)
                source += TTraits::Lattice::DT * 0.5 * 
                        GradientDensity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz) *
                        Velocity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz);
            //this->pressure[k] = Stencil::Cs2/(1.0-Stencil::Weights[0])*(this->computeDensity(distribution, k)-distribution[0]+source+Stencil::Weights[0]*this->density[k]*velfactor);
            this->pressure[k] = Stencil::Cs2*(this->computeDensity(distribution, k)+source);

            
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits>
inline double FlowFieldPressureWellBalanced2<TLattice, TTraits>::computeEquilibrium(int k, int idx) {
    double A = 0.5-(1./InverseTau<>::get<TLattice>(k)-0.5);
    double div_v =
        (Gradient<Velocity<>>::get<TLattice,TLattice::NDIM,TLattice::NDIM>(k,0,0));
    if constexpr (TTraits::Stencil::D > 1)
        div_v +=
        (Gradient<Velocity<>>::get<TLattice,TLattice::NDIM,TLattice::NDIM>(k,1,1));
    if constexpr (TTraits::Stencil::D > 2)
        div_v +=
        (Gradient<Velocity<>>::get<TLattice,TLattice::NDIM,TLattice::NDIM>(k,2,2));
    double gradUprod = Stencil::Ci_x[idx] * Stencil::Ci_x[idx] * Gradient<Velocity<>>::get<TLattice,TLattice::NDIM,TLattice::NDIM>(k,0,0) +
        Stencil::Ci_x[idx] * Stencil::Ci_y[idx] * Gradient<Velocity<>>::get<TLattice,TLattice::NDIM,TLattice::NDIM>(k,0,1) +
        Stencil::Ci_y[idx] * Stencil::Ci_x[idx] * Gradient<Velocity<>>::get<TLattice,TLattice::NDIM,TLattice::NDIM>(k,1,0) +
        Stencil::Ci_y[idx] * Stencil::Ci_y[idx] * Gradient<Velocity<>>::get<TLattice,TLattice::NDIM,TLattice::NDIM>(k,1,1) - Stencil::Cs2*div_v;
    double gradUTprod = Stencil::Ci_x[idx] * Stencil::Ci_x[idx] * Gradient<Velocity<>>::get<TLattice,TLattice::NDIM,TLattice::NDIM>(k,0,0) +
        Stencil::Ci_y[idx] * Stencil::Ci_x[idx] * Gradient<Velocity<>>::get<TLattice,TLattice::NDIM,TLattice::NDIM>(k,0,1) +
        Stencil::Ci_x[idx] * Stencil::Ci_y[idx] * Gradient<Velocity<>>::get<TLattice,TLattice::NDIM,TLattice::NDIM>(k,1,0) +
        Stencil::Ci_y[idx] * Stencil::Ci_y[idx] * Gradient<Velocity<>>::get<TLattice,TLattice::NDIM,TLattice::NDIM>(k,1,1) - Stencil::Cs2*div_v;
    double viscterm = (gradUprod+gradUTprod)/(2*Stencil::Cs2);
    double velocityFactor = CollisionBase<TLattice, Stencil>::computeVelocityFactor(&(this->velocity[k * mNDIM]), idx);
    if (idx==0)
        return (this->pressure[k])/Stencil::Cs2 +Stencil::Weights[idx] * (this->density[k] * (velocityFactor + A*div_v));
    else {
        return Stencil::Weights[idx] * (this->density[k] * (velocityFactor+viscterm*A));
    }
    //return (idx==0)* /*Stencil::Weights[idx] * */ (this->pressure[k])/Stencil::Cs2 +Stencil::Weights[idx] * (this->density[k] * velocityFactor)+force;
}


template <class TLattice, class TTraits = DefaultTraitPressureWellBalanced<TLattice>>
class FlowFieldPressureWellBalanced3Evap
    : public FlowFieldPressure<TLattice, TTraits> {  // Inherit from base class to avoid repetition of common
    // calculations

    using Stencil = typename TTraits::Stencil;
    static constexpr int mNDIM = TLattice::NDIM;

   public:
    virtual inline void initialise() override;  // Initialisation step

    /**
     * \brief Virtual function to compute the pressure. May be overridden.
     * \param k The index of the current node in the lattice.
     */
    inline virtual double computePressure(int k);

    inline void collide() override;  // Collision step

    virtual inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    virtual inline double computeEquilibrium(int k, int idx) override;  // Calculate equilibrium in direction idx

    virtual inline double computeConcentration(int k, int iFluid) {
        double sum = 0;
        for (int i = 0; i < TTraits::NumberOfComponents; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            if (i==iFluid) return orderParameter;
            sum += orderParameter;
        }
        if (iFluid == TTraits::NumberOfComponents-1) return 1-sum;
        else {
            throw std::invalid_argument("Invalid fluid number, should be less than the number of components.");
        }
    }

    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions

    template <typename... TTaus>
    inline void setTaus(TTaus... tau) {
        static_assert(sizeof...(TTaus) == TTraits::NumberOfComponents,
                      "Number of relaxation times must correspond to the number of components.");

        std::vector<double> t{{tau...}};
        mv_Tau = t;

        this->mTauMax = *std::max_element(std::begin(t), std::end(t));
        this->mTauMin = *std::min_element(std::begin(t), std::end(t));
    }

    inline void setTaus(std::vector<double> tau) {
        if ((int)tau.size() != TTraits::NumberOfComponents)
            throw std::runtime_error("Number of relaxation times must correspond to the number of components.");
        mv_Tau = tau;

        this->mTauMax = *std::max_element(std::begin(mv_Tau), std::end(mv_Tau));
        this->mTauMin = *std::min_element(std::begin(mv_Tau), std::end(mv_Tau));
    }

    template <typename... TDensities>
    inline void setDensities(TDensities... density) {
        static_assert(sizeof...(TDensities) == TTraits::NumberOfComponents,
                      "Number of densities times must correspond to the number of components.");

        std::vector<double> d{{density...}};
        mv_Density = d;
    }

    inline void setDensities(std::vector<double> densities) {
        if ((int)densities.size() != TTraits::NumberOfComponents)
            throw std::runtime_error("Number of densities times must correspond to the number of components.");
        mv_Density = densities;
    }

   private:
    std::vector<double> mv_Tau;
    std::vector<double> mv_Density;

};

template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced3Evap<TLattice, TTraits>::collide() {  // Collision step

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        if (this->isCollisionNode(k)) {
            double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

            double equilibriums[Stencil::Q];
            
            for (int idx = 0; idx < Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
            }

            this->collisionQ(equilibriums, old_distribution, InverseTau<>::get<TLattice>(k),
                             k);  // CHANGE NEEDED If no forces, don't require them to be passed
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicateDistribution();
}


template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced3Evap<TLattice, TTraits>::initialise() {  // Initialise model
    this->initialiseProcessors();

    ModelBase<TLattice, TTraits>::mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<Stencil>::template initialise<TLattice>(this->mt_Forces, this->mTauMin, this->mTauMax);

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
        double* old_distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionOldPointer(k);

        Pressure<>::initialise<TLattice>(1.0, k);  // Set density to 1 initially (This will change)
        double dens = 0;
        double sumorderparameter = 0;
        for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            sumorderparameter += orderParameter;
            dens += orderParameter * mv_Density[i];
        }
        dens += (1.0 - sumorderparameter) * mv_Density.back();

        Density<>::template initialise<TLattice>(dens, k);

        double invtau = 0;
        sumorderparameter = 0;
        for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            sumorderparameter += orderParameter;
            invtau += orderParameter / mv_Tau[i];
        }
        invtau += (1.0 - sumorderparameter) / mv_Tau.back();
        InverseTau<>::template initialise<TLattice>(invtau, k);
        Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, x);
        Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, y);
        if constexpr (TLattice::NDIM == 3) Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, z);

        for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(BoundaryLabels<TTraits::Lattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced3Evap<TLattice, TTraits>::computeMomenta() {  // Calculate Density<> and Velocity

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);

            //pressure[k] = this->computeDensity(distribution, k);  // Calculate density
            
            
            double dens = 0;
            double sumorderparameter = 0;
            double fact = 1.0;
            if (Geometry<TLattice>::getBoundaryType(k)==0||Geometry<TLattice>::getBoundaryType(k)==6||Geometry<TLattice>::getBoundaryType(k)==9||Geometry<TLattice>::getBoundaryType(k)==8) {
                fact = 1.0/(1.0-Humidity<>::get<TLattice>(k));
            }
            for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
                double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
                sumorderparameter += orderParameter;
                dens += orderParameter * mv_Density[i]*(1+(i==1?fact:1));
            }
            dens += (1.0 - sumorderparameter) * mv_Density.back()*((TTraits::NumberOfComponents==2?fact:1));

            this->density[k] = dens;

            double invtau = 0;
            sumorderparameter = 0;
            double maxorderparameter = 0;
            int maxindex = 0;
            for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
                double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
                sumorderparameter += orderParameter;
                if (orderParameter>maxorderparameter&&i!=1) {
                    maxorderparameter = orderParameter;
                    maxindex=i;
                }
                invtau += (orderParameter>0)*orderParameter / mv_Tau[i];
            }
            invtau += (1.0 - (sumorderparameter>0)*sumorderparameter) / mv_Tau.back();
            if ((1.0 - sumorderparameter)>maxorderparameter) maxindex=TTraits::NumberOfComponents-1;
            //invtau = 1.0/mv_Tau[maxindex];
            this->itau[k] = invtau;
            // + source; 
            this->velocity[k * TTraits::Stencil::D + x] =
                //1. / (TTraits::Stencil::Cs2) *
                this->computeVelocity(distribution, this->mt_Forces, this->density[k], x, k);  // Calculate velocities
            this->velocity[k * TTraits::Stencil::D + y] =
                //1. / (TTraits::Stencil::Cs2) * 
                this->computeVelocity(distribution, this->mt_Forces, this->density[k], y, k);
            if constexpr (TLattice::NDIM == 3)
                this->velocity[k * TTraits::Stencil::D + z] =
                    //1. / (TTraits::Stencil::Cs2) *
                    this->computeVelocity(distribution, this->mt_Forces, this->density[k], z, k);
            //double velfactor = CollisionBase<TLattice, Stencil>::computeVelocityFactor(&this->velocity[k * mNDIM], 0);
            double source = 0;
            for (int xyz = 0; xyz < TTraits::Lattice::NDIM; xyz++)
                source += TTraits::Lattice::DT * 0.5 * 
                        GradientDensity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz) *
                        Velocity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz);
            //this->pressure[k] = Stencil::Cs2/(1.0-Stencil::Weights[0])*(this->computeDensity(distribution, k)-distribution[0]+source+Stencil::Weights[0]*this->density[k]*velfactor);
            this->pressure[k] = Stencil::Cs2*(this->computeDensity(distribution, k)+source);

            
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits>
inline double FlowFieldPressureWellBalanced3Evap<TLattice, TTraits>::computeEquilibrium(int k, int idx) {
    double force = 0;//-0.5*mForcingScheme1.compute<TTraits>(idx, k);
    double velocityFactor = CollisionBase<TLattice, Stencil>::computeVelocityFactor(&(this->velocity[k * mNDIM]), idx);
    return  /*Stencil::Weights[idx] * */ Stencil::Weights[idx] *(this->pressure[k])/Stencil::Cs2 +Stencil::Weights[idx] * (this->density[k] * velocityFactor)+force;
}

template <class TLattice, class TTraits>
inline double FlowFieldPressureWellBalanced3Evap<TLattice, TTraits>::computePressure(int k) {
    double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
    double velfactor = CollisionBase<TLattice, Stencil>::computeVelocityFactor(&this->velocity[k * mNDIM], 0);
    double source = 0;
    for (int xyz = 0; xyz < TTraits::Lattice::NDIM; xyz++)
        source += TTraits::Lattice::DT * 0.5 * 
                GradientDensity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz) *
                Velocity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz);
    return Stencil::Cs2*(this->computeDensity(distribution, k)+source);
    //this->pressure[k] = Stencil::Cs2*(this->computeDensity(distribution, k)+source);
}

template <class TLattice, class TTraits = DefaultTraitPressureWellBalanced<TLattice>>
class FlowFieldPressureWellBalanced2Frozen
    : public FlowFieldPressure<TLattice, TTraits> {  // Inherit from base class to avoid repetition of common
    // calculations

    using Stencil = typename TTraits::Stencil;
    static constexpr int mNDIM = TLattice::NDIM;

   public:
    virtual inline void initialise() override;  // Initialisation step

    inline void collide() override;  // Collision step

    virtual inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    virtual inline double computeEquilibrium(int k, int idx) override;  // Calculate equilibrium in direction idx

    virtual inline double computeConcentration(int k, int iFluid) {
        double sum = 0;
        for (int i = 0; i < TTraits::NumberOfComponents; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            if (i==iFluid) return orderParameter;
            sum += orderParameter;
        }
        if (iFluid == TTraits::NumberOfComponents-1) return 1-sum;
        else {
            throw std::invalid_argument("Invalid fluid number, should be less than the number of components.");
        }
    }

    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions

    template <typename... TTaus>
    inline void setTaus(TTaus... tau) {
        static_assert(sizeof...(TTaus) == TTraits::NumberOfComponents,
                      "Number of relaxation times must correspond to the number of components.");

        std::vector<double> t{{tau...}};
        mv_Tau = t;

        this->mTauMax = *std::max_element(std::begin(t), std::end(t));
        this->mTauMin = *std::min_element(std::begin(t), std::end(t));
    }

    inline void setTaus(std::vector<double> tau) {
        if ((int)tau.size() != TTraits::NumberOfComponents)
            throw std::runtime_error("Number of relaxation times must correspond to the number of components.");
        mv_Tau = tau;

        this->mTauMax = *std::max_element(std::begin(mv_Tau), std::end(mv_Tau));
        this->mTauMin = *std::min_element(std::begin(mv_Tau), std::end(mv_Tau));
    }

    template <typename... TDensities>
    inline void setDensities(TDensities... density) {
        static_assert(sizeof...(TDensities) == TTraits::NumberOfComponents,
                      "Number of densities times must correspond to the number of components.");

        std::vector<double> d{{density...}};
        mv_Density = d;
    }

    inline void setDensities(std::vector<double> densities) {
        if ((int)densities.size() != TTraits::NumberOfComponents)
            throw std::runtime_error("Number of densities times must correspond to the number of components.");
        mv_Density = densities;
    }

    int mFrozenComponent = 1;
    void setFrozenComponent(int comp) {mFrozenComponent = comp;}

   private:
   
    std::vector<double> mv_Tau;
    std::vector<double> mv_Density;

};

template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced2Frozen<TLattice, TTraits>::collide() {  // Collision step

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        if (this->isCollisionNode(k)) {
            double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

            double equilibriums[Stencil::Q];
            
            for (int idx = 0; idx < Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
            }

            this->collisionQ(equilibriums, old_distribution, InverseTau<>::get<TLattice>(k),
                             k);  // CHANGE NEEDED If no forces, don't require them to be passed
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicateDistribution();
}


template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced2Frozen<TLattice, TTraits>::initialise() {  // Initialise model
    this->initialiseProcessors();

    ModelBase<TLattice, TTraits>::mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<Stencil>::template initialise<TLattice>(this->mt_Forces, this->mTauMin, this->mTauMax);

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
        double* old_distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionOldPointer(k);

        Pressure<>::initialise<TLattice>(1.0, k);  // Set density to 1 initially (This will change)
        double dens = 0;
        double sumorderparameter = 0;
        for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            sumorderparameter += orderParameter;
            dens += orderParameter * mv_Density[i];
        }
        dens += (1.0 - sumorderparameter) * mv_Density.back();

        Density<>::template initialise<TLattice>(dens, k);

        double invtau = 0;
        sumorderparameter = 0;
        for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            sumorderparameter += orderParameter;
            invtau += orderParameter / mv_Tau[i];
        }
        invtau += (1.0 - sumorderparameter) / mv_Tau.back();
        InverseTau<>::template initialise<TLattice>(invtau, k);
        Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, x);
        Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, y);
        if constexpr (TLattice::NDIM == 3) Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, z);

        for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(BoundaryLabels<TTraits::Lattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced2Frozen<TLattice, TTraits>::computeMomenta() {  // Calculate Density<> and Velocity

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);

            //pressure[k] = this->computeDensity(distribution, k);  // Calculate density
            
            
            double dens = 0;
            double sumorderparameter = 0;
            for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
                double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
                sumorderparameter += orderParameter;
                dens += orderParameter * mv_Density[i];
            }
            dens += (1.0 - sumorderparameter) * mv_Density.back();

            this->density[k] = dens;

            double invtau = 0;
            sumorderparameter = 0;
            double maxorderparameter = 0;
            int maxindex = 0;
            for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
                double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
                sumorderparameter += orderParameter;
                if (orderParameter>maxorderparameter&&i!=mFrozenComponent) {
                    maxorderparameter = orderParameter;
                    maxindex=i;
                }
                invtau += orderParameter / mv_Tau[i];
            }
            invtau += (1.0 - sumorderparameter) / mv_Tau.back();
            if ((1.0 - sumorderparameter)>maxorderparameter) maxindex=TTraits::NumberOfComponents-1;
            invtau = 1.0/mv_Tau[maxindex];
            this->itau[k] = invtau;
            // + source; 
            this->velocity[k * TTraits::Stencil::D + x] =
                //1. / (TTraits::Stencil::Cs2) *
                this->computeVelocity(distribution, this->mt_Forces, this->density[k], x, k);  // Calculate velocities
            this->velocity[k * TTraits::Stencil::D + y] =
                //1. / (TTraits::Stencil::Cs2) * 
                this->computeVelocity(distribution, this->mt_Forces, this->density[k], y, k);
            if constexpr (TLattice::NDIM == 3)
                this->velocity[k * TTraits::Stencil::D + z] =
                    //1. / (TTraits::Stencil::Cs2) *
                    this->computeVelocity(distribution, this->mt_Forces, this->density[k], z, k);
            //double velfactor = CollisionBase<TLattice, Stencil>::computeVelocityFactor(&this->velocity[k * mNDIM], 0);
            double source = 0;
            for (int xyz = 0; xyz < TTraits::Lattice::NDIM; xyz++)
                source += TTraits::Lattice::DT * 0.5 * 
                        GradientDensity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz) *
                        Velocity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz);
            //this->pressure[k] = Stencil::Cs2/(1.0-Stencil::Weights[0])*(this->computeDensity(distribution, k)-distribution[0]+source+Stencil::Weights[0]*this->density[k]*velfactor);
            this->pressure[k] = Stencil::Cs2*(this->computeDensity(distribution, k)+source);

            
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits>
inline double FlowFieldPressureWellBalanced2Frozen<TLattice, TTraits>::computeEquilibrium(int k, int idx) {
    double force = 0;//-0.5*mForcingScheme1.compute<TTraits>(idx, k);
    double velocityFactor = CollisionBase<TLattice, Stencil>::computeVelocityFactor(&(this->velocity[k * mNDIM]), idx);
    return (idx==0)* /*Stencil::Weights[idx] * */ (this->pressure[k])/Stencil::Cs2 +Stencil::Weights[idx] * (this->density[k] * velocityFactor)+force;
}


template <class TLattice, class TTraits = DefaultTraitPressureWellBalanced<TLattice>>
class FlowFieldPressureWellBalanced3
    : public FlowFieldPressure<TLattice, TTraits> {  // Inherit from base class to avoid repetition of common
    // calculations

    using Stencil = typename TTraits::Stencil;
    static constexpr int mNDIM = TLattice::NDIM;

   public:
    virtual inline void initialise() override;  // Initialisation step

    inline void collide() override;  // Collision step

    virtual inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    virtual inline double computeEquilibrium(int k, int idx) override;  // Calculate equilibrium in direction idx

    virtual inline double computeConcentration(int k, int iFluid) {
        double sum = 0;
        for (int i = 0; i < TTraits::NumberOfComponents; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            if (i==iFluid) return orderParameter;
            sum += orderParameter;
        }
        if (iFluid == TTraits::NumberOfComponents-1) return 1-sum;
        else {
            throw std::invalid_argument("Invalid fluid number, should be less than the number of components.");
        }
    }

    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions

    template <typename... TTaus>
    inline void setTaus(TTaus... tau) {
        static_assert(sizeof...(TTaus) == TTraits::NumberOfComponents,
                      "Number of relaxation times must correspond to the number of components.");

        std::vector<double> t{{tau...}};
        mv_Tau = t;

        this->mTauMax = *std::max_element(std::begin(t), std::end(t));
        this->mTauMin = *std::min_element(std::begin(t), std::end(t));
    }

    inline void setTaus(std::vector<double> tau) {
        if ((int)tau.size() != TTraits::NumberOfComponents)
            throw std::runtime_error("Number of relaxation times must correspond to the number of components.");
        mv_Tau = tau;

        this->mTauMax = *std::max_element(std::begin(mv_Tau), std::end(mv_Tau));
        this->mTauMin = *std::min_element(std::begin(mv_Tau), std::end(mv_Tau));
    }

    template <typename... TDensities>
    inline void setDensities(TDensities... density) {
        static_assert(sizeof...(TDensities) == TTraits::NumberOfComponents,
                      "Number of densities times must correspond to the number of components.");

        std::vector<double> d{{density...}};
        mv_Density = d;
    }

    inline void setDensities(std::vector<double> densities) {
        if ((int)densities.size() != TTraits::NumberOfComponents)
            throw std::runtime_error("Number of densities times must correspond to the number of components.");
        mv_Density = densities;
    }

   private:
    std::vector<double> mv_Tau;
    std::vector<double> mv_Density;

};

template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced3<TLattice, TTraits>::collide() {  // Collision step

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        if (this->isCollisionNode(k)) {
            double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

            double equilibriums[Stencil::Q];
            
            for (int idx = 0; idx < Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
            }

            this->collisionQ(equilibriums, old_distribution, InverseTau<>::get<TLattice>(k),
                             k);  // CHANGE NEEDED If no forces, don't require them to be passed
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicateDistribution();
}


template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced3<TLattice, TTraits>::initialise() {  // Initialise model
    this->initialiseProcessors();

    ModelBase<TLattice, TTraits>::mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<Stencil>::template initialise<TLattice>(this->mt_Forces, this->mTauMin, this->mTauMax);

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
        double* old_distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionOldPointer(k);

        Pressure<>::initialise<TLattice>(1.0, k);  // Set density to 1 initially (This will change)
        double dens = 0;
        double sumorderparameter = 0;
        for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            sumorderparameter += orderParameter;
            dens += orderParameter * mv_Density[i];
        }
        dens += (1.0 - sumorderparameter) * mv_Density.back();

        Density<>::template initialise<TLattice>(dens, k);

        double invtau = 0;
        sumorderparameter = 0;
        for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
            double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
            sumorderparameter += orderParameter;
            invtau += orderParameter / mv_Tau[i];
        }
        invtau += (1.0 - sumorderparameter) / mv_Tau.back();
        InverseTau<>::template initialise<TLattice>(invtau, k);
        Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, x);
        Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, y);
        if constexpr (TLattice::NDIM == 3) Velocity<>::initialise<TLattice, TLattice::NDIM>(0.0, k, z);

        for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }
    }
    ModelBase<TLattice, TTraits>::mData.communicate(BoundaryLabels<TTraits::Lattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits>
inline void FlowFieldPressureWellBalanced3<TLattice, TTraits>::computeMomenta() {  // Calculate Density<> and Velocity

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);

            double dens = 0;
            double sumorderparameter = 0;
            for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
                double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
                sumorderparameter += orderParameter;
                dens += orderParameter * mv_Density[i];
            }
            dens += (1.0 - sumorderparameter) * mv_Density.back();

            this->density[k] = dens;

            /*double invtau = 0;
            sumorderparameter = 0;
            double maxorderparameter = 0;
            int maxindex = 0;
            for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
                double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
                sumorderparameter += orderParameter;
                if (orderParameter>maxorderparameter) {//&&i!=1) {
                    maxorderparameter = orderParameter;
                    maxindex=i;
                }
                invtau += orderParameter / mv_Tau[i];
            }
            invtau += (1.0 - sumorderparameter) / mv_Tau.back();
            if ((1.0 - sumorderparameter)>maxorderparameter) maxindex=TTraits::NumberOfComponents-1;
            invtau = 1.0/mv_Tau[maxindex];*/
            double invtau = 0;
            sumorderparameter = 0;
            double maxorderparameter = 0;
            int maxindex = 0;
            for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
                double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
                sumorderparameter += orderParameter;
                if (orderParameter>maxorderparameter&&i!=1) {
                    maxorderparameter = orderParameter;
                    maxindex=i;
                }
                invtau += (orderParameter>0)*orderParameter / mv_Tau[i];
            }
            invtau += (1.0 - (sumorderparameter>0)*sumorderparameter) / mv_Tau.back();
            if ((1.0 - sumorderparameter)>maxorderparameter) maxindex=TTraits::NumberOfComponents-1;
            //if ((1.0 - OrderParameter<>::get<TLattice>(k)-OrderParameter<1>::get<TLattice>(k))<OrderParameter<>::get<TLattice>(k)) maxindex=0;
            //if (OrderParameter<1>::get<TLattice>(k)>OrderParameter<>::get<TLattice>(k)&&OrderParameter<1>::get<TLattice>(k)>(1.0 - OrderParameter<>::get<TLattice>(k)-OrderParameter<1>::get<TLattice>(k))) maxindex=0;
            //invtau = 1.0/mv_Tau[maxindex];
            /*int yy = computeY(TLattice::LY, TLattice::LZ, k);
            double t = yy / (TLattice::LY - 1.0);
            double rhoIn = mv_Tau[1]/100.;
            double rhoOut = mv_Tau[1];
            if (maxindex==1) this->itau[k] = rhoIn*(1-t) + rhoOut*t;
            else */this->itau[k] = invtau;
            // + source; 
            this->velocity[k * TTraits::Stencil::D + x] =
                //1. / (TTraits::Stencil::Cs2) *
                this->computeVelocity(distribution, this->mt_Forces, this->density[k], x, k);  // Calculate velocities
            this->velocity[k * TTraits::Stencil::D + y] =
                //1. / (TTraits::Stencil::Cs2) * 
                this->computeVelocity(distribution, this->mt_Forces, this->density[k], y, k);
            if constexpr (TLattice::NDIM == 3)
                this->velocity[k * TTraits::Stencil::D + z] =
                    //1. / (TTraits::Stencil::Cs2) *
                    this->computeVelocity(distribution, this->mt_Forces, this->density[k], z, k);
            //double velfactor = CollisionBase<TLattice, Stencil>::computeVelocityFactor(&this->velocity[k * mNDIM], 0);
            double source = 0;
            for (int xyz = 0; xyz < TTraits::Lattice::NDIM; xyz++)
                source += TTraits::Lattice::DT * 0.5 * 
                        GradientDensity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz) *
                        Velocity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz);
            //this->pressure[k] = Stencil::Cs2/(1.0-Stencil::Weights[0])*(this->computeDensity(distribution, k)-distribution[0]+source+Stencil::Weights[0]*this->density[k]*velfactor);
            this->pressure[k] = Stencil::Cs2*(this->computeDensity(distribution, k)+source);

            
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits>
inline double FlowFieldPressureWellBalanced3<TLattice, TTraits>::computeEquilibrium(int k, int idx) {
    double velocityFactor = CollisionBase<TLattice, Stencil>::computeVelocityFactor(&(this->velocity[k * mNDIM]), idx);
    return Stencil::Weights[idx] *  (this->pressure[k])/Stencil::Cs2 +Stencil::Weights[idx] * (this->density[k] * velocityFactor);
}