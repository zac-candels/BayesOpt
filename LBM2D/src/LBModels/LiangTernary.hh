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
#include "WellBalanced.hh"

class LiangMomentumCorrection : public ForceBase<SimpleForcing<NoTauDependence>> { //ForceBase?
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return Velocity<>::template get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz)*((rho1-rho3)*mobility1*LaplacianChemicalPotential<0>::template get<typename TTraitsF::Lattice>(k)+(rho2-rho3)*mobility2*LaplacianChemicalPotential<1>::template get<typename TTraitsF::Lattice>(k));
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return 0;
    }
    void setRhos(double r1, double r2, double r3) {
        rho1 = r1;
        rho2 = r2;
        rho3 = r3;
    }
    void setMobilities(double m1, double m2) {
        mobility1 = m1;
        mobility2 = m2;
    }
    private:
        double rho1=1;
        double rho2=1;
        double rho3=1;
        double mobility1=1;
        double mobility2=1;
};

template<int Nmax,class TLattice>
using DefaultTraitPressureLiangN = typename DefaultTrait<TLattice,Nmax> :: template SetBoundary<BounceBack>
                                                                :: template SetProcessor<GradientsMultiStencil<Density<>,CentralXYZBounceBack,CentralQBounceBack>>
                                                               :: template SetForce< ChemicalForce<LiangForce<GuoPrefactor>>, LiangMomentumCorrection >;//, PressureForceWellBalancedTau<SimpleForcingQ<SplitGuoPrefactor>,ChemicalForce>

template<int N=0>
class NoSourceLiang : public ChemicalForceBinaryMu<LiangCHForce<N>, Gradient> { //ForceBase?
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

template<int N, int Nmax, class TLattice>
using DefaultTraitLiangCH1 = typename DefaultTrait<TLattice,Nmax> :: template SetBoundary<BounceBack>
/* Change this so only done for N=0 except for order param*/   :: template SetProcessor<std::tuple<GradientsMultiStencil<OrderParameter<N>,CentralXYZBounceBack,LaplacianCentralWetting/*,BiasedQBounceBack,BiasedXYZBounceBack*/,CentralQBounceBack>>,
                                                                                        std::tuple<ChemicalPotentialCalculatorTernaryLee>,
                                                                                        std::tuple<GradientsMultiStencil<ChemicalPotential<>,LaplacianCentralBounceBack>>>
                                                               :: template SetForce< NoSourceLiang<N> >;

template<int N, int Nmax, class TLattice>
using DefaultTraitLiangCH2 = typename DefaultTrait<TLattice,Nmax> :: template SetBoundary<BounceBack>
/* Change this so only done for N=0 except for order param*/   :: template SetProcessor<GradientsMultiStencil<OrderParameter<N>,CentralXYZBounceBack,LaplacianCentralWetting/*,BiasedQBounceBack,BiasedXYZBounceBack*/,CentralQBounceBack>>
                                                               :: template SetForce< NoSourceLiang<N> >;

template<int N, int Nmax, class TLattice, class TTraits = DefaultTraitLiangCH1<N, Nmax, TLattice>>
class LiangCH : public CollisionBase<TLattice,typename TTraits::Stencil>, public ModelBase<TLattice, TTraits> { //Inherit from base class to avoid repetition of common
                                                         //calculations

    using Stencil = typename TTraits::Stencil;  
    static constexpr int mNDIM = TLattice::NDIM; 

    public:
        std::vector<double>& density; //Reference to vector of TDensities
        std::vector<double>& orderparameter;
        std::vector<double>& velocity; //Reference to vector of velocities
        std::vector<double>& itau; 

        LiangCH() : CollisionBase<TLattice,typename TTraits::Stencil>(), ModelBase<TLattice,TTraits>(),
                           density(Density<>::get<TLattice>()), orderparameter(OrderParameter<N>::template get<TLattice>()),
                           velocity(Velocity<>::get<TLattice,TLattice::NDIM>()), itau(InverseTau<>::get<TLattice>()) {}

        LiangCH(LiangCH<N,Nmax,TLattice, TTraits>& other) : CollisionBase<TLattice,typename TTraits::Stencil>(), ModelBase<TLattice,TTraits>(other),
                    density(Density<>::get<TLattice>()), orderparameter(OrderParameter<N>::template get<TLattice>()),
                           velocity(Velocity<>::get<TLattice,TLattice::NDIM>()), itau(InverseTau<>::get<TLattice>()) {}

        LiangCH<N,Nmax,TLattice, TTraits>& operator=(const LiangCH<N,Nmax,TLattice, TTraits>& other) {
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

        inline void setMij(std::vector<double> mij) {
            #pragma omp parallel
            {
            mWeightedMuSum.setMij(mij);
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
thread_local WellBalancedEQ<N> LiangCH<N, Nmax, TLattice, TTraits>::mWeightedMuSum;

template<int N, int Nmax, class TLattice, class TTraits>
inline void LiangCH<N, Nmax, TLattice, TTraits>::collide() { //Collision step

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
inline void LiangCH<N, Nmax, TLattice, TTraits>::initialise() { //Initialise model
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
inline void LiangCH<N, Nmax, TLattice, TTraits>::computeMomenta() { //Calculate Density<> and Velocity

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
inline double LiangCH<N, Nmax, TLattice, TTraits>::computeEquilibrium(int k, int idx) {
    double velocityFactor = CollisionBase<TLattice, Stencil>::computeVelocityFactorFirstOrder(&(this->velocity[k * mNDIM]), idx);
    if (idx == 0) {
        return ((orderparameter[k] - (1 - Stencil::Weights[0]) * (gamma * (mWeightedMuSum.template compute<TTraits>(idx,k)))))+Stencil::Weights[idx] *orderparameter[k]*velocityFactor;
    } else {
        return Stencil::Weights[idx] * gamma * (mWeightedMuSum.template compute<TTraits>(idx,k))+Stencil::Weights[idx] *orderparameter[k]*velocityFactor;
    }

}