#pragma once
#include <algorithm>
#include <utility>

#include "../AddOns/AddOns.hh"
#include "../BoundaryModels/Boundaries.hh"
#include "../Collide.hh"
#include "../Data.hh"
#include "../Forces/Forces.hh"
#include "../Parallel.hh"
#include "../Parameters.hh"
#include "ModelBase.hh"

// NComponent.hh: Contains the details of the LBM model to solve an equation for phase separation. Each
// Model is given a "TTraits" class that contains stencil, data, force and boundary information

template <class TLattice, int N, int Nmax>
using DefaultTraitNComponent = typename DefaultTrait<TLattice, Nmax>::template SetBoundary<
    BounceBack>::template SetForce<AllenCahnSource<AllenCahnSourceMethod, N>>;

template <class TLattice, int N, int Nmax, class TTraits = DefaultTraitNComponent<TLattice, N, Nmax>>
class NComponent : public CollisionBase<TLattice, typename TTraits::Stencil>,
                   public ModelBase<TLattice, TTraits> {  // Inherit from base class to avoid repetition of common
    // calculations

    using Stencil = typename TTraits::Stencil;
    static constexpr int mNDIM = TLattice::NDIM;

   public:
    inline void collide() override;  // Collision step

    inline void initialise() override;  // Initialisation step

    inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    inline double computeEquilibrium(int k, int idx) override;  // Calculate equilibrium in direction idx

    inline void setTau(const std::vector<double>& tau) {
        if ((int)tau.size() != TTraits::NumberOfComponents)
            throw std::runtime_error("Number of relaxation times must correspond to the number of components.");
        mTau = tau[N];
        mGamma = mMobility / (TLattice::DT * Stencil::Cs2 * (2 * TLattice::DT / mTau - TLattice::DT) / 2.0);
    }

    inline void setTau(double tau) {
        mTau = tau;
        mGamma = mMobility / (TLattice::DT * Stencil::Cs2 * (2 * TLattice::DT / tau - TLattice::DT) / 2.0);
    }

    inline void setMobility(double mobility) {
        mMobility = mobility;
        mGamma = mobility / (TLattice::DT * Stencil::Cs2 * (2 * TLattice::DT / mTau - TLattice::DT) / 2.0);
    }

    template <typename TTau>
    inline void setTauAndMobility(TTau tau, double mobility) {
        setTau(tau);
        setMobility(mobility);
    }

   private:
    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions

    double mGamma = 0.01;
    double mMobility;

    double mTau = 1.0;

    std::vector<double>& density = Density<>::get<TLattice>();  // Reference to vector of order parameters
    std::vector<double>& orderparameter =
        OrderParameter<N>::template get<TLattice>();  // Reference to vector of order parameters
    std::vector<double>& velocity = Velocity<>::get<TLattice, TLattice::NDIM>();  // Reference to vector of velocities
    std::vector<double>& itau = InverseTau<>::get<TLattice>();                    // Reference to vector of velocities
};

template <class TLattice, int N, int Nmax, class TTraits>
inline void NComponent<TLattice, N, Nmax, TTraits>::collide() {
#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        if (this->isCollisionNode(k)) {
            double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

            double equilibriums[TTraits::Stencil::Q];

            for (int idx = 1; idx < TTraits::Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
            }
            equilibriums[0] = computeEquilibrium(k, 0);

            this->collisionQ(equilibriums, old_distribution, 1.0 / mTau,
                             k);  // CHANGE NEEDED If no forces, don't require them to be passed
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicateDistribution();
}

template <class TLattice, int N, int Nmax, class TTraits>
inline void NComponent<TLattice, N, Nmax, TTraits>::initialise() {  // Initialise model
    this->initialiseProcessors();

    ModelBase<TLattice, TTraits>::mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<typename TTraits::Stencil>::template initialise<TLattice>(this->mt_Forces, mTau,
                                                                                               mTau);

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
        double* old_distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionOldPointer(k);

        ChemicalPotential<N>::template initialise<TLattice>(0, k);
        OrderParameter<N>::template initialise<TLattice>(1.0, k);

        for (int idx = 1; idx < TTraits::Stencil::Q; ++idx) {
            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }

        double equilibrium = computeEquilibrium(k, 0);
        distribution[0] = equilibrium;  // Set distributions to equillibrium
        old_distribution[0] = equilibrium;
    }

    TLattice::template communicate<OrderParameter, TTraits::NumberOfComponents - 1>();
    ModelBase<TLattice, TTraits>::mData.communicate(BoundaryLabels<TLattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::template getInstance<TLattice>());
}

template <class TLattice, int N, int Nmax, class TTraits>
inline void NComponent<TLattice, N, Nmax, TTraits>::computeMomenta() {  // Calculate order parameter

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);

            orderparameter[k] = this->computeDensity(distribution, k);
        }
    }

    TLattice::template communicate<OrderParameter, TTraits::NumberOfComponents - 1>();
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, int N, int Nmax, class TTraits>
inline double NComponent<TLattice, N, Nmax, TTraits>::computeEquilibrium(int k, int idx) {
    double orderparam = orderparameter[k];
    double velocityFactor = CollisionBase<TLattice, Stencil>::computeVelocityFactor(&velocity[k * Stencil::D], idx);

    if (idx == 0) {
        return Stencil::Weights[idx] * orderparam *
               ((1 - (1 - Stencil::Weights[0]) * mGamma) / Stencil::Weights[0] + velocityFactor);
    } else {
        return Stencil::Weights[idx] * orderparam * (mGamma + velocityFactor);
    }
}

template <class TMethod>
class PressureForceNComp : public ChemicalForce<TMethod> {
   public:
    template <class TTraits>
    inline double computeXYZ(const int xyz, const int k) {
        return ChemicalForce<TMethod>::template computeXYZ<TTraits>(xyz, k);
    }
    template <class TTraits>
    inline double computeQ(const int idx, const int k) {
        double sum = 0;
        for (int xyz = 0; xyz < TTraits::Lattice::NDIM; xyz++) {
            sum += (TTraits::Stencil::Cs2 *
                        GradientDensity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz) +
                    ChemicalForce<TMethod>::template computeXYZ<TTraits>(xyz, k)) *
                   TTraits::Stencil::Ci_xyz(xyz)[idx];
        }
        return sum;
    }
    template <class TTraits>
    inline double computeDensitySource(int k) {  // SHOULD BE CENTRAL GRADIENTS
        double source = 0;
        for (int xyz = 0; xyz < TTraits::Lattice::NDIM; xyz++)
            source += TTraits::Lattice::DT * 0.5 * TTraits::Stencil::Cs2 *
                      GradientDensity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz) *
                      Velocity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz);
        return source;
    }

   private:
};

template <class TLattice, int TNMax>
using DefaultTraitFlowFieldPressureNComp =
    typename DefaultTraitFlowFieldPressure<TLattice, TNMax>::template SetForce<PressureForceNComp<NCompForce>>;

template <class TLattice, int Nmax, class TTraits = DefaultTraitFlowFieldPressureNComp<TLattice, Nmax>>
class FlowFieldPressureNComp
    : public FlowFieldPressure<TLattice, TTraits> {  // Inherit from base class to avoid repetition of common
    // calculations

    using Stencil = typename TTraits::Stencil;
    static constexpr int mNDIM = TLattice::NDIM;

   public:
    FlowFieldPressureNComp() : mv_Tau(Nmax, 1.), mv_Density(Nmax, 1.) {}

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
                      "Number of relaxation times must correspond to the number of components.");

        std::vector<double> d{{density...}};
        mv_Density = d;
    }

    inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    inline void initialise() override;

    std::vector<double>& pressure = Pressure<>::get<TLattice>();                  // Reference to vector of TDensities
    std::vector<double>& density = Density<>::get<TLattice>();                    // Reference to vector of TDensities
    std::vector<double>& velocity = Velocity<>::get<TLattice, TLattice::NDIM>();  // Reference to vector of velocities
    std::vector<double>& itau = InverseTau<>::get<TLattice>();

    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions

   private:
    std::vector<double> mv_Tau;
    std::vector<double> mv_Density;
};

template <class TLattice, int Nmax, class TTraits>
inline void FlowFieldPressureNComp<TLattice, Nmax, TTraits>::initialise() {  // Initialise model
    this->initialiseProcessors();

    this->mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<Stencil>::template initialise<TLattice>(this->mt_Forces, this->mTauMin,
                                                                             this->mTauMax);

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        double* distribution = this->mDistribution.getDistributionPointer(k);
        double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

        Pressure<>::template initialise<TLattice>(1.0, k);  // Set density to 1 initially (This will change)

        Velocity<>::template initialise<TLattice, TLattice::NDIM>(0.0, k, x);
        Velocity<>::template initialise<TLattice, TLattice::NDIM>(0.0, k, y);
        if constexpr (TLattice::NDIM == 3) Velocity<>::template initialise<TLattice, TLattice::NDIM>(0.0, k, z);

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

        for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
            double equilibrium = this->computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }
    }

    this->mData.communicate(Pressure<>::getInstance<TLattice>());
    this->mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
}

template <class TLattice, int Nmax, class TTraits>
inline void FlowFieldPressureNComp<TLattice, Nmax, TTraits>::computeMomenta() {  // Calculate Density<> and Velocity

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = this->mDistribution.getDistributionPointer(k);

            pressure[k] = this->computeDensity(distribution, k);  // Calculate density

            double dens = 0;
            double sumorderparameter = 0;
            for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
                double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
                sumorderparameter += orderParameter;
                dens += orderParameter * mv_Density[i];
            }
            dens += (1.0 - sumorderparameter) * mv_Density.back();

            density[k] = dens;

            velocity[k * TTraits::Stencil::D + x] =
                1. / (TTraits::Stencil::Cs2) *
                this->computeVelocity(distribution, this->mt_Forces, density[k], x, k);  // Calculate velocities
            velocity[k * TTraits::Stencil::D + y] =
                1. / (TTraits::Stencil::Cs2) * this->computeVelocity(distribution, this->mt_Forces, density[k], y, k);
            if constexpr (TLattice::NDIM == 3)
                velocity[k * TTraits::Stencil::D + z] =
                    1. / (TTraits::Stencil::Cs2) *
                    this->computeVelocity(distribution, this->mt_Forces, density[k], z, k);

            double invtau = 0;
            sumorderparameter = 0;
            for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
                double orderParameter = getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TLattice>(i)[k];
                sumorderparameter += orderParameter;
                invtau += orderParameter / mv_Tau[i];
            }
            invtau += (1.0 - sumorderparameter) / mv_Tau.back();
            itau[k] = invtau;
        }
    }

    this->mData.communicate(Pressure<>::getInstance<TLattice>());
    this->mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
}
