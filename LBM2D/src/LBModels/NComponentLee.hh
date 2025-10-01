#pragma once
#include <utility>

#include "../Collide.hh"
#include "../Data.hh"
#include "../Parameters.hh"
#include "FlowField.hh"
#include "ModelBase.hh"

template <int N>
class LeeNCompSource : public ChemicalForceMu<Lee, GradientMixed> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return (
            MixedGradientOrderParameter<N>::template get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) -
            OrderParameter<N>::template get<typename TTraitsF::Lattice>(k) /
                (TTraitsF::Stencil::Cs2 * Density<>::get<typename TTraitsF::Lattice>(k)) *
                (MixedGradientPressure<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) +
                 ChemicalForceMu<Lee, GradientMixed>::computeXYZ<TTraitsF>(xyz, k)));
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return (MixedGradientOrderParameter<N>::template get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) -
                OrderParameter<N>::template get<typename TTraitsF::Lattice>(k) /
                    (TTraitsF::Stencil::Cs2 * Density<>::get<typename TTraitsF::Lattice>(k)) *
                    (MixedGradientPressure<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) +
                     ChemicalForceMu<Lee, GradientMixed>::computeQ<TTraitsF>(idx, k)));
    }
};

template <int N>
class LeeNCompSourceCentral : public ChemicalForceMu<Lee, Gradient> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return (GradientOrderParameter<N>::template get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) -
                OrderParameter<N>::template get<typename TTraitsF::Lattice>(k) /
                    (TTraitsF::Stencil::Cs2 * Density<>::get<typename TTraitsF::Lattice>(k)) *
                    (GradientPressure<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) +
                     ChemicalForceMu<Lee, Gradient>::computeXYZ<TTraitsF>(xyz, k)));
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return (GradientOrderParameter<N>::template get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) -
                OrderParameter<N>::template get<typename TTraitsF::Lattice>(k) /
                    (TTraitsF::Stencil::Cs2 * Density<>::get<typename TTraitsF::Lattice>(k)) *
                    (GradientPressure<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) +
                     ChemicalForceMu<Lee, Gradient>::computeQ<TTraitsF>(idx, k)));
    }
};

template <int N>
class LeeNCompSourceEquilibrium : public ChemicalForceMu<Lee, Gradient> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return -0.5 *
               (GradientOrderParameter<N>::template get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) -
                OrderParameter<N>::template get<typename TTraitsF::Lattice>(k) /
                    (TTraitsF::Stencil::Cs2 * Density<>::get<typename TTraitsF::Lattice>(k)) *
                    (GradientPressure<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) +
                     ChemicalForceMu<Lee, Gradient>::computeXYZ<TTraitsF>(xyz, k)));
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return -0.5 *
               (GradientOrderParameter<N>::template get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) -
                OrderParameter<N>::template get<typename TTraitsF::Lattice>(k) /
                    (TTraitsF::Stencil::Cs2 * Density<>::get<typename TTraitsF::Lattice>(k)) *
                    (GradientPressure<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) +
                     ChemicalForceMu<Lee, Gradient>::computeQ<TTraitsF>(idx, k)));
    }
};

template <int N>
class MuNCompSourceLocal : public ForceBase<LeeMuLocal> {
   public:
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return LaplacianChemicalPotential<N>::template get<typename TTraitsF::Lattice>(k) * mMobility;
    }
    void setMobility(double mobility) { mMobility = mobility; }
    void setBetaInterfaceWidth(double beta, double intefacewidth) {
        mMobility = intefacewidth * mBeta * mMobility / beta / mInterfaceWidth;
        mBeta = beta;
        mInterfaceWidth = intefacewidth;
    }

   private:
    double mBeta = 0.025;
    double mInterfaceWidth = 4;
    double mMobility = mInterfaceWidth * 0.02 / mBeta / 12.0;
};

template <int N>
class MuNCompSourceNonLocal : public ForceBase<LeeMuNonLocal> {
   public:
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        using data = Data_Base<typename TTraitsF::Lattice, typename TTraitsF::Stencil>;

        for (int i : Geometry<typename TTraitsF::Lattice>::mFluidVals) {
            if (Geometry<typename TTraitsF::Lattice>::getBoundaryType(
                    data::getInstance().getNeighbors()[k * TTraitsF::Stencil::Q + idx]) == i) {
                return LaplacianChemicalPotential<N>::template get<typename TTraitsF::Lattice>(
                           data::getInstance().getNeighbors()[k * TTraitsF::Stencil::Q + idx]) *
                       mMobility;
            }
        }
        return LaplacianChemicalPotential<N>::template get<typename TTraitsF::Lattice>(k) * mMobility;
    }
    template <class TTraits>
    inline void communicate() {
        using Lattice = typename TTraits::Lattice;
        Lattice::template communicate<ChemicalPotential, TTraits::NumberOfComponents>();
        Lattice::template communicate<LaplacianChemicalPotential, TTraits::NumberOfComponents>();
    }

    void setMobility(double mobility) { mMobility = mobility; }
    void setBetaInterfaceWidth(double beta, double intefacewidth) {
        mMobility = intefacewidth * mBeta * mMobility / beta / mInterfaceWidth;
        mBeta = beta;
        mInterfaceWidth = intefacewidth;
    }

   private:
    double mBeta = 0.025;
    double mInterfaceWidth = 4;
    double mMobility = mInterfaceWidth * 0.02 / mBeta / 12.0;
};

template <class TLattice, int N, int Nmax>
using DefaultTraitNCompLee =
    typename DefaultTrait<TLattice, Nmax>::template SetBoundary<BounceBack>::template SetProcessor<
        GradientsMultiInstance<OrderParameter, Nmax - 1, CentralXYZMirrorSolid, CentralQMirrorSolid,
                               MixedXYZMirrorSolid, MixedQMirrorSolid, LaplacianCentralMirrorSolid>>::
        template AddProcessor<
            std::tuple<GradientsMultiInstance<ChemicalPotential, Nmax, CentralXYZMirrorSolid, CentralQMirrorSolid,
                                              MixedXYZMirrorSolid, MixedQMirrorSolid, LaplacianCentralMirrorSolid>>>::
            template SetForce<LeeNCompSource<N>, MuNCompSourceLocal<N>, MuNCompSourceNonLocal<N>>;

template <class TLattice, int N, int Nmax, class TTraits = DefaultTraitNCompLee<TLattice, N, Nmax>>
class NCompLee : public CollisionBase<TLattice, typename TTraits::Stencil>,
                 public ModelBase<TLattice, TTraits> {  // Inherit from base class to avoid repetition of common
    // calculations

    using Stencil = typename TTraits::Stencil;
    static constexpr int mNDIM = TLattice::NDIM;

   public:
    virtual inline void collide() override;  // Collision step

    virtual inline void initialise() override;  // Initialisation step

    virtual inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    inline double computeEquilibrium(int k, int idx) override;  // Calculate equilibrium in direction idx

    template <typename... TTaus>
    inline void setTaus(TTaus... tau) {
        static_assert(sizeof...(TTaus) == TTraits::NumberOfComponents,
                      "Number of relaxation times must correspond to the number of components.");

        std::vector<double> t{{tau...}};
        mv_Tau = t;

        mTauMax = *std::max_element(std::begin(t), std::end(t));
        mTauMin = *std::min_element(std::begin(t), std::end(t));
    }

    inline void setTaus(std::vector<double> tau) {
        if ((int)tau.size() != TTraits::NumberOfComponents)
            throw std::runtime_error("Number of relaxation times must correspond to the number of components.");
        mv_Tau = tau;

        mTauMax = *std::max_element(std::begin(mv_Tau), std::end(mv_Tau));
        mTauMin = *std::min_element(std::begin(mv_Tau), std::end(mv_Tau));
    }

   private:
    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions

    double mGamma = 1;

    std::vector<double> mv_Tau{};

    std::vector<double>& density = Density<>::get<TLattice>();  // Reference to vector of order parameters
    std::vector<double>& velocity =
        Velocity<>::get<TLattice, TTraits::Lattice::NDIM>();    // Reference to vector of velocities
    std::vector<double>& itau = InverseTau<>::get<TLattice>();  // Reference to vector of velocities

    std::tuple<LeeNCompSourceEquilibrium<N>> mt_ModelForceEq;

    using data = Data_Base<TLattice, typename TTraits::Stencil>;
    data& mData = data::getInstance();

    double mTauMin = 1;
    double mTauMax = 1;

    thread_local static Lee mForcingScheme;
};

template <class TLattice, int N, int Nmax, class TTraits>
thread_local Lee NCompLee<TLattice, N, Nmax, TTraits>::mForcingScheme;

template <class TLattice, int N, int Nmax, class TTraits>
inline void NCompLee<TLattice, N, Nmax, TTraits>::collide() {
    TLattice::template communicate<LaplacianChemicalPotential, Nmax>();
#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        if (this->isCollisionNode(k)) {
            double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

            double equilibriums[Stencil::Q] = {};
            mForcingScheme.reset();
            mForcingScheme.precompute<TTraits>(std::get<0>(mt_ModelForceEq), k);

            for (int idx = 0; idx < Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
            }

            this->collisionQ(equilibriums, old_distribution, InverseTau<>::get<TLattice>(k), k);
        }
    }
}

template <class TLattice, int N, int Nmax, class TTraits>
inline void NCompLee<TLattice, N, Nmax, TTraits>::initialise() {  // Initialise model
    this->initialiseProcessors();

    ModelBase<TLattice, TTraits>::mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<Stencil>::template initialise<TLattice>(this->mt_Forces, mTauMin, mTauMax);

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        ChemicalPotential<0>::template initialise<TLattice>(0, k);
        ChemicalPotential<1>::template initialise<TLattice>(0, k);
        ChemicalPotential<2>::template initialise<TLattice>(0, k);
        OrderParameter<N>::template initialise<TLattice>(0.0, k);
        // OrderParameter<1>::initialise<TLattice>(0.0,k);
        Density<>::initialise<TLattice>(1.0, k);
        InverseTau<>::initialise<TLattice>(1.0, k);
        Velocity<>::initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, x);
        if constexpr (TLattice::NDIM >= 2) Velocity<>::initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, y);
        if constexpr (TLattice::NDIM == 3) Velocity<>::initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, z);
    }
    TLattice::template communicate<ChemicalPotential, Nmax>();
    TLattice::template communicate<OrderParameter, Nmax - 1>();
    ModelBase<TLattice, TTraits>::mData.communicate(
        BoundaryLabels<TTraits::Lattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TTraits::Lattice::NDIM>());
#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {
        double* distribution = this->mDistribution.getDistributionPointer(k);
        double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

        mForcingScheme.precompute<TTraits>(std::get<0>(mt_ModelForceEq), k);

        for (int idx = 0; idx < TTraits::Stencil::Q; ++idx) {
            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }
    }
}

template <class TLattice, int N, int Nmax, class TTraits>
inline void NCompLee<TLattice, N, Nmax, TTraits>::computeMomenta() {  // Calculate order parameter

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);

            OrderParameter<N>::template get<TLattice>(k) = this->computeDensity(distribution, k);
        }
    }

    TLattice::template communicate<ChemicalPotential, Nmax>();
    TLattice::template communicate<LaplacianChemicalPotential, Nmax>();
    TLattice::template communicate<OrderParameter, Nmax - 1>();
    this->mData.communicate(BoundaryLabels<TLattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, int N, int Nmax, class TTraits>
inline double NCompLee<TLattice, N, Nmax, TTraits>::computeEquilibrium(int k, int idx) {
    double force = mForcingScheme.compute<TTraits>(idx, k);
    double gamma =
        CollisionBase<TLattice, typename TTraits::Stencil>::computeGamma(&velocity[k * TTraits::Stencil::D], idx);

    return OrderParameter<N>::template get<TLattice>(k) * gamma + force;
}

// FlowField.hh: Contains the details of the LBM model to solve the Navier-Stokes and continuity equation. Each
// Model is given a "TTraits" class that contains stencil, data, force and boundary information

class PressureNCompLeeForce : public ChemicalForceMu<Lee, GradientMixed> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return (TTraitsF::Stencil::Cs2 *
                    MixedGradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) -
                ChemicalForceMu<Lee, GradientMixed>::computeXYZ<TTraitsF>(xyz, k));
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return (TTraitsF::Stencil::Cs2 *
                    MixedGradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) -
                ChemicalForceMu<Lee, GradientMixed>::computeQ<TTraitsF>(idx, k));
    }
    template <class TTraitsF>
    inline double computeDensitySource(int k) {  // SHOULD BE CENTRAL GRADIENTS
        double source = 0;
        for (int xyz = 0; xyz < TTraitsF::Lattice::NDIM; xyz++)
            source += TTraitsF::Lattice::DT * 0.5 * TTraitsF::Stencil::Cs2 *
                      GradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) *
                      Velocity<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz);
        return source;
    }
    template <class TTraitsF>
    inline double computeVelocitySource(const int xyz, const int k) {  // Need to correct velocity

        return -TTraitsF::Stencil::Cs2 * mChemForceCentral.computeXYZ<TTraitsF>(xyz, k) * TTraitsF::Lattice::DT / (2.0);
    }

   private:
    ChemicalForceMu<Lee, Gradient> mChemForceCentral;
};

class PressureNCompLeeForce2 : public ForceBase<LeeGamma0> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return -TTraitsF::Stencil::Cs2 *
               MixedGradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz);
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return -TTraitsF::Stencil::Cs2 *
               MixedGradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx);
    }
};

class PressureNCompLeeForceEquilibrium : public ChemicalForceMu<Lee, Gradient> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return -0.5 * (TTraitsF::Stencil::Cs2 *
                           GradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) -
                       ChemicalForceMu<Lee, Gradient>::computeXYZ<TTraitsF>(xyz, k));
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return -0.5 * (TTraitsF::Stencil::Cs2 *
                           GradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) -
                       ChemicalForceMu<Lee, Gradient>::computeQ<TTraitsF>(idx, k));
    }
};

class PressureNCompLeeForce2Equilibrium : public ForceBase<LeeGamma0> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return 0.5 * TTraitsF::Stencil::Cs2 *
               GradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz);
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return 0.5 * TTraitsF::Stencil::Cs2 *
               GradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx);
    }
};

template <class TLattice, int Nmax>
using DefaultTraitPressureNCompLee =
    typename DefaultTrait<TLattice, Nmax>::template SetBoundary<BounceBack>  // template SetBoundary<BounceBack>
    ::template SetProcessor<
        GradientsMultiStencil<Density<>, CentralXYZBounceBack, CentralQBounceBack, MixedXYZBounceBack,
                              MixedQBounceBack>,
        GradientsMultiStencil<Pressure<>, CentralXYZBounceBack, CentralQBounceBack, MixedXYZBounceBack,
                              MixedQBounceBack>>::template SetForce<PressureNCompLeeForce, PressureNCompLeeForce2>;

template <class TLattice, int Nmax, class TTraits = DefaultTraitPressureNCompLee<TLattice, Nmax>>
class PressureNCompLee : public FlowFieldPressure<TLattice, TTraits> {  // Inherit from base class to avoid repetition
                                                                        // of common calculations

   public:
    PressureNCompLee() : mv_Tau(Nmax, 1.), mv_Density(Nmax, 1.) {}

    template <typename... TTaus>
    inline void setTaus(TTaus... tau) {
        static_assert(sizeof...(TTaus) == TTraits::NumberOfComponents,
                      "Number of relaxation times must correspond to the number of components.");

        std::vector<double> t{{tau...}};
        mv_Tau = t;

        mTauMax = *std::max_element(std::begin(t), std::end(t));
        mTauMin = *std::min_element(std::begin(t), std::end(t));
    }

    inline void setTaus(std::vector<double> tau) {
        if ((int)tau.size() != TTraits::NumberOfComponents)
            throw std::runtime_error("Number of relaxation times must correspond to the number of components.");
        mv_Tau = tau;

        mTauMax = *std::max_element(std::begin(mv_Tau), std::end(mv_Tau));
        mTauMin = *std::min_element(std::begin(mv_Tau), std::end(mv_Tau));
    }

    template <typename... TDensities>
    inline void setDensities(TDensities... density) {
        static_assert(sizeof...(TDensities) == TTraits::NumberOfComponents,
                      "Number of relaxation times must correspond to the number of components.");

        std::vector<double> d{{density...}};
        mv_Density = d;
    }

    inline void setDensities(std::vector<double> density) {
        if ((int)density.size() != TTraits::NumberOfComponents)
            throw std::runtime_error("Number of relaxation times must correspond to the number of components.");

        mv_Density = density;
    }

    virtual inline void collide() override;  // Collision step

    virtual inline void initialise() override;  // Initialisation step

    virtual inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    inline double computeEquilibrium(int k, int idx) override;  // Calculate equilibrium in direction idx

   private:
    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions

    std::tuple<PressureNCompLeeForceEquilibrium> mt_ModelForceEq;
    std::tuple<PressureNCompLeeForce2Equilibrium> mt_ModelForce2Eq;

    std::vector<double>& density = Density<>::get<TLattice>();    // Reference to vector of order parameters
    std::vector<double>& pressure = Pressure<>::get<TLattice>();  // Reference to vector of order parameters
    std::vector<double>& velocity =
        Velocity<>::get<TLattice, TTraits::Lattice::NDIM>();    // Reference to vector of velocities
    std::vector<double>& itau = InverseTau<>::get<TLattice>();  // Reference to vector of velocities

    std::vector<double> mv_Tau{};

    std::vector<double> mv_Density{};

    double mTauMin = 1;
    double mTauMax = 1;

    thread_local static Lee mForcingScheme1;
    thread_local static LeeGamma0 mForcingScheme2;
};

template <class TLattice, int Nmax, class TTraits>
thread_local Lee PressureNCompLee<TLattice, Nmax, TTraits>::mForcingScheme1;
template <class TLattice, int Nmax, class TTraits>
thread_local LeeGamma0 PressureNCompLee<TLattice, Nmax, TTraits>::mForcingScheme2;

template <class TLattice, int Nmax, class TTraits>
inline void PressureNCompLee<TLattice, Nmax, TTraits>::collide() {  // Collision step

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        if (this->isCollisionNode(k)) {
            double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

            double equilibriums[TTraits::Stencil::Q] = {};

            mForcingScheme1.reset();
            mForcingScheme1.precompute<TTraits>(std::get<0>(mt_ModelForceEq), k);
            mForcingScheme2.reset();
            mForcingScheme2.precompute<TTraits>(std::get<0>(mt_ModelForce2Eq), k);

            for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
            }

            this->collisionQ(equilibriums, old_distribution, InverseTau<>::get<TLattice>(k), k);
        }
    }
}

template <class TLattice, int Nmax, class TTraits>
inline void PressureNCompLee<TLattice, Nmax, TTraits>::initialise() {  // Initialise model
    this->initialiseProcessors();

    ModelBase<TLattice, TTraits>::mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<typename TTraits::Stencil>::template initialise<TLattice>(this->mt_Forces, mTauMin,
                                                                                               mTauMax);

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        Pressure<>::initialise<TLattice>(1.0, k);  // Set density to 1 initially (This will change)
        Velocity<>::initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, x);
        if constexpr (TLattice::NDIM >= 2) Velocity<>::initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, y);
        if constexpr (TLattice::NDIM == 3) Velocity<>::initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, z);

        double dens = 0;
        double sumorderparameter = 0;
        for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
            double orderParameter =
                getInstance<OrderParameter, TTraits::NumberOfComponents - 1, typename TTraits::Lattice>(i)[k];
            sumorderparameter += orderParameter;
            dens += (orderParameter)*mv_Density[i];
        }
        dens += (1.0 - sumorderparameter) * mv_Density.back();

        Density<>::initialise<TLattice>(dens, k);

        double invtau = 0;
        sumorderparameter = 0;
        for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
            double orderParameter =
                getInstance<OrderParameter, TTraits::NumberOfComponents - 1, typename TTraits::Lattice>(i)[k];
            sumorderparameter += orderParameter;
            invtau += (orderParameter) / mv_Tau[i];
        }
        invtau += (1.0 - sumorderparameter) / mv_Tau.back();
        InverseTau<>::initialise<TLattice>(invtau, k);

        PressureOld<>::initialise<TLattice>(1.0, k);  // Set density to 1 initially (This will change)
    }

    TLattice::template communicate<ChemicalPotential, Nmax>();
    TLattice::template communicate<OrderParameter, Nmax - 1>();
    TLattice::template communicate<GradientOrderParameter, Nmax - 1, TLattice::NDIM>();
    ModelBase<TLattice, TTraits>::mData.communicate(BoundaryLabels<TLattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {
        double* distribution = this->mDistribution.getDistributionPointer(k);
        double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

        mForcingScheme1.precompute<TTraits>(std::get<0>(mt_ModelForceEq), k);
        mForcingScheme2.precompute<TTraits>(std::get<0>(mt_ModelForce2Eq), k);
        for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TTraits::Lattice::NDIM>());
}

template <class TLattice, int Nmax, class TTraits>
inline void PressureNCompLee<TLattice, Nmax, TTraits>::computeMomenta() {  // Calculate Density<> and Velocity

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);

            this->velocity[k * TTraits::Stencil::D + x] =
                1. / (TTraits::Stencil::Cs2) *
                this->computeVelocity(distribution, this->mt_Forces, this->density[k], x, k);  // Calculate velocities
            if constexpr (TLattice::NDIM >= 2)
                this->velocity[k * TTraits::Stencil::D + y] =
                    1. / (TTraits::Stencil::Cs2) *
                    this->computeVelocity(distribution, this->mt_Forces, this->density[k], y, k);
            if constexpr (TLattice::NDIM == 3)
                this->velocity[k * TTraits::Stencil::D + z] =
                    1. / (TTraits::Stencil::Cs2) *
                    this->computeVelocity(distribution, this->mt_Forces, this->density[k], z, k);

            this->pressure[k] = this->computeDensity(distribution, k);  // Calculate density

            double dens = 0;
            double sumorderparameter = 0;
            for (int i = 0; i < TTraits::NumberOfComponents - 1; i++) {
                double orderParameter =
                    getInstance<OrderParameter, TTraits::NumberOfComponents - 1, typename TTraits::Lattice>(i)[k];
                sumorderparameter += orderParameter;
                dens += (orderParameter)*mv_Density[i];
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
                double orderParameter =
                    getInstance<OrderParameter, TTraits::NumberOfComponents - 1, typename TTraits::Lattice>(i)[k];
                sumorderparameter += orderParameter;
                invtau += (orderParameter) / mv_Tau[i];
            }
            invtau += (1.0 - sumorderparameter) / mv_Tau.back();
            itau[k] = invtau;
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
}

template <class TLattice, int Nmax, class TTraits>
inline double PressureNCompLee<TLattice, Nmax, TTraits>::computeEquilibrium(int k, int idx) {
    double force = mForcingScheme1.compute<TTraits>(idx, k) + mForcingScheme2.compute<TTraits>(idx, k);
    double velocityFactor = CollisionBase<TLattice, typename TTraits::Stencil>::computeVelocityFactor(
        &velocity[k * TTraits::Stencil::D], idx);
    // Equilibrium is density times gamma in this case
    return TTraits::Stencil::Weights[idx] * (pressure[k] + density[k] * TTraits::Stencil::Cs2 * velocityFactor) + force;
}
