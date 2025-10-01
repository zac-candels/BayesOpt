#pragma once
#include <utility>

#include "../Collide.hh"
#include "../Data.hh"
#include "../Parameters.hh"
#include "FlowField.hh"
#include "ModelBase.hh"

// BinaryLee.hh: Contains the details of the LBM model to solve an equation for phase separation. Each
// Model is given a "TTraits" class that contains stencil, data, force and boundary information

template <int N, template<class, template<class> class> class chemicalforce = ChemicalForce>
class LeeTernarySource : public chemicalforce<Lee, GradientMixed> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        using Lattice = typename TTraitsF::Lattice;
        double gradOP = MixedGradientOrderParameter<N>::template get<Lattice, Lattice::NDIM>(k, xyz);
        double orderParam = OrderParameter<N>::template get<Lattice>(k);
        double density = Density<>::template get<Lattice>(k);
        double gradP = MixedGradientPressure<>::template get<Lattice, Lattice::NDIM>(k, xyz);
        double force = chemicalforce<Lee, GradientMixed>::template computeXYZ<TTraitsF>(xyz, k);
        return gradOP - orderParam / (TTraitsF::Stencil::Cs2 * density) * (gradP - force);
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        using Lattice = typename TTraitsF::Lattice;
        double gradOP = MixedGradientOrderParameter<N>::template get<Lattice, TTraitsF::Stencil::Q>(k, idx);
        double orderParam = OrderParameter<N>::template get<Lattice>(k);
        double density = Density<>::template get<Lattice>(k);
        double gradP = MixedGradientPressure<>::template get<Lattice, TTraitsF::Stencil::Q>(k, idx);
        double force = chemicalforce<Lee, GradientMixed>::template computeQ<TTraitsF>(idx, k);
        return gradOP - orderParam / (TTraitsF::Stencil::Cs2 * density) * (gradP - force);
    }
};

template <int N, template<class, template<class> class> class chemicalforce = ChemicalForce>
class LeeTernarySourceEquilibrium : public chemicalforce<Lee, Gradient> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return -0.5 *
               (GradientOrderParameter<N>::template get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) -
                OrderParameter<N>::template get<typename TTraitsF::Lattice>(k) /
                    (TTraitsF::Stencil::Cs2 * Density<>::template get<typename TTraitsF::Lattice>(k)) *
                    (GradientPressure<>::template get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) -
                     chemicalforce<Lee, Gradient>::template computeXYZ<TTraitsF>(xyz, k)));
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return -0.5 *
               (GradientOrderParameter<N>::template get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) -
                OrderParameter<N>::template get<typename TTraitsF::Lattice>(k) /
                    (TTraitsF::Stencil::Cs2 * Density<>::template get<typename TTraitsF::Lattice>(k)) *
                    (GradientPressure<>::template get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) -
                     chemicalforce<Lee, Gradient>::template computeQ<TTraitsF>(idx, k)));
    }
};

template <int N>
class MuTernarySourceLocal : public ForceBase<LeeMuLocal> {
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
    double mMobility = mInterfaceWidth * 0.02 / mBeta / 6.0;
};

template <int N>
class MuTernarySourceNonLocal : public ForceBase<LeeMuNonLocal> {
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
        using data = Data_Base<typename TTraits::Lattice, typename TTraits::Stencil>;
        data::getInstance().communicate(ChemicalPotential<0>::template getInstance<typename TTraits::Lattice>());
        data::getInstance().communicate(ChemicalPotential<1>::template getInstance<typename TTraits::Lattice>());
        data::getInstance().communicate(ChemicalPotential<2>::template getInstance<typename TTraits::Lattice>());
        data::getInstance().communicate(
            LaplacianChemicalPotential<0>::template getInstance<typename TTraits::Lattice>());
        data::getInstance().communicate(
            LaplacianChemicalPotential<1>::template getInstance<typename TTraits::Lattice>());
        data::getInstance().communicate(
            LaplacianChemicalPotential<2>::template getInstance<typename TTraits::Lattice>());
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
    double mMobility = mInterfaceWidth * 0.02 / mBeta / 6.0;
};

template <class TLattice, int N>
using DefaultTraitTernaryLee =
    typename DefaultTrait<TLattice, 3>::template SetBoundary<BounceBack>::template SetProcessor<
        GradientsMultiInstance<OrderParameter, 2, CentralXYZMirrorSolid, CentralQMirrorSolid, MixedXYZMirrorSolid,
                               MixedQMirrorSolid, LaplacianCentralMirrorSolid>>::
        template AddProcessor<std::tuple<GradientsMultiInstance<ChemicalPotential, 3, LaplacianCentralMirrorSolid>>>::
            template SetForce<LeeTernarySource<N>, MuTernarySourceLocal<N>, MuTernarySourceNonLocal<N>>;

template <class TLattice, int N, class TTraits = DefaultTraitTernaryLee<TLattice, N>>
class TernaryLee : public CollisionBase<TLattice, typename TTraits::Stencil>,
                   public ModelBase<TLattice, TTraits> {  // Inherit from base class to avoid repetition of common
    // calculations

    using Stencil = typename TTraits::Stencil;
    static constexpr int mNDIM = TLattice::NDIM;

   public:
    inline void setTau1(double val) { mTau1 = val; }
    inline void setTau2(double val) { mTau2 = val; }
    inline void setTau3(double val) { mTau3 = val; }

    virtual inline void collide() override;  // Collision step

    virtual inline void initialise() override;  // Initialisation step

    virtual inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    inline double computeEquilibrium(int k, int idx) override;  // Calculate equilibrium in direction idx

    /**
     * \brief Function to compute the fractional concentration of each fluid (between 0 and 1).
     * \param k The index of the node on the lattice.
     * \param iFluid The index of the fluid (0, 1, or 2).
     */
    inline double computeConcentration(int k, int iFluid) override;

    template <class, class>
    friend class TernaryLeeHumidity;

   private:
    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions

    double mGamma = 1;

    double mTau1 = 1;
    double mTau2 = 1;
    double mTau3 = 1;

    std::vector<double>& density = Density<>::template get<TLattice>();  // Reference to vector of order parameters
    std::vector<double>& orderparameter1 = OrderParameter<0>::template get<TLattice>();
    std::vector<double>& orderparameter2 = OrderParameter<1>::template get<TLattice>();
    std::vector<double>& velocity =
        Velocity<>::template get<TLattice, TTraits::Lattice::NDIM>();    // Reference to vector of velocities
    std::vector<double>& itau = InverseTau<>::template get<TLattice>();  // Reference to vector of velocities

    std::tuple<LeeTernarySourceEquilibrium<N>> mt_ModelForceEq;

    using data = Data_Base<TLattice, typename TTraits::Stencil>;
    data& mData = data::getInstance();

    thread_local static Lee mForcingScheme;
};

template <class TLattice, int N, class TTraits>
thread_local Lee TernaryLee<TLattice, N, TTraits>::mForcingScheme;

template <class TLattice, int N, class TTraits>
inline void TernaryLee<TLattice, N, TTraits>::collide() {
    ModelBase<TLattice, TTraits>::mData.communicate(LaplacianChemicalPotential<0>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(LaplacianChemicalPotential<1>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(LaplacianChemicalPotential<2>::template getInstance<TLattice>());
#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        if (this->isCollisionNode(k)) {
            double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

            double equilibriums[Stencil::Q] = {};
            mForcingScheme.reset();
            mForcingScheme.precompute<TTraits>(std::template get<0>(mt_ModelForceEq), k);

            for (int idx = 0; idx < Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
            }

            this->collisionQ(equilibriums, old_distribution, InverseTau<>::template get<TLattice>(k), k);
        }
    }
}

template <class TLattice, int N, class TTraits>
inline void TernaryLee<TLattice, N, TTraits>::initialise() {  // Initialise model
    this->initialiseProcessors();

    TLattice::ResetParallelTracking();
    ModelBase<TLattice, TTraits>::mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<Stencil>::template initialise<TLattice>(this->mt_Forces, mTau1, mTau2);

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        ChemicalPotential<0>::template initialise<TLattice>(0, k);
        ChemicalPotential<1>::template initialise<TLattice>(0, k);
        ChemicalPotential<2>::template initialise<TLattice>(0, k);
        OrderParameter<N>::template initialise<TLattice>(0.0, k);
        // Density<>::template initialise<TLattice>(1.0, k);
        // InverseTau<>::template initialise<TLattice>(1.0, k);
        Velocity<>::template initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, x);
        if constexpr (TLattice::NDIM >= 2) Velocity<>::template initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, y);
        if constexpr (TLattice::NDIM == 3) Velocity<>::template initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, z);
    }

    this->mData.communicate(ChemicalPotential<0>::template getInstance<TLattice>());
    this->mData.communicate(ChemicalPotential<1>::template getInstance<TLattice>());
    this->mData.communicate(ChemicalPotential<2>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(
        BoundaryLabels<TTraits::Lattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<0>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<1>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(
        Velocity<>::template getInstance<TLattice, TTraits::Lattice::NDIM>());
#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {
        double* distribution = this->mDistribution.getDistributionPointer(k);
        double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

        mForcingScheme.precompute<TTraits>(std::template get<0>(mt_ModelForceEq), k);

        for (int idx = 0; idx < TTraits::Stencil::Q; ++idx) {
            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }
    }
}

template <class TLattice, int N, class TTraits>
inline void TernaryLee<TLattice, N, TTraits>::computeMomenta() {  // Calculate order parameter
    TLattice::ResetParallelTracking();
#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);

            OrderParameter<N>::template get<TLattice>(k) = this->computeDensity(distribution, k);
        }
    }

    this->mData.communicate(ChemicalPotential<0>::template getInstance<TLattice>());
    this->mData.communicate(ChemicalPotential<1>::template getInstance<TLattice>());
    this->mData.communicate(ChemicalPotential<2>::template getInstance<TLattice>());
    this->mData.communicate(LaplacianChemicalPotential<0>::template getInstance<TLattice>());
    this->mData.communicate(LaplacianChemicalPotential<1>::template getInstance<TLattice>());
    this->mData.communicate(LaplacianChemicalPotential<2>::template getInstance<TLattice>());
    this->mData.communicate(BoundaryLabels<TLattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<0>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<1>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::template getInstance<TLattice>());
}

template <class TLattice, int N, class TTraits>
inline double TernaryLee<TLattice, N, TTraits>::computeEquilibrium(int k, int idx) {
    double force = mForcingScheme.compute<TTraits>(idx, k);
    double gamma =
        CollisionBase<TLattice, typename TTraits::Stencil>::computeGamma(&velocity[k * TTraits::Stencil::D], idx);
    return OrderParameter<N>::template get<TLattice>(k) * gamma + force;
}

template <class TLattice, int N, class TTraits>
inline double TernaryLee<TLattice, N, TTraits>::computeConcentration(int k, int iFluid) {
    if (iFluid == 0) {
        return orderparameter1[k];
    } else if (iFluid == 1) {
        return orderparameter2[k];
    } else if (iFluid == 2) {
        return 1 - orderparameter1[k] - orderparameter2[k];
    } else {
        throw std::invalid_argument("Invalid fluid number. Must be 0, 1, or 2.");
    }
}

// FlowField.hh: Contains the details of the LBM model to solve the Navier-Stokes and continuity equation. Each
// Model is given a "TTraits" class that contains stencil, data, force and boundary information

template<template<class, template<class> class> class chemicalforce = ChemicalForce>
class PressureTernaryLeeForce : public chemicalforce<Lee, GradientMixed> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return (TTraitsF::Stencil::Cs2 *
                    MixedGradientDensity<>::template get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) +
                chemicalforce<Lee, GradientMixed>::template computeXYZ<TTraitsF>(xyz, k));
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return (TTraitsF::Stencil::Cs2 *
                    MixedGradientDensity<>::template get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) +
                chemicalforce<Lee, GradientMixed>::template computeQ<TTraitsF>(idx, k));
    }
    template <class TTraitsF>
    inline double computeDensitySource(int k) {  // SHOULD BE CENTRAL GRADIENTS
        double source = 0;
        for (int xyz = 0; xyz < TTraitsF::Lattice::NDIM; xyz++)
            source += TTraitsF::Lattice::DT * 0.5 * TTraitsF::Stencil::Cs2 *
                      GradientDensity<>::template get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) *
                      Velocity<>::template get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz);
        return source;
    }
    template <class TTraitsF>
    inline double computeVelocitySource(const int xyz, const int k) {  // Need to correct velocity

        return TTraitsF::Stencil::Cs2 * mChemForceCentral.template computeXYZ<TTraitsF>(xyz, k) * TTraitsF::Lattice::DT / (2.0);
    }

   private:
    chemicalforce<Lee, Gradient> mChemForceCentral;
};

class PressureTernaryLeeForce2 : public ForceBase<LeeGamma0> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return -TTraitsF::Stencil::Cs2 *
               MixedGradientDensity<>::template get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz);
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return -TTraitsF::Stencil::Cs2 *
               MixedGradientDensity<>::template get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx);
    }
};

template<template<class, template<class> class> class chemicalforce = ChemicalForce>
class PressureTernaryLeeForceEquilibrium : public chemicalforce<Lee, Gradient> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return -0.5 *
               (TTraitsF::Stencil::Cs2 *
                    GradientDensity<>::template get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) +
                chemicalforce<Lee, Gradient>::template computeXYZ<TTraitsF>(xyz, k));
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return -0.5 * (TTraitsF::Stencil::Cs2 *
                           GradientDensity<>::template get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) +
                       chemicalforce<Lee, Gradient>::template computeQ<TTraitsF>(idx, k));
    }
};

class PressureTernaryLeeForce2Equilibrium : public ForceBase<LeeGamma0> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return 0.5 * TTraitsF::Stencil::Cs2 *
               GradientDensity<>::template get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz);
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return 0.5 * TTraitsF::Stencil::Cs2 *
               GradientDensity<>::template get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx);
    }
};

template <class TLattice>
using DefaultTraitPressureTernaryLee =
    typename DefaultTrait<TLattice, 3>::template SetBoundary<BounceBack>  // template SetBoundary<BounceBack>
    ::template SetProcessor<
        GradientsMultiStencil<Density<>, CentralXYZBounceBack, CentralQBounceBack, MixedXYZBounceBack,
                              MixedQBounceBack>,
        GradientsMultiStencil<Pressure<>, CentralXYZBounceBack, CentralQBounceBack, MixedXYZBounceBack,
                              MixedQBounceBack>>::template SetForce<PressureTernaryLeeForce<>, PressureTernaryLeeForce2>;

template <class TLattice, class TTraits = DefaultTraitPressureTernaryLee<TLattice>>
class PressureTernaryLee : public FlowFieldPressure<TLattice, TTraits> {  // Inherit from base class to avoid repetition
                                                                          // of common calculations

   public:
    inline void setTau1(double val) { mTau1 = val; }
    inline void setTau2(double val) { mTau2 = val; }
    inline void setTau3(double val) { mTau3 = val; }

    void setDensity1(double density) { mDensityC1 = density; }

    void setDensity2(double density) { mDensityC2 = density; }

    void setDensity3(double density) { mDensityC3 = density; }

    virtual inline void collide() override;  // Collision step

    virtual inline void initialise() override;  // Initialisation step

    virtual inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    inline double computeEquilibrium(int k, int idx) override;  // Calculate equilibrium in direction idx

    /**
     * \brief Function to compute the fractional concentration of each fluid (between 0 and 1).
     * \param k The index of the node on the lattice.
     * \param iFluid The index of the fluid (0, 1, or 2).
     */
    inline double computeConcentration(int k, int iFluid) override;

    template <class, class>
    friend class PressureTernaryLeeHumidity;

   private:
    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions

    std::tuple<PressureTernaryLeeForceEquilibrium<>> mt_ModelForceEq;
    std::tuple<PressureTernaryLeeForce2Equilibrium> mt_ModelForce2Eq;

    std::vector<double>& density = Density<>::template get<TLattice>();    // Reference to vector of order parameters
    std::vector<double>& pressure = Pressure<>::template get<TLattice>();  // Reference to vector of order parameters
    std::vector<double>& velocity =
        Velocity<>::template get<TLattice, TTraits::Lattice::NDIM>();    // Reference to vector of velocities
    std::vector<double>& itau = InverseTau<>::template get<TLattice>();  // Reference to vector of velocities

    double mTau1 = 1;
    double mTau2 = 1;
    double mTau3 = 1;

    double mDensityC1 = 1;
    double mDensityC2 = 1;
    double mDensityC3 = 1;

    thread_local static Lee mForcingScheme1;
    thread_local static LeeGamma0 mForcingScheme2;
};

template <class TLattice, class TTraits>
thread_local Lee PressureTernaryLee<TLattice, TTraits>::mForcingScheme1;
template <class TLattice, class TTraits>
thread_local LeeGamma0 PressureTernaryLee<TLattice, TTraits>::mForcingScheme2;

template <class TLattice, class TTraits>
inline void PressureTernaryLee<TLattice, TTraits>::collide() {  // Collision step

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        if (this->isCollisionNode(k)) {
            double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

            double equilibriums[TTraits::Stencil::Q] = {};

            mForcingScheme1.reset();
            mForcingScheme1.precompute<TTraits>(std::template get<0>(mt_ModelForceEq), k);
            mForcingScheme2.reset();
            mForcingScheme2.precompute<TTraits>(std::template get<0>(mt_ModelForce2Eq), k);

            for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
            }

            this->collisionQ(equilibriums, old_distribution, InverseTau<>::template get<TLattice>(k), k);
        }
    }
}

template <class TLattice, class TTraits>
inline void PressureTernaryLee<TLattice, TTraits>::initialise() {  // Initialise model
    this->initialiseProcessors();

    ModelBase<TLattice, TTraits>::mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<typename TTraits::Stencil>::template initialise<TLattice>(this->mt_Forces, mTau1,
                                                                                               mTau2);

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        Pressure<>::template initialise<TLattice>(1.0, k);  // Set density to 1 initially (This will change)
        Velocity<>::template initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, x);
        if constexpr (TLattice::NDIM >= 2) Velocity<>::template initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, y);
        if constexpr (TLattice::NDIM == 3) Velocity<>::template initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, z);

        Density<>::template initialise<TLattice>(
            (mDensityC3 + (OrderParameter<0>::template get<TLattice>(k)) * (mDensityC1 - mDensityC3) +
             (OrderParameter<1>::template get<TLattice>(k)) * (mDensityC2 - mDensityC3)),
            k);
        InverseTau<>::template initialise<TLattice>(
            1.0 / (mTau3 + (OrderParameter<0>::template get<TLattice>(k)) * (mTau1 - mTau3) +
                   (OrderParameter<1>::template get<TLattice>(k)) * (mTau2 - mTau3)),
            k);

        // PressureOld<>::template initialise<TLattice>(1.0, k);  // Set density to 1 initially (This will change)
    }

    ModelBase<TLattice, TTraits>::mData.communicate(
        GradientOrderParameter<0>::template getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(
        GradientOrderParameter<1>::template getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(BoundaryLabels<TLattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<0>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<1>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(ChemicalPotential<0>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(ChemicalPotential<1>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(ChemicalPotential<2>::template getInstance<TLattice>());

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {
        double* distribution = this->mDistribution.getDistributionPointer(k);
        double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

        mForcingScheme1.precompute<TTraits>(std::template get<0>(mt_ModelForceEq), k);
        mForcingScheme2.precompute<TTraits>(std::template get<0>(mt_ModelForce2Eq), k);
        for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(
        Velocity<>::template getInstance<TLattice, TTraits::Lattice::NDIM>());
}

template <class TLattice, class TTraits>
inline void PressureTernaryLee<TLattice, TTraits>::computeMomenta() {  // Calculate Density<> and Velocity

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);

            density[k] = mDensityC3 + (OrderParameter<0>::template get<TLattice>(k)) * (mDensityC1 - mDensityC3) +
                         (OrderParameter<1>::template get<TLattice>(k)) * (mDensityC2 - mDensityC3);

            itau[k] = 1.0 / (mTau3 + (OrderParameter<0>::template get<TLattice>(k)) * (mTau1 - mTau3) +
                             (OrderParameter<1>::template get<TLattice>(k)) * (mTau2 - mTau3));

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

            
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::template getInstance<TLattice, TLattice::NDIM>());
}

template <class TLattice, class TTraits>
inline double PressureTernaryLee<TLattice, TTraits>::computeEquilibrium(int k, int idx) {
    double force = mForcingScheme1.compute<TTraits>(idx, k) + mForcingScheme2.compute<TTraits>(idx, k);
    double velocityFactor = CollisionBase<TLattice, typename TTraits::Stencil>::computeVelocityFactor(
        &velocity[k * TTraits::Stencil::D], idx);
    return TTraits::Stencil::Weights[idx] * (pressure[k] + density[k] * TTraits::Stencil::Cs2 * velocityFactor) +
           force;  // Equilibrium is density times gamma in this case
}

template <class TLattice, class TTraits>
inline double PressureTernaryLee<TLattice, TTraits>::computeConcentration(int k, int iFluid) {
    std::vector<double>& orderparameter1 = OrderParameter<0>::template get<TLattice>();
    std::vector<double>& orderparameter2 = OrderParameter<1>::template get<TLattice>();
    if (iFluid == 0) {
        return orderparameter1[k];
    } else if (iFluid == 1) {
        return orderparameter2[k];
    } else if (iFluid == 2) {
        return 1 - orderparameter1[k] - orderparameter2[k];
    } else {
        throw std::invalid_argument("Invalid fluid number. Must be 0, 1, or 2.");
    }
}
