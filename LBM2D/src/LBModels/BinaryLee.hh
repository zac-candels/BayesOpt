#pragma once
#include <utility>

#include "../Collide.hh"
#include "../Data.hh"
#include "../Parameters.hh"
#include "FlowField.hh"
#include "ModelBase.hh"

// BinaryLee.hh: Contains the details of the LBM model to solve an equation for phase separation. Each
// Model is given a "TTraits" class that contains stencil, data, force and boundary information

class LeeBinarySource : public ChemicalForceBinary<Lee, GradientMixed> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return (MixedGradientOrderParameter<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) -
                OrderParameter<>::get<typename TTraitsF::Lattice>(k) /
                    (TTraitsF::Stencil::Cs2 * Density<>::get<typename TTraitsF::Lattice>(k)) *
                    (MixedGradientPressure<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) -
                     ChemicalForceBinary<Lee, GradientMixed>::computeXYZ<TTraitsF>(xyz, k)));
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return (MixedGradientOrderParameter<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) -
                OrderParameter<>::get<typename TTraitsF::Lattice>(k) /
                    (TTraitsF::Stencil::Cs2 * Density<>::get<typename TTraitsF::Lattice>(k)) *
                    (MixedGradientPressure<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) -
                     ChemicalForceBinary<Lee, GradientMixed>::computeQ<TTraitsF>(idx, k)));
    }
};

class LeeBinarySourceCentral : public ChemicalForceBinary<Lee, Gradient> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return (GradientOrderParameter<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) -
                OrderParameter<>::get<typename TTraitsF::Lattice>(k) /
                    (TTraitsF::Stencil::Cs2 * Density<>::get<typename TTraitsF::Lattice>(k)) *
                    (GradientPressure<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) -
                     ChemicalForceBinary<Lee, Gradient>::computeXYZ<TTraitsF>(xyz, k)));
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return (GradientOrderParameter<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) -
                OrderParameter<>::get<typename TTraitsF::Lattice>(k) /
                    (TTraitsF::Stencil::Cs2 * Density<>::get<typename TTraitsF::Lattice>(k)) *
                    (GradientPressure<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) -
                     ChemicalForceBinary<Lee, Gradient>::computeQ<TTraitsF>(idx, k)));
    }
};

class LeeBinarySourceEquilibrium : public ChemicalForceBinaryMu<Lee, Gradient> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return -0.5 * (GradientOrderParameter<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) -
                       OrderParameter<>::get<typename TTraitsF::Lattice>(k) /
                           (TTraitsF::Stencil::Cs2 * Density<>::get<typename TTraitsF::Lattice>(k)) *
                           (GradientPressure<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) -
                            ChemicalForceBinaryMu<Lee, Gradient>::computeXYZ<TTraitsF>(xyz, k)));
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return -0.5 * (GradientOrderParameter<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) -
                       OrderParameter<>::get<typename TTraitsF::Lattice>(k) /
                           (TTraitsF::Stencil::Cs2 * Density<>::get<typename TTraitsF::Lattice>(k)) *
                           (GradientPressure<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) -
                            ChemicalForceBinaryMu<Lee, Gradient>::computeQ<TTraitsF>(idx, k)));
    }
};

class MuSourceLocal : public ForceBase<LeeMuLocal> {
   public:
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return LaplacianChemicalPotential<>::get<typename TTraitsF::Lattice>(k) * mMobility;
    }
    void setMobility(double mobility) { mMobility = mobility; }
    void setBeta(double beta) {
        mMobility = mBeta * mMobility / beta;
        mBeta = beta;
    }

   private:
    double mBeta = 0.025;
    double mMobility = 0.02 / mBeta;
};

class MuSourceNonLocal : public ForceBase<LeeMuNonLocal> {
   public:
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        using data = Data_Base<typename TTraitsF::Lattice, typename TTraitsF::Stencil>;
        
        for (int i : Geometry<typename TTraitsF::Lattice>::mFluidVals) {
            if (Geometry<typename TTraitsF::Lattice>::getBoundaryType(
                    data::getInstance().getNeighbors()[k * TTraitsF::Stencil::Q + idx]) == i)
                return LaplacianChemicalPotential<>::get<typename TTraitsF::Lattice>(
                           data::getInstance().getNeighbors()[k * TTraitsF::Stencil::Q + idx]) *
                       mMobility;
        }
        return LaplacianChemicalPotential<>::get<typename TTraitsF::Lattice>(k) * mMobility;
        // return LaplacianChemicalPotential<>::get<typename TTraitsF::Lattice>(data::getInstance().getNeighbors()[k *
        // TTraitsF::Stencil::Q + idx])*mMobility;
    }
    template <class TTraits>
    inline void communicate() {
        using data = Data_Base<typename TTraits::Lattice, typename TTraits::Stencil>;
        data::getInstance().communicate(LaplacianChemicalPotential<>::getInstance<typename TTraits::Lattice>());
    }

    void setMobility(double mobility) { mMobility = mobility; }
    void setBeta(double beta) {
        mMobility = mBeta * mMobility / beta;
        mBeta = beta;
    }

   private:
    double mBeta = 0.025;
    double mMobility = 0.02 / mBeta;
};


class MuSourceNonLocalFullWay : public ForceBase<LeeMuNonLocal> {
   public:
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        using data = Data_Base<typename TTraitsF::Lattice, typename TTraitsF::Stencil>;
        
        for (int i : Geometry<typename TTraitsF::Lattice>::mFluidVals) {
            if (Geometry<typename TTraitsF::Lattice>::getBoundaryType(
                    data::getInstance().getNeighbors()[k * TTraitsF::Stencil::Q + idx]) == i)
                return LaplacianChemicalPotential<>::get<typename TTraitsF::Lattice>(
                           data::getInstance().getNeighbors()[k * TTraitsF::Stencil::Q + idx]) *
                       mMobility;
        }
        return LaplacianChemicalPotential<>::get<typename TTraitsF::Lattice>(data::getInstance().getNeighbors()[k * TTraitsF::Stencil::Q + TTraitsF::Stencil::Opposites[idx]]) * mMobility;
        // return LaplacianChemicalPotential<>::get<typename TTraitsF::Lattice>(data::getInstance().getNeighbors()[k *
        // TTraitsF::Stencil::Q + idx])*mMobility;
    }
    template <class TTraits>
    inline void communicate() {
        using data = Data_Base<typename TTraits::Lattice, typename TTraits::Stencil>;
        data::getInstance().communicate(LaplacianChemicalPotential<>::getInstance<typename TTraits::Lattice>());
    }

    void setMobility(double mobility) { mMobility = mobility; }
    void setBeta(double beta) {
        mMobility = mBeta * mMobility / beta;
        mBeta = beta;
    }

   private:
    double mBeta = 0.025;
    double mMobility = 0.02 / mBeta;
};

template <class TLattice>
using DefaultTraitBinaryLee =
    typename DefaultTrait<TLattice, 2>::template SetBoundary<BounceBack>::template SetProcessor<
        GradientsMultiStencil<OrderParameter<>, CentralXYZMirrorSolid, CentralQMirrorSolid, MixedXYZMirrorSolid,
                              MixedQMirrorSolid, LaplacianCentralWetting>,
        GradientsMultiStencil<Pressure<>, CentralXYZMirrorSolid, CentralQMirrorSolid, MixedXYZMirrorSolid,
                              MixedQMirrorSolid>,
        ChemicalPotentialCalculatorBinaryLee>::
        template AddProcessor<std::tuple<GradientsMultiStencil<ChemicalPotential<>, LaplacianCentralBounceBack>>>::
            template SetForce<LeeBinarySource, MuSourceLocal, MuSourceNonLocal>;

template <class TLattice>
using DefaultTraitBinaryLeeCentral =
    typename DefaultTrait<TLattice, 2>::template SetBoundary<BounceBack>::template SetProcessor<
        GradientsMultiStencil<OrderParameter<>, CentralXYZMirrorSolid, LaplacianCentralWetting>,
        GradientsMultiStencil<Pressure<>, CentralXYZMirrorSolid>, ChemicalPotentialCalculatorBinaryLee>::
        template AddProcessor<std::tuple<Gradients<ChemicalPotential<>, LaplacianCentralMirrorSolid>>>::
            template SetForce<LeeBinarySourceCentral, MuSourceLocal, MuSourceNonLocal>;

template <class TLattice, class TTraits = DefaultTraitBinaryLee<TLattice>, bool TSimpleForcing = false>
class BinaryLee : public CollisionBase<TLattice, typename TTraits::Stencil>,
                  public ModelBase<TLattice, TTraits> {  // Inherit from base class to avoid repetition of common
    // calculations

    using Stencil = typename TTraits::Stencil;
    using EquilibriumForceType = std::conditional_t<TSimpleForcing, He, Lee>;
    static constexpr int mNDIM = TLattice::NDIM;

   public:
    inline void setTau1(double val) { mTau1 = val; }
    inline void setTau2(double val) { mTau2 = val; }

    virtual inline void collide() override;  // Collision step

    virtual inline void initialise() override;  // Initialisation step

    virtual inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    inline double computeEquilibrium(int k, int idx) override;  // Calculate equilibrium in direction idx

    /**
     * \brief Function to compute the fractional concentration of each fluid (between 0 and 1).
     * \param k The index of the node on the lattice.
     * \param iFluid The index of the fluid (0 or 1).
     */
    inline double computeConcentration(int k, int iFluid) override;

    void setDensity1(double density) { mDensityC1 = density; }

    void setDensity2(double density) { mDensityC2 = density; }

    void setDensities(double density1, double density2) {
        setDensity1(density1);
        setDensity2(density2);
    }

    template <class, class>
    friend class BinaryLeeHumidity;

   private:
    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions

    double mGamma = 1;

    double mTau1 = 1;
    double mTau2 = 1;

    double mDensityC1 = 1;
    double mDensityC2 = 1;

    std::vector<double>& density = Density<>::get<TLattice>();                // Reference to vector of order parameters
    std::vector<double>& orderparameter = OrderParameter<>::get<TLattice>();  // Reference to vector of order parameters
    std::vector<double>& velocity =
        Velocity<>::get<TLattice, TTraits::Lattice::NDIM>();    // Reference to vector of velocities
    std::vector<double>& itau = InverseTau<>::get<TLattice>();  // Reference to vector of velocities

    std::tuple<LeeBinarySourceEquilibrium> mt_ModelForceEq;

    using data = Data_Base<TLattice, typename TTraits::Stencil>;
    data& mData = data::getInstance();

    thread_local static EquilibriumForceType mForcingScheme;
};

template <class TLattice, class TTraits, bool TSimpleForcing>
thread_local std::conditional_t<TSimpleForcing, He, Lee> BinaryLee<TLattice, TTraits, TSimpleForcing>::mForcingScheme;

template <class TLattice, class TTraits, bool TSimpleForcing>
inline void BinaryLee<TLattice, TTraits, TSimpleForcing>::collide() {
    ModelBase<TLattice, TTraits>::mData.communicate(LaplacianChemicalPotential<>::getInstance<TLattice>());
#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        if (this->isCollisionNode(k)) {
            double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

            double equilibriums[Stencil::Q] = {};
            mForcingScheme.reset();
            mForcingScheme.template precompute<TTraits>(std::get<0>(mt_ModelForceEq), k);

            for (int idx = 0; idx < Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
            }

            this->collisionQ(equilibriums, old_distribution, InverseTau<>::get<TLattice>(k), k);
        }
    }

    // ModelBase<TLattice, TTraits>::mData.communicateDistribution();
}

template <class TLattice, class TTraits, bool TSimpleForcing>
inline void BinaryLee<TLattice, TTraits, TSimpleForcing>::initialise() {  // Initialise model
    this->initialiseProcessors();

    ModelBase<TLattice, TTraits>::mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<Stencil>::template initialise<TLattice>(this->mt_Forces, mTau1, mTau2);

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        ChemicalPotential<>::initialise<TLattice>(0, k);
        OrderParameter<>::initialise<TLattice>(1.0, k);
        Density<>::initialise<TLattice>(((OrderParameter<>::get<TLattice>(k)) * (mDensityC1 - mDensityC2) + mDensityC2),
                                        k);
        InverseTau<>::initialise<TLattice>(
            ((OrderParameter<>::get<TLattice>(k)) / mTau1 + (1. - OrderParameter<>::get<TLattice>(k)) / mTau2), k);
    }

    ModelBase<TLattice, TTraits>::mData.communicate(GradientOrderParameter<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(BoundaryLabels<TLattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(ChemicalPotential<>::getInstance<TLattice>());

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {
        double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
        double* old_distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionOldPointer(k);

        mForcingScheme.reset();
        mForcingScheme.template precompute<TTraits>(std::get<0>(mt_ModelForceEq), k);

        for (int idx = 0; idx < TTraits::Stencil::Q; ++idx) {
            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(
        BoundaryLabels<TTraits::Lattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits, bool TSimpleForcing>
inline void BinaryLee<TLattice, TTraits, TSimpleForcing>::computeMomenta() {  // Calculate order parameter

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);

            orderparameter[k] = this->computeDensity(distribution, k);
            //if(orderparameter[k]<0) orderparameter[k]=0;

            density[k] = ((OrderParameter<>::get<TLattice>(k)) * (mDensityC1 - mDensityC2) + mDensityC2);

            //const double wid = 0.48201379003;
            //double& C1 = orderparameter[k];
            //itau[k]=1/((C1>0.5+wid)*(mTau1-mTau2)+(C1>0.5-wid&&C1<=0.5+wid)*0.5*(1+(C1-0.5)/wid+1/M_PI*sin(M_PI*(C1-0.5)/wid))*(mTau1-mTau2)
            //        +mTau2);
            itau[k] =
                ((OrderParameter<>::get<TLattice>(k)) / mTau1 + (1. - OrderParameter<>::get<TLattice>(k)) / mTau2);
        }
    }

    this->mData.communicate(ChemicalPotential<>::template getInstance<TLattice>());
    this->mData.communicate(LaplacianChemicalPotential<>::template getInstance<TLattice>());
    this->mData.communicate(BoundaryLabels<TLattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits, bool TSimpleForcing>
inline double BinaryLee<TLattice, TTraits, TSimpleForcing>::computeEquilibrium(int k, int idx) {
    double force = mForcingScheme.template compute<TTraits>(idx, k);
    double gamma =
        CollisionBase<TLattice, typename TTraits::Stencil>::computeGamma(&velocity[k * TTraits::Stencil::D], idx);
    return orderparameter[k] * gamma + force;
}

template <class TLattice, class TTraits, bool TSimpleForcing>
inline double BinaryLee<TLattice, TTraits, TSimpleForcing>::computeConcentration(int k, int iFluid) {
    if (iFluid == 0) {
        return orderparameter[k];
    } else if (iFluid == 1) {
        return 1.0 - orderparameter[k];
    } else {
        throw std::invalid_argument("Invalid fluid number. Must be 0 or 1.");
    }
}

// FlowField.hh: Contains the details of the LBM model to solve the Navier-Stokes and continuity equation. Each
// Model is given a "TTraits" class that contains stencil, data, force and boundary information

class PressureLeeForce : public ChemicalForceBinary<Lee, GradientMixed> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return (TTraitsF::Stencil::Cs2 *
                    MixedGradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) +
                ChemicalForceBinary<Lee, GradientMixed>::computeXYZ<TTraitsF>(xyz, k));
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return (TTraitsF::Stencil::Cs2 *
                    MixedGradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) +
                ChemicalForceBinary<Lee, GradientMixed>::computeQ<TTraitsF>(idx, k));
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
    /*template <class TTraitsF>
    inline double computeVelocitySource(const int xyz, const int k) {  // Need to correct velocity

        return TTraitsF::Stencil::Cs2 * mChemForceCentral.computeXYZ<TTraitsF>(xyz, k) * TTraitsF::Lattice::DT / (2.0);
    }

    private:
        ChemicalForceMu<Lee, Gradient> mChemForceCentral;
    */
    template <class TTraitsF>
    inline double computeVelocitySource(const int xyz, const int k) {  // Need to correct velocity

         return +TTraitsF::Stencil::Cs2*ChemicalPotential<>::get<typename
         TTraitsF::Lattice>(k)*GradientOrderParameter<>::get<typename
         TTraitsF::Lattice,TTraitsF::Lattice::NDIM>(k,xyz) * TTraitsF::Lattice::DT / (2.0);
        //return -TTraitsF::Stencil::Cs2 * OrderParameter<>::get<typename TTraitsF::Lattice>(k) *
        //       Gradient<ChemicalPotential<>>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) *
        //       TTraitsF::Lattice::DT / (2.0);
    }
};

class PressureLeeForce2 : public ForceBase<LeeGamma0> {
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

class PressureLeeForceEquilibrium : public ChemicalForceBinary<Lee, Gradient> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return -0.5 * (TTraitsF::Stencil::Cs2 *
                           GradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz) +
                       ChemicalForceBinary<Lee, Gradient>::computeXYZ<TTraitsF>(xyz, k));
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return -0.5 * (TTraitsF::Stencil::Cs2 *
                           GradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx) +
                       ChemicalForceBinary<Lee, Gradient>::computeQ<TTraitsF>(idx, k));
    }
};

class PressureLeeForce2Equilibrium : public ForceBase<LeeGamma0> {
   public:
    template <class TTraitsF>
    inline double computeXYZ(const int xyz, const int k) {
        return +0.5 * TTraitsF::Stencil::Cs2 *
               GradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Lattice::NDIM>(k, xyz);
    }
    template <class TTraitsF>
    inline double computeQ(const int idx, const int k) {
        return +0.5 * TTraitsF::Stencil::Cs2 *
               GradientDensity<>::get<typename TTraitsF::Lattice, TTraitsF::Stencil::Q>(k, idx);
    }
};

template <class TLattice>
using DefaultTraitPressureLee =
    typename DefaultTrait<TLattice, 2>::template SetBoundary<BounceBack>  // template SetBoundary<BounceBack>
    ::template SetProcessor<
        GradientsMultiStencil<Density<>, CentralXYZBounceBack, CentralQBounceBack, MixedXYZBounceBack,
                              MixedQBounceBack>>::template SetForce<PressureLeeForce, PressureLeeForce2>;

template <class TLattice, class TTraits = DefaultTraitPressureLee<TLattice>, bool TSimpleForcing = false>
class PressureLee
    : public FlowFieldPressure<TLattice, TTraits> {  // Inherit from base class to avoid repetition of common
    // calculations
    using EquilibriumForceType = std::conditional_t<TSimpleForcing, He, Lee>;
    using EquilibriumForceTypeGamma0 = std::conditional_t<TSimpleForcing, HeGamma0, LeeGamma0>;

   public:
    inline void setTau1(double val) { mTau1 = val; }
    inline void setTau2(double val) { mTau2 = val; }

    virtual inline void collide() override;  // Collision step

    virtual inline void initialise() override;  // Initialisation step

    virtual inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    inline double computeEquilibrium(int k, int idx) override;  // Calculate equilibrium in direction idx

    /**
     * \brief Function to compute the fractional concentration of each fluid (between 0 and 1).
     * \param k The index of the node on the lattice.
     * \param iFluid The index of the fluid (0 or 1).
     */
    inline double computeConcentration(int k, int iFluid) override;

    template <class, class>
    friend class PressureLeeHumidity;

   private:
    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions

    std::tuple<PressureLeeForceEquilibrium> mt_ModelForceEq;
    std::tuple<PressureLeeForce2Equilibrium> mt_ModelForce2Eq;

    std::vector<double>& density = Density<>::get<TLattice>();    // Reference to vector of order parameters
    std::vector<double>& pressure = Pressure<>::get<TLattice>();  // Reference to vector of order parameters
    std::vector<double>& velocity =
        Velocity<>::get<TLattice, TTraits::Lattice::NDIM>();    // Reference to vector of velocities
    std::vector<double>& itau = InverseTau<>::get<TLattice>();  // Reference to vector of velocities

    double mTau1 = 1;
    double mTau2 = 1;

    thread_local static EquilibriumForceType mForcingScheme1;
    thread_local static EquilibriumForceTypeGamma0 mForcingScheme2;
};

template <class TLattice, class TTraits, bool TSimpleForcing>
thread_local std::conditional_t<TSimpleForcing, He, Lee>
    PressureLee<TLattice, TTraits, TSimpleForcing>::mForcingScheme1;
template <class TLattice, class TTraits, bool TSimpleForcing>
thread_local std::conditional_t<TSimpleForcing, HeGamma0, LeeGamma0>
    PressureLee<TLattice, TTraits, TSimpleForcing>::mForcingScheme2;

template <class TLattice, class TTraits, bool TSimpleForcing>
inline void PressureLee<TLattice, TTraits, TSimpleForcing>::collide() {  // Collision step

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        if (this->isCollisionNode(k)) {
            double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

            double equilibriums[TTraits::Stencil::Q] = {};

            mForcingScheme1.reset();
            mForcingScheme1.template precompute<TTraits>(std::get<0>(mt_ModelForceEq), k);
            mForcingScheme2.reset();
            mForcingScheme2.template precompute<TTraits>(std::get<0>(mt_ModelForce2Eq), k);

            for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
            }

            this->collisionQ(equilibriums, old_distribution, InverseTau<>::get<TLattice>(k), k);
        }
    }

    // ModelBase<TLattice, TTraits>::mData.communicateDistribution();
}

template <class TLattice, class TTraits, bool TSimpleForcing>
inline void PressureLee<TLattice, TTraits, TSimpleForcing>::initialise() {  // Initialise model
    this->initialiseProcessors();

    ModelBase<TLattice, TTraits>::mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<typename TTraits::Stencil>::template initialise<TLattice>(this->mt_Forces, mTau1,
                                                                                               mTau2);

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        Pressure<>::initialise<TLattice>(1.0, k);  // Set density to 1 initially (This will change)
        Velocity<>::initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, x);
        if constexpr (TLattice::NDIM >= 2) Velocity<>::initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, y);
        if constexpr (TLattice::NDIM == 3) Velocity<>::initialise<TLattice, TTraits::Lattice::NDIM>(0.0, k, z);

        PressureOld<>::initialise<TLattice>(1.0, k);  // Set density to 1 initially (This will change)
    }

    ModelBase<TLattice, TTraits>::mData.communicate(GradientOrderParameter<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(BoundaryLabels<TLattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(ChemicalPotential<>::getInstance<TLattice>());

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {
        // InverseTau<>::initialise<TLattice>(1,k);
        double* distribution = this->mDistribution.getDistributionPointer(k);
        double* old_distribution = this->mDistribution.getDistributionOldPointer(k);
        mForcingScheme1.reset();
        mForcingScheme1.template precompute<TTraits>(std::get<0>(mt_ModelForceEq), k);
        mForcingScheme2.reset();
        mForcingScheme2.template precompute<TTraits>(std::get<0>(mt_ModelForce2Eq), k);
        for (int idx = 0; idx < TTraits::Stencil::Q; idx++) {
            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TTraits::Lattice::NDIM>());
}

template <class TLattice, class TTraits, bool TSimpleForcing>
inline void PressureLee<TLattice, TTraits, TSimpleForcing>::computeMomenta() {  // Calculate Density<> and Velocity

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

        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
}

template <class TLattice, class TTraits, bool TSimpleForcing>
inline double PressureLee<TLattice, TTraits, TSimpleForcing>::computeEquilibrium(int k, int idx) {
    double force =
        mForcingScheme1.template compute<TTraits>(idx, k) + mForcingScheme2.template compute<TTraits>(idx, k);
    double velocityFactor = CollisionBase<TLattice, typename TTraits::Stencil>::computeVelocityFactor(
        &velocity[k * TTraits::Stencil::D], idx);
    return TTraits::Stencil::Weights[idx] * (pressure[k] + density[k] * TTraits::Stencil::Cs2 * velocityFactor) +
           force;  // Equilibrium is density times gamma in this case
}

template <class TLattice, class TTraits, bool TSimpleForcing>
inline double PressureLee<TLattice, TTraits, TSimpleForcing>::computeConcentration(int k, int iFluid) {
    std::vector<double>& orderparameter = OrderParameter<>::get<TLattice>();
    if (iFluid == 0) {
        return orderparameter[k];
    } else if (iFluid == 1) {
        return 1.0 - orderparameter[k];
    } else {
        throw std::invalid_argument("Invalid fluid number. Must be 0 or 1.");
    }
}
