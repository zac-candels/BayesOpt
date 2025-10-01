#pragma once
#include <utility>

#include "../AddOns/AddOns.hh"
#include "../BoundaryModels/Boundaries.hh"
#include "../Collide.hh"
#include "../Data.hh"
#include "../Forces/Forces.hh"
#include "../GradientStencils/GradientStencils.hh"
#include "../Parallel.hh"
#include "../Parameters.hh"
#include "BinaryLee.hh"
#include "ModelBase.hh"
#include "TernaryLee.hh"

// FlowField.hh: Contains the details of the LBM model to solve the Navier-Stokes and continuity equation. Each
// Model is given a "TTraits" class that contains stencil, data, force and boundary information

template <class TLattice>
using DefaultTraitHumidity = typename DefaultTrait<TLattice>::template SetBoundary<InterpolatedDirichlet, Dirichlet,
                                                                                   Refill<Humidity<>>, FreeSlip>::
    template SetForce<EvaporationHumiditySource<EvaporationSourceMethod>>::template SetProcessor<
        Gradients<Humidity<>, CentralXYZInterfaceMirrorSolid>>  //
    ::template AddProcessor<std::tuple<HumidityBoundaryLabels, SetHumidityLiquid>>;

template <class TLattice, class TTraits = DefaultTraitHumidity<TLattice>>
class EvaporationHumidity
    : public CollisionBase<TLattice, typename TTraits::Stencil>,
      public ModelBase<TLattice, TTraits> {  // Inherit from base class to avoid repetition of common
    // calculations

    using Stencil = typename TTraits::Stencil;
    static constexpr int mNDIM = TLattice::NDIM;

   public:
    inline void collide() override;  // Collision step

    inline void initialise() override;  // Initialisation step

    inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    inline double computeEquilibrium(int k, int idx) override;  // Calculate equilibrium in direction idx

    inline void setDiffusivity(double D) {
        mTau = D / Stencil::Cs2 + 0.5 * TLattice::DT;
        mInverseTau = 1.0 / mTau;
    }

   private:
    double mTau = 0.02 / Stencil::Cs2 + 0.5 * TLattice::DT;  // TEMPORARY relaxation time
    double mInverseTau = 1.0 / mTau;                         // TEMPORARY inverse relaxation time

    std::vector<double>& humidity = Humidity<>::get<TLattice>();         // Reference to vector of TDensities
    std::vector<double>& velocity = Velocity<>::get<TLattice, mNDIM>();  // Reference to vector of velocities
    std::vector<double>& orderparameter = OrderParameter<>::get<TLattice>();

    enum { x = 0, y = 1, z = 2 };  // Indices corresponding to x, y, z directions
};

template <class TLattice, class TTraits>
inline void EvaporationHumidity<TLattice, TTraits>::collide() {  // Collision step

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        if (this->isCollisionNode(k)) {
            double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

            double equilibriums[Stencil::Q];

            for (int idx = 0; idx < Stencil::Q; idx++) {
                equilibriums[idx] = computeEquilibrium(k, idx);
            }

            this->collisionQ(equilibriums, old_distribution, mInverseTau,
                             k);  // CHANGE NEEDED If no forces, don't require them to be passed
        }
    }
}

template <class TLattice, class TTraits>
inline void EvaporationHumidity<TLattice, TTraits>::initialise() {  // Initialise model
    this->initialiseProcessors();

    this->mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<Stencil>::template initialise<TLattice>(this->mt_Forces, mTau, mTau);

#pragma omp parallel for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // loop over k

        Humidity<>::initialise<TLattice>(0.0, k);  // Set density to 1 initially (This will change)
        HumidityOld<>::initialise<TLattice>(Humidity<>::get<TLattice>(k),
                                            k);  // Set density to 1 initially (This will change)
    }

    this->mData.communicate(BoundaryLabels<TLattice::NDIM>::template getInstance<TLattice>());
    this->mData.communicate(Humidity<>::getInstance<TLattice>());
    this->mData.communicate(HumidityOld<>::getInstance<TLattice>());

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {
        double* distribution = this->mDistribution.getDistributionPointer(k);
        double* old_distribution = this->mDistribution.getDistributionOldPointer(k);

        for (int idx = 0; idx < Stencil::Q; idx++) {
            double equilibrium = computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }

        // std::cout<<Geometry<TLattice>::getBoundaryType(k)<<std::endl;
    }
    TLattice::communicateDistributionAll(this->mDistribution);
    TLattice::communicateDistributionAllOld(this->mDistribution);
}

template <class TLattice, class TTraits>
inline void EvaporationHumidity<TLattice, TTraits>::computeMomenta() {  // Calculate Density<> and Velocity

    Humidity<>::get<typename TTraits::Lattice>().swap(HumidityOld<>::get<typename TTraits::Lattice>());

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = this->mDistribution.getDistributionPointer(k);

            humidity[k] = this->computeDensity(distribution, k);  // Calculate density
        }
    }

    this->mData.communicate(BoundaryLabels<TLattice::NDIM>::template getInstance<TLattice>());
    this->mData.communicate(HumidityOld<>::getInstance<TLattice>());
    this->mData.communicate(Humidity<>::getInstance<TLattice>());
    this->mData.communicate(OrderParameter<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits>
inline double EvaporationHumidity<TLattice, TTraits>::computeEquilibrium(int k, int idx) {
    double gamma = CollisionBase<TLattice, Stencil>::computeGamma(&velocity[k * mNDIM], idx);
    return humidity[k] * gamma;
}

template <class TLattice>
using DefaultTraitPressureLeeHumidity = typename DefaultTraitPressureLee<TLattice>::template AddForce<EvaporationPressureSource<EvaporationSourceMethod>>;//template AddProcessor<std::tuple<NoFluxSolid<Pressure<>>>>::

template <class TLattice, class TTraits = DefaultTraitPressureLeeHumidity<TLattice>>
class PressureLeeHumidity : public PressureLee<TLattice, TTraits> {  // Inherit from base class to avoid repetition of
                                                                     // common calculations

   public:
    inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    inline double computePressure(int k) override;

    inline double computeEquilibrium(int k, int idx) override;

   private:
    std::vector<double>& velocity = Velocity<>::get<TLattice, TLattice::NDIM>();
    std::vector<double>& velocityold = VelocityOld<>::get<TLattice, TLattice::NDIM>();
};

template <class TLattice, class TTraits>
inline void PressureLeeHumidity<TLattice, TTraits>::computeMomenta() {  // Calculate Density<> and Velocity

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = this->mDistribution.getDistributionPointer(k);

            velocityold[k * TTraits::Stencil::D + x] = velocity[k * TTraits::Stencil::D + x];
            if constexpr (TLattice::NDIM >= 2)
                velocityold[k * TTraits::Stencil::D + y] = velocity[k * TTraits::Stencil::D + y];
            if constexpr (TLattice::NDIM == 3)
                velocityold[k * TTraits::Stencil::D + z] = velocity[k * TTraits::Stencil::D + z];

            if((BoundaryLabels<TLattice::NDIM>::template get<TLattice>(k).Id!=8&&BoundaryLabels<TLattice::NDIM>::template get<TLattice>(k).Id!=9)) {
            velocity[k * TTraits::Stencil::D + x] =
                1. / (TTraits::Stencil::Cs2) *
                this->computeVelocity(distribution, this->mt_Forces, this->density[k], x, k);  // Calculate velocities
            if constexpr (TLattice::NDIM >= 2){
                velocity[k * TTraits::Stencil::D + y] =
                    1. / (TTraits::Stencil::Cs2) *
                    this->computeVelocity(distribution, this->mt_Forces, this->density[k], y, k);
            }
            if constexpr (TLattice::NDIM == 3)
            {
                velocity[k * TTraits::Stencil::D + z] =
                    1. / (TTraits::Stencil::Cs2) *
                    this->computeVelocity(distribution, this->mt_Forces, this->density[k], z, k);
            }
            double velfactor = this->computeVelocityFactor(&velocity[k*TLattice::NDIM], 0);
            this->pressure[k] = computePressure(k);//+1.0/(1-TTraits::Stencil::Weights[0])*TTraits::Stencil::Cs2 * TTraits::Stencil::Weights[0]*Density<>::get<TLattice>(k)*velfactor;
            }

        }
    }

    this->mData.communicate(BoundaryLabels<TLattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(VelocityOld<>::getInstance<TLattice, TLattice::NDIM>());
}

template <class TLattice, class TTraits>
inline double PressureLeeHumidity<TLattice, TTraits>::computeEquilibrium(int k, int idx) {
    double force = this->mForcingScheme1.template compute<TTraits>(idx, k) +
                   this->mForcingScheme2.template compute<TTraits>(idx, k);
    double velocityFactor = CollisionBase<TLattice, typename TTraits::Stencil>::computeVelocityFactor(
        &(this->velocity[k * TTraits::Stencil::D]), idx);
    return TTraits::Stencil::Weights[idx] *
           (this->pressure[k]/*/TTraits::Stencil::Cs2*/ + this->density[k] * TTraits::Stencil::Cs2 * velocityFactor) + force - 0*(idx==0) * this->pressure[k];
}

template <class TLattice, class TTraits>
inline double PressureLeeHumidity<TLattice, TTraits>::computePressure(int k) {
    double* distribution = this->mDistribution.getDistributionPointer(k);
    auto& forcetuple = this->mt_Forces;

    /*if constexpr (std::tuple_size<typename TTraits::Forces>::value != 0) {
        return (1.0/(1-TTraits::Stencil::Weights[0])*(CollisionBase<TLattice, typename TTraits::Stencil>::computeZerothMoment(distribution)-distribution[0]) +
                std::apply([k, forcetuple](
                               auto&... forces) { return (forces.template computeDensitySource<TTraits>(k) + ...); },
                           forcetuple)) *
               std::apply(
                   [k, forcetuple](auto&... forces) {
                       return (forces.template computeDensitySourceMultiplicative<TTraits>(k) * ...);
                   },
                   forcetuple);

    }*/
    if constexpr (std::tuple_size<typename TTraits::Forces>::value != 0) {
        return (CollisionBase<TLattice, typename TTraits::Stencil>::computeZerothMoment(distribution) +
                std::apply([k, forcetuple](
                               auto&... forces) { return (forces.template computeDensitySource<TTraits>(k) + ...); },
                           forcetuple)) *
               std::apply(
                   [k, forcetuple](auto&... forces) {
                       return (forces.template computeDensitySourceMultiplicative<TTraits>(k) * ...);
                   },
                   forcetuple);

    } else
        return (CollisionBase<TLattice, typename TTraits::Stencil>::computeZerothMoment(distribution));
}

template <class TLattice>
using DefaultTraitBinaryLeeHumidity = typename DefaultTraitBinaryLee<TLattice>::template SetProcessor<
    
    GradientsMultiStencil<Pressure<>, CentralXYZBounceBack, CentralQBounceBack, MixedXYZBounceBack, MixedQBounceBack>,
    ChemicalPotentialCalculatorBinaryLee>::
    template AddProcessor<std::tuple<
        GradientsMultiStencil<OrderParameter<>, CentralXYZMirrorSolid, CentralQMirrorSolid, MixedXYZMirrorSolid,
                              MixedQMirrorSolid, LaplacianCentralWetting>,
        MassLossCalculatorInterpolated>>::template AddForce<EvaporationPhaseSource<EvaporationSourceMethod>>::
        template SetBoundary<ExtrapolationOutflow, BounceBack>::template AddProcessor<
            std::tuple<GradientsMultiStencil<ChemicalPotential<>, CentralXYZMirrorSolid, CentralQMirrorSolid,
                                             MixedXYZMirrorSolid, MixedQMirrorSolid, LaplacianCentralMirrorSolid>>>;

template <class TTrait, class TWetting>
struct SetWetting : TTrait {
    using Wetting = TWetting;
};

template <class TLattice, class TTraits = SetWetting<SetWetting<DefaultTraitBinaryLeeHumidity<TLattice>, LinearWetting>,
                                                     CubicWetting>>  // DefaultTraitBinaryLeeHumidity<TLattice>>
class BinaryLeeHumidity
    : public BinaryLee<TLattice, TTraits> {  // Inherit from base class to avoid repetition of common
    // calculations

    using Stencil = typename TTraits::Stencil;
    static constexpr int mNDIM = TLattice::NDIM;

   public:
    inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    inline void initialise() override;  // Momenta (density, velocity) calculation

   private:
    std::vector<double>& humidity = Humidity<>::get<TLattice>();  // Reference to vector of order parameters
    std::vector<double>& orderparameter = OrderParameter<>::get<TLattice>();
};

template <class TLattice, class TTraits>
inline void BinaryLeeHumidity<TLattice, TTraits>::initialise() {  // Initialise model
    this->initialiseProcessors();

    ModelBase<TLattice, TTraits>::mData.generateNeighbors();  // Fill array of neighbor values (See Data.hh)
    TTraits::template CollisionModel<Stencil>::template initialise<TLattice>(this->mt_Forces, this->mTau1, this->mTau2);

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        ChemicalPotential<>::initialise<TLattice>(0, k);
        OrderParameter<>::initialise<TLattice>(1.0, k);
        Density<>::initialise<TLattice>(
            ((OrderParameter<>::get<TLattice>(k)) * (this->mDensityC1 - this->mDensityC2) + this->mDensityC2), k);
        InverseTau<>::initialise<TLattice>(((OrderParameter<>::get<TLattice>(k)) / this->mTau1 +
                                            (1. - OrderParameter<>::get<TLattice>(k)) / this->mTau2),
                                           k);

        DensityOld<>::initialise<TLattice>(
            ((OrderParameter<>::get<TLattice>(k)) * (this->mDensityC1 - this->mDensityC2) + this->mDensityC2),
            k);  // Set density to 1 initially (This will change)
        OrderParameterOld<>::initialise<TLattice>(OrderParameter<>::get<TLattice>(k),
                                                  k);  // Set density to 1 initially (This will change)
    }

    ModelBase<TLattice, TTraits>::mData.communicate(GradientOrderParameter<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(BoundaryLabels<TLattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(ChemicalPotential<>::getInstance<TLattice>());

#pragma omp parallel for schedule(guided)
    for (int k = 0; k < TLattice::N; k++) {  // loop over k

        double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
        double* old_distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionOldPointer(k);

        this->mForcingScheme.template precompute<TTraits>(std::get<0>(this->mt_ModelForceEq), k);

        for (int idx = 0; idx < TTraits::Stencil::Q; ++idx) {
            double equilibrium = this->computeEquilibrium(k, idx);
            distribution[idx] = equilibrium;  // Set distributions to equillibrium
            old_distribution[idx] = equilibrium;
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(GradientOrderParameter<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(BoundaryLabels<TLattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(ChemicalPotential<>::getInstance<TLattice>());
}

template <class TLattice, class TTraits>
inline void BinaryLeeHumidity<TLattice, TTraits>::computeMomenta() {  // Calculate order parameter

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = ModelBase<TLattice, TTraits>::mDistribution.getDistributionPointer(k);
            orderparameter[k] = this->computeDensity(distribution, k);

            double fact = 1.0;
            if (Geometry<TLattice>::getBoundaryType(k)==0||Geometry<TLattice>::getBoundaryType(k)==6||Geometry<TLattice>::getBoundaryType(k)==9||Geometry<TLattice>::getBoundaryType(k)==8) {
                fact = 1.0/(1.0-Humidity<>::get<TLattice>(k));
            }

            this->density[k] =
                ((OrderParameter<>::get<TLattice>(k)) * (this->mDensityC1 - this->mDensityC2*fact) + this->mDensityC2*fact);

            this->itau[k] = ((OrderParameter<>::get<TLattice>(k)) / this->mTau1 +
                             (1. - OrderParameter<>::get<TLattice>(k)) / this->mTau2);
        }
    }

    ModelBase<TLattice, TTraits>::mData.communicate(GradientOrderParameter<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(OrderParameter<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Density<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(ChemicalPotential<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(LaplacianChemicalPotential<>::getInstance<TLattice>());
}

template <class TLattice>
using DefaultTraitPressureTernaryLeeHumidity = typename DefaultTraitPressureTernaryLee<TLattice>::template AddProcessor<
    std::tuple<Swapper<Velocity<>, VelocityOld<>, TLattice::NDIM>,
               NoFluxSolid<Pressure<>>>>::template AddForce<EvaporationPressureSource<EvaporationSourceMethod>>;

template <class TLattice, class TTraits = DefaultTraitPressureTernaryLeeHumidity<TLattice>>
class PressureTernaryLeeHumidity : public PressureTernaryLee<TLattice, TTraits> {  // Inherit from base class to avoid
                                                                                   // repetition of common calculations

   public:
    inline void computeMomenta() override;  // Momenta (density, velocity) calculation

    inline double computePressure(int k) override;

   private:
    std::vector<double>& velocityold = VelocityOld<>::get<TLattice, TLattice::NDIM>();
};

template <class TLattice, class TTraits>
inline void PressureTernaryLeeHumidity<TLattice, TTraits>::computeMomenta() {  // Calculate Density<> and Velocity

#pragma omp for schedule(guided)
    for (int k = TLattice::HaloSize; k < TLattice::N - TLattice::HaloSize; k++) {  // Loop over k

        if (this->isCollisionNode(k)) {
            double* distribution = this->mDistribution.getDistributionPointer(k);

            this->pressure[k] = computePressure(k);

            velocityold[k * TTraits::Stencil::D + x] =
                1. / (TTraits::Stencil::Cs2) *
                this->computeVelocity(distribution, this->mt_Forces, this->density[k], x, k);  // Calculate velocities
            velocityold[k * TTraits::Stencil::D + y] =
                1. / (TTraits::Stencil::Cs2) *
                this->computeVelocity(distribution, this->mt_Forces, this->density[k], y, k);
            if constexpr (TLattice::NDIM == 3)
                velocityold[k * TTraits::Stencil::D + z] =
                    1. / (TTraits::Stencil::Cs2) *
                    this->computeVelocity(distribution, this->mt_Forces, this->density[k], z, k);
        }
    }

    this->mData.communicate(BoundaryLabels<TLattice::NDIM>::template getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Pressure<>::getInstance<TLattice>());
    ModelBase<TLattice, TTraits>::mData.communicate(Velocity<>::getInstance<TLattice, TLattice::NDIM>());
    ModelBase<TLattice, TTraits>::mData.communicate(VelocityOld<>::getInstance<TLattice, TLattice::NDIM>());
}

template <class TLattice, class TTraits>
inline double PressureTernaryLeeHumidity<TLattice, TTraits>::computePressure(int k) {
    double* distribution = this->mDistribution.getDistributionPointer(k);
    auto& forcetuple = this->mt_Forces;

    if constexpr (std::tuple_size<typename TTraits::Forces>::value != 0) {
        return 1.0 / (1 - TTraits::Stencil::Weights[0]) *
               (CollisionBase<TLattice, typename TTraits::Stencil>::computeZerothMoment(distribution) -
                distribution[0] +
                TTraits::Stencil::Weights[0] * Density<>::get<TLattice>(k) *
                    CollisionBase<TLattice, typename TTraits::Stencil>::computeVelocityFactor(
                        &(Velocity<>::get<TLattice, TLattice::NDIM>(k, 0)), 0) +
                std::apply([k, forcetuple](
                               auto&... forces) { return (forces.template computeDensitySource<TTraits>(k) + ...); },
                           forcetuple)) *
               std::apply(
                   [k, forcetuple](auto&... forces) {
                       return (forces.template computeDensitySourceMultiplicative<TTraits>(k) * ...);
                   },
                   forcetuple);

    } else
        return 1.0 / (1 - TTraits::Stencil::Weights[0]) *
               (CollisionBase<TLattice, typename TTraits::Stencil>::computeZerothMoment(distribution) -
                distribution[0] +
                TTraits::Stencil::Weights[0] * Density<>::get<TLattice>(k) *
                    CollisionBase<TLattice, typename TTraits::Stencil>::computeVelocityFactor(
                        &(Velocity<>::get<TLattice, TLattice::NDIM>(k, 0)), 0));
}
