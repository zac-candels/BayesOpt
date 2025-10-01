#pragma once
#include <algorithm>
#include <map>

#include "../Geometry.hh"
#include "../LBModels/ModelBase.hh"
#include "BoundaryBase.hh"

// ZouHe.hh: Contains the Zou-He boundary condition for setting a constant density or velocity
// (https://doi.org/10.1063/1.869307). Use ZouHeDensity for constant density boundaries and ZouHeVelocity for constant
// velocity.

class ZouHe : public BoundaryBase {
   public:
    template <class TTraits, class TDistributionType>
    inline void compute(TDistributionType& mDistribution, int k);

    // Used to set a non-perpendicular velocity for constant density boundaries
    void setAngledVelocity(std::vector<double> velocityDirection);

    enum BoundaryTypes { DENSITY = 0, VELOCITY = 1, PRESSURE = 2, PRESSUREEVAPORATION = 3 };
    int boundaryType;

   private:
    template <class TLattice>
    inline void initialiseBoundaryValue(int k);
    std::map<int, double> mBoundaryValues;

    template <class TLattice>
    inline void angleVelocity(int k, int normalDim);
    bool mUseAngledVelocity = false;
    std::vector<double> mVelocityDirection;
};

class ZouHeDensity : public ZouHe {
   public:
    ZouHeDensity() { boundaryType = DENSITY; };
};

class ZouHeVelocity : public ZouHe {
   public:
    ZouHeVelocity() { boundaryType = VELOCITY; };
};

class ZouHePressure : public ZouHe {
   public:
    ZouHePressure() { boundaryType = PRESSURE; };
};

class ZouHePressureEvaporation : public ZouHe {
   public:
    ZouHePressureEvaporation() { boundaryType = PRESSUREEVAPORATION; };
};

template <class TLattice>
inline void getNormal(int k, int& normalDim, int& normalDir) {
    auto normalVec = BoundaryLabels<TLattice::NDIM>::template get<TLattice>(k).NormalDirection;
    normalDim = std::distance(begin(normalVec),
                              std::find_if(begin(normalVec), end(normalVec), [](int8_t ni) { return (int)ni != 0; }));
    // std::cout<<(int)normalVec[0]<<std::endl;
    normalDir = (int)normalVec[normalDim];
}

template <class TLattice>
inline void ZouHe::initialiseBoundaryValue(int k) {
    switch (boundaryType) {
        case DENSITY:
#pragma omp critical
            mBoundaryValues.insert({k, Density<>::template get<TLattice>(k)});
            break;
        case PRESSURE:
#pragma omp critical
            mBoundaryValues.insert({k, Pressure<>::template get<TLattice>(k)});
            break;
        case PRESSUREEVAPORATION:
#pragma omp critical
            mBoundaryValues.insert({k, Pressure<>::template get<TLattice>(k)});
            break;
        case VELOCITY: {
            int normalDim, normalDir;
            getNormal<TLattice>(k, normalDim, normalDir);
            double velOut = normalDir * Velocity<>::template get<TLattice, TLattice::NDIM>(k, normalDim);
#pragma omp critical
            mBoundaryValues.insert({k, velOut});
            break;
        }
        default:
            throw std::invalid_argument("Invalid boundary type");
    }
}

template <class TTraits, class TDistributionType>
inline void ZouHe::compute(TDistributionType& distribution, int k) {
    using Lattice = typename TTraits::Lattice;
    if (!apply<Lattice>(k)) return;
    if (mBoundaryValues.count(k) == 0)
        initialiseBoundaryValue<Lattice>(k);  // Initialise the fixed value if it is not already

    // Get necessary values
    int normalDim, normalDir;
    getNormal<Lattice>(k, normalDim, normalDir);
    auto distrK = distribution.getDistributionPointer(k);
    auto model = static_cast<ModelBase<Lattice, TTraits>*>(mModel);

    // Sum the distribution function groups and get unknowns
    double distOut = 0;
    double distNeutral = 0;
    double diffTangent = 0;

    //auto normalVec = BoundaryLabels<Lattice::NDIM>::template get<Lattice>(k).NormalDirection;
    std::vector<int> tangent1 = {normalDim!=0,normalDim==0,0};
    std::vector<int> tangent2 = {0,normalDim!=1,normalDim==1};
    //tangentDim = std::distance(begin(normalVec),
    //                          std::find_if(begin(normalVec), end(normalVec), [](int8_t ni) { return (int)ni != 0; }));

    std::vector<int> unknowns;
    unknowns.reserve(TTraits::Stencil::Q / 2);
    for (int iQ = 0; iQ < TTraits::Stencil::Q; iQ++) {
        int cNormal = normalDir * TTraits::Stencil::Ci_xyz(normalDim)[iQ];
        if (cNormal == 0) {
            distNeutral += distrK[iQ];
            if (iQ>0) diffTangent += distrK[iQ]*TTraits::Stencil::Ci_xyz(normalDim!=1)[iQ];
        } else if (cNormal < 0) {
            distOut += distrK[iQ];
        } else {
            unknowns.push_back(iQ);
        }
    }
    
    // Compute the unknown velocity or density / pressure
    double* density = Density<>::template getAddress<Lattice>(k);
    double* pressure = Pressure<>::template getAddress<Lattice>(k);
    double* velocity = Velocity<>::template getAddress<Lattice, Lattice::NDIM>(k, normalDim);
    double dotvel = 0;

    switch (boundaryType) {
        case DENSITY:
            *density = mBoundaryValues[k];
            *velocity = normalDir * (1 - (distNeutral + 2 * distOut) / *density);
            if (mUseAngledVelocity) angleVelocity<Lattice>(k, normalDim);
            break;
        case PRESSURE:

            *pressure = mBoundaryValues[k];
            *velocity = normalDir * ((*pressure) / (*density * TTraits::Stencil::Cs2) -
                                     (distNeutral + 2 * distOut) / (*density * TTraits::Stencil::Cs2))/(1+normalDir*TTraits::Lattice::DT * 0.5 * TTraits::Stencil::Cs2 *
                        GradientDensity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, 0) / (*density * TTraits::Stencil::Cs2)); 

            if (mUseAngledVelocity) angleVelocity<Lattice>(k, normalDim);
            break;
        case PRESSUREEVAPORATION:

            *pressure = mBoundaryValues[k];
            //for (int iter = 0; iter < 5; iter++) {
                dotvel = 0;
                for (int xyz = 0; xyz < TTraits::Lattice::NDIM; xyz++) {
                dotvel += Velocity<>::get<Lattice, TTraits::Lattice::NDIM>(k, xyz) *
                          Velocity<>::get<Lattice, TTraits::Lattice::NDIM>(k, xyz);
                }
                for (int xyz = 0; xyz < TTraits::Lattice::NDIM; xyz++)
                *velocity =
                    normalDir * ((*pressure) * (1 - TTraits::Stencil::Weights[0]) / (*density * TTraits::Stencil::Cs2) -
                                (distNeutral + 2 * distOut - distrK[0] -
                                TTraits::Stencil::Cs2 * Density<>::get<Lattice>(k) * TTraits::Stencil::Weights[0] *
                                    (dotvel) / (2 * TTraits::Stencil::Cs2)) /
                                    (*density * TTraits::Stencil::Cs2))/(1 + (1 - TTraits::Stencil::Weights[0]) / (*density * TTraits::Stencil::Cs2) * normalDir*TTraits::Lattice::DT * 0.5 * TTraits::Stencil::Cs2 *
                        GradientDensity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, 0)); 
            //}
            
            
            if (mUseAngledVelocity) angleVelocity<Lattice>(k, normalDim);
            break;
        case VELOCITY:  // TODO: Velocity boundary for pressure model
            *velocity = normalDir * mBoundaryValues[k];
            *density = (distNeutral + 2 * distOut) / (1 - mBoundaryValues[k]);
            break;
        default:
            throw std::invalid_argument("Invalid boundary type");
    }

    // Compute the unknown distribution functions
    for (int iQ : unknowns) {
        
        int Ct1=0;
        int Ct2=0;
        int C2=0;
        for (int t=0; t<Lattice::NDIM; t++) {
            Ct1 += TTraits::Stencil::Ci_xyz(t)[iQ]*tangent1[t];
            Ct2 += TTraits::Stencil::Ci_xyz(t)[iQ]*tangent2[t];
            C2 += TTraits::Stencil::Ci_xyz(t)[iQ]*TTraits::Stencil::Ci_xyz(t)[iQ];
        }

        int iQOpp = distribution.getOpposite(iQ);
        double distrEq = model->computeEquilibrium(k, iQ);
        double distrEqOpp = model->computeEquilibrium(k, iQOpp);
        distrK[iQ] = distrK[iQOpp] + (distrEq - distrEqOpp)-0.5*diffTangent*Ct1/sqrt(C2)-0.5*diffTangent*Ct2/sqrt(C2);
    }
}

void ZouHe::setAngledVelocity(std::vector<double> velocityDirection) {
    mUseAngledVelocity = true;
    mVelocityDirection = velocityDirection;
}

template <class TLattice>
inline void ZouHe::angleVelocity(int k, int normalDim) {
    double* velocity = Velocity<>::template getAddress<TLattice, TLattice::NDIM>(k, 0);
    double vPerp = velocity[normalDim];
    // (De)project the perpendicular velocity onto the velocity direction
    double projectionFactor = 1 / mVelocityDirection[normalDim];
    for (int iDim = 0; iDim < TLattice::NDIM; iDim++) {
        velocity[iDim] = vPerp * mVelocityDirection[iDim] * projectionFactor;
    }
}
