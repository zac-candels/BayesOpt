#pragma once
#include <math.h>
#include <array>
#include <functional>
#include <map>
#include "../Geometry.hh"
#include "../Lattice.hh"
#include "../Parameters.hh"
#include "AddOnBase.hh"
#include "../Service.hh"

class CubicWettingDani : public AddOnBase {   //this is a class template
   public:
    inline void setTheta(double theta);

    inline void setThetaDegrees(double theta);

    inline void setEta(double eta);

    template <class TTraits>
    inline void compute(int k);     //we are only worrying about this compute function   this is the literal compute information about the neighbours (will comppute the concentration in the solid nodes only)

    template <class TTraits>
    inline void communicate();

   private:
    double mEta = 2;
    double mPrefactor = 0;
};

template <class TTraits>
inline void CubicWettingDani::compute(int k) {
    using Lattice = typename TTraits::Lattice;    //information about the grid, lattice points k
    using Stencil = typename TTraits::Stencil;   //rules for how to access neighbouring k points
    using data = Data_Base<Lattice, Stencil>;   //this uses information from stencil and has functions to access neighbouring points k

    if (!this->apply<Lattice>(k)) return;   //don't touch this for now (if not on a solid node just 'return' do nothing for the current k (ignore everything below this line).. otherwise if true we take info from lattice, stencil, etc.)
                                            //this is the part that tells us to skip a k and do nothing with it.. 
    // Calculate prefactor
    double prefactor;   //prefactor is the xi
    prefactor = mPrefactor;

    // Get average order parameter from the neighbours -> approximate order parameter c_1^0 in the fluid adjacent to the solid in the normal direction
    double CAvg = 0;
    int count = 0;
    for (int idx = 1; idx < Stencil::Q; idx++) {   //so each lattice point is a k. If I am a point k, I can see Q points k(idx) around me, including myself. idx is a number that corresponds to one of these neighboring points. 0<=idx<Q.

        int k_neighbor = data::getInstance().getNeighbors()[k * Stencil::Q + idx]; //get k of the neighbor in direction idx

        if (!Geometry<Lattice>::isBoundary(k_neighbor)) { //If the neighboring is not a solid, do the following (inside the {...})
            CAvg += OrderParameter<0>::get<Lattice>(k_neighbor);
            count++;
        }
    }
    CAvg /= count; //Approximate order parameter c_1^0 in the fluid adjacent to the solid in the normal direction as the mean of the order parameter in the fluid nodes around the solid node
    std::cout<<CAvg + prefactor * (CAvg - pow(CAvg, 2))<<std::endl;
    // Set the order parameter on the solid node
    OrderParameter<0>::get<Lattice>(k) = CAvg + prefactor * (CAvg - pow(CAvg, 2));  //Set c_1^+1, (look at equation halim wrote)
}

inline void CubicWettingDani::setTheta(double theta) { mPrefactor = cos(theta) / (sqrt(2.0) * mEta); }

inline void CubicWettingDani::setThetaDegrees(double theta) { setTheta(theta / 180.0 * M_PI); }

inline void CubicWettingDani::setEta(double eta) {
    mPrefactor *= mEta / eta;
    mEta = eta;
}

template <class TTraits>
inline void CubicWettingDani::communicate() {
    using Lattice = typename TTraits::Lattice;
    Lattice::communicate(OrderParameter<>::getInstance<Lattice>());
}
