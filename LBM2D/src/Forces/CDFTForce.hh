#pragma once
#include <iostream>
#include <stdexcept>

#include "../Forcing.hh"
#include "../Lattice.hh"
#include "../Parameters.hh"
#include "ForceBase.hh"

/**
 * \file ExternalForce.hh
 * \brief Contains the force class for a constant applied body force in a given direction.
 */

/**
 * \brief BodyForce can be used to apply an external force to the whole fluid or a single component
 */
template <class TMethod = Guo<>>
class CDFTForce : public ForceBase<TMethod> {
   public:

    //! Return force at lattice point k in direction xyz
    template <class TTraits>
    inline double computeXYZ(int xyz, int k);

    //! Return force at lattice point k along lattice direction idx
    template <class TTraits>
    inline double computeQ(int idx, int k);

    //! Calculate any possible source/correction term for velocity
    template <class TTraits>
    inline double computeVelocitySource(int xyz, int k);

   private:

};

template <class TMethod>
template <class TTraits>
inline double CDFTForce<TMethod>::computeXYZ(int xyz, int k) {
    // Density of fluid being forced
    double density = Density<>::template get<typename TTraits::Lattice>(k);
    /*#pragma omp single
    {
    
    if(TIME==1) std::cout<<TTraits::Stencil::Cs2*density*Gradient<C1<>>::template get<typename TTraits::Lattice,TTraits::Lattice::NDIM>(k,xyz)<<" "<<-TTraits::Stencil::Cs2*Gradient<Density<>>::template get<typename TTraits::Lattice,TTraits::Lattice::NDIM>(k,xyz)<<std::endl;
    //if(TIME==0)  std::cout<<k<<" "<<density<<" "<<C1<>::get<typename TTraits::Lattice>(k)<<" "<<ElectricField<>::get<typename TTraits::Lattice>(k)<<std::endl;

    }*/
    return TTraits::Stencil::Cs2*density*Gradient<C1<>>::template get<typename TTraits::Lattice,TTraits::Lattice::NDIM>(k,xyz);//-TTraits::Stencil::Cs2*Gradient<Density<>>::template get<typename TTraits::Lattice,TTraits::Lattice::NDIM>(k,xyz);
}

template <class TMethod>
template <class TTraits>
inline double CDFTForce<TMethod>::computeQ(int idx, int k) {
    // Density of fluid being forced
    double density = Density<>::template get<typename TTraits::Lattice>(k);
    return TTraits::Stencil::Cs2*density*Gradient<C1<>>::template get<typename TTraits::Lattice,TTraits::Stencil::Q>(k,idx);
}

template <class TMethod>
template <class TTraits>
inline double CDFTForce<TMethod>::computeVelocitySource(int xyz, int k) {
    return +computeXYZ<TTraits>(xyz, k) * TTraits::Lattice::DT / (2.0);
}
