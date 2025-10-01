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
 * \brief RepulsiveForce can be used to apply an external force to the whole fluid or a single component
 */
template <int TCompID, int TOtherID, class TMethod = Guo<>>
class RepulsiveForce : public ForceBase<TMethod> {
   public:

    inline void setMagnitude(double magnitude);  //!< Set the z-component of the force
    inline void setInterfaceWidth(double width);  //!< Set the z-component of the force

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
    double mMagnitude = 0;
    double mInterfaceWidth = 4.0;  //!< Width of the diffuse interface
};

template <int TCompID, int TOtherID, class TMethod>
inline void RepulsiveForce<TCompID, TOtherID, TMethod>::setMagnitude(double magnitude) {
    mMagnitude = magnitude;
}

template <int TCompID, int TOtherID, class TMethod>
inline void RepulsiveForce<TCompID, TOtherID, TMethod>::setInterfaceWidth(double width) {
    mInterfaceWidth = width;
}


template <int TCompID, int TOtherID, class TMethod>
template <class TTraits>
inline double RepulsiveForce<TCompID, TOtherID, TMethod>::computeXYZ(int xyz, int k) {
    // Density of fluid being forced
    double x0 = computeX(TTraits::Lattice::LYdiv, TTraits::Lattice::LZdiv, k);
    double y0 = computeY(TTraits::Lattice::LYdiv, TTraits::Lattice::LZdiv, k);
    
    if (OrderParameter<TCompID>::template get<typename TTraits::Lattice>(k) < 0.005 || OrderParameter<TCompID>::template get<typename TTraits::Lattice>(k) > 0.995) {
        return 0;
    }

    double grad = getGradientInstance<Gradient, OrderParameter, TTraits::NumberOfComponents-1, typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                TCompID)[k * TTraits::Lattice::NDIM + xyz];
    double grad2 = getGradientInstance<Gradient, OrderParameter, TTraits::NumberOfComponents-1, typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                TCompID)[k * TTraits::Lattice::NDIM + (!xyz)];
    double normal = grad/(sqrt(pow(grad,2)+pow(grad2,2)));
    double normal2 = grad2/(sqrt(pow(grad,2)+pow(grad2,2)));

    if ((sqrt(pow(grad,2)+pow(grad2,2))) < 1e-8) {
        return 0;
    }
    
    double dist0=-0.5 * mInterfaceWidth * atanh((OrderParameter<TCompID>::template get<typename TTraits::Lattice>(k) - 0.5) / 0.5);
    double x0_corr = x0-dist0 * (xyz==0?normal:normal2);
    double y0_corr = y0-dist0 * (xyz==0?normal2:normal);

    double forcesum = 0; 

    for (int x1=((x0-10)<0 ? 0 : (x0-10)); x1<((x0+10)>TTraits::Lattice::LXdiv ? TTraits::Lattice::LXdiv : (x0+10)); x1++) {
        for (int y1=((y0-10)<0 ? 0 : (y0-10)); y1<((y0+10)>TTraits::Lattice::LYdiv ? TTraits::Lattice::LYdiv : (y0+10)); y1++) {
            int kk=computeK<typename TTraits::Lattice>(x1, y1, 0);
            double gradother1 = getGradientInstance<Gradient, OrderParameter, TTraits::NumberOfComponents-1, typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                TOtherID)[kk * TTraits::Lattice::NDIM + xyz];
            double gradother12 = getGradientInstance<Gradient, OrderParameter, TTraits::NumberOfComponents-1, typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                TOtherID)[kk * TTraits::Lattice::NDIM + (!xyz)];
            double dist1=-0.5 * mInterfaceWidth * atanh((OrderParameter<TOtherID>::template get<typename TTraits::Lattice>(kk) - 0.5) / 0.5);
            double normal1 = gradother1/(sqrt(pow(gradother1,2)+pow(gradother12,2)));
            double normal12 = gradother12/(sqrt(pow(gradother1,2)+pow(gradother12,2)));
            
            double x1_corr = x1 - dist1 * (xyz==0?normal1:normal12);
            double y1_corr = y1 - dist1 * (xyz==0?normal12:normal1);

            double dist = sqrt(pow(x1_corr-x0_corr,2) + pow(y1_corr-y0_corr,2));
        
            int kk2=kk;//computeK<typename TTraits::Lattice>((int)x1_corr, (int)y1_corr, 0);
            std::vector<double> l={-((double)(x1_corr-x0_corr))/sqrt(pow((double)(x1_corr-x0_corr),2)+pow((double)(y1_corr-y0_corr),2)),-((double)(y1_corr-y0_corr))/sqrt(pow((double)(x1_corr-x0_corr),2)+pow((double)(y1_corr-y0_corr),2))};
            double gradother = getGradientInstance<Gradient, OrderParameter, TTraits::NumberOfComponents-1, typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                TOtherID)[kk2 * TTraits::Lattice::NDIM + xyz];
            double gradother2 = getGradientInstance<Gradient, OrderParameter, TTraits::NumberOfComponents-1, typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                TOtherID)[kk2 * TTraits::Lattice::NDIM + (!xyz)];
            
            if (OrderParameter<TOtherID>::template get<typename TTraits::Lattice>(kk) > 0.001 && (grad*gradother+grad2*gradother2)<0 && (l[xyz]*grad+l[!xyz]*grad2<0) && (l[xyz]*gradother+l[!xyz]*gradother2>0) && dist < 10 && sqrt(pow(gradother,2)+pow(gradother2,2)) > 1e-4) {
                forcesum += sqrt(pow(gradother,2)+pow(gradother2,2))*(0.0169 * pow(dist,4) - 0.5049 * pow(dist,3) + 5.1703 * pow(dist,2) - 22.422 * dist + 39.359)*l[xyz];
            }
        }
    }
    ForceRepulsive<>::template get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k,xyz)= -0.5*mMagnitude * sqrt(pow(grad,2)+pow(grad2,2)) *forcesum;
    return -0.5*mMagnitude * sqrt(pow(grad,2)+pow(grad2,2)) *forcesum;
}

template <int TCompID, int TOtherID, class TMethod>
template <class TTraits>
inline double RepulsiveForce<TCompID, TOtherID, TMethod>::computeQ(int idx, int k) {
    return 0;/*
    double x0 = computeX(TTraits::Lattice::LYdiv, TTraits::Lattice::LZdiv, k);
    double y0 = computeY(TTraits::Lattice::LYdiv, TTraits::Lattice::LZdiv, k);
    
    if (OrderParameter<TCompID>::template get<typename TTraits::Lattice>(k) < 0.05 || OrderParameter<TCompID>::template get<typename TTraits::Lattice>(k) > 0.95) {
        return 0;
    }
    int xyz==0;
    double grad = getGradientInstance<Gradient, OrderParameter, TTraits::NumberOfComponents-1, typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                TCompID)[k * TTraits::Lattice::NDIM + xyz];
    double grad2 = getGradientInstance<Gradient, OrderParameter, TTraits::NumberOfComponents-1, typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                TCompID)[k * TTraits::Lattice::NDIM + (!xyz)];
    double normal = grad/(sqrt(pow(grad,2)+pow(grad2,2)));
    double normal2 = grad2/(sqrt(pow(grad,2)+pow(grad2,2)));

    if ((sqrt(pow(grad,2)+pow(grad2,2))) < 1e-8) {
        return 0;
    }
    
    double dist0=-0.5 * mInterfaceWidth * atanh((OrderParameter<TCompID>::template get<typename TTraits::Lattice>(k) - 0.5) / 0.5);
    x0 -= dist0 * (xyz==0?normal:normal2);
    y0 -= dist0 * (xyz==0?normal2:normal);

    double forcesum = 0; 

    for (int x1=((x0-10)<0 ? 0 : (x0-10)); x1<((x0+10)>TTraits::Lattice::LXdiv ? TTraits::Lattice::LXdiv : (x0+10)); x1++) {
        for (int y1=((y0-10)<0 ? 0 : (y0-10)); y1<((y0+10)>TTraits::Lattice::LYdiv ? TTraits::Lattice::LYdiv : (y0+10)); y1++) {
            int kk=computeK<typename TTraits::Lattice>(x1, y1, 0);
            double gradother1 = getGradientInstance<Gradient, OrderParameter, TTraits::NumberOfComponents-1, typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                TOtherID)[kk * TTraits::Lattice::NDIM + xyz];
            double gradother12 = getGradientInstance<Gradient, OrderParameter, TTraits::NumberOfComponents-1, typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                TOtherID)[kk * TTraits::Lattice::NDIM + (!xyz)];
            double dist1=-0.5 * mInterfaceWidth * atanh((OrderParameter<TOtherID>::template get<typename TTraits::Lattice>(kk) - 0.5) / 0.5);
            double normal1 = gradother1/(sqrt(pow(gradother1,2)+pow(gradother12,2)));
            double normal12 = gradother12/(sqrt(pow(gradother1,2)+pow(gradother12,2)));
            
            double x1_corr = x1 - dist1 * (xyz==0?normal1:normal12);
            double y1_corr = y1 - dist1 * (xyz==0?normal12:normal1);

            double dist = sqrt(pow(x1_corr-x0,2) + pow(y1_corr-y0,2));
        
            int kk2=computeK<typename TTraits::Lattice>((int)x1_corr, (int)y1_corr, 0);
            std::vector<double> l={-((double)(x1_corr-x0))/(pow((double)(x1_corr-x0),2)+pow((double)(y1_corr-y0),2)),-((double)(y1_corr-y0))/(pow((double)(x1_corr-x0),2)+pow((double)(y1_corr-y0),2))};
            double gradother = getGradientInstance<Gradient, OrderParameter, TTraits::NumberOfComponents-1, typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                TOtherID)[kk2 * TTraits::Lattice::NDIM + xyz];
            double gradother2 = getGradientInstance<Gradient, OrderParameter, TTraits::NumberOfComponents-1, typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                TOtherID)[kk2 * TTraits::Lattice::NDIM + (!xyz)];
            
            if (OrderParameter<TOtherID>::template get<typename TTraits::Lattice>(kk) > 0.001 && (grad*gradother+grad2*gradother2)<0 && (l[xyz]*grad+l[xyz]*grad2<0) && (l[xyz]*gradother+l[xyz]*gradother2>0) && dist < 10 && sqrt(pow(gradother,2)+pow(gradother2,2)) > 1e-4) {
                forcesum += gradother*(0.0169 * pow(dist,4) - 0.5049 * pow(dist,3) + 5.1703 * pow(dist,2) - 22.422 * dist + 39.359)*l[xyz];
            }
        }
    }
    
    return -mMagnitude * getGradientInstance<Gradient, OrderParameter, TTraits::NumberOfComponents-1, typename TTraits::Lattice, TTraits::Stencil::Q>(
                TCompID)[k * TTraits::Stencil::Q + idx] *forcesum;*/
}

template <int TCompID, int TOtherID, class TMethod>
template <class TTraits>
inline double RepulsiveForce<TCompID, TOtherID, TMethod>::computeVelocitySource(int xyz, int k) {
    return +computeXYZ<TTraits>(xyz, k) * TTraits::Lattice::DT / (2.0);
}
