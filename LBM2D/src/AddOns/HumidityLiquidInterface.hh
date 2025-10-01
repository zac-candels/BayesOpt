#pragma once
#include <iostream>

class SetHumidityLiquid : public AddOnBase {
   public:
    template <class TTraits>
    inline void compute(int k);

    template <class TTraits>
    inline void communicate();

    inline void setInterfaceVal(double val) { mInterfaceHumidity = val; };

   private:
    double mInterfaceHumidity;
};

template <class TTraits>
inline void SetHumidityLiquid::compute(int k) {
    using Lattice = typename TTraits::Lattice;
    const int NDIM = TTraits::Lattice::NDIM;

    if (BoundaryLabels<NDIM>::template get<Lattice>(k).Id == 5) {
        Humidity<>::get<Lattice>(k) = mInterfaceHumidity;
        HumidityOld<>::get<Lattice>(k) = mInterfaceHumidity;
    }
}

template <class TTraits>
inline void SetHumidityLiquid::communicate() {}