#pragma once
#include <iostream>

class HumidityBoundaryLabels : public AddOnBase {
   public:
    template <class TTraits>
    inline void compute(int k);

    template <class TTraits>
    inline void communicate();

    inline void setInterfaceCondition(bool (*condition)(const double& val, int k)) {
        evalInterfaceCondition = condition;
    }

   private:
    static bool defaultCondition(const double& val, int k) { return true; }

    bool (*evalInterfaceCondition)(const double& val, int k) = &defaultCondition;
};

template <class TTraits>
inline void HumidityBoundaryLabels::compute(int k) {
    using Lattice = typename TTraits::Lattice;
    const int NDIM = TTraits::Lattice::NDIM;

    if (Geometry<Lattice>::isBulkSolid(k)) return;

    int& boundarylabel = BoundaryLabels<NDIM>::template get<Lattice>(k).Id;

    if (evalInterfaceCondition(OrderParameter<>::get<Lattice>(k), k) && boundarylabel == 5) {
        boundarylabel = 6;

    } else if (!evalInterfaceCondition(OrderParameter<>::get<Lattice>(k), k) &&
               (boundarylabel == 6 || boundarylabel == 0)) {
        boundarylabel = 5;

    } else if ((boundarylabel == 6 || boundarylabel == 5) &&
               evalInterfaceCondition(OrderParameter<>::get<Lattice>(k), k))
        boundarylabel = 0;
}

template <class TTraits>
inline void HumidityBoundaryLabels::communicate() {}

class HumidityBoundaryLabelsTernary : public AddOnBase {
   public:
    template <class TTraits>
    inline void compute(int k);

    template <class TTraits>
    inline void communicate();

    inline void setInterfaceCondition(bool (*condition)(int k)) { evalInterfaceCondition = condition; }

   private:
    static bool defaultCondition(int k) { return true; }

    bool (*evalInterfaceCondition)(int k) = &defaultCondition;
};

template <class TTraits>
inline void HumidityBoundaryLabelsTernary::compute(int k) {
    using Lattice = typename TTraits::Lattice;
    const int NDIM = TTraits::Lattice::NDIM;

    if (Geometry<Lattice>::isBulkSolid(k)) return;

    int& boundarylabel = BoundaryLabels<NDIM>::template get<Lattice>(k).Id;

    if (evalInterfaceCondition(k) && boundarylabel == 5) {
        boundarylabel = 6;
    } else if (!evalInterfaceCondition(k) && (boundarylabel == 6 || boundarylabel == 0)) {
        boundarylabel = 5;
    } else if ((boundarylabel == 6 || boundarylabel == 5) && evalInterfaceCondition(k)) {
        boundarylabel = 0;
    }
}

template <class TTraits>
inline void HumidityBoundaryLabelsTernary::communicate() {}