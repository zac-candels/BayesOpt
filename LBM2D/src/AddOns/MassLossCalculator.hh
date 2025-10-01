#pragma once
#include <math.h>

#include <iostream>
#include <utility>

#include "../Lattice.hh"
#include "../Parameters.hh"
#include "../Service.hh"
#include "AddOnBase.hh"

class MassLossCalculator : public AddOnBase {
   public:
    template <class TTraits>
    inline void compute(int k);

    inline void setInterfaceHumidity(double humidityi);

    inline void setDiffusivity(double D);

    inline void setGasDensity(double density);

   private:
    double mHumidityI = 0.0;
    double mDiffusivity = 0.02;
    double mDensityGas = 1.0;
    double mPrefactor = mDiffusivity * mDensityGas / (1.0 - mHumidityI);
};

template <class TTraits>
inline void MassLossCalculator::compute(int k) {
    using Lattice = typename TTraits::Lattice;
    const int NDIM = Lattice::NDIM;

    if (Geometry<Lattice>::isBulkSolid(k)) return;

    double dotproduct = 0;

    for (int xyz = 0; xyz < Lattice::NDIM; xyz++) {
        dotproduct +=
            GradientHumidity<>::get<Lattice, NDIM>(k, xyz) * GradientOrderParameter<>::get<Lattice, NDIM>(k, xyz);
    }

    MassSink<>::get<Lattice>(k) = -mPrefactor * dotproduct;
}

inline void MassLossCalculator::setInterfaceHumidity(double humidityi) {
    mHumidityI = humidityi;
    mPrefactor = mDiffusivity * mDensityGas / (1.0 - mHumidityI);
}

inline void MassLossCalculator::setDiffusivity(double D) {
    mDiffusivity = D;
    mPrefactor = mDiffusivity * mDensityGas / (1.0 - mHumidityI);
}

inline void MassLossCalculator::setGasDensity(double density) {
    mDensityGas = density;
    mPrefactor = mDiffusivity * mDensityGas / (1.0 - mHumidityI);
}

template <int TNdim>
struct InterfacePoint {
    InterfacePoint(int idx, double dist[TNdim]) : index(idx) { std::copy(dist, dist + TNdim, distance); }
    double distance[TNdim];
    int index;
};

class MassLossCalculatorInterpolated : public AddOnBase {
   public:
    template <int ndim = 3>
    struct InterfacePoint {
        InterfacePoint(int idx, double dist[ndim], bool issafe = true) : index(idx), safe(issafe) {
            std::copy(dist, dist + ndim, distance);
        }
        double distance[ndim];
        int index;
        bool safe = true;
    };

    template <class TTraits>
    inline void compute(int k);

    inline void toggleCalculate(bool calc) { mCalculate = calc; }

    inline void setInterfaceHumidity(double humidityi);

    inline void setDiffusivity(double D);

    inline void setGasDensity(double density);

    inline void setLiquidDensity(double density);

    inline void setInterfaceWidth(double width) { mAlpha = width; }

    inline void setNumberIterations(int num) { mNumIterations = num; }

    inline void setLiquidId(int id) { mLiquidId = id; }

    inline void setGasId(int id) { mGasId = id; }

    inline void setPhiGasLiquid(double phiGas, double phiLiquid) {
        mPhiMax = std::max(phiGas, phiLiquid);
        mPhiMin = std::min(phiGas, phiLiquid);
        mPhiGas = (phiGas >= phiLiquid);
    }

    template <class TTraits>
    inline std::vector<double> calcHumidityGradientInterfaceNearestNeighbor(int k);

    template <class TTraits>
    inline void calcHumidityGradientInterfaceGlobal();

    template <class TTraits>
    inline std::vector<double> calcHumidityGradientInterface(int k);
    template <class TTraits>
    inline std::vector<double> calcHumidityGradientInterface(int k, double hsat);

    template <class TTraits>
    inline double calcInterfaceDistance(int k);

    template <class TTraits>
    inline double calcInterfaceDistanceRebalance(int k);

    template <class TTraits>
    inline double calcOrderParamGradientRebalance(int direction, int k);

    template <class Tlattice>
    inline std::vector<double> calcNormal(const double* gradorderparam);

    template <int TNDIM>
    inline double calcNeighborDistance(const std::vector<double>& normal);

    template <typename T>
    int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }

    template <class TTraits>
    inline auto findClosestKToInterface(const double& interfacedistance, const double& neighbordistance,
                                        const std::vector<double>& normal, int k);

    template <class TTraits>
    inline InterfacePoint<TTraits::Lattice::NDIM> findClosestPointToInterface(const double& interfacedistance,
                                                                              const double& neighbordistance,
                                                                              const std::vector<double>& normal, int k,
                                                                              int ktoignore1 = -1, int ktoignore2 = -1);

    inline double calcMPIOffset();

    template <class TTraits>
    inline std::vector<double> calculateHumidityGradientInterfaceClosest(const double& interfacedistance,
                                                                         const double& neighbordistance,
                                                                         const std::vector<double>& normal, int k);

    template <class TTraits>
    inline std::vector<double> calculateHumidityGradientInterface(const double& interfacedistance,
                                                                  const double& neighbordistance,
                                                                  const std::vector<double>& normal, int k);

    template <class TTraits>
    inline std::vector<double> calculateHumidityGradientInterface(const double& interfacedistance,
                                                                  const double& neighbordistance,
                                                                  const std::vector<double>& normal, int k,
                                                                  double hsat);

    inline std::vector<double> calculateHumidityGradientInterfaceTwoPoints();

    inline std::vector<double> calculateHumidityGradientInterfaceThreePoints();

    inline void setInterfaceCondition(bool (*condition)(const double& val, int k)) {
        evalInterfaceCondition = condition;
    }

    inline void setInterfaceConditionSafe(bool (*condition)(const double& val, int k)) {
        evalInterfaceCondition = condition;
    }

    template <class TTraits>
    inline void communicate();

   private:
    static bool defaultCondition(const double& val, int k) { return true; }

    bool (*evalInterfaceCondition)(const double& val, int k) = &defaultCondition;
    bool (*evalInterfaceConditionSafe)(const double& val, int k) = &defaultCondition;

    double mHumidityI = 0.0;
    double mDiffusivity = 0.02;
    double mDensityGas = 1.0;
    double mDensityLiquid = 1.0;
    double mPrefactor = mDiffusivity * mDensityGas / (1.0 - mHumidityI);
    double mAlpha;
    std::vector<double> mGradH;
    bool mPhiGas = 0;
    bool mCalculate = true;
    int mNumIterations = 0;
    int mPhiMin = 0;
    int mPhiMax = 1;
    int mLiquidId = 0;
    int mGasId = 0;
};

template <class TTraits>
inline void MassLossCalculatorInterpolated::communicate() {
    using Lattice = typename TTraits::Lattice;
    Lattice::communicate(GradientHumidity<>::getInstance<Lattice, Lattice::NDIM>());
    calcHumidityGradientInterfaceGlobal<TTraits>();
}

template <class TTraits>
inline void MassLossCalculatorInterpolated::compute(int k) {
    using Lattice = typename TTraits::Lattice;
    const int NDIM = TTraits::Lattice::NDIM;
    const int N = TTraits::NumberOfComponents;

    if (Geometry<Lattice>::isBulkSolid(k) || mCalculate == false || (getInstance<OrderParameter, TTraits::NumberOfComponents - 1, typename TTraits::Lattice>(mLiquidId)[k]<0.005) || (getInstance<OrderParameter, TTraits::NumberOfComponents - 1, typename TTraits::Lattice>(mLiquidId)[k]>0.995) || k < Lattice::HaloSize ||
        k >= Lattice::N - Lattice::HaloSize){
            MassSink<>::get<typename TTraits::Lattice>(k) = 0;
            return;
        }
        

    std::vector<double> gradhumidity(NDIM, 0);
    double hsatnew = mHumidityI;
    double massloss;

    gradhumidity = calcHumidityGradientInterface<TTraits>(
        k);  // mGradH;//calcHumidityGradientInterface<TTraits>(k);//mGradH;//{0,-mDensityGas/(1.0-mHumidityI) *
             // mDiffusivity / ((Lattice::LY-82.0-4)) *
             // log(1.0/(1.0-mHumidityI))};//mGradH;//calcHumidityGradientInterface<TTraits>(k);
    // if(TIME==50000) std::cout<<gradhumidity[1]<<" "<<mDensityGas<<" "<<mHumidityI<<" "<<mDiffusivity<<"
    // "<<(Lattice::LY-82.0-4)<<std::endl;
    double dotproduct = 0;
    //double dotproduct2 = 0;

    const std::vector<double>& gradLiquid = getInstance<GradientOrderParameter, N-(N==2), Lattice, NDIM>(mLiquidId);
    const std::vector<double>& gradGas = getInstance<GradientOrderParameter, N-(N==2), Lattice, NDIM>(mLiquidId);

    if (mGasId < N - 1) {
        for (int xyz = 0; xyz < NDIM; xyz++) {
            dotproduct += gradhumidity[xyz] * sgn(gradLiquid[k * NDIM + xyz]) *
                          sqrt(fabs(gradLiquid[k * NDIM + xyz] * gradGas[k * NDIM + xyz]));
            //dotproduct2 += gradhumidity[xyz]*gradhumidity[xyz];
        }
    } else if (N == 3) {
        for (int xyz = 0; xyz < TTraits::Lattice::NDIM; xyz++) {
            /*dotproduct += GradientHumidity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz) *
                          sgn(gradGas[k * NDIM + xyz]) * gradLiquid[k * NDIM + xyz]) *
                          sqrt(fabs(gradLiquid[k * NDIM + xyz] *
                               (GradientOrderParameter<0>::template get<
                                    typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz) +
                                GradientOrderParameter<1>::template get<
                                    typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz))));*/
            dotproduct += gradhumidity[xyz] * calcOrderParamGradientRebalance<TTraits>(xyz, k);
            /*sqrt(fabs(gradLiquid[k * NDIM + xyz] *
                      (GradientOrderParameter<0>::template get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                           k, xyz) +
                       GradientOrderParameter<1>::template get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                           k, xyz))));*/
        }
    }
    double dens = mDensityGas;  //*(1-mHumidityI);
    massloss = dens * mDiffusivity / ((1.0 - hsatnew) * (1.0 - hsatnew)) * (dotproduct);
    //double massloss2 = dens * mDiffusivity / ((1.0 - hsatnew) * (1.0 - hsatnew)) * (dotproduct2);
    //if (massloss2 >= 0) MassSink2<>::get<typename TTraits::Lattice>(k) = massloss2;

    hsatnew = mHumidityI / (1. - 0.5 * massloss * (1.0 / (mDensityGas / (1.0 - mHumidityI)) - 1.0 / mDensityLiquid));

    for (int i = 0; i < mNumIterations - 1; i++) {
        gradhumidity = calcHumidityGradientInterface<TTraits>(k, hsatnew);
        dotproduct = 0;

        if (mGasId < N - 1) {
            for (int xyz = 0; xyz < NDIM; xyz++) {
                dotproduct += gradhumidity[xyz] * sgn(gradLiquid[k * NDIM + xyz]) *
                              sqrt(fabs(gradLiquid[k * NDIM + xyz] * gradGas[k * NDIM + xyz]));
            }
        } else if (N == 3) {
            for (int xyz = 0; xyz < TTraits::Lattice::NDIM; xyz++) {
                /*dotproduct += GradientHumidity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, xyz) *
                            sgn(gradGas[k * NDIM + xyz] * gradLiquid[k * NDIM + xyz] *
                            sqrt(fabs(gradLiquid[k * NDIM + xyz] *
                                (GradientOrderParameter<N - 1>::template get<
                                        typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, 0, xyz) +
                                    GradientOrderParameter<N - 1>::template get<
                                        typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, 1, xyz))));*/
                dotproduct += gradhumidity[xyz] * calcOrderParamGradientRebalance<TTraits>(xyz, k);
                /*sqrt(fabs(
                    gradLiquid[k * NDIM + xyz] *
                    (GradientOrderParameter<N - 1>::template get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                        k, 0, xyz) +
                    GradientOrderParameter<N - 1>::template get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(
                        k, 1, xyz))));*/
            }
        }
        massloss = dens * mDiffusivity / ((1.0 - hsatnew) * (1.0 - hsatnew)) * (dotproduct);

        hsatnew =
            mHumidityI / (1. - 0.5 * massloss * (1.0 / (mDensityGas / (1.0 - mHumidityI)) - 1.0 / mDensityLiquid));
    }
    // if (masslos<0)
    if (massloss >= 0) MassSink<>::get<typename TTraits::Lattice>(k) = massloss;
}

inline void MassLossCalculatorInterpolated::setInterfaceHumidity(double humidityi) {
    mHumidityI = humidityi;
    mPrefactor = mDiffusivity * mDensityGas / (1.0 - mHumidityI);
}

inline void MassLossCalculatorInterpolated::setDiffusivity(double D) {
    mDiffusivity = D;
    mPrefactor = mDiffusivity * mDensityGas / (1.0 - mHumidityI);
}

inline void MassLossCalculatorInterpolated::setGasDensity(double density) {
    mDensityGas = density;
    mPrefactor = mDiffusivity * mDensityGas / (1.0 - mHumidityI);
}

inline void MassLossCalculatorInterpolated::setLiquidDensity(double density) { mDensityLiquid = density; }

template <class TTraits>
inline std::vector<double> MassLossCalculatorInterpolated::calcHumidityGradientInterfaceNearestNeighbor(int k) {
    using Lattice = typename TTraits::Lattice;
    const int NDIM = TTraits::Lattice::NDIM;
    const int N = TTraits::NumberOfComponents;

    double distance;
    if constexpr (TTraits::NumberOfComponents == 2)
        distance = calcInterfaceDistance<TTraits>(k);
    else
        distance = calcInterfaceDistanceRebalance<TTraits>(k);

    if (std::isnan(abs(distance))) {
        std::vector<double> vec(TTraits::Lattice::NDIM, 0);
        return vec;
    }

    const double* gradorderparam = &getInstance<GradientOrderParameter, N, Lattice, NDIM>(mLiquidId)[k * NDIM];

    std::vector<double> normal = calcNormal<typename TTraits::Lattice>(gradorderparam);

    double neighbordistance = calcNeighborDistance<TTraits::Lattice::NDIM>(normal);
#ifdef MPIPARALLEL
    if (abs(distance) > TTraits::Lattice::Parallel.Width * neighbordistance) {
        std::vector<double> vec(TTraits::Lattice::NDIM, 0);
        return vec;
    }
#endif
    return calculateHumidityGradientInterfaceClosest<TTraits>(distance, neighbordistance, normal, k);
}

template <class TTraits>
inline void MassLossCalculatorInterpolated::calcHumidityGradientInterfaceGlobal() {
#pragma omp master
    {
        using Lattice = typename TTraits::Lattice;
        const int NDIM = TTraits::Lattice::NDIM;
        const int N = TTraits::NumberOfComponents;

        double closestdist = Lattice::LX * Lattice::LY * Lattice::LZ;
        int k = 0;

        for (int kk = Lattice::HaloSize; kk < Lattice::N - Lattice::HaloSize; kk++) {  // loop over k

            double distance;
            distance = fabs(calcInterfaceDistance<TTraits>(kk));

            if (std::isnan(abs(distance))) {
                distance = Lattice::LX * Lattice::LY * Lattice::LZ;
            }

            if (distance < closestdist &&
                (Geometry<Lattice>::getBoundaryType(kk) == 0 || Geometry<Lattice>::getBoundaryType(kk) == 6)) {
                closestdist = distance;
                k = kk;
            }
        }

        if (closestdist == Lattice::LX * Lattice::LY * Lattice::LZ) {
            std::vector<double> vec(NDIM, 0);
            mGradH = vec;
        } else {
            double distarray[NDIM] = {};
            int axyz[NDIM] = {};
            axyz[0] = computeX(Lattice::LYdiv, Lattice::LZdiv, k);

            if constexpr (NDIM >= 2) {
                if (Lattice::LYdiv > 1)
                    axyz[1] = computeY(Lattice::LYdiv, Lattice::LZdiv, k);
                else
                    axyz[1] = computeZ(Lattice::LYdiv, Lattice::LZdiv, k);
            }
            if constexpr (NDIM == 3) axyz[2] = computeZ(Lattice::LYdiv, Lattice::LZdiv, k);

            double pos[NDIM];

            const double* gradorderparam = &getInstance<GradientOrderParameter, N, Lattice, NDIM>(mLiquidId)[k * NDIM];

            std::vector<double> normal = calcNormal<Lattice>(gradorderparam);

            pos[0] = closestdist * normal[0];

            if constexpr (NDIM >= 2) pos[1] = closestdist * normal[1];
            if constexpr (NDIM == 3) pos[2] = closestdist * normal[2];

            distarray[0] = fabs(pos[0]);
            distarray[1] = fabs(pos[1]);
            distarray[2] = fabs(pos[2]);
            InterfacePoint<NDIM> closest(k, distarray);
            // if (TIME%1000==0) std::cout<<closest.distance[0]<<" "<<closest.distance[1]<<std::endl;
            const double* gradhumidity = &GradientHumidity<>::get<Lattice, NDIM>(closest.index, 0);

            if (closest.index < 0 || closest.index >= Lattice::N) {
                const double* gradhumidity2 = &GradientHumidity<>::get<Lattice, NDIM>(k, 0);
                std::vector<double> grads(gradhumidity2, gradhumidity2 + NDIM);

                mGradH = grads;
            }

            else if (closest.index < Lattice::HaloSize || closest.index >= Lattice::N - Lattice::HaloSize) {
                std::vector<double> grads(NDIM);

                for (int xyz = 0; xyz < NDIM; xyz++) {
                    grads[xyz] = gradhumidity[xyz];
                }

                mGradH = grads;
            }

            else {
                // int dir1;
                using Stencil = typename TTraits::Stencil;
                using DataType = Data_Base<Lattice, Stencil>;

                DataType& data = DataType::getInstance();
                const double(&dist1)[NDIM] = closest.distance;
                double grad[NDIM] = {};
                for (int xyz = 0; xyz < NDIM; xyz++) {
                    if (normal[xyz] < 0) {
                        double neighborhumidity, neighborhumidity2;
                        std::array<int8_t, NDIM> direction{};
                        direction[xyz] = (int8_t)-1;

                        if (Geometry<Lattice>::getBoundaryType(
                                data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second)) != 0 &&
                            Geometry<Lattice>::getBoundaryType(
                                data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second)) != 6) {
                            neighborhumidity = Humidity<>::get<Lattice>(closest.index);
                            if (Geometry<Lattice>::getBoundaryType(data.getNeighbor(
                                    data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                    Stencil::QMap.find(direction)->second)) != 0 &&
                                Geometry<Lattice>::getBoundaryType(data.getNeighbor(
                                    data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                    Stencil::QMap.find(direction)->second)) != 6)
                                neighborhumidity2 = Humidity<>::get<Lattice>(closest.index);
                            else
                                neighborhumidity2 = Humidity<>::get<Lattice>(data.getNeighbor(
                                    closest.index, Stencil::Opposites[Stencil::QMap.find(direction)->second]));
                        } else if (Geometry<Lattice>::getBoundaryType(data.getNeighbor(
                                       data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                       Stencil::QMap.find(direction)->second)) != 0 &&
                                   Geometry<Lattice>::getBoundaryType(data.getNeighbor(
                                       data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                       Stencil::QMap.find(direction)->second)) != 6) {
                            neighborhumidity = Humidity<>::get<Lattice>(
                                data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
                            neighborhumidity2 = Humidity<>::get<Lattice>(
                                data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
                        } else {
                            neighborhumidity = Humidity<>::get<Lattice>(
                                data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
                            neighborhumidity2 = Humidity<>::get<Lattice>(
                                data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                                 Stencil::QMap.find(direction)->second));
                        }
                        grad[xyz] = gradhumidity[xyz] + dist1[xyz] * ((Humidity<>::get<Lattice>(closest.index)) -
                                                                      2 * neighborhumidity + neighborhumidity2);
                    }
                    if (normal[xyz] > 0) {
                        double neighborhumidity, neighborhumidity2;
                        std::array<int8_t, NDIM> direction{};
                        direction[xyz] = (int8_t)1;

                        if (Geometry<Lattice>::getBoundaryType(
                                data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second)) != 0 &&
                            Geometry<Lattice>::getBoundaryType(
                                data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second)) != 6) {
                            neighborhumidity = Humidity<>::get<Lattice>(closest.index);
                            if (Geometry<Lattice>::getBoundaryType(data.getNeighbor(
                                    data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                    Stencil::QMap.find(direction)->second)) != 0 &&
                                Geometry<Lattice>::getBoundaryType(data.getNeighbor(
                                    data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                    Stencil::QMap.find(direction)->second)) != 6)
                                neighborhumidity2 = Humidity<>::get<Lattice>(closest.index);
                            else
                                neighborhumidity2 = Humidity<>::get<Lattice>(data.getNeighbor(
                                    closest.index, Stencil::Opposites[Stencil::QMap.find(direction)->second]));
                        } else if (Geometry<Lattice>::getBoundaryType(data.getNeighbor(
                                       data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                       Stencil::QMap.find(direction)->second)) != 0 &&
                                   Geometry<Lattice>::getBoundaryType(data.getNeighbor(
                                       data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                       Stencil::QMap.find(direction)->second)) != 6) {
                            neighborhumidity = Humidity<>::get<Lattice>(
                                data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
                            neighborhumidity2 = Humidity<>::get<Lattice>(
                                data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
                        } else {
                            neighborhumidity = Humidity<>::get<Lattice>(
                                data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
                            neighborhumidity2 = Humidity<>::get<Lattice>(
                                data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                                 Stencil::QMap.find(direction)->second));
                        }
                        grad[xyz] = gradhumidity[xyz] - dist1[xyz] * ((Humidity<>::get<Lattice>(closest.index)) -
                                                                      2 * neighborhumidity + neighborhumidity2);
                    }
                }

                std::vector<double> grads(grad, grad + NDIM);

                mGradH = grads;
            }
        }
    }
#pragma omp barrier
}

template <class TTraits>
inline std::vector<double> MassLossCalculatorInterpolated::calcHumidityGradientInterface(int k, double hsat) {
    using Lattice = typename TTraits::Lattice;
    const int N = TTraits::NumberOfComponents;

    double distance;
    if constexpr (N == 2)
        distance = calcInterfaceDistance<TTraits>(k);
    else
        distance = calcInterfaceDistanceRebalance<TTraits>(k);

    if (std::isnan(abs(distance))) {
        std::vector<double> vec(Lattice::NDIM, 0);

        return vec;
    }

    const double* gradorderparam =
        &getInstance<GradientOrderParameter, N, Lattice, Lattice::NDIM>(mLiquidId)[k * Lattice::NDIM];

    std::vector<double> normal = calcNormal<Lattice>(gradorderparam);

    double neighbordistance = calcNeighborDistance<Lattice::NDIM>(normal);
#ifdef MPIPARALLEL
    if (abs(distance) > Lattice::Parallel.Width * neighbordistance) {
        std::vector<double> vec(Lattice::NDIM, 0);

        return vec;
    }
#endif
    return calculateHumidityGradientInterface<TTraits>(distance, neighbordistance, normal, k, hsat);
}

template <class TTraits>
inline std::vector<double> MassLossCalculatorInterpolated::calcHumidityGradientInterface(int k) {
    using Lattice = typename TTraits::Lattice;
    const int NDIM = TTraits::Lattice::NDIM;
    const int N = TTraits::NumberOfComponents;

    double distance;
    if constexpr (TTraits::NumberOfComponents == 2)
        distance = calcInterfaceDistance<TTraits>(k);
    else
        distance = calcInterfaceDistanceRebalance<TTraits>(k);

    if (std::isnan(abs(distance))|| std::isinf(abs(distance))|| abs(distance) >= 1000) {
        std::vector<double> vec(TTraits::Lattice::NDIM, 0);

        return vec;
    }

    const double* gradorderparam = &getInstance<GradientOrderParameter, N, Lattice, NDIM>(mLiquidId)[k * NDIM];

    std::vector<double> normal = calcNormal<typename TTraits::Lattice>(gradorderparam);

    double neighbordistance = calcNeighborDistance<TTraits::Lattice::NDIM>(normal);
#ifdef MPIPARALLEL
    if (abs(distance) > TTraits::Lattice::Parallel.Width * neighbordistance) {
        std::vector<double> vec(TTraits::Lattice::NDIM, 0);

        return vec;
    }
#endif
    return calculateHumidityGradientInterface<TTraits>(distance, neighbordistance, normal, k);
}

// Calculates the distance to the interface assuming a tanh profile of concentration

template <class TTraits>
inline double MassLossCalculatorInterpolated::calcInterfaceDistance(int k) {
    
    if (getInstance<OrderParameter, TTraits::NumberOfComponents - 1, typename TTraits::Lattice>(mLiquidId)[k]>0.01&&getInstance<OrderParameter, TTraits::NumberOfComponents - 1, typename TTraits::Lattice>(mLiquidId)[k]<0.99) return -mAlpha / 2. *
           atanh((getInstance<OrderParameter, TTraits::NumberOfComponents - 1, typename TTraits::Lattice>(mLiquidId)[k] -
                  (mPhiMin + mPhiMax) / 2.0) /
                 (fabs(mPhiMin) + fabs(mPhiMax)) * 2.0);
    else return 1000;
}

template <class TTraits>
inline double MassLossCalculatorInterpolated::calcInterfaceDistanceRebalance(int k) {
    double sum = 0;
    double sumall = 0;
    for (int component = 0; component < TTraits::NumberOfComponents - 1; component++) {
        double orderParam =
            getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TTraits::Lattice>(component)[k];
        sumall += orderParam;
        if (component != mLiquidId && component != mGasId) sum += orderParam;
    }
    if ((TTraits::NumberOfComponents > 2) && mLiquidId != TTraits::NumberOfComponents - 1 &&
        mGasId != TTraits::NumberOfComponents - 1)
        sum += 1.0 - sumall;
    const double& orderparamliquid =
        getInstance<OrderParameter, TTraits::NumberOfComponents - 1, TTraits::Lattice>(mLiquidId)[k];
    // const double& orderparamgas = getInstance<OrderParameter, TTraits::NumberOfComponents - 1,
    // TTraits::Lattice>(mGasId)[k];
    double orderparam;
    if (sum <= 0.5)
        orderparam = orderparamliquid * (1.0 + (sum) / (1.0 - sum));
    else
        orderparam = orderparamliquid;
    return -mAlpha / 2. * atanh((orderparam - (mPhiMin + mPhiMax) / 2.0) / (fabs(mPhiMin) + fabs(mPhiMax)) * 2.0);
}

template <class TTraits>
inline double MassLossCalculatorInterpolated::calcOrderParamGradientRebalance(int direction, int k) {
    using Lattice = typename TTraits::Lattice;
    using Stencil = typename TTraits::Stencil;
    const int N = TTraits::NumberOfComponents;

    using DataType = Data_Base<Lattice, Stencil>;
    DataType& data = DataType::getInstance();

    if (Geometry<Lattice>::isBoundary(k)) return 0;

    double gradientsum = 0;
    // const static auto& param = TParameter::template get<Lattice>();

    for (int idx = 1; idx < Stencil::Q; idx++) {
        if ((Geometry<Lattice>::isBoundary(data.getNeighbor(k, idx)))) {
            double sum = 0;
            double sumall = 0;
            for (int component = 0; component < N - 1; component++) {
                double orderParam = getInstance<OrderParameter, N - 1, typename TTraits::Lattice>(component)[k];
                sumall += orderParam;
                if (component != mLiquidId && component != mGasId) sum += orderParam;
            }
            if ((N > 2) && mLiquidId != N - 1 && mGasId != N - 1) sum += 1.0 - sumall;
            const double& orderparamliquid = getInstance<OrderParameter, N - 1, typename TTraits::Lattice>(mLiquidId)[k];
            // const double& orderparamgas = getInstance<OrderParameter, N - 1, TTraits::Lattice>(mGasId)[k];
            double orderparam = orderparamliquid * (1.0 + (sum) / (1.0 - sum));  ///(orderparamliquid+orderparamgas));
            if (sum <= 0.5) gradientsum += Stencil::Weights[idx] * Stencil::Ci_xyz(direction)[idx] * (orderparam);

        } else {
            double sum = 0;
            double sumall = 0;
            for (int component = 0; component < N - 1; component++) {
                double orderParam =
                    getInstance<OrderParameter, N - 1, typename TTraits::Lattice>(component)[data.getNeighbor(k, idx)];
                sumall += orderParam;
                if (component != mLiquidId && component != mGasId) sum += orderParam;
            }
            if ((N > 2) && mLiquidId != N - 1 && mGasId != N - 1) sum += 1.0 - sumall;
            const double& orderparamliquid =
                getInstance<OrderParameter, N - 1, typename TTraits::Lattice>(mLiquidId)[data.getNeighbor(k, idx)];
            // const double& orderparamgas = getInstance<OrderParameter, N - 1,
            // TTraits::Lattice>(mGasId)[data.getNeighbor(k, idx)];

            double orderparam = orderparamliquid * (1.0 + (sum) / (1.0 - sum));  ///(orderparamliquid+orderparamgas));
            if (sum <= 0.5) gradientsum += Stencil::Weights[idx] * Stencil::Ci_xyz(direction)[idx] * (orderparam);
        }
    }
    return 1.0 / (Stencil::Cs2 * Lattice::DT) * gradientsum;
}

// Returns a vector of the normal direction to the interface at the current point

template <class TLattice>
inline std::vector<double> MassLossCalculatorInterpolated::calcNormal(const double* gradorderparam) {
    double magnitudegradient2 = 0;

    for (int xyz = 0; xyz < TLattice::NDIM; xyz++) {
        magnitudegradient2 += pow(gradorderparam[xyz], 2);
    }

    double magnitudegradient = sqrt(magnitudegradient2);

    std::vector<double> normal(TLattice::NDIM, 0);

    for (int xyz = 0; xyz < TLattice::NDIM; xyz++) {
        normal[xyz] = -(1 - 2 * mPhiGas) * gradorderparam[xyz] / magnitudegradient;
    }

    return normal;
}

// Calculates the distance from the point we are on to the next cell in the normal direction. This allows us to set a
// cutoff distance that will always contain a given number of lattice points.

template <int TNDIM>
inline double MassLossCalculatorInterpolated::calcNeighborDistance(const std::vector<double>& normal) {
    double zfactor = 1;
    double xyfactor = 1;
    double magxy = sqrt(pow(normal[0], 2) + pow(normal[1], 2));
    if constexpr (TNDIM >= 3) {
        if (normal[2] < magxy && magxy > 0) {
            zfactor = cos(atan(normal[2] / magxy));
        } else if (normal[2] > 0) {
            zfactor = cos(atan(magxy / normal[2]));
        } else {
            zfactor = 0;
        }
    }
    if constexpr (TNDIM >= 2) {
        if (fabs(normal[1]) < fabs(normal[0])) {
            xyfactor = cos(atan(fabs(normal[1]) / fabs(normal[0])));
        } else if (fabs(normal[0]) <= fabs(normal[1])) {
            xyfactor = cos(atan(fabs(normal[0]) / fabs(normal[1])));
        } else {
            xyfactor = 0;
        }
    }
    double neighbordistance = 1. / (xyfactor * zfactor);
    if (abs(neighbordistance < 1)) return 1;
    if (abs(neighbordistance) > sqrt(2)) return sqrt(2);
    return neighbordistance;
}

// We want to find the point on the interface intersected by the normal vector from the original 'k' point. This
// approximates this by taking the closest value of k to the interface point.

template <class TTraits>
inline auto MassLossCalculatorInterpolated::findClosestKToInterface(const double& interfacedistance,
                                                                    const double& neighbordistance,
                                                                    const std::vector<double>& normal, int k) {
    double distarray[TTraits::Lattice::NDIM] = {};
    int axyz[TTraits::Lattice::NDIM] = {};
    axyz[0] = computeX(TTraits::Lattice::LYdiv, TTraits::Lattice::LZdiv, k);

    if constexpr (TTraits::Lattice::NDIM >= 2) {
        if (TTraits::Lattice::LYdiv > 1)
            axyz[1] = computeY(TTraits::Lattice::LYdiv, TTraits::Lattice::LZdiv, k);
        else
            axyz[1] = computeZ(TTraits::Lattice::LYdiv, TTraits::Lattice::LZdiv, k);
    }
    if constexpr (TTraits::Lattice::NDIM == 3) axyz[2] = computeZ(TTraits::Lattice::LYdiv, TTraits::Lattice::LZdiv, k);

    double pos[TTraits::Lattice::NDIM];

    pos[0] = axyz[0] - interfacedistance * normal[0];

    if constexpr (TTraits::Lattice::NDIM >= 2) pos[1] = axyz[1] - interfacedistance * normal[1];
    if constexpr (TTraits::Lattice::NDIM == 3) pos[2] = axyz[2] - interfacedistance * normal[2];
    
  
    int closest = -1;
#ifdef MPIPARALLEL
    double lowestdist = TTraits::Lattice::Parallel.Width * neighbordistance;
#else
    double lowestdist = 999;
#endif

    for (int xi = -1; xi < 4 * round(mAlpha / 4.0)-1; xi++) {
        for (int yi = -1; yi < 2 * (1 + (TTraits::Lattice::NDIM >= 2)) * round(mAlpha / 4.0)-1; yi++) {
            for (int zi = 0; zi < (1 + (TTraits::Lattice::NDIM >= 3)) * round(mAlpha / 4.0); zi++) {
                int kidx;
                double dist2;
                double tempdist[TTraits::Lattice::NDIM] = {};

                if constexpr (TTraits::Lattice::NDIM == 2) {
                    kidx = computeK<typename TTraits::Lattice>((int)pos[0] + xi,// - TTraits::Lattice::HaloXWidth,
                                                               (int)pos[1] + yi, 0);
                    tempdist[0] = fabs(pos[0] - ((int)pos[0] + xi));
                    tempdist[1] = fabs(pos[1] - ((int)pos[1] + yi));
                    dist2 = pow(tempdist[0], 2) + pow(tempdist[1], 2);
                    
                    if (kidx<0||kidx>=TTraits::Lattice::N) {
                        dist2 = 9999999;
                    }
                    else{
                        dist2 = pow(fabs(calcInterfaceDistance<TTraits>(kidx)),2);
                        dist2 = pow(tempdist[0], 2) + pow(tempdist[1], 2);
                    }
                } else if constexpr (TTraits::Lattice::NDIM == 3) {
                    kidx = computeK<typename TTraits::Lattice>((int)pos[0] + xi - TTraits::Lattice::HaloXWidth,
                                                               (int)pos[1] + yi, (int)pos[2] + zi);
                    tempdist[0] = fabs(pos[0] - ((int)pos[0] + xi));
                    tempdist[1] = fabs(pos[1] - ((int)pos[1] + yi));
                    tempdist[2] = fabs(pos[2] - ((int)pos[2] + zi));
                    dist2 = pow(tempdist[0], 2) + pow(tempdist[1], 2) + pow(tempdist[2], 2);
                } else {
                    kidx = (int)pos[0] + xi;
                    tempdist[0] = fabs(pos[0] - ((int)pos[0] + xi));
                    dist2 = pow(tempdist[0], 2);
                }
                double dist = sqrt(dist2);
                if (dist < lowestdist) {
                    if (Geometry<typename TTraits::Lattice>::getBoundaryType(kidx) == 0 ||
                        Geometry<typename TTraits::Lattice>::getBoundaryType(kidx) == 6) {
                        lowestdist = dist;
                        std::copy(tempdist, tempdist + TTraits::Lattice::NDIM, distarray);
                        closest = kidx;
                    }
                }
            }
        }
    }

    /*if (closest<0){
        //closest = computeK<typename TTraits::Lattice>(round(pos[0]),round(pos[1]),round(pos[2]));

        if constexpr (TTraits::Lattice::NDIM==2) {
            int c = computeK<typename TTraits::Lattice>(round(pos[0]),round(pos[1]),0);
            if (Geometry<typename TTraits::Lattice>::getBoundaryType(c) == 0 ||
                        Geometry<typename TTraits::Lattice>::getBoundaryType(c) == 6) closest = c;
        }
        else if constexpr (TTraits::Lattice::NDIM==3) {
            int c = computeK<typename TTraits::Lattice>(round(pos[0]),round(pos[1]),round(pos[2]));
            if (Geometry<typename TTraits::Lattice>::getBoundaryType(c) == 0 ||
                        Geometry<typename TTraits::Lattice>::getBoundaryType(c) == 6) closest = c;
        }
        else {

            int c = round(pos[0]);//computeK<typename TTraits::Lattice>((int)pos[0]+xi,(int)pos[1]+yi,(int)pos[2]+zi);
            if (Geometry<typename TTraits::Lattice>::getBoundaryType(c) == 0 ||
                        Geometry<typename TTraits::Lattice>::getBoundaryType(c) == 6) closest = c;

        }
        double disttoclosest = calcInterfaceDistanceRebalance<TTraits>(k);
        if (std::isnan(abs(disttoclosest))) disttoclosest=TTraits::Lattice::Parallel.Width+1;

        for(int xi=0;xi<2;xi++){
            for(int yi=0;yi<(1+(TTraits::Lattice::NDIM>=2));yi++){
                for(int zi=0;zi<(1+(TTraits::Lattice::NDIM>=3));zi++){
                    int kidx;
                    double dist2;
                    double tempdist[TTraits::Lattice::NDIM] = {};

                    if constexpr (TTraits::Lattice::NDIM==2) {
                        kidx = computeK<typename TTraits::Lattice>(round(pos[0])+xi,round(pos[1])+yi,0);
                        tempdist[0] = fabs(pos[0]-(round(pos[0])+xi));
                        tempdist[1] = fabs(pos[1]-(round(pos[1])+yi));
                        dist2 = pow(tempdist[0],2) + pow(tempdist[1],2);
                    }
                    if constexpr (TTraits::Lattice::NDIM==3) {
                        kidx = computeK<typename TTraits::Lattice>(round(pos[0])+xi,round(pos[1])+yi,round(pos[2])+zi);
                        tempdist[0] = fabs(pos[0]-(round(pos[0])+xi));
                        tempdist[1] = fabs(pos[1]-(round(pos[1])+yi));
                        tempdist[2] = fabs(pos[2]-(round(pos[2])+zi));
                        dist2 = pow(tempdist[0],2) + pow(tempdist[1],2) + pow(tempdist[2],2);
                    }
                    else{
                        kidx = round(pos[0])+xi;//computeK<typename
                        tempdist[0] = fabs(pos[0]-(round(pos[0])+xi));
                        dist2 = pow(tempdist[0],2);
                    }

                    double dist=sqrt(dist2);

                    if (dist<lowestdist) { //&&dist<9999*neighbordistance
                        if (Geometry<typename TTraits::Lattice>::getBoundaryType(kidx)==0||Geometry<typename
    TTraits::Lattice>::getBoundaryType(kidx)==6){ lowestdist = dist;
                            std::copy(tempdist,tempdist+TTraits::Lattice::NDIM,distarray);
                            closest = kidx;

                        }
                    }
                }
            }
        }
    }

    if (closest<0){
        //closest = computeK<typename TTraits::Lattice>(round(pos[0]),round(pos[1]),round(pos[2]));

        if constexpr (TTraits::Lattice::NDIM==2) {
            int c = computeK<typename TTraits::Lattice>((int)pos[0]+1,(int)pos[1]+1,0);
            if (Geometry<typename TTraits::Lattice>::getBoundaryType(c) == 0 ||
                        Geometry<typename TTraits::Lattice>::getBoundaryType(c) == 6) closest = c;
        }
        else if constexpr (TTraits::Lattice::NDIM==3) {
            int c = computeK<typename TTraits::Lattice>((int)pos[0]+1,(int)pos[1]+1,(int)pos[2]+1);
            if (Geometry<typename TTraits::Lattice>::getBoundaryType(c) == 0 ||
                        Geometry<typename TTraits::Lattice>::getBoundaryType(c) == 6) closest = c;
        }
        else {

            int c = round((int)pos[0]+1);//computeK<typename
    TTraits::Lattice>((int)pos[0]+xi,(int)pos[1]+yi,(int)pos[2]+zi); if (Geometry<typename
    TTraits::Lattice>::getBoundaryType(c) == 0 || Geometry<typename TTraits::Lattice>::getBoundaryType(c) == 6) closest
    = c;

        }
        double disttoclosest = calcInterfaceDistanceRebalance<TTraits>(k);
        if (std::isnan(abs(disttoclosest))) disttoclosest=TTraits::Lattice::Parallel.Width+1;

        for (int xi = 0; xi < 2; xi++) {
        for (int yi = 0; yi < (1 + (TTraits::Lattice::NDIM >= 2)); yi++) {
            for (int zi = 0; zi < (1 + (TTraits::Lattice::NDIM >= 3)); zi++) {
                int kidx;
                double dist2;
                double tempdist[TTraits::Lattice::NDIM] = {};

                if constexpr (TTraits::Lattice::NDIM == 2) {
                    kidx = computeK<typename TTraits::Lattice>((int)pos[0] + 2*xi - TTraits::Lattice::HaloXWidth,
                                                               (int)pos[1] + 2*yi, 0);
                    tempdist[0] = fabs(pos[0] - ((int)pos[0] + 2*xi));
                    tempdist[1] = fabs(pos[1] - ((int)pos[1] + 2*yi));
                    dist2 = pow(tempdist[0], 2) + pow(tempdist[1], 2);
                } else if constexpr (TTraits::Lattice::NDIM == 3) {
                    kidx = computeK<typename TTraits::Lattice>((int)pos[0] + 2*xi - TTraits::Lattice::HaloXWidth,
                                                               (int)pos[1] + 2*yi, (int)pos[2] +1 + zi);
                    tempdist[0] = fabs(pos[0] - ((int)pos[0] + 2*xi));
                    tempdist[1] = fabs(pos[1] - ((int)pos[1] + 2*yi));
                    tempdist[2] = fabs(pos[2] - ((int)pos[2] + 2*zi));
                    dist2 = pow(tempdist[0], 2) + pow(tempdist[1], 2) + pow(tempdist[2], 2);
                } else {
                    kidx = (int)pos[0] + 2*xi;
                    tempdist[0] = fabs(pos[0] - ((int)pos[0] + 2*xi));
                    dist2 = pow(tempdist[0], 2);
                }
                double dist = sqrt(dist2);
                if (dist < lowestdist) {
                    if (Geometry<typename TTraits::Lattice>::getBoundaryType(kidx) == 0 ||
                        Geometry<typename TTraits::Lattice>::getBoundaryType(kidx) == 6) {
                        lowestdist = dist;
                        std::copy(tempdist, tempdist + TTraits::Lattice::NDIM, distarray);
                        closest = kidx;
                    }
                }
            }
        }
    }
    }*/
    
    if (closest < 0) {
        distarray[TTraits::Lattice::NDIM] = {};
        InterfacePoint<TTraits::Lattice::NDIM> tempP(-1, distarray);
        return tempP;
    }

    InterfacePoint<TTraits::Lattice::NDIM> tempP(closest, distarray);
    return tempP;
    

}

template <class TTraits>
inline std::vector<double> MassLossCalculatorInterpolated::calculateHumidityGradientInterfaceClosest(
    const double& interfacedistance, const double& neighbordistance, const std::vector<double>& normal, int k) {
    InterfacePoint<TTraits::Lattice::NDIM> closest =
        findClosestKToInterface<TTraits>(interfacedistance, neighbordistance, normal, k);

    const double* gradhumidity =
        &GradientHumidity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(closest.index, 0);

    if (closest.index < 0 || closest.index >= TTraits::Lattice::N) {
        const double* gradhumidity2 = &GradientHumidity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, 0);
        std::vector<double> grads(gradhumidity2, gradhumidity2 + TTraits::Lattice::NDIM);

        return grads;
    }

    std::vector<double> grads(TTraits::Lattice::NDIM, 0);

    for (int xyz = 0; xyz < TTraits::Lattice::NDIM; xyz++) {
        grads[xyz] = gradhumidity[xyz];
    }

    return grads;
}

template <class TTraits>
inline std::vector<double> MassLossCalculatorInterpolated::calculateHumidityGradientInterface(
    const double& interfacedistance, const double& neighbordistance, const std::vector<double>& normal, int k) {
    using Lattice = typename TTraits::Lattice;
    using Stencil = typename TTraits::Stencil;

    using DataType = Data_Base<Lattice, Stencil>;

    DataType& data = DataType::getInstance();

    InterfacePoint<Lattice::NDIM> closest =
        findClosestKToInterface<TTraits>(interfacedistance, neighbordistance, normal, k);

    const double* gradhumidity =
        &GradientHumidity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(closest.index, 0);

    
    if (closest.index < 0 || closest.index >= Lattice::N) {
        const double* gradhumidity2 = &GradientHumidity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, 0);
        std::vector<double> grads(gradhumidity2, gradhumidity2 + Lattice::NDIM);

        return grads;
    }

    else if (closest.index < Lattice::HaloSize || closest.index >= Lattice::N - Lattice::HaloSize) {
        std::vector<double> grads(Lattice::NDIM);

        for (int xyz = 0; xyz < Lattice::NDIM; xyz++) {
            grads[xyz] = gradhumidity[xyz];
        }

        return grads;
    }


    // int dir1;
    const double(&dist1)[Lattice::NDIM] = closest.distance;
    double grad[Lattice::NDIM] = {};
    for (int xyz = 0; xyz < Lattice::NDIM; xyz++) {
        if (normal[xyz] < 0) {
            double neighborhumidity, neighborhumidity2;
            std::array<int8_t, Lattice::NDIM> direction{};
            direction[xyz] = (int8_t)-1;

            if (Geometry<Lattice>::getBoundaryType(
                    data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second)) != 0 &&
                Geometry<Lattice>::getBoundaryType(
                    data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second)) != 6) {
                neighborhumidity = Humidity<>::get<Lattice>(closest.index);
                if (Geometry<Lattice>::getBoundaryType(
                        data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                         Stencil::QMap.find(direction)->second)) != 0 &&
                    Geometry<Lattice>::getBoundaryType(
                        data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                         Stencil::QMap.find(direction)->second)) != 6)
                    neighborhumidity2 = Humidity<>::get<Lattice>(closest.index);
                else
                    neighborhumidity2 = Humidity<>::get<Lattice>(
                        data.getNeighbor(closest.index, Stencil::Opposites[Stencil::QMap.find(direction)->second]));
            } else if (Geometry<Lattice>::getBoundaryType(
                           data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                            Stencil::QMap.find(direction)->second)) != 0 &&
                       Geometry<Lattice>::getBoundaryType(
                           data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                            Stencil::QMap.find(direction)->second)) != 6) {
                neighborhumidity =
                    Humidity<>::get<Lattice>(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
                neighborhumidity2 =
                    Humidity<>::get<Lattice>(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
            } else {
                neighborhumidity =
                    Humidity<>::get<Lattice>(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
                neighborhumidity2 = Humidity<>::get<Lattice>(
                    data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                     Stencil::QMap.find(direction)->second));
            }
            grad[xyz] = gradhumidity[xyz] + dist1[xyz] * ((Humidity<>::get<Lattice>(closest.index)) -
                                                          2 * neighborhumidity + neighborhumidity2);
        }
        if (normal[xyz] > 0) {
            double neighborhumidity, neighborhumidity2;
            std::array<int8_t, Lattice::NDIM> direction{};
            direction[xyz] = (int8_t)1;

            if (Geometry<Lattice>::getBoundaryType(
                    data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second)) != 0 &&
                Geometry<Lattice>::getBoundaryType(
                    data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second)) != 6) {
                neighborhumidity = Humidity<>::get<Lattice>(closest.index);
                if (Geometry<Lattice>::getBoundaryType(
                        data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                         Stencil::QMap.find(direction)->second)) != 0 &&
                    Geometry<Lattice>::getBoundaryType(
                        data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                         Stencil::QMap.find(direction)->second)) != 6)
                    neighborhumidity2 = Humidity<>::get<Lattice>(closest.index);
                else
                    neighborhumidity2 = Humidity<>::get<Lattice>(
                        data.getNeighbor(closest.index, Stencil::Opposites[Stencil::QMap.find(direction)->second]));
            } else if (Geometry<Lattice>::getBoundaryType(
                           data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                            Stencil::QMap.find(direction)->second)) != 0 &&
                       Geometry<Lattice>::getBoundaryType(
                           data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                            Stencil::QMap.find(direction)->second)) != 6) {
                neighborhumidity =
                    Humidity<>::get<Lattice>(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
                neighborhumidity2 =
                    Humidity<>::get<Lattice>(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
            } else {
                neighborhumidity =
                    Humidity<>::get<Lattice>(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
                neighborhumidity2 = Humidity<>::get<Lattice>(
                    data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                     Stencil::QMap.find(direction)->second));
            }
            grad[xyz] = gradhumidity[xyz] - dist1[xyz] * ((Humidity<>::get<Lattice>(closest.index)) -
                                                          2 * neighborhumidity + neighborhumidity2);
        }
    }

    std::vector<double> grads(grad, grad + Lattice::NDIM);

    return grads;
}

template <class Lattice>
double distancefunctest(int k, int idx) {
    using Stencil = std::conditional_t<Lattice::NDIM == 1, D1Q3, std::conditional_t<Lattice::NDIM == 2, D2Q9, D3Q19>>;
    ;

    double normaldist = -0.5 * 4 * atanh((OrderParameter<>::get<Lattice>(k) - 0.5) / 0.5);
    double* gradorderparam = GradientOrderParameter<>::getAddress<Lattice, Lattice::NDIM>(k, 0);

    std::vector<double> normal;
    if constexpr (Lattice::NDIM == 1)
        normal = {-gradorderparam[0] / sqrt(pow(gradorderparam[0], 2))};
    else if constexpr (Lattice::NDIM == 2)
        normal = {-gradorderparam[0] / sqrt(pow(gradorderparam[0], 2) + pow(gradorderparam[1], 2)),
                  -gradorderparam[1] / sqrt(pow(gradorderparam[0], 2) + pow(gradorderparam[1], 2))};
    else
        normal = {-gradorderparam[0] /
                      sqrt(pow(gradorderparam[0], 2) + pow(gradorderparam[1], 2) + pow(gradorderparam[2], 2)),
                  -gradorderparam[1] /
                      sqrt(pow(gradorderparam[0], 2) + pow(gradorderparam[1], 2) + pow(gradorderparam[2], 2)),
                  -gradorderparam[2] /
                      sqrt(pow(gradorderparam[0], 2) + pow(gradorderparam[1], 2) + pow(gradorderparam[2], 2))};
    double normdotci = 0;
    double magci = 0;
    for (int xyz = 0; xyz < Lattice::NDIM; xyz++) {
        normdotci += normal[xyz] * Stencil::Ci_xyz(xyz)[idx];
        magci += Stencil::Ci_xyz(xyz)[idx] * Stencil::Ci_xyz(xyz)[idx];
    }
    double dist = fabs(normaldist / normdotci);
    // return 0.5;
    if (idx == 0 || std::isnan(dist) || std::isinf(dist)) {
        return 0.5;
    } else if (dist <= 0.0001) {
        return 0.0001;
    } else if (dist >= 0.9999) {
        return 0.9999;
    }

    return dist;
}

#include "../GradientStencils/CentralXYZInterface.hh"

template <class TTraits>
inline std::vector<double> MassLossCalculatorInterpolated::calculateHumidityGradientInterface(
    const double& interfacedistance, const double& neighbordistance, const std::vector<double>& normal, int k,
    double hsat) {
    using Lattice = typename TTraits::Lattice;
    using Stencil = typename TTraits::Stencil;

    using DataType = Data_Base<Lattice, Stencil>;

    DataType& data = DataType::getInstance();

    InterfacePoint<Lattice::NDIM> closest =
        findClosestKToInterface<TTraits>(interfacedistance, neighbordistance, normal, k);

    // const double* gradhumidity =
    //     &GradientHumidity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(closest.index, 0);
    CentralXYZInterface gradscheme;
    gradscheme.setInterfaceDistance(distancefunctest<Lattice>);
    gradscheme.setInterfaceVal(hsat);
    gradscheme.setBoundaryID({1, 3, 4, 5});

    if (closest.index < 0 || closest.index >= Lattice::N) {
        const double* gradhumidity2 = &GradientHumidity<>::get<typename TTraits::Lattice, TTraits::Lattice::NDIM>(k, 0);
        std::vector<double> grads(gradhumidity2, gradhumidity2 + Lattice::NDIM);

        return grads;
    }
    std::vector<double> gradhumidity(Lattice::NDIM);
    for (int xyz = 0; xyz < Lattice::NDIM; xyz++) {
        gradhumidity[xyz] = gradscheme.compute<TTraits, Humidity<>>(xyz, closest.index);
    }
    if (closest.index < Lattice::HaloSize || closest.index >= Lattice::N - Lattice::HaloSize) {
        std::vector<double> grads(Lattice::NDIM);

        for (int xyz = 0; xyz < Lattice::NDIM; xyz++) {
            grads[xyz] = gradhumidity[xyz];
        }

        return grads;
    }

    // int dir1;
    const double(&dist1)[Lattice::NDIM] = closest.distance;
    double grad[Lattice::NDIM] = {};
    for (int xyz = 0; xyz < Lattice::NDIM; xyz++) {
        if (normal[xyz] < 0) {
            double neighborhumidity, neighborhumidity2;
            std::array<int8_t, Lattice::NDIM> direction{};
            direction[xyz] = (int8_t)-1;

            if (Geometry<Lattice>::getBoundaryType(
                    data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second)) != 0 &&
                Geometry<Lattice>::getBoundaryType(
                    data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second)) != 6) {
                neighborhumidity = Humidity<>::get<Lattice>(closest.index);
                if (Geometry<Lattice>::getBoundaryType(
                        data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                         Stencil::QMap.find(direction)->second)) != 0 &&
                    Geometry<Lattice>::getBoundaryType(
                        data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                         Stencil::QMap.find(direction)->second)) != 6)
                    neighborhumidity2 = Humidity<>::get<Lattice>(closest.index);
                else
                    neighborhumidity2 = Humidity<>::get<Lattice>(
                        data.getNeighbor(closest.index, Stencil::Opposites[Stencil::QMap.find(direction)->second]));
            } else if (Geometry<Lattice>::getBoundaryType(
                           data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                            Stencil::QMap.find(direction)->second)) != 0 &&
                       Geometry<Lattice>::getBoundaryType(
                           data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                            Stencil::QMap.find(direction)->second)) != 6) {
                neighborhumidity =
                    Humidity<>::get<Lattice>(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
                neighborhumidity2 =
                    Humidity<>::get<Lattice>(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
            } else {
                neighborhumidity =
                    Humidity<>::get<Lattice>(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
                neighborhumidity2 = Humidity<>::get<Lattice>(
                    data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                     Stencil::QMap.find(direction)->second));
            }
            grad[xyz] = gradhumidity[xyz] + dist1[xyz] * ((Humidity<>::get<Lattice>(closest.index)) -
                                                          2 * neighborhumidity + neighborhumidity2);
        }
        if (normal[xyz] > 0) {
            double neighborhumidity, neighborhumidity2;
            std::array<int8_t, Lattice::NDIM> direction{};
            direction[xyz] = (int8_t)1;

            if (Geometry<Lattice>::getBoundaryType(
                    data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second)) != 0 &&
                Geometry<Lattice>::getBoundaryType(
                    data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second)) != 6) {
                neighborhumidity = Humidity<>::get<Lattice>(closest.index);
                if (Geometry<Lattice>::getBoundaryType(
                        data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                         Stencil::QMap.find(direction)->second)) != 0 &&
                    Geometry<Lattice>::getBoundaryType(
                        data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                         Stencil::QMap.find(direction)->second)) != 6)
                    neighborhumidity2 = Humidity<>::get<Lattice>(closest.index);
                else
                    neighborhumidity2 = Humidity<>::get<Lattice>(
                        data.getNeighbor(closest.index, Stencil::Opposites[Stencil::QMap.find(direction)->second]));
            } else if (Geometry<Lattice>::getBoundaryType(
                           data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                            Stencil::QMap.find(direction)->second)) != 0 &&
                       Geometry<Lattice>::getBoundaryType(
                           data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                            Stencil::QMap.find(direction)->second)) != 6) {
                neighborhumidity =
                    Humidity<>::get<Lattice>(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
                neighborhumidity2 =
                    Humidity<>::get<Lattice>(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
            } else {
                neighborhumidity =
                    Humidity<>::get<Lattice>(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second));
                neighborhumidity2 = Humidity<>::get<Lattice>(
                    data.getNeighbor(data.getNeighbor(closest.index, Stencil::QMap.find(direction)->second),
                                     Stencil::QMap.find(direction)->second));
            }
            grad[xyz] = gradhumidity[xyz] - dist1[xyz] * ((Humidity<>::get<Lattice>(closest.index)) -
                                                          2 * neighborhumidity + neighborhumidity2);
        }
    }

    std::vector<double> grads(grad, grad + Lattice::NDIM);

    return grads;
}

class MassLossCalculatorClausiusClapeyron : public MassLossCalculatorInterpolated {
   public:
    template <class TTraits>
    inline void compute(int k);

    template <class TTraits>
    inline std::vector<double> calcHumidityGradientInterfaceNearestNeighbor(int k);

    template <class TTraits>
    inline std::vector<double> calcHumidityGradientInterface(int k);

    template <class TTraits>
    inline std::vector<double> calculateHumidityGradientInterfaceClosest(const double& interfacedistance,
                                                                         const double& neighbordistance,
                                                                         const std::vector<double>& normal, int k);

    template <class TTraits>
    inline std::vector<double> calculateHumidityGradientInterface(const double& interfacedistance,
                                                                  const double& neighbordistance,
                                                                  const std::vector<double>& normal, int k);

    inline std::vector<double> calculateHumidityGradientInterfaceTwoPoints();

    inline std::vector<double> calculateHumidityGradientInterfaceThreePoints();

    template <class TTraits>
    inline void communicate();
};
