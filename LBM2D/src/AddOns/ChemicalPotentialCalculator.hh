#pragma once
#include "../Geometry.hh"
#include "../Lattice.hh"
#include "../Parameters.hh"
#include "../Service.hh"
#include "AddOnBase.hh"

class ChemicalPotentialCalculatorBinary : public AddOnBase {
   public:
    template <class TTraits>
    inline void compute(int k);

    inline void setA(double A);

    inline void setKappa(double kappa);

    double mA;
    double mKappa;
};

template <class TTraits>
inline void ChemicalPotentialCalculatorBinary::compute(int k) {
    using Lattice = typename TTraits::Lattice;

    if (Geometry<Lattice>::isBulkSolid(k)) return;

    const double& orderparameter = OrderParameter<>::get<Lattice>(k);
    const double& laplacian = LaplacianOrderParameter<>::get<Lattice>(k);

    ChemicalPotential<>::get<Lattice>(k) = -mA * orderparameter + mA * pow(orderparameter, 3) - mKappa * laplacian;
}

inline void ChemicalPotentialCalculatorBinary::setA(double A) { mA = A; }

inline void ChemicalPotentialCalculatorBinary::setKappa(double kappa) { mKappa = kappa; }

class ChemicalPotentialCalculatorRho : public AddOnBase {
   public:
    template <class TTraits>
    inline void compute(int k);

    inline void setA(const double A);

    inline void setKappa(const double kappa);

    inline void setRhoLiquid(const double rho) { mRhoLiquid = rho; };

    inline void setRhoGas(const double rho) { mRhoVapor = rho; };

    inline void setRho(const double rhol, const double rhov) {
        mRhoVapor = rhov;
        mRhoLiquid = rhol;
    };

    double mA;
    double mKappa;
    double mRhoLiquid;
    double mRhoVapor;
};

template <class TTraits>
inline void ChemicalPotentialCalculatorRho::compute(int k) {
    using Lattice = typename TTraits::Lattice;

    if (Geometry<Lattice>::isBulkSolid(k)) return;

    const double& density = Density<>::get<Lattice>(k);
    const double& laplacian = LaplacianDensity<>::get<Lattice>(k);

    ChemicalPotential<>::get<Lattice>(k) =
        2 * mA * (density - mRhoLiquid) * (density - mRhoVapor) * (2 * density - mRhoLiquid - mRhoVapor) -
        mKappa * laplacian;
}

inline void ChemicalPotentialCalculatorRho::setA(double A) { mA = A; }

inline void ChemicalPotentialCalculatorRho::setKappa(double kappa) { mKappa = kappa; }

class ChemicalPotentialCalculatorBinaryLee : public AddOnBase {
   public:
    template <class TTraits>
    inline void compute(int k);

    template <class TTraits>
    inline void communicate() {
        using Lattice = typename TTraits::Lattice;
        Lattice::communicate(ChemicalPotential<>::getInstance<Lattice>());
    }

    inline void setA(double A);

    inline void setKappa(double kappa);

    inline void setOmega(double omega);

    double mA;
    double mKappa;
    double mOmega = 0.000;
};

template <class TTraits>
inline void ChemicalPotentialCalculatorBinaryLee::compute(int k) {
    using Lattice = typename TTraits::Lattice;

    if (Geometry<Lattice>::isBulkSolid(k)) return;

    const double& orderparameter = OrderParameter<>::get<Lattice>(k);
    const double& laplacian = LaplacianOrderParameter<>::get<Lattice>(k);

    ChemicalPotential<>::get<Lattice>(k) = 2 * mA * orderparameter - 6 * mA * pow(orderparameter, 2) +
                                           4 * mA * pow(orderparameter, 3) -
                                           //2 * mOmega * (orderparameter < 0) * orderparameter-
                                           //2 * mOmega * ((1-orderparameter) < 0) * (1-orderparameter)-
                                            mKappa * laplacian;
    ChemicalPotential2<>::get<Lattice>(k) = 2 * mA * orderparameter - 6 * mA * pow(orderparameter, 2) +
                                           4 * mA * pow(orderparameter, 3) +
                                           2 * mOmega * (orderparameter < 0) * orderparameter-
                                           //2 * mOmega * ((1-orderparameter) < 0) * (1-orderparameter)-
                                            mKappa * laplacian;

}

inline void ChemicalPotentialCalculatorBinaryLee::setA(double A) { mA = A; }

inline void ChemicalPotentialCalculatorBinaryLee::setKappa(double kappa) { mKappa = kappa; }

inline void ChemicalPotentialCalculatorBinaryLee::setOmega(double omega) { mOmega = omega; }


class ChemicalPotentialCalculatorFrozen : public AddOnBase {
   public:
    template <class TTraits>
    inline void compute(int k);

    template <class TTraits>
    inline void communicate() {
        using Lattice = typename TTraits::Lattice;
        Lattice::communicate(ChemicalPotential<>::getInstance<Lattice>());
    }

    inline void setSurfaceTension(double A);

    inline void setKappas(double kappa1,double kappa2);

    inline void setInterfaceWidth(double w);

    inline void setOmega(double omega) { mOmega = omega; }

    double mOmega = 0.000;

    double mS;
    double mKappa1;
    double mKappa2;
    double mW = 4.0;
};

template <class TTraits>
inline void ChemicalPotentialCalculatorFrozen::compute(int k) {
    using Lattice = typename TTraits::Lattice;

    if (Geometry<Lattice>::isBulkSolid(k)) return;

    const double& c = OrderParameter<>::get<Lattice>(k);
    const double& cF = OrderParameter<1>::get<Lattice>(k);
    const double& laplacian = LaplacianOrderParameter<>::get<Lattice>(k);
    const double& laplacianF = LaplacianOrderParameter<1>::get<Lattice>(k);

    ChemicalPotential<>::get<Lattice>(k) = mKappa1*c*c*(2*c-2)/2.+(mKappa2*(2*c + 2*cF)*pow((c + cF - 1),2))/2. 
                                           + (mKappa2*pow((c + cF),2)*(2*c + 2*cF - 2))/2. + mKappa1*c*pow((c - 1),2)
                                           - mW*mW/16.*(mKappa1+mKappa2)*laplacian-mW*mW/16.*(mKappa2)*laplacianF+
                                           2 * mOmega * (c < 0) * c-
                                           2 * mOmega * ((1-c) < 0) * (1-c);

}

inline void ChemicalPotentialCalculatorFrozen::setSurfaceTension(double s) { mKappa2 *= s/mS;
                                                                             mKappa1 *= s/mS;
                                                                             mS = s; }

inline void ChemicalPotentialCalculatorFrozen::setKappas(double kappa1,double kappa2) { mKappa2 = 6*mS/mW*kappa2;
                                                                          mKappa1 = 6*mS/mW*kappa1;}

inline void ChemicalPotentialCalculatorFrozen::setInterfaceWidth(double w) { mKappa2 *= mW/w;
                                                                             mKappa1 *= mW/w;
                                                                             mW = w; }


class ChemicalPotentialCalculatorTernaryLee : public AddOnBase {
   public:
    template <class TTraits>
    inline void compute(int k);

    inline void setSurfaceTension(double sigma12, double sigma13, double sigma23);

    inline void setOmega(double omega);

    inline void setLambda(double lambda);

    inline void setInterfaceWidth(double interfacewidth);

    template <class TTraits>
    inline void communicate() {
        using Lattice = typename TTraits::Lattice;
        Lattice::communicate(ChemicalPotential<0>::getInstance<Lattice>());
        Lattice::communicate(ChemicalPotential<1>::getInstance<Lattice>());
        Lattice::communicate(ChemicalPotential<2>::getInstance<Lattice>());
    }

    std::array<double, 3> mSigma;
    std::array<double, 3> mGamma;
    double mGammaT;
    double mOmega = 0.000;
    double mLambda = 0;
    double mInterfaceWidth;
};

template <class TTraits>
inline void ChemicalPotentialCalculatorTernaryLee::compute(int k) {
    using Lattice = typename TTraits::Lattice;

    if (Geometry<Lattice>::isBulkSolid(k)) return;

    const double C1 = OrderParameter<0>::get<Lattice>(k);  // C1, C2, C3
    const double C2 = OrderParameter<1>::get<Lattice>(k);
    const double C3 = 1 - C1 - C2;
    using Lap1 = LaplacianOrderParameter<0>;
    using Lap2 = LaplacianOrderParameter<1>;

    double dEd12 = mGamma[0] * C1 * (1 - 3 * C1 + 2 * C1 * C1) - mGamma[1] * C2 * (1 - 3 * C2 + 2 * C2 * C2) +
                   2 * mLambda * C3 * C3 * (C2 * C2 * C1 - C1 * C1 * C2) + (C1 < 0) * 2 * mOmega * C1 -
                   (C2 < 0) * 2 * mOmega * C2 - 2 * mOmega * ((1-C1) < 0) * (1-C1) + 2 * mOmega * ((1-C2) < 0) * (1-C2);

    double dEd13 = mGamma[0] * C1 * (1 - 3 * C1 + 2 * C1 * C1) - mGamma[2] * C3 * (1 - 3 * C3 + 2 * C3 * C3) +
                   2 * mLambda * C2 * C2 * (C3 * C3 * C1 - C1 * C1 * C3) + (C1 < 0) * 2 * mOmega * C1 -
                   (C3 < 0) * 2 * mOmega * C3 - 2 * mOmega * ((1-C1) < 0) * (1-C1) + 2 * mOmega * ((1-C3) < 0) * (1-C3);

    double dEd23 = mGamma[1] * C2 * (1 - 3 * C2 + 2 * C2 * C2) - mGamma[2] * C3 * (1 - 3 * C3 + 2 * C3 * C3) +
                   2 * mLambda * C1 * C1 * (C3 * C3 * C2 - C2 * C2 * C3) + (C2 < 0) * 2 * mOmega * C2 -
                   (C3 < 0) * 2 * mOmega * C3 - 2 * mOmega * ((1-C2) < 0) * (1-C2) + 2 * mOmega * ((1-C3) < 0) * (1-C3);

    ChemicalPotential<0>::get<Lattice>(k) =
        (4.0 * mGammaT / mInterfaceWidth * (1 / mGamma[1] * dEd12 + 1 / mGamma[2] * dEd13) -
         3.0 / 4.0 * mInterfaceWidth * mGamma[0] * (Lap1::get<Lattice>(k)));

    ChemicalPotential<1>::get<Lattice>(k) =
        (4.0 * mGammaT / mInterfaceWidth * (-1 / mGamma[0] * dEd12 + 1 / mGamma[2] * dEd23) -
         3.0 / 4.0 * mInterfaceWidth * mGamma[1] * (Lap2::get<Lattice>(k)));

    ChemicalPotential<2>::get<Lattice>(k) =
        (4.0 * mGammaT / mInterfaceWidth * (-1 / mGamma[0] * dEd13 - 1 / mGamma[1] * dEd23) -
         3.0 / 4.0 * mInterfaceWidth * mGamma[2] * (-Lap1::get<Lattice>(k) - Lap2::get<Lattice>(k)));
}

inline void ChemicalPotentialCalculatorTernaryLee::setSurfaceTension(double sigma12, double sigma13, double sigma23) {
    mSigma[0] = sigma12;
    mSigma[1] = sigma13;
    mSigma[2] = sigma23;

    mGamma[0] = sigma12 + sigma13 - sigma23;
    mGamma[1] = sigma12 + sigma23 - sigma13;
    mGamma[2] = sigma13 + sigma23 - sigma12;

    mGammaT = 3.0 / (1.0 / mGamma[0] + 1.0 / mGamma[1] + 1.0 / mGamma[2]);
}

inline void ChemicalPotentialCalculatorTernaryLee::setOmega(double omega) { mOmega = omega; }

inline void ChemicalPotentialCalculatorTernaryLee::setLambda(double lambda) { mLambda = lambda; }

inline void ChemicalPotentialCalculatorTernaryLee::setInterfaceWidth(double interfacewidth) {
    mInterfaceWidth = interfacewidth;
}

class ChemicalPotentialCalculatorTernary : public AddOnBase {
   public:
    template <class TTraits>
    inline void compute(int k);

    inline void setSurfaceTension(double sigma12, double sigma13, double sigma23);

    inline void setOmega(double omega);

    inline void setLambda(double lambda);

    inline void setInterfaceWidth(double interfacewidth);

    template <class TTraits>
    inline void communicate() {
        using Lattice = typename TTraits::Lattice;
        Lattice::communicate(ChemicalPotential<0>::getInstance<Lattice>());
        Lattice::communicate(ChemicalPotential<1>::getInstance<Lattice>());
        Lattice::communicate(ChemicalPotential<2>::getInstance<Lattice>());
    }

    std::array<double, 3> mSigma;
    std::array<double, 3> mGamma;
    double mGammaT;
    double mOmega = 0.0001;
    double mLambda = 0;
    double mInterfaceWidth;
};

template <class TTraits>
inline void ChemicalPotentialCalculatorTernary::compute(int k) {
    using Lattice = typename TTraits::Lattice;

    if (Geometry<Lattice>::isBulkSolid(k)) return;

    const double C1 = OrderParameter<0>::get<Lattice>(k);  // C1, C2, C3
    const double C2 = OrderParameter<1>::get<Lattice>(k);
    const double C3 = 1 - C1 - C2;
    using Lap1 = LaplacianOrderParameter<0>;
    using Lap2 = LaplacianOrderParameter<1>;

    double dEd1 = mGamma[0] * C1 * (1 - 3 * C1 + 2 * C1 * C1);
    double dEd2 = mGamma[1] * C2 * (1 - 3 * C2 + 2 * C2 * C2);
    double dEd3 = mGamma[2] * C3 * (1 - 3 * C3 + 2 * C3 * C3);

    double lapsum = Lap1::get<Lattice>(k) + Lap2::get<Lattice>(k);

    ChemicalPotential<0>::get<Lattice>(k) =
        (4.0 * mGammaT / mInterfaceWidth * (dEd1) -
         3.0 / 4.0 * mInterfaceWidth * mGamma[0] * (Lap1::get<Lattice>(k)));

    ChemicalPotential<1>::get<Lattice>(k) =
        (4.0 * mGammaT / mInterfaceWidth * (dEd2) -
         3.0 / 4.0 * mInterfaceWidth * mGamma[1] * (Lap2::get<Lattice>(k)));

    ChemicalPotential<2>::get<Lattice>(k) =
        (4.0 * mGammaT / mInterfaceWidth * (dEd3) -
         3.0 / 4.0 * mInterfaceWidth * mGamma[2] * (-Lap1::get<Lattice>(k) - Lap2::get<Lattice>(k)));
}

inline void ChemicalPotentialCalculatorTernary::setSurfaceTension(double sigma12, double sigma13, double sigma23) {
    mSigma[0] = sigma12;
    mSigma[1] = sigma13;
    mSigma[2] = sigma23;

    mGamma[0] = sigma12 + sigma13 - sigma23;
    mGamma[1] = sigma12 + sigma23 - sigma13;
    mGamma[2] = sigma13 + sigma23 - sigma12;

    mGammaT = 3.0 / (1.0 / mGamma[0] + 1.0 / mGamma[1] + 1.0 / mGamma[2]);
}

inline void ChemicalPotentialCalculatorTernary::setOmega(double omega) { mOmega = omega; }

inline void ChemicalPotentialCalculatorTernary::setLambda(double lambda) { mLambda = lambda; }

inline void ChemicalPotentialCalculatorTernary::setInterfaceWidth(double interfacewidth) {
    mInterfaceWidth = interfacewidth;
}

class ChemicalPotentialCalculatorTernaryLeeExtraPotential : public ChemicalPotentialCalculatorTernaryLee {
   public:
    template <class TTraits>
    inline void compute(int k);
    inline void setPreOmega(double preomega) { mPreOmega = preomega; }
    inline void setPotentialComponent(double component) { mComponent = component; }
    double mPreOmega = 0.00333;  // 1;
    inline void setPotentialCondition(double (*condition)(int k)) { evalPotentialCondition = condition; }

   private:
    static double defaultCondition(int k) { return true; }
    double (*evalPotentialCondition)(int k) = &defaultCondition;
    int mComponent = 1;
};

template <class TTraits>
inline void ChemicalPotentialCalculatorTernaryLeeExtraPotential::compute(int k) {
    using Lattice = typename TTraits::Lattice;

    if (Geometry<Lattice>::isBulkSolid(k)) return;

    const double C1 = OrderParameter<0>::get<Lattice>(k);  // C1, C2, C3
    const double C2 = OrderParameter<1>::get<Lattice>(k);
    const double C3 = 1 - C1 - C2;
    double chem1 = 0.0;
    double chem2 = 0.0;
    double chem3 = 0.0;

    double dEd12 = mGamma[0] * C1 * (1 - 3 * C1 + 2 * C1 * C1) - mGamma[1] * C2 * (1 - 3 * C2 + 2 * C2 * C2) +
                   2 * mLambda * C3 * C3 * (C2 * C2 * C1 - C1 * C1 * C2) + (C1 < 0) * 2 * mOmega * C1 -
                   (C2 < 0) * 2 * mOmega * C2;

    double dEd13 = mGamma[0] * C1 * (1 - 3 * C1 + 2 * C1 * C1) - mGamma[2] * C3 * (1 - 3 * C3 + 2 * C3 * C3) +
                   2 * mLambda * C2 * C2 * (C3 * C3 * C1 - C1 * C1 * C3) + (C1 < 0) * 2 * mOmega * C1 -
                   (C3 < 0) * 2 * mOmega * C3;

    double dEd23 = mGamma[1] * C2 * (1 - 3 * C2 + 2 * C2 * C2) - mGamma[2] * C3 * (1 - 3 * C3 + 2 * C3 * C3) +
                   2 * mLambda * C1 * C1 * (C3 * C3 * C2 - C2 * C2 * C3) + (C2 < 0) * 2 * mOmega * C2 -
                   (C3 < 0) * 2 * mOmega * C3;

    //* Extra Potential
    double pre_omega = mPreOmega;
    double Chem_P_extra[3] = {0.0, 0.0, 0.0};

    Chem_P_extra[mComponent] = evalPotentialCondition(k) * pre_omega * (2.0 * C2 - 1.0);

    dEd12 += Chem_P_extra[0] - Chem_P_extra[1];
    dEd13 += Chem_P_extra[0] - Chem_P_extra[2];
    dEd23 += Chem_P_extra[1] - Chem_P_extra[2];

    chem1 = (4.0 * mGammaT / mInterfaceWidth * (1 / mGamma[1] * dEd12 + 1 / mGamma[2] * dEd13) -
             3.0 / 4.0 * mInterfaceWidth * mGamma[0] * (LaplacianOrderParameter<0>::get<Lattice>(k)));

    chem2 = (4.0 * mGammaT / mInterfaceWidth * (-1 / mGamma[0] * dEd12 + 1 / mGamma[2] * dEd23) -
             3.0 / 4.0 * mInterfaceWidth * mGamma[1] * (LaplacianOrderParameter<1>::get<Lattice>(k)));

    chem3 = (4.0 * mGammaT / mInterfaceWidth * (-1 / mGamma[0] * dEd13 - 1 / mGamma[1] * dEd23) -
             3.0 / 4.0 * mInterfaceWidth * mGamma[2] *
                 (-LaplacianOrderParameter<0>::get<Lattice>(k) - LaplacianOrderParameter<1>::get<Lattice>(k)));

    ChemicalPotential<0>::get<Lattice>(k) = chem1;
    ChemicalPotential<1>::get<Lattice>(k) = chem2;
    ChemicalPotential<2>::get<Lattice>(k) = chem3;
}

class ChemicalPotentialCalculatorNComp : public AddOnBase {
   public:
    ChemicalPotentialCalculatorNComp() : mv_Beta({{0}}), mv_Gamma({{0}}) {}

    template <class traits>
    inline void compute(const int k);

    inline void setA(double** A) { ma_Beta = A; }

    inline void setKappa(double** kappa) { ma_Gamma = kappa; }

    inline void setBeta(int i, int j, double beta) {
        if ((int)mv_Beta.size() - 1 < i || (int)mv_Beta.size() - 1 < j) {
            mv_Beta.resize(mv_Beta.size() + std::max<int>(i - (int)mv_Beta.size() + 1, j - (int)mv_Beta.size() + 1));
            for (int l = 0; l < (int)mv_Beta.size(); l++) {
                mv_Beta[l].resize(mv_Beta[l].size() +
                                  std::max<int>(i - (int)mv_Beta[l].size() + 1, j - (int)mv_Beta[l].size() + 1));
                mv_Beta[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Beta[i][j] = beta;
            mv_Beta[j][i] = beta;
        }
    }
    inline void setGamma(int i, int j, double gamma) {
        if ((int)mv_Gamma.size() - 1 < i || (int)mv_Gamma.size() - 1 < j) {
            mv_Gamma.resize(mv_Gamma.size() +
                            std::max<int>(i - (int)mv_Gamma.size() + 1, j - (int)mv_Gamma.size() + 1));
            for (int l = 0; l < (int)mv_Gamma.size(); l++) {
                mv_Gamma[l].resize(mv_Gamma[l].size() +
                                   std::max<int>(i - (int)mv_Gamma[l].size() + 1, j - (int)mv_Gamma[l].size() + 1));
                mv_Gamma[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Gamma[i][j] = gamma;
            mv_Gamma[j][i] = gamma;
        }
    }

    inline void setBetaAndGamma(int i, int j, double beta, double gamma) {
        setBeta(i, j, beta);
        setGamma(i, j, gamma);
    }

    inline void setBeta(std::vector<std::vector<double>>& beta) { mv_Beta = beta; }
    inline void setGamma(std::vector<std::vector<double>>& gamma) { mv_Gamma = gamma; }
    inline void setBetaAndGamma(std::vector<std::vector<double>>& beta, std::vector<std::vector<double>>& gamma) {
        mv_Beta = beta;
        mv_Gamma = gamma;
    }

    inline void setD(double d) {
        for (int l = 0; l < (int)mv_Gamma.size(); l++) {
            for (int m = 0; m < (int)mv_Gamma.size(); m++) {
                mv_Gamma[l][m] *= d / mD;
            }
        }
        for (int l = 0; l < (int)mv_Beta.size(); l++) {
            for (int m = 0; m < (int)mv_Beta.size(); m++) {
                mv_Beta[l][m] *= mD / d;
            }
        }
        mD = d;
    }

    inline void setSurfaceTension(int i, int j, double surfacetension) {
        setBeta(i, j, 3.0 * surfacetension / mD);
        setGamma(i, j, -3.0 * mD * surfacetension / 4.0);
    }

   public:
    template <int numberofcomponents>
    inline bool checkValid() {
        if ((int)mv_Beta.size() != numberofcomponents || (int)mv_Beta[0].size() != numberofcomponents ||
            (int)mv_Gamma.size() != numberofcomponents || (int)mv_Gamma[0].size() != numberofcomponents) {
            throw std::runtime_error(
                "Number of beta/gamma parameters does not match the number of "
                "components.");
            return false;
        }
        return true;
    }

    bool mIsValid;
    double** ma_Gamma;
    double** ma_Beta;
    std::vector<std::vector<double>> mv_Beta;
    std::vector<std::vector<double>> mv_Gamma;
    double mD = 5;
    double mS = 0.0;//100;
};

template <class TTraits>
inline void ChemicalPotentialCalculatorNComp::compute(const int k) {
    if (Geometry<typename TTraits::Lattice>::isBulkSolid(k)) return;

    [[maybe_unused]] static bool isvalid = checkValid<TTraits::NumberOfComponents>();
    mIsValid=isvalid;
    using Lattice = typename TTraits::Lattice;
    constexpr int N = TTraits::NumberOfComponents;
    double chempot=0.0;
    double sumci=0.0;
    double sumcj=0.0;

    double onlylaplacian=0.0;
    double gammalaplacian=0.0;
    double onlylaplaciansum=0.0;
    double gammalaplaciansum=0.0;       

    for (int i=0;i<TTraits::NumberOfComponents-1;i++){
        chempot=0.0;
        const double& ci=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
        sumci += ci;

        sumcj=0.0;
        gammalaplaciansum=0.0;
        onlylaplaciansum=0.0;

        for (int j=0;j<TTraits::NumberOfComponents-1;j++){
            
            const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
            sumcj+=cj;
            
            onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];
            
            onlylaplaciansum += onlylaplacian;                
            gammalaplacian=mv_Gamma[i][j]*onlylaplacian;
            gammalaplaciansum += gammalaplacian;

            chempot+=2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
            
        }// j

        const double cj=1.0-sumcj;
        
        chempot+=2*mv_Beta[i][TTraits::NumberOfComponents-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) + mv_Gamma[i][TTraits::NumberOfComponents-1]*onlylaplaciansum;
        
        double P=0;
        for (int j=0;j<N-1;j++){
            if(i!=j){
                const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
                for (int m=j+1;m<N-1;m++){
                    const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
                    if(i!=m) {
                        P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
                        
                    }
                }
                P+=2*ci*pow(cj,2)*pow(1-sumcj,2);
            }
        }

        getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot+mS*P;
        
    } // i=0:N-2
    

    int i = TTraits::NumberOfComponents-1;
    chempot=0.0;
    const double& ci=1.0-sumci;
    

    sumcj=0.0;
    gammalaplaciansum=0.0;
    onlylaplaciansum=0.0;
    
    for (int j=0;j<TTraits::NumberOfComponents-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        sumcj+=cj;
        
        onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];                         
        onlylaplaciansum += onlylaplacian;

        gammalaplacian = mv_Gamma[i][j]*onlylaplacian;
        gammalaplaciansum += gammalaplacian;
        
        if (i!=j){
            chempot+=2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
        }
    }

    double P = 0;
    for (int j=0;j<N-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        for (int m=j+1;m<N-1;m++){
            const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
            P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
        }
    }

    getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot+mS*P;
    
}


class ChemicalPotentialCalculatorNCompAll : public AddOnBase {
   public:
    ChemicalPotentialCalculatorNCompAll() : mv_Beta({{0}}), mv_Gamma({{0}}) {}

    template <class traits>
    inline void compute(const int k);

    inline void setA(double** A) { ma_Beta = A; }

    inline void setKappa(double** kappa) { ma_Gamma = kappa; }

    inline void setBeta(int i, int j, double beta) {
        if ((int)mv_Beta.size() - 1 < i || (int)mv_Beta.size() - 1 < j) {
            mv_Beta.resize(mv_Beta.size() + std::max<int>(i - (int)mv_Beta.size() + 1, j - (int)mv_Beta.size() + 1));
            for (int l = 0; l < (int)mv_Beta.size(); l++) {
                mv_Beta[l].resize(mv_Beta[l].size() +
                                  std::max<int>(i - (int)mv_Beta[l].size() + 1, j - (int)mv_Beta[l].size() + 1));
                mv_Beta[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Beta[i][j] = beta;
            mv_Beta[j][i] = beta;
        }
    }
    inline void setGamma(int i, int j, double gamma) {
        if ((int)mv_Gamma.size() - 1 < i || (int)mv_Gamma.size() - 1 < j) {
            mv_Gamma.resize(mv_Gamma.size() +
                            std::max<int>(i - (int)mv_Gamma.size() + 1, j - (int)mv_Gamma.size() + 1));
            for (int l = 0; l < (int)mv_Gamma.size(); l++) {
                mv_Gamma[l].resize(mv_Gamma[l].size() +
                                   std::max<int>(i - (int)mv_Gamma[l].size() + 1, j - (int)mv_Gamma[l].size() + 1));
                mv_Gamma[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Gamma[i][j] = gamma;
            mv_Gamma[j][i] = gamma;
        }
    }

    inline void setBetaAndGamma(int i, int j, double beta, double gamma) {
        setBeta(i, j, beta);
        setGamma(i, j, gamma);
    }

    inline void setBeta(std::vector<std::vector<double>>& beta) { mv_Beta = beta; }
    inline void setGamma(std::vector<std::vector<double>>& gamma) { mv_Gamma = gamma; }
    inline void setBetaAndGamma(std::vector<std::vector<double>>& beta, std::vector<std::vector<double>>& gamma) {
        mv_Beta = beta;
        mv_Gamma = gamma;
    }

    inline void setD(double d) {
        for (int l = 0; l < (int)mv_Gamma.size(); l++) {
            for (int m = 0; m < (int)mv_Gamma.size(); m++) {
                mv_Gamma[l][m] *= d / mD;
            }
        }
        for (int l = 0; l < (int)mv_Beta.size(); l++) {
            for (int m = 0; m < (int)mv_Beta.size(); m++) {
                mv_Beta[l][m] *= mD / d;
            }
        }
        mD = d;
    }

    inline void setSurfaceTension(int i, int j, double surfacetension) {
        setBeta(i, j, 3.0 * surfacetension / mD);
        setGamma(i, j, -3.0 * mD * surfacetension / 4.0);
    }

   public:
    template <int numberofcomponents>
    inline bool checkValid() {
        if ((int)mv_Beta.size() != numberofcomponents || (int)mv_Beta[0].size() != numberofcomponents ||
            (int)mv_Gamma.size() != numberofcomponents || (int)mv_Gamma[0].size() != numberofcomponents) {
            throw std::runtime_error(
                "Number of beta/gamma parameters does not match the number of "
                "components.");
            return false;
        }
        return true;
    }

    bool mIsValid;
    double** ma_Gamma;
    double** ma_Beta;
    std::vector<std::vector<double>> mv_Beta;
    std::vector<std::vector<double>> mv_Gamma;
    double mD = 5;
    double mS = 0;//100;
};

template <class TTraits>
inline void ChemicalPotentialCalculatorNCompAll::compute(const int k) {
    if (Geometry<typename TTraits::Lattice>::isBulkSolid(k)) return;

    [[maybe_unused]] static bool isvalid = checkValid<TTraits::NumberOfComponents>();
    mIsValid=isvalid;
    using Lattice = typename TTraits::Lattice;
    constexpr int N = TTraits::NumberOfComponents;
    double chempot=0.0;
    double chempot_bulk=0.0;
    double chempot_tension=0.0;
    double sumci=0.0;
    double sumcj=0.0;

    double onlylaplacian=0.0;
    double gammalaplacian=0.0;
    double onlylaplaciansum=0.0;
    double gammalaplaciansum=0.0;       

    for (int i=0;i<TTraits::NumberOfComponents;i++){
        chempot=0.0;
        chempot_bulk=0.0;
        chempot_tension=0.0;
        const double& ci=getInstance<OrderParameter, N, Lattice>(i)[k];
        sumci += ci;

        sumcj=0.0;
        gammalaplaciansum=0.0;
        onlylaplaciansum=0.0;

        for (int j=0;j<TTraits::NumberOfComponents;j++){
            
            const double& cj=getInstance<OrderParameter, N, Lattice>(j)[k];
            sumcj+=cj;
            
            onlylaplacian = getInstance<LaplacianOrderParameter, N, Lattice>(j)[k];
            
            onlylaplaciansum += onlylaplacian;                
            gammalaplacian=mv_Gamma[i][j]*onlylaplacian;
            gammalaplaciansum += gammalaplacian;

            chempot+=2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
            chempot_bulk += 2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj);
            chempot_tension += - gammalaplacian;
        }// j

        getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot;
        
    } // i=0:N-2
    
}

class ChemicalPotentialCalculatorNCompLee : public AddOnBase {
   
    public:
 
        ChemicalPotentialCalculatorNCompLee() : mv_Beta({{0}}), mv_Gamma({{0}}) {}
 
        double **ma_Gamma;
 
        double **ma_Beta;
 
        std::vector<std::vector<double>> mv_Beta;
        std::vector<std::vector<double>> mv_Gamma;
 
        double m_M0;
        double mobility = 0.00333;
 
        double mD=4;   //// mD_old = 5  /////////////////////////////////////////////////////////////////////////////
 
        template<class traits>
        inline void compute(const int k); //Perform any neccessary computations before force is computed
       
        //template<class traits>
        //inline void compute_M0(double mobility); // link mobility, m_M0 and lambda !!!
 
        inline void setA(double **A) {ma_Beta=A;}
        inline void setKappa(double **kappa) {ma_Gamma=kappa;}
       
        inline void setBeta(int i, int j, double beta) {
            if ((int)mv_Beta.size()-1<i||(int)mv_Beta.size()-1<j){
                mv_Beta.resize(mv_Beta.size()+std::max<int>(i-(int)mv_Beta.size()+1,j-(int)mv_Beta.size()+1));
                for (int l=0;l<(int)mv_Beta.size();l++) {
                    mv_Beta[l].resize(mv_Beta[l].size()+std::max<int>(i-(int)mv_Beta[l].size()+1,j-(int)mv_Beta[l].size()+1));
                    mv_Beta[l][l]=0;
                }
            }
            if (i!=j) {
                mv_Beta[i][j] = beta;
                mv_Beta[j][i] = beta;
            }            
        }
        inline void setGamma(int i, int j, double gamma) {
            if ((int)mv_Gamma.size()-1<i||(int)mv_Gamma.size()-1<j){
                mv_Gamma.resize(mv_Gamma.size()+std::max<int>(i-(int)mv_Gamma.size()+1,j-(int)mv_Gamma.size()+1));
                for (int l=0;l<(int)mv_Gamma.size();l++) {
                    mv_Gamma[l].resize(mv_Gamma[l].size()+std::max<int>(i-(int)mv_Gamma[l].size()+1,j-(int)mv_Gamma[l].size()+1));
                    mv_Gamma[l][l]=0;
                }
            }
            if (i!=j) {
                mv_Gamma[i][j] = gamma;
                mv_Gamma[j][i] = gamma;
            }
        }
 
        inline void setBetaAndGamma(int i, int j, double beta, double gamma) {
            setBeta(i,j,beta);
            setGamma(i,j,gamma);    
        }
 
        inline void setBeta(std::vector<std::vector<double>>& beta) {
            mv_Beta=beta;
        }
        inline void setGamma(std::vector<std::vector<double>>& gamma) {
            mv_Gamma=gamma;
        }
        inline void setBetaAndGamma(std::vector<std::vector<double>>& beta,std::vector<std::vector<double>>& gamma) {
            mv_Beta=beta;
            mv_Gamma=gamma;
        }
 
        inline void setD(double d){
            for (int l=0;l<(int)mv_Gamma.size();l++) {
                for (int m=0;m<(int)mv_Gamma.size();m++) {
                    mv_Gamma[l][m]*=d/mD;
                }
            }
            for (int l=0;l<(int)mv_Beta.size();l++) {
                for (int m=0;m<(int)mv_Beta.size();m++) {
                    mv_Beta[l][m]*=mD/d;
                }
            }
            mD=d;
        }
 
        inline void setSurfaceTension(int i, int j, double surfacetension) {
            setBeta(i,j,3.0*surfacetension/mD);
            setGamma(i,j,-3.0*mD*surfacetension/4.0);    
        }
       
        inline void setMobility_ij(double sigma12, double sigma13, double sigma23){
            for (int n=0;n<4;n++){
                for (int m=0;m<4;m++){
                    if(n==m) mobility_ij[n][m]=3;
                    else mobility_ij[n][m]=-1;
                }
            }
            /*
            mobility_ij[0][0]=  m0*(sigma12+sigma13)/(4.0*sigma12*sigma13);
            mobility_ij[0][1]= -m0        /(2.0*sigma12);
            mobility_ij[0][2]= -m0        /(2.0*sigma13);
            mobility_ij[1][0]= -m0        /(2.0*sigma12);
            mobility_ij[1][1]=  m0*(sigma12+sigma23)/(4.0*sigma12*sigma23);
            mobility_ij[1][2]= -m0        /(2.0*sigma23);
            mobility_ij[2][0]= -m0        /(2.0*sigma13);
            mobility_ij[2][1]= -m0        /(2.0*sigma23);
            mobility_ij[2][2]=  m0*(sigma13+sigma23)/(4.0*sigma13*sigma23);*/
 
        }
 
        void setConstBetaInterfaceWidth(double constbeta, double interfacewidth) {mBeta=constbeta; mD=interfacewidth; m0=constbeta*interfacewidth/6.0;}
   
    public:
        template<int numberofcomponents>
        inline bool checkValid(){
           
            if ((int)mv_Beta.size() != numberofcomponents || (int)mv_Beta[0].size() != numberofcomponents
                || (int)mv_Gamma.size() != numberofcomponents || (int)mv_Gamma[0].size() != numberofcomponents){
                throw std::runtime_error("Number of beta/gamma parameters does not match the number of components.");
                return false;
            }
            return true;
        }
 
        bool mIsValid;
 
    private:
        double mobility_ij [4][4];
        double mBeta;
        double mInterfaceWidth;
        double mMobility;
        double m0;
 
};
 
/*
template<class TTraits>
inline void ChemicalPotentialCalculatorNCompLee::compute_M0(double mobility){
 
    double sum = 0.0;
    for (int i=0;i<TTraits::NumberOfComponents;i++){
        sum += mobility* mv_Gamma[i][1];
    }
 
    m_M0=sum;
    std::cout<<"m_M0 = "<<m_M0<<std::endl;
}
*/
 
template<class TTraits>
inline void ChemicalPotentialCalculatorNCompLee::compute(const int k) {
    constexpr int N = TTraits::NumberOfComponents;
    [[maybe_unused]] static const bool isValid = checkValid<N>();
    mIsValid = isValid;
 
    using Lattice = typename TTraits::Lattice;

    std::array<double, N> laplacian_ci{};
    std::array<double, N> chempot_bulk_ci{};
    std::array<double, N> chempot_ci{};
 
    // Calculate laplacian_ci
    double sum_temp = 0.0;
    for (int i = 0; i < N - 1; ++i) {
        laplacian_ci[i] = getInstance<LaplacianOrderParameter, N-1, Lattice>(i)[k];
        sum_temp += laplacian_ci[i];
    }
    laplacian_ci[N-1] = -sum_temp;
 
    // Calculate sum_mij
    double sum_mij = 0.0;
    double Beta_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            sum_mij += mobility_ij[i][j];
            Beta_sum += mv_Beta[i][j];
        }
    }
 
    // Calculate sum_beta1
    double sum_beta1 = 0.0;
    for (int i = 0; i < N; ++i) {   /////// note: loop from 1 to N !!!
        for (int j = 0; j < N; ++j) {  /////// note: loop from 1 to N !!!
            for (int kk = 0; kk < N; ++kk) {
                sum_beta1 += mobility_ij[i][j] * mv_Gamma[j][kk] * laplacian_ci[kk];
            }
        }
    }
 
    double sumci = 0.0;
    // Calculate chemical potentials for components 0 to N-2
    for (int i = 0; i < N - 1; ++i) {
        double chempot = 0.0;
        double chempot_bulk = 0.0;
        double chempot_tension = 0.0;
 
        const double ci = getInstance<OrderParameter, N-1, Lattice>(i)[k];
        sumci += ci;
 
        double sumcj = 0.0;
        double gammalaplaciansum = 0.0;
        double onlylaplaciansum = 0.0;
        double stable_term = 1.0;
 
        for (int j = 0; j < N - 1; ++j) {
            const double cj = getInstance<OrderParameter, N-1, Lattice>(j)[k];
            sumcj += cj;
           
            double onlylaplacian = getInstance<LaplacianOrderParameter, N-1, Lattice>(j)[k];
            onlylaplaciansum += onlylaplacian;
           
            double gammalaplacian = mv_Gamma[i][j] * onlylaplacian;
            gammalaplaciansum += gammalaplacian;
 
            if(i != j ){stable_term = stable_term*cj*cj;} // c_2^2;
 
            double common_term = 2.0 * mv_Beta[i][j] * (-12*ci*ci*cj - 12*ci*cj*cj + 12*ci*cj - 4*cj*cj*cj + 6*cj*cj - 2*cj);
            chempot += common_term - gammalaplacian;
            chempot_bulk += common_term;
            chempot_tension -= gammalaplacian;
        }
 
        const double cj = 1.0 - sumcj;
        stable_term = stable_term*cj*cj; // c_2^2 * c_3^2;
        double common_term = 2.0 * mv_Beta[i][N-1] * (-12*ci*ci*cj - 12*ci*cj*cj + 12*ci*cj - 4*cj*cj*cj + 6*cj*cj - 2*cj)  + 4.0*stable_term*Beta_sum*ci ;
       
        chempot += common_term + mv_Gamma[i][N-1] * onlylaplaciansum;
        chempot_bulk += common_term;
        chempot_bulk_ci[i] = chempot_bulk;
        chempot_tension += mv_Gamma[i][N-1] * onlylaplaciansum;
 
        chempot_ci[i] = chempot;
        //ChemicalPotential<N>::template get<typename TTraits::Lattice>(k, i) = chempot;
    }
 
    // Calculate chemical potential for component N-1
    {
        int i = N - 1;
        double chempot = 0.0;
        double chempot_bulk = 0.0;
        double chempot_tension = 0.0;
        const double ci = 1.0 - sumci;
 
        double sumcj = 0.0;
        double gammalaplaciansum = 0.0;
        double onlylaplaciansum = 0.0;
        double stable_term = 1.0;
        for (int j = 0; j < N - 1; ++j) {
            const double cj = getInstance<OrderParameter, N-1, Lattice>(j)[k];
            sumcj += cj;
           
            double onlylaplacian = getInstance<LaplacianOrderParameter, N-1, Lattice>(j)[k];
            onlylaplaciansum += onlylaplacian;
 
            double gammalaplacian = mv_Gamma[i][j] * onlylaplacian;
            gammalaplaciansum += gammalaplacian;
           
            if (i != j) {
                stable_term = stable_term*cj*cj;
                double common_term = 2.0 * mv_Beta[i][j] * (-12*ci*ci*cj - 12*ci*cj*cj + 12*ci*cj - 4*cj*cj*cj + 6*cj*cj - 2*cj);
                chempot += common_term - gammalaplacian;
                chempot_bulk += common_term;
                chempot_tension -= gammalaplacian;
            }
        }
 
        chempot_bulk_ci[N-1] = chempot_bulk  + 4.0*stable_term*Beta_sum*ci ;
        chempot_ci[N-1] = chempot  + 4.0*stable_term*Beta_sum*ci ;
        //ChemicalPotential<N>::template get<typename TTraits::Lattice>(k, i) = chempot;
    }
 
   
    // Calculate sum_beta2
    double sum_beta2 = 0.0;
    for (int i = 0; i < N ; ++i) {
        for (int j = 0; j < N ; ++j) {
            sum_beta2 += mobility_ij[i][j] * chempot_bulk_ci[j];
            //if (k==10000){std::cout<<"sum_beta2_ij= "<<  sum_beta2 <<std::endl;}
        }
    }

 
    // Final chemical potential calculation
    for (int i = 0; i < N; ++i) {
        getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot_ci[i]  + (sum_beta1 - sum_beta2) / sum_mij;
    }

}

//vvvvvvvvvvvvvvvvvvvvvvvvvvvvSHOULD BE TEMPORARYvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
class ChemicalPotentialCalculator4ComponentBoyer : public AddOnBase {
   public:
    ChemicalPotentialCalculator4ComponentBoyer() : mv_Beta({{0}}), mv_Gamma({{0}}) {
        //atemp.push_back({-148000./907.,   66000./907.,   92000./907.,  -10000./907.});
        //atemp.push_back({  66000./907., -152000./907.,    160./907.,   78000./907.});
        //atemp.push_back({  92000./907.,    160./907., -167500./907.,   67500./907.});
        //atemp.push_back({ -10000./907.,   78000./907.,   67500./907., -135500./907.});
        /*atemp.push_back({-150.,   50.,   50.,  50.});
        atemp.push_back({  50., -150.,    50.,   50.});
        atemp.push_back({  50.,    50., -150,   50.});
        atemp.push_back({ 50.,   50.,   50., -150.});
        th.push_back({2,   2*atemp[0][1]/atemp[0][0],   2*atemp[0][2]/atemp[0][0],  2*atemp[0][3]/atemp[0][0]});
        th.push_back({  2*atemp[1][0]/atemp[1][1], 2,    2*atemp[1][2]/atemp[1][1],   2*atemp[1][3]/atemp[1][1]});
        th.push_back({  2*atemp[2][0]/atemp[2][2],    2*atemp[2][1]/atemp[2][2], 2,   2*atemp[2][3]/atemp[2][2]});
        th.push_back({ 2*atemp[3][0]/atemp[3][3],   2*atemp[3][1]/atemp[3][3],   2*atemp[3][2]/atemp[3][3], 2});*/
    }

    std::vector<std::vector<double>> atemp;
    std::vector<std::vector<double>> th;

    inline void setAlphas(std::vector<std::vector<double>> a) { atemp=a; 
        
        th.push_back({2,   2*atemp[0][1]/atemp[0][0],   2*atemp[0][2]/atemp[0][0],  2*atemp[0][3]/atemp[0][0]});
        th.push_back({  2*atemp[1][0]/atemp[1][1], 2,    2*atemp[1][2]/atemp[1][1],   2*atemp[1][3]/atemp[1][1]});
        th.push_back({  2*atemp[2][0]/atemp[2][2],    2*atemp[2][1]/atemp[2][2], 2,   2*atemp[2][3]/atemp[2][2]});
        th.push_back({ 2*atemp[3][0]/atemp[3][3],   2*atemp[3][1]/atemp[3][3],   2*atemp[3][2]/atemp[3][3], 2});
    }

    template <class traits>
    inline void compute(const int k);

    inline void setA(double** A) { ma_Beta = A; }
    inline void setOmega(double Omega) { mOmega = Omega; }
    inline void setKappa(double** kappa) { ma_Gamma = kappa; }

    inline double calcdH1(double c1, double c2) {
        if (fabs(c1)==0&&fabs(c2)==0) return 0;
        return -2*c1*fabs(c2)*c2/pow(fabs(c2)+c1*c1,2);
    }

    inline double calcdH2(double c1, double c2) {
        if (fabs(c1)==0&&fabs(c2)==0) return 1;
        return (2*fabs(c2)*c1*c1+c2*c2)/pow(fabs(c2)+c1*c1,2);
    }

    inline double calcH(double c1, double c2) {
        if (fabs(c1)==0&&fabs(c2)==0) return 0;
        return (fabs(c2)*c2)/(fabs(c2)+c1*c1);
    }

    inline double calcdP(std::vector<double> c,int i,int j, int k, int l){

        return pow(c[k],2)*pow(c[l],2)*(th[i][j]*(calcdH1(c[i], c[i]*c[j]) + c[j]*calcdH2(c[i], c[i]*c[j])) + th[j][i]*c[j]*calcdH2(c[j], c[i]*c[j])) + pow(c[j],2)*pow(c[l],2)*(th[i][k]*(calcdH1(c[i], c[i]*c[k]) + c[k]*calcdH2(c[i], c[i]*c[k])) + th[k][i]*c[k]*calcdH2(c[k], c[i]*c[k])) + pow(c[j],2)*pow(c[k],2)*(th[i][l]*(calcdH1(c[i], c[i]*c[l]) + c[l]*calcdH2(c[i], c[i]*c[l])) + th[l][i]*c[l]*calcdH2(c[l], c[i]*c[l])) + 2*c[i]*pow(c[l],2)*(th[j][k]*calcH(c[j], c[j]*c[k]) + th[k][j]*calcH(c[k], c[j]*c[k])) + 2*c[i]*pow(c[k],2)*(th[j][l]*calcH(c[j], c[j]*c[l]) + th[l][j]*calcH(c[l], c[j]*c[l])) + 2*c[i]*pow(c[j],2)*(th[k][l]*calcH(c[k], c[k]*c[l]) + th[l][k]*calcH(c[l], c[k]*c[l]));
    }

    inline double calcdPSimple(std::vector<double> c,int i,int j, int k, int l){

        return pow(c[k],2)*pow(c[l],2)*(th[i][j]*c[j]) + pow(c[j],2)*pow(c[l],2)*(th[i][k]*c[k]) + pow(c[j],2)*pow(c[k],2)*(th[i][l]*c[l]);
    }

    inline void setBeta(int i, int j, double beta) {
        if ((int)mv_Beta.size() - 1 < i || (int)mv_Beta.size() - 1 < j) {
            mv_Beta.resize(mv_Beta.size() + std::max<int>(i - (int)mv_Beta.size() + 1, j - (int)mv_Beta.size() + 1));
            for (int l = 0; l < (int)mv_Beta.size(); l++) {
                mv_Beta[l].resize(mv_Beta[l].size() +
                                  std::max<int>(i - (int)mv_Beta[l].size() + 1, j - (int)mv_Beta[l].size() + 1));
                mv_Beta[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Beta[i][j] = beta;
            mv_Beta[j][i] = beta;
        }
    }
    inline void setGamma(int i, int j, double gamma) {
        if ((int)mv_Gamma.size() - 1 < i || (int)mv_Gamma.size() - 1 < j) {
            mv_Gamma.resize(mv_Gamma.size() +
                            std::max<int>(i - (int)mv_Gamma.size() + 1, j - (int)mv_Gamma.size() + 1));
            for (int l = 0; l < (int)mv_Gamma.size(); l++) {
                mv_Gamma[l].resize(mv_Gamma[l].size() +
                                   std::max<int>(i - (int)mv_Gamma[l].size() + 1, j - (int)mv_Gamma[l].size() + 1));
                mv_Gamma[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Gamma[i][j] = gamma;
            mv_Gamma[j][i] = gamma;
        }
    }

    inline void setBetaAndGamma(int i, int j, double beta, double gamma) {
        setBeta(i, j, beta);
        setGamma(i, j, gamma);
    }

    inline void setBeta(std::vector<std::vector<double>>& beta) { mv_Beta = beta; }
    inline void setGamma(std::vector<std::vector<double>>& gamma) { mv_Gamma = gamma; }
    inline void setLambdas(std::vector<double>& lambda) { mv_Lambda = lambda; }
    template <typename... TLambdas>
    inline void setLambdas(TLambdas... lambda) {

        std::vector<double> l{{lambda...}};
        mv_Lambda = l;
    }
    inline void setBetaAndGamma(std::vector<std::vector<double>>& beta, std::vector<std::vector<double>>& gamma) {
        mv_Beta = beta;
        mv_Gamma = gamma;
    }

    inline void setD(double d) {
        for (int l = 0; l < (int)mv_Gamma.size(); l++) {
            for (int m = 0; m < (int)mv_Gamma.size(); m++) {
                mv_Gamma[l][m] *= d / mD;
            }
        }
        for (int l = 0; l < (int)mv_Beta.size(); l++) {
            for (int m = 0; m < (int)mv_Beta.size(); m++) {
                mv_Beta[l][m] *= mD / d;
            }
        }
        mD = d;
    }

    inline void setSurfaceTension(int i, int j, double surfacetension) {
        setBeta(i, j, 3.0 * surfacetension / mD);
        setGamma(i, j, -3.0 * mD * surfacetension / 4.0);
    }

   public:
    template <int numberofcomponents>
    inline bool checkValid() {
        if ((int)mv_Beta.size() != numberofcomponents || (int)mv_Beta[0].size() != numberofcomponents ||
            (int)mv_Gamma.size() != numberofcomponents || (int)mv_Gamma[0].size() != numberofcomponents ||
            (int)mv_Lambda.size() != numberofcomponents) {
            throw std::runtime_error(
                "Number of beta/gamma parameters does not match the number of "
                "components.");
            return false;
        }
        return true;
    }

    bool mIsValid;
    double** ma_Gamma;
    double** ma_Beta;
    std::vector<std::vector<double>> mv_Beta;
    std::vector<std::vector<double>> mv_Gamma;
    std::vector<double> mv_Lambda;
    double mD = 5;
    double mOmega = 0.0;
    double cutoff = 1e-10;

};

template <class TTraits>
inline void ChemicalPotentialCalculator4ComponentBoyer::compute(const int k) {
    if (Geometry<typename TTraits::Lattice>::isBulkSolid(k)) return;

    [[maybe_unused]] static bool isvalid = checkValid<4>();
    mIsValid=isvalid;
    using Lattice = typename TTraits::Lattice;
    constexpr int N = 4;
    double chempot=0.0;
    double chempot_bulk=0.0;
    double chempot_tension=0.0;
    double sumci=0.0;
    double sumcj=0.0;

    double onlylaplacian=0.0;
    double gammalaplacian=0.0;
    double onlylaplaciansum=0.0;
    double gammalaplaciansum=0.0;   

    std::vector<double> CVec = {getInstance<OrderParameter, N - 1, Lattice>(0)[k],
                                getInstance<OrderParameter, N - 1, Lattice>(1)[k],
                                getInstance<OrderParameter, N - 1, Lattice>(2)[k],
                                1-getInstance<OrderParameter, N - 1, Lattice>(0)[k]-getInstance<OrderParameter, N - 1, Lattice>(1)[k]-getInstance<OrderParameter, N - 1, Lattice>(2)[k]};    

    for (int i=0;i<N-1;i++){
        chempot=0.0;
        chempot_bulk=0.0;
        chempot_tension=0.0;
        const double& ci=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
        sumci += ci;

        sumcj=0.0;
        gammalaplaciansum=0.0;
        onlylaplaciansum=0.0;
        double product = 1;
        double P=0;
        for (int j=0;j<N-1;j++){
            
            const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
            sumcj+=cj;
            
            onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];
            
            onlylaplaciansum += onlylaplacian;                
            gammalaplacian=mv_Gamma[i][j]*onlylaplacian;
            gammalaplaciansum += gammalaplacian;

            chempot+=2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
            chempot_bulk += 2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj);
            chempot_tension += - gammalaplacian;
            if(i!=j) {
                product *= cj;
            
                
            }
        }// j
        /*
        for (int j=0;j<N-1;j++){
            if(i!=j){
                const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
                for (int m=j+1;m<N-1;m++){
                    const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
                    if(i!=m) {
                        P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
                        P+=-2*atemp[i][3]/atemp[i][i]*(1-sumcj)*cj*cj*cm*cm;
                    }
                }
                P+=2*ci*pow(cj,2)*pow(1-sumcj,2);
                for (int m=0;m<N-1;m++){
                    const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
                    if(i!=m&&j!=m) {
                        P+=-2*atemp[i][m]/atemp[i][i]*cm*cj*cj*pow(1-sumcj,2);
                    }
                }
            }
        }*/

       for (int j=0;j<N-1;j++){
            if(i!=j){
                const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
                for (int m=j+1;m<N-1;m++){
                    const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
                    if(i!=m) {
                        P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
                        
                    }
                }
                P+=2*ci*pow(cj,2)*pow(1-sumcj,2);
            }
        }
        if (i==0) P+=-calcdPSimple(CVec,0,1,2,3);
        if (i==1) P+=-calcdPSimple(CVec,1,0,2,3);
        if (i==2) P+=-calcdPSimple(CVec,2,0,1,3);

        const double cj=1.0-sumcj;

        product *= cj;

        chempot+=2*mv_Beta[i][N-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) + mv_Gamma[i][N-1]*onlylaplaciansum;
        chempot_bulk += 2*mv_Beta[i][N-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj);
        chempot_tension += mv_Gamma[i][N-1]*onlylaplaciansum;

        //getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + mv_Lambda[i]*product;
        
        double lambdadH2 = 0;
        for (int j=0;j<N-1;j++){
            lambdadH2+=mv_Lambda[j]*calcdH2(getInstance<OrderParameter, N - 1, Lattice>(j)[k],ci*product);
            
        }
        lambdadH2+=mv_Lambda[N-1]*calcdH2(1-sumcj,ci*product);
        
        //lambdadH2=mv_Lambda[i];
        getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + 2 * mOmega * (ci < 0) * ci/*-
                                           2 * mOmega * ((1-ci) < 0) * (1-ci)*/ + mv_Lambda[i]*product;//+0.5*P;//*calcdH1(ci,ci*product)+product*(lambdadH2)+0.5*P;
        //getInstance<ChemicalPotential2, N, Lattice>(i)[k] = chempot;
    } // i=0:N-2
    

    int i = N-1;
    chempot=0.0;
    chempot_bulk=0.0;
    chempot_tension=0.0;
    const double& ci=1.0-sumci;
    

    sumcj=0.0;
    gammalaplaciansum=0.0;
    onlylaplaciansum=0.0;
    double product = 1;
    for (int j=0;j<N-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        sumcj+=cj;
        
        onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];                         
        onlylaplaciansum += onlylaplacian;

        gammalaplacian = mv_Gamma[i][j]*onlylaplacian;
        gammalaplaciansum += gammalaplacian;
        
        if (i!=j){
            chempot+=2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
            chempot_bulk += 2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj);
            chempot_tension += - gammalaplacian;
        }
        product *= cj;
    }
    /*
    double P = 0;
    for (int j=0;j<N-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        for (int m=j+1;m<N-1;m++){
            const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
            P+=ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
            for (int n=0;n<N-1;n++){
                const double& cn=getInstance<OrderParameter, N - 1, Lattice>(n)[k];
                if(n!=i&&n!=j&&n!=m) P+=-2*atemp[3][n]/atemp[3][3]*cn*cj*cj*cm*cm;
            }
        }
    }*/

    double P = 0;
    for (int j=0;j<N-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        for (int m=j+1;m<N-1;m++){
            const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
            P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
        }
    }
    P+=-calcdPSimple(CVec,3,0,1,2);
    //getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot+mv_Lambda[i]*product;//+0.5*P;
    double lambdadH2 = 0;
    for (int j=0;j<N-1;j++){
        lambdadH2+=mv_Lambda[j]*calcdH2(getInstance<OrderParameter, N - 1, Lattice>(j)[k],ci*product);
        
    }
    lambdadH2+=mv_Lambda[N-1]*calcdH2(1-sumcj,ci*product);
    
    getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + 2 * mOmega * (ci < 0) * ci/*-
                                           2 * mOmega * ((1-ci) < 0) * (1-ci)*/ + mv_Lambda[i]*product;//+0.5*P;//mv_Lambda[i]*calcdH1(ci,ci*product)+product*(lambdadH2)+0.5*P;
    //getInstance<ChemicalPotential2, N, Lattice>(i)[k] = chempot;
}



//vvvvvvvvvvvvvvvvvvvvvvvvvvvvSHOULD BE TEMPORARYvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
class ChemicalPotentialCalculator4ComponentBoyerSplit : public AddOnBase {
    public:
     ChemicalPotentialCalculator4ComponentBoyerSplit() : mv_Beta({{0}}), mv_Gamma({{0}}) {
         //atemp.push_back({-148000./907.,   66000./907.,   92000./907.,  -10000./907.});
         //atemp.push_back({  66000./907., -152000./907.,    160./907.,   78000./907.});
         //atemp.push_back({  92000./907.,    160./907., -167500./907.,   67500./907.});
         //atemp.push_back({ -10000./907.,   78000./907.,   67500./907., -135500./907.});
         /*atemp.push_back({-150.,   50.,   50.,  50.});
         atemp.push_back({  50., -150.,    50.,   50.});
         atemp.push_back({  50.,    50., -150,   50.});
         atemp.push_back({ 50.,   50.,   50., -150.});
         th.push_back({2,   2*atemp[0][1]/atemp[0][0],   2*atemp[0][2]/atemp[0][0],  2*atemp[0][3]/atemp[0][0]});
         th.push_back({  2*atemp[1][0]/atemp[1][1], 2,    2*atemp[1][2]/atemp[1][1],   2*atemp[1][3]/atemp[1][1]});
         th.push_back({  2*atemp[2][0]/atemp[2][2],    2*atemp[2][1]/atemp[2][2], 2,   2*atemp[2][3]/atemp[2][2]});
         th.push_back({ 2*atemp[3][0]/atemp[3][3],   2*atemp[3][1]/atemp[3][3],   2*atemp[3][2]/atemp[3][3], 2});*/
     }
 
     std::vector<std::vector<double>> atemp;
     std::vector<std::vector<double>> th;
 
     inline void setAlphas(std::vector<std::vector<double>> a) { atemp=a; 
         
         th.push_back({2,   2*atemp[0][1]/atemp[0][0],   2*atemp[0][2]/atemp[0][0],  2*atemp[0][3]/atemp[0][0]});
         th.push_back({  2*atemp[1][0]/atemp[1][1], 2,    2*atemp[1][2]/atemp[1][1],   2*atemp[1][3]/atemp[1][1]});
         th.push_back({  2*atemp[2][0]/atemp[2][2],    2*atemp[2][1]/atemp[2][2], 2,   2*atemp[2][3]/atemp[2][2]});
         th.push_back({ 2*atemp[3][0]/atemp[3][3],   2*atemp[3][1]/atemp[3][3],   2*atemp[3][2]/atemp[3][3], 2});
     }
 
     template <class traits>
     inline void compute(const int k);
 
     inline void setA(double** A) { ma_Beta = A; }
     inline void setOmega(double Omega) { mOmega = Omega; }
     inline void setKappa(double** kappa) { ma_Gamma = kappa; }
 
     inline double calcdH1(double c1, double c2) {
         if (fabs(c1)==0&&fabs(c2)==0) return 0;
         return -2*c1*fabs(c2)*c2/pow(fabs(c2)+c1*c1,2);
     }
 
     inline double calcdH2(double c1, double c2) {
         if (fabs(c1)==0&&fabs(c2)==0) return 1;
         return (2*fabs(c2)*c1*c1+c2*c2)/pow(fabs(c2)+c1*c1,2);
     }
 
     inline double calcH(double c1, double c2) {
         if (fabs(c1)==0&&fabs(c2)==0) return 0;
         return (fabs(c2)*c2)/(fabs(c2)+c1*c1);
     }
 
     inline double calcdP(std::vector<double> c,int i,int j, int k, int l){
 
         return pow(c[k],2)*pow(c[l],2)*(th[i][j]*(calcdH1(c[i], c[i]*c[j]) + c[j]*calcdH2(c[i], c[i]*c[j])) + th[j][i]*c[j]*calcdH2(c[j], c[i]*c[j])) + pow(c[j],2)*pow(c[l],2)*(th[i][k]*(calcdH1(c[i], c[i]*c[k]) + c[k]*calcdH2(c[i], c[i]*c[k])) + th[k][i]*c[k]*calcdH2(c[k], c[i]*c[k])) + pow(c[j],2)*pow(c[k],2)*(th[i][l]*(calcdH1(c[i], c[i]*c[l]) + c[l]*calcdH2(c[i], c[i]*c[l])) + th[l][i]*c[l]*calcdH2(c[l], c[i]*c[l])) + 2*c[i]*pow(c[l],2)*(th[j][k]*calcH(c[j], c[j]*c[k]) + th[k][j]*calcH(c[k], c[j]*c[k])) + 2*c[i]*pow(c[k],2)*(th[j][l]*calcH(c[j], c[j]*c[l]) + th[l][j]*calcH(c[l], c[j]*c[l])) + 2*c[i]*pow(c[j],2)*(th[k][l]*calcH(c[k], c[k]*c[l]) + th[l][k]*calcH(c[l], c[k]*c[l]));
     }
 
     inline double calcdPSimple(std::vector<double> c,int i,int j, int k, int l){
 
         return pow(c[k],2)*pow(c[l],2)*(th[i][j]*c[j]) + pow(c[j],2)*pow(c[l],2)*(th[i][k]*c[k]) + pow(c[j],2)*pow(c[k],2)*(th[i][l]*c[l]);
     }
 
     inline void setBeta(int i, int j, double beta) {
         if ((int)mv_Beta.size() - 1 < i || (int)mv_Beta.size() - 1 < j) {
             mv_Beta.resize(mv_Beta.size() + std::max<int>(i - (int)mv_Beta.size() + 1, j - (int)mv_Beta.size() + 1));
             for (int l = 0; l < (int)mv_Beta.size(); l++) {
                 mv_Beta[l].resize(mv_Beta[l].size() +
                                   std::max<int>(i - (int)mv_Beta[l].size() + 1, j - (int)mv_Beta[l].size() + 1));
                 mv_Beta[l][l] = 0;
             }
         }
         if (i != j) {
             mv_Beta[i][j] = beta;
             mv_Beta[j][i] = beta;
         }
     }
     inline void setGamma(int i, int j, double gamma) {
         if ((int)mv_Gamma.size() - 1 < i || (int)mv_Gamma.size() - 1 < j) {
             mv_Gamma.resize(mv_Gamma.size() +
                             std::max<int>(i - (int)mv_Gamma.size() + 1, j - (int)mv_Gamma.size() + 1));
             for (int l = 0; l < (int)mv_Gamma.size(); l++) {
                 mv_Gamma[l].resize(mv_Gamma[l].size() +
                                    std::max<int>(i - (int)mv_Gamma[l].size() + 1, j - (int)mv_Gamma[l].size() + 1));
                 mv_Gamma[l][l] = 0;
             }
         }
         if (i != j) {
             mv_Gamma[i][j] = gamma;
             mv_Gamma[j][i] = gamma;
         }
     }
 
     inline void setBetaAndGamma(int i, int j, double beta, double gamma) {
         setBeta(i, j, beta);
         setGamma(i, j, gamma);
     }
 
     inline void setBeta(std::vector<std::vector<double>>& beta) { mv_Beta = beta; }
     inline void setGamma(std::vector<std::vector<double>>& gamma) { mv_Gamma = gamma; }
     inline void setLambdas(std::vector<double>& lambda) { mv_Lambda = lambda; }
     template <typename... TLambdas>
     inline void setLambdas(TLambdas... lambda) {
 
         std::vector<double> l{{lambda...}};
         mv_Lambda = l;
     }
     inline void setBetaAndGamma(std::vector<std::vector<double>>& beta, std::vector<std::vector<double>>& gamma) {
         mv_Beta = beta;
         mv_Gamma = gamma;
     }
 
     inline void setD(double d) {
         for (int l = 0; l < (int)mv_Gamma.size(); l++) {
             for (int m = 0; m < (int)mv_Gamma.size(); m++) {
                 mv_Gamma[l][m] *= d / mD;
             }
         }
         for (int l = 0; l < (int)mv_Beta.size(); l++) {
             for (int m = 0; m < (int)mv_Beta.size(); m++) {
                 mv_Beta[l][m] *= mD / d;
             }
         }
         mD = d;
     }
 
     inline void setSurfaceTension(int i, int j, double surfacetension) {
         setBeta(i, j, 3.0 * surfacetension / mD);
         setGamma(i, j, -3.0 * mD * surfacetension / 4.0);
     }
 
    public:
     template <int numberofcomponents>
     inline bool checkValid() {
         if ((int)mv_Beta.size() != numberofcomponents || (int)mv_Beta[0].size() != numberofcomponents ||
             (int)mv_Gamma.size() != numberofcomponents || (int)mv_Gamma[0].size() != numberofcomponents ||
             (int)mv_Lambda.size() != numberofcomponents) {
             throw std::runtime_error(
                 "Number of beta/gamma parameters does not match the number of "
                 "components.");
             return false;
         }
         return true;
     }
 
     bool mIsValid;
     double** ma_Gamma;
     double** ma_Beta;
     std::vector<std::vector<double>> mv_Beta;
     std::vector<std::vector<double>> mv_Gamma;
     std::vector<double> mv_Lambda;
     double mD = 5;
     double mOmega = 0.0;
     double cutoff = 1e-10;
 
 };
 
 template <class TTraits>
 inline void ChemicalPotentialCalculator4ComponentBoyerSplit::compute(const int k) {
     if (Geometry<typename TTraits::Lattice>::isBulkSolid(k)) return;
 
     [[maybe_unused]] static bool isvalid = checkValid<4>();
     mIsValid=isvalid;
     using Lattice = typename TTraits::Lattice;
     constexpr int N = 4;
     double chempot=0.0;
     double chempot_bulk=0.0;
     double chempot_tension=0.0;
     double sumci=0.0;
     double sumcj=0.0;
 
     double onlylaplacian=0.0;
     double gammalaplacian=0.0;
     double onlylaplaciansum=0.0;
     double gammalaplaciansum=0.0;   
 
     std::vector<double> CVec = {getInstance<OrderParameter, N - 1, Lattice>(0)[k],
                                 getInstance<OrderParameter, N - 1, Lattice>(1)[k],
                                 getInstance<OrderParameter, N - 1, Lattice>(2)[k],
                                 1-getInstance<OrderParameter, N - 1, Lattice>(0)[k]-getInstance<OrderParameter, N - 1, Lattice>(1)[k]-getInstance<OrderParameter, N - 1, Lattice>(2)[k]};    
 
     for (int i=0;i<N-1;i++){
         chempot=0.0;
         chempot_bulk=0.0;
         chempot_tension=0.0;
         const double& ci=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
         sumci += ci;
 
         sumcj=0.0;
         gammalaplaciansum=0.0;
         onlylaplaciansum=0.0;
         double product = 1;
         double P=0;
         for (int j=0;j<N-1;j++){
             
             const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
             sumcj+=cj;
             
             onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];
             
             onlylaplaciansum += onlylaplacian;                
             gammalaplacian=mv_Gamma[i][j]*onlylaplacian;
             gammalaplaciansum += gammalaplacian;
 
             chempot+=2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
             chempot_bulk += 2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj);
             chempot_tension += - gammalaplacian;
             if(i!=j) {
                 product *= cj;
             
                 
             }
         }// j
         /*
         for (int j=0;j<N-1;j++){
             if(i!=j){
                 const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
                 for (int m=j+1;m<N-1;m++){
                     const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
                     if(i!=m) {
                         P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
                         P+=-2*atemp[i][3]/atemp[i][i]*(1-sumcj)*cj*cj*cm*cm;
                     }
                 }
                 P+=2*ci*pow(cj,2)*pow(1-sumcj,2);
                 for (int m=0;m<N-1;m++){
                     const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
                     if(i!=m&&j!=m) {
                         P+=-2*atemp[i][m]/atemp[i][i]*cm*cj*cj*pow(1-sumcj,2);
                     }
                 }
             }
         }*/
 
        for (int j=0;j<N-1;j++){
             if(i!=j){
                 const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
                 for (int m=j+1;m<N-1;m++){
                     const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
                     if(i!=m) {
                         P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
                         
                     }
                 }
                 P+=2*ci*pow(cj,2)*pow(1-sumcj,2);
             }
         }
         if (i==0) P+=-calcdPSimple(CVec,0,1,2,3);
         if (i==1) P+=-calcdPSimple(CVec,1,0,2,3);
         if (i==2) P+=-calcdPSimple(CVec,2,0,1,3);
 
         const double cj=1.0-sumcj;
 
         product *= cj;
 
         chempot+=2*mv_Beta[i][N-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) + mv_Gamma[i][N-1]*onlylaplaciansum;
         chempot_bulk += 2*mv_Beta[i][N-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj);
         chempot_tension += mv_Gamma[i][N-1]*onlylaplaciansum;
 
         //getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + mv_Lambda[i]*product;
         
         double lambdadH2 = 0;
         for (int j=0;j<N-1;j++){
             lambdadH2+=mv_Lambda[j]*calcdH2(getInstance<OrderParameter, N - 1, Lattice>(j)[k],ci*product);
             
         }
         lambdadH2+=mv_Lambda[N-1]*calcdH2(1-sumcj,ci*product);
         
         //lambdadH2=mv_Lambda[i];
         getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + 2 * mOmega * (ci < 0) * ci/*-
                                            2 * mOmega * ((1-ci) < 0) * (1-ci)*/ + mv_Lambda[i]*product;//+0.5*P;//*calcdH1(ci,ci*product)+product*(lambdadH2)+0.5*P;
         getInstance<ChemicalPotential2, N, Lattice>(i)[k] = chempot;
     } // i=0:N-2
     
 
     int i = N-1;
     chempot=0.0;
     chempot_bulk=0.0;
     chempot_tension=0.0;
     const double& ci=1.0-sumci;
     
 
     sumcj=0.0;
     gammalaplaciansum=0.0;
     onlylaplaciansum=0.0;
     double product = 1;
     for (int j=0;j<N-1;j++){
         const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
         sumcj+=cj;
         
         onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];                         
         onlylaplaciansum += onlylaplacian;
 
         gammalaplacian = mv_Gamma[i][j]*onlylaplacian;
         gammalaplaciansum += gammalaplacian;
         
         if (i!=j){
             chempot+=2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
             chempot_bulk += 2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj);
             chempot_tension += - gammalaplacian;
         }
         product *= cj;
     }
     /*
     double P = 0;
     for (int j=0;j<N-1;j++){
         const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
         for (int m=j+1;m<N-1;m++){
             const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
             P+=ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
             for (int n=0;n<N-1;n++){
                 const double& cn=getInstance<OrderParameter, N - 1, Lattice>(n)[k];
                 if(n!=i&&n!=j&&n!=m) P+=-2*atemp[3][n]/atemp[3][3]*cn*cj*cj*cm*cm;
             }
         }
     }*/
 
     double P = 0;
     for (int j=0;j<N-1;j++){
         const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
         for (int m=j+1;m<N-1;m++){
             const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
             P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
         }
     }
     P+=-calcdPSimple(CVec,3,0,1,2);
     //getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot+mv_Lambda[i]*product;//+0.5*P;
     double lambdadH2 = 0;
     for (int j=0;j<N-1;j++){
         lambdadH2+=mv_Lambda[j]*calcdH2(getInstance<OrderParameter, N - 1, Lattice>(j)[k],ci*product);
         
     }
     lambdadH2+=mv_Lambda[N-1]*calcdH2(1-sumcj,ci*product);
     
     getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + 2 * mOmega * (ci < 0) * ci/*-
                                            2 * mOmega * ((1-ci) < 0) * (1-ci)*/ + mv_Lambda[i]*product;//+0.5*P;//mv_Lambda[i]*calcdH1(ci,ci*product)+product*(lambdadH2)+0.5*P;
     getInstance<ChemicalPotential2, N, Lattice>(i)[k] = chempot;
 }
 


template<int TN>
class ChemicalPotentialCalculatorNCompBoyerFinal : public AddOnBase {
    public:
     ChemicalPotentialCalculatorNCompBoyerFinal() : mv_Beta({{0}}), mv_Gamma({{0}}) {}

     std::array<std::array<std::array<std::array<double,TN>,TN>,TN>,TN> th;
     std::array<std::array<std::array<std::array<double,TN>,TN>,TN>,TN> lambda;

     inline void setLambda(int i,int j,int k,int l,double lambdaijkl) { lambda[i][j][k][l] = lambdaijkl; lambda[i][j][l][k] = lambdaijkl; lambda[i][l][j][k] = lambdaijkl; lambda[i][l][k][j] = lambdaijkl; lambda[i][k][j][l] = lambdaijkl; lambda[i][k][l][j] = lambdaijkl;}

     inline void setTheta(int i,int j,int k,int l,double thetaijkl) { th[i][j][k][l]=thetaijkl;}
 
     template <class traits>
     inline void compute(const int k);
 
     inline void setA(double** A) { ma_Beta = A; }
     inline void setOmega(double Omega) { mOmega = Omega; }
     inline void setKappa(double** kappa) { ma_Gamma = kappa; }
 
     inline double calcdPSimple(std::vector<double> c,int i,int j, int k, int l){
 
         return pow(c[j],2)*pow(c[k],2)*(th[j][k][i][l]*c[l]);// + pow(c[j],2)*pow(c[l],2)*(th[j][l][i][k]*c[k]) + pow(c[j],2)*pow(c[k],2)*(th[j][k][i][l]*c[l]);
     }
 
     inline void setBeta(int i, int j, double beta) {
         if ((int)mv_Beta.size() - 1 < i || (int)mv_Beta.size() - 1 < j) {
             mv_Beta.resize(mv_Beta.size() + std::max<int>(i - (int)mv_Beta.size() + 1, j - (int)mv_Beta.size() + 1));
             for (int l = 0; l < (int)mv_Beta.size(); l++) {
                 mv_Beta[l].resize(mv_Beta[l].size() +
                                   std::max<int>(i - (int)mv_Beta[l].size() + 1, j - (int)mv_Beta[l].size() + 1));
                 mv_Beta[l][l] = 0;
             }
         }
         if (i != j) {
             mv_Beta[i][j] = beta;
             mv_Beta[j][i] = beta;
         }
     }
     inline void setGamma(int i, int j, double gamma) {
         if ((int)mv_Gamma.size() - 1 < i || (int)mv_Gamma.size() - 1 < j) {
             mv_Gamma.resize(mv_Gamma.size() +
                             std::max<int>(i - (int)mv_Gamma.size() + 1, j - (int)mv_Gamma.size() + 1));
             for (int l = 0; l < (int)mv_Gamma.size(); l++) {
                 mv_Gamma[l].resize(mv_Gamma[l].size() +
                                    std::max<int>(i - (int)mv_Gamma[l].size() + 1, j - (int)mv_Gamma[l].size() + 1));
                 mv_Gamma[l][l] = 0;
             }
         }
         if (i != j) {
             mv_Gamma[i][j] = gamma;
             mv_Gamma[j][i] = gamma;
         }
     }
 
     inline void setBetaAndGamma(int i, int j, double beta, double gamma) {
         setBeta(i, j, beta);
         setGamma(i, j, gamma);
     }
 
     inline void setBeta(std::vector<std::vector<double>>& beta) { mv_Beta = beta; }
     inline void setGamma(std::vector<std::vector<double>>& gamma) { mv_Gamma = gamma; }

     inline void setBetaAndGamma(std::vector<std::vector<double>>& beta, std::vector<std::vector<double>>& gamma) {
         mv_Beta = beta;
         mv_Gamma = gamma;
     }
 
     inline void setD(double d) {
         for (int l = 0; l < (int)mv_Gamma.size(); l++) {
             for (int m = 0; m < (int)mv_Gamma.size(); m++) {
                 mv_Gamma[l][m] *= d / mD;
             }
         }
         for (int l = 0; l < (int)mv_Beta.size(); l++) {
             for (int m = 0; m < (int)mv_Beta.size(); m++) {
                 mv_Beta[l][m] *= mD / d;
             }
         }
         mD = d;
     }
 
     inline void setSurfaceTension(int i, int j, double surfacetension) {
         setBeta(i, j, 3.0 * surfacetension / mD);
         setGamma(i, j, -3.0 * mD * surfacetension / 4.0);
     }
 
    public:
     template <int numberofcomponents>
     inline bool checkValid() {
         if ((int)mv_Beta.size() != numberofcomponents || (int)mv_Beta[0].size() != numberofcomponents ||
             (int)mv_Gamma.size() != numberofcomponents || (int)mv_Gamma[0].size() != numberofcomponents) {
             throw std::runtime_error(
                 "Number of beta/gamma parameters does not match the number of "
                 "components.");
             return false;
         }
         return true;
     }
 
     bool mIsValid;
     double** ma_Gamma;
     double** ma_Beta;
     std::vector<std::vector<double>> mv_Beta;
     std::vector<std::vector<double>> mv_Gamma;
     double mD = 5;
     double mOmega = 0.0;
 
 };
 
 template<int TN>
 template <class TTraits>
 inline void ChemicalPotentialCalculatorNCompBoyerFinal<TN>::compute(const int k) {
     if (Geometry<typename TTraits::Lattice>::isBulkSolid(k)) return;
 
     [[maybe_unused]] static bool isvalid = checkValid<TN>();
     mIsValid=isvalid;
     using Lattice = typename TTraits::Lattice;
     constexpr int N = TN;
     double chempot=0.0;
     
     double sumci=0.0;
     double sumcj=0.0;
 
     double onlylaplacian=0.0;
     double gammalaplacian=0.0;
     double onlylaplaciansum=0.0;
 
     std::vector<double> CVec = {};    
 
    double sum = 0;
    for (int i=0;i<N-1;i++){
        CVec.push_back(getInstance<OrderParameter, N - 1, Lattice>(i)[k]);
        sum+=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
    }
    CVec.push_back(1-sum);

     for (int i=0;i<N-1;i++){
         chempot=0.0;

         const double& ci=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
         sumci += ci;
 
         sumcj=0.0;
         onlylaplaciansum=0.0;
         double product = 1;
         double P=0;
         for (int j=0;j<N-1;j++){
             
             const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
             sumcj+=cj;
             
             onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];
             
             onlylaplaciansum += onlylaplacian;                
             gammalaplacian=mv_Gamma[i][j]*onlylaplacian;
 
             chempot+=2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
             if(i!=j) {
                 product *= cj;
             
                 
             }
         }// j

         for (int j=0;j<N-1;j++){
            if(i!=j){
                const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
                for (int m=j+1;m<N-1;m++){
                    const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
                    if(i!=m) {
                        P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
                        
                    }
                }
                P+=2*ci*pow(cj,2)*pow(1-sumcj,2);
            }
        }

        for (int j=0;j<N;j++){
            if(i!=j){
                for (int m=j+1;m<N;m++){
                    if(i!=m){
                        for (int n=0;n<N;n++){
                            if(i!=n&&j!=n&&m!=n) {
                                P+=-calcdPSimple(CVec,i,j,m,n);
                            }
                        }
                    }
                }
            }
        }
        double lambdaterm=0;

        for (int j=0;j<N;j++){
            if (j!=i) {
            for (int m=j+1;m<N;m++){
                if (m!=i) {
                for (int n=m+1;n<N;n++){
                    if (n!=i) {

                    lambdaterm += lambda[i][j][m][n]*CVec[j]*CVec[m]*CVec[n];
                    
                    }
                }
                }
            }
            }
        }
 
         const double cj=1.0-sumcj;
 
         product *= cj;
 
         chempot+=2*mv_Beta[i][N-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) + mv_Gamma[i][N-1]*onlylaplaciansum;
         
         double chemforcecorr = 0;
         for (int j=0;j<N;j++){
             if (i!=j) {
                 chemforcecorr += 2*mv_Beta[i][j]*(pow(CVec[i]+CVec[j],2)*(CVec[i]+CVec[j]-1)+(CVec[i]+CVec[j],2)*pow(CVec[i]+CVec[j]-1,2));
             }
         }

         getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + 2 * mOmega * (ci < 0) * ci-
                                            2 * mOmega * ((1-ci) < 0) * (1-ci) + lambdaterm+1./6.*P;//- 1./6.*chemforcecorr;// + lambdaterm+0.8*P;//*calcdH1(ci,ci*product)+product*(lambdadH2)+0.5*P;
        //getInstance<ChemicalPotential2, N, Lattice>(i)[k] = chempot + lambdaterm;//*calcdH1(ci,ci*product)+product*(lambdadH2)+0.5*P;
         
 
     } // i=0:N-2
     
 
     int i = N-1;
     chempot=0.0;

     const double& ci=1.0-sumci;
     
 
     sumcj=0.0;
     onlylaplaciansum=0.0;
     double product = 1;
     for (int j=0;j<N-1;j++){
         const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
         sumcj+=cj;
         
         onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];                         
         onlylaplaciansum += onlylaplacian;
 
         gammalaplacian = mv_Gamma[i][j]*onlylaplacian;
         
         if (i!=j){
             chempot+=2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
         }
         product *= cj;
    }

    double P = 0;
    for (int j=0;j<N-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        for (int m=j+1;m<N-1;m++){
            const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
            P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
        }
    }
    for (int j=0;j<N;j++){
        if(i!=j){
            for (int m=j+1;m<N;m++){
                if(i!=m){
                    for (int n=0;n<N;n++){
                        if(i!=n&&j!=n&&m!=n) {
                            P+=-calcdPSimple(CVec,i,j,m,n);
                        }
                    }
                }
            }
        }
    }

    double lambdaterm = 0;
    for (int j=0;j<N;j++){
        if (j!=i) {
        for (int m=j+1;m<N;m++){
            if (m!=i) {
            for (int n=m+1;n<N;n++){
                if (n!=i) {

                lambdaterm += lambda[i][j][m][n]*CVec[j]*CVec[m]*CVec[n];
                
                }
            }
            }
        }
        }
    }

    double chemforcecorr = 0;
    for (int j=0;j<N;j++){
        if (i!=j) {
            chemforcecorr += 2*mv_Beta[i][j]*(pow(CVec[i]+CVec[j],2)*(CVec[i]+CVec[j]-1)+(CVec[i]+CVec[j],2)*pow(CVec[i]+CVec[j]-1,2));
        }
    }

     getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + 2 * mOmega * (ci < 0) * ci-
                                            2 * mOmega * ((1-ci) < 0) * (1-ci)  + lambdaterm+1./6.*P;// - 1./6.*chemforcecorr;// + lambdaterm+0.8*P;//mv_Lambda[i]*calcdH1(ci,ci*product)+product*(lambdadH2)+0.5*P;
    //getInstance<ChemicalPotential2, N, Lattice>(i)[k] =  chempot + lambdaterm;//mv_Lambda[i]*calcdH1(ci,ci*product)+product*(lambdadH2)+0.5*P;
     
     
 }







class ChemicalPotentialCalculator5ComponentBoyer : public AddOnBase {
   public:
    ChemicalPotentialCalculator5ComponentBoyer() : mv_Beta({{0}}), mv_Gamma({{0}}) {
        //atemp.push_back({-248000.0/151.,   84000.0/151.,  132000.0/151.,   16000.0/151.,   16000.0/151.});
        //atemp.push_back({  84000.0/151., -272000.0/151.,    4000.0/151.,   92000.0/151.,   92000.0/151.});
        //atemp.push_back({  132000.0/151.,    4000.0/151., -844000.0/453.,  218000.0/453.,  218000.0/453.});
        //atemp.push_back({ 16000.0/151.,   92000.0/151.,  218000.0/453., -724000.0/453.,  182000.0/453.});
        //atemp.push_back({ 16000.0/151.,   92000.0/151.,  218000.0/453.,  182000.0/453., -724000.0/453.});
        atemp.push_back({-163.535911602210, 59.6685082872928, 90.6077348066298, 13.2596685082873});
        atemp.push_back({59.6685082872928, -156.906077348066, 20.9944751381215, 76.2430939226520});
        atemp.push_back({90.6077348066298, 20.9944751381216, -171.823204419889, 60.2209944751381});
        atemp.push_back({13.2596685082873, 76.2430939226519, 60.2209944751381, -149.723756906077});
        /*atemp.push_back({-160.,   40.,  40.,   40.,   40.});
        atemp.push_back({  40., -160.,    40.,   40.,   40.});
        atemp.push_back({  40.,    40., -160.,  40.,  40.});
        atemp.push_back({ 40.,   40.,  40., -160.,  40.});
        atemp.push_back({ 40.,   40.,  40.,  40., -160.});*/
        //th = getTh();
        /*th.push_back({2,   2*atemp[0][1]/atemp[0][0],   2*atemp[0][2]/atemp[0][0],  2*atemp[0][3]/atemp[0][0],  2*atemp[0][4]/atemp[0][0]});
        th.push_back({  2*atemp[1][0]/atemp[1][1], 2,    2*atemp[1][2]/atemp[1][1],   2*atemp[1][3]/atemp[1][1],   2*atemp[1][4]/atemp[1][1]});
        th.push_back({  2*atemp[2][0]/atemp[2][2],    2*atemp[2][1]/atemp[2][2], 2,   2*atemp[2][3]/atemp[2][2],   2*atemp[2][4]/atemp[2][2]});
        th.push_back({ 2*atemp[3][0]/atemp[3][3],   2*atemp[3][1]/atemp[3][3],   2*atemp[3][2]/atemp[3][3], 2, 2*atemp[3][4]/atemp[3][3]});
        th.push_back({ 2*atemp[4][0]/atemp[4][4],   2*atemp[4][1]/atemp[4][4],   2*atemp[4][2]/atemp[4][4],  2*atemp[4][3]/atemp[4][4], 2});*/
    
    }

    std::array<std::array<std::array<std::array<double,5>,5>,5>,5> getTh() {
        std::array<std::array<std::array<std::array<double,5>,5>,5>,5> _th;
        for (int i=0;i<5;i++){
            for (int j=0;j<5;j++){
                for (int k=0;k<5;k++){
                    if (k==i||k==j) continue;
                    for (int l=0;l<5;l++){
                        if (l==i||l==j) continue;
                        int _n=0;
                        for (int n=0;n<5;n++) if (n!=i&&n!=j&&n!=k&&n!=l) _n=n;
                        if ((atemp[l][_n]*atemp[k][k]-atemp[k][_n]*atemp[l][k])==(atemp[k][k]*atemp[l][l]-atemp[l][k]*atemp[k][l])) _th[i][j][k][l]=0;
                        else _th[i][j][k][l]=2*(atemp[l][_n]*atemp[k][k]-atemp[k][_n]*atemp[l][k])/(atemp[k][k]*atemp[l][l]-atemp[l][k]*atemp[k][l]);
                        //if (TIME==0) std::cout<<k<<" "<<l<<" "<<2*atemp[k][l]/atemp[k][k]<<" "<<_th[i][j][k][l]<<std::endl;
                    }
                }
            }
        }
        return _th;
    }

    std::vector<std::vector<double>> atemp;
    std::array<std::array<std::array<std::array<double,5>,5>,5>,5> th;
    //std::vector<std::vector<double>> th;

    template <class traits>
    inline void compute(const int k);

    inline void setA(double** A) { ma_Beta = A; }

    inline void setP(double P) { m_P = P; }
    inline void setOmega(double Omega) { mOmega = Omega; }

    inline void setKappa(double** kappa) { ma_Gamma = kappa; }

    inline double calcdH1(double c1, double c2) {
        if (fabs(c1)==0&&fabs(c2)==0) return 0;
        return -2*c1*fabs(c2)*c2/pow(fabs(c2)+c1*c1,2);
    }

    inline double calcdH2(double c1, double c2) {
        if (fabs(c1)==0&&fabs(c2)==0) return 1;
        //return (2*fabs(c2)*c1*c1+c2*c2)/pow(fabs(c2)+c1*c1,2);
        return (2*fabs(c2)*c1*c1+c2*c2)/pow(fabs(c2)+c1*c1,2);
    }

    inline double calcH(double c1, double c2) {
        if (fabs(c1)==0&&fabs(c2)==0) return 0;
        //return (2*fabs(c2)*c1*c1+c2*c2)/pow(fabs(c2)+c1*c1,2);
        return (fabs(c2)*c2)/(fabs(c2)+c1*c1);
    }

    inline double calcdP(std::vector<double> c,int i,int j, int k, int l){
        return pow(c[k],2)*pow(c[l],2)*(th[k][l][i][j]*(calcdH1(c[i], c[i]*c[j]) + c[j]*calcdH2(c[i], c[i]*c[j])) + th[k][l][j][i]*c[j]*calcdH2(c[j], c[i]*c[j])) + pow(c[j],2)*pow(c[l],2)*(th[j][l][i][k]*(calcdH1(c[i], c[i]*c[k]) + c[k]*calcdH2(c[i], c[i]*c[k])) + th[j][l][k][i]*c[k]*calcdH2(c[k], c[i]*c[k])) + pow(c[j],2)*pow(c[k],2)*(th[j][k][i][l]*(calcdH1(c[i], c[i]*c[l]) + c[l]*calcdH2(c[i], c[i]*c[l])) + th[j][k][l][i]*c[l]*calcdH2(c[l], c[i]*c[l])) + 2*c[i]*pow(c[l],2)*(th[i][l][j][k]*calcH(c[j], c[j]*c[k]) + th[i][l][k][j]*calcH(c[k], c[j]*c[k])) + 2*c[i]*pow(c[k],2)*(th[i][k][j][l]*calcH(c[j], c[j]*c[l]) + th[i][k][l][j]*calcH(c[l], c[j]*c[l])) + 2*c[i]*pow(c[j],2)*(th[i][j][k][l]*calcH(c[k], c[k]*c[l]) + th[i][j][l][k]*calcH(c[l], c[k]*c[l]));
    }
    /*inline double calcdP(std::vector<double> c,int i,int j, int k, int l){
        return pow(c[k],2)*pow(c[l],2)*(th[i][j]*(calcdH1(c[i], c[i]*c[j]) + c[j]*calcdH2(c[i], c[i]*c[j])) + th[j][i]*c[j]*calcdH2(c[j], c[i]*c[j])) + pow(c[j],2)*pow(c[l],2)*(th[i][k]*(calcdH1(c[i], c[i]*c[k]) + c[k]*calcdH2(c[i], c[i]*c[k])) + th[k][i]*c[k]*calcdH2(c[k], c[i]*c[k])) + pow(c[j],2)*pow(c[k],2)*(th[i][l]*(calcdH1(c[i], c[i]*c[l]) + c[l]*calcdH2(c[i], c[i]*c[l])) + th[l][i]*c[l]*calcdH2(c[l], c[i]*c[l])) + 2*c[i]*pow(c[l],2)*(th[j][k]*calcH(c[j], c[j]*c[k]) + th[k][j]*calcH(c[k], c[j]*c[k])) + 2*c[i]*pow(c[k],2)*(th[j][l]*calcH(c[j], c[j]*c[l]) + th[l][j]*calcH(c[l], c[j]*c[l])) + 2*c[i]*pow(c[j],2)*(th[k][l]*calcH(c[k], c[k]*c[l]) + th[l][k]*calcH(c[l], c[k]*c[l]));
    }*/

    inline void setBeta(int i, int j, double beta) {
        if ((int)mv_Beta.size() - 1 < i || (int)mv_Beta.size() - 1 < j) {
            mv_Beta.resize(mv_Beta.size() + std::max<int>(i - (int)mv_Beta.size() + 1, j - (int)mv_Beta.size() + 1));
            for (int l = 0; l < (int)mv_Beta.size(); l++) {
                mv_Beta[l].resize(mv_Beta[l].size() +
                                  std::max<int>(i - (int)mv_Beta[l].size() + 1, j - (int)mv_Beta[l].size() + 1));
                mv_Beta[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Beta[i][j] = beta;
            mv_Beta[j][i] = beta;
        }
    }
    inline void setGamma(int i, int j, double gamma) {
        if ((int)mv_Gamma.size() - 1 < i || (int)mv_Gamma.size() - 1 < j) {
            mv_Gamma.resize(mv_Gamma.size() +
                            std::max<int>(i - (int)mv_Gamma.size() + 1, j - (int)mv_Gamma.size() + 1));
            for (int l = 0; l < (int)mv_Gamma.size(); l++) {
                mv_Gamma[l].resize(mv_Gamma[l].size() +
                                   std::max<int>(i - (int)mv_Gamma[l].size() + 1, j - (int)mv_Gamma[l].size() + 1));
                mv_Gamma[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Gamma[i][j] = gamma;
            mv_Gamma[j][i] = gamma;
        }
    }

    inline void setBetaAndGamma(int i, int j, double beta, double gamma) {
        setBeta(i, j, beta);
        setGamma(i, j, gamma);
    }

    inline void setBeta(std::vector<std::vector<double>>& beta) { mv_Beta = beta; }
    inline void setGamma(std::vector<std::vector<double>>& gamma) { mv_Gamma = gamma; }
    inline void setLambdas(std::vector<double> lambda0123,std::vector<double> lambda0124,std::vector<double> lambda0134,std::vector<double> lambda0234,std::vector<double> lambda1234) { mv_Lambda_0123 = lambda0123; mv_Lambda_0124 = lambda0124; mv_Lambda_0134 = lambda0134; mv_Lambda_0234 = lambda0234; mv_Lambda_1234 = lambda1234; }

    inline void setBetaAndGamma(std::vector<std::vector<double>>& beta, std::vector<std::vector<double>>& gamma) {
        mv_Beta = beta;
        mv_Gamma = gamma;
    }

    inline void setD(double d) {
        for (int l = 0; l < (int)mv_Gamma.size(); l++) {
            for (int m = 0; m < (int)mv_Gamma.size(); m++) {
                mv_Gamma[l][m] *= d / mD;
            }
        }
        for (int l = 0; l < (int)mv_Beta.size(); l++) {
            for (int m = 0; m < (int)mv_Beta.size(); m++) {
                mv_Beta[l][m] *= mD / d;
            }
        }
        mD = d;
    }

    inline void setSurfaceTension(int i, int j, double surfacetension) {
        setBeta(i, j, 3.0 * surfacetension / mD);
        setGamma(i, j, -3.0 * mD * surfacetension / 4.0);
    }

   public:
    template <int numberofcomponents>
    inline bool checkValid() {
        if ((int)mv_Beta.size() != numberofcomponents || (int)mv_Beta[0].size() != numberofcomponents ||
            (int)mv_Gamma.size() != numberofcomponents || (int)mv_Gamma[0].size() != numberofcomponents ||
            (int)mv_Lambda_0123.size() != numberofcomponents) {
            throw std::runtime_error(
                "Number of beta/gamma parameters does not match the number of "
                "components.");
            return false;
        }
        return true;
    }

    bool mIsValid;
    double** ma_Gamma;
    double** ma_Beta;
    std::vector<std::vector<double>> mv_Beta;
    std::vector<std::vector<double>> mv_Gamma;
    std::vector<double> mv_Lambda_0123;
    std::vector<double> mv_Lambda_0124;
    std::vector<double> mv_Lambda_0134;
    std::vector<double> mv_Lambda_0234;
    std::vector<double> mv_Lambda_1234;
    double mD = 5;
    double m_P = 0.25;
    double mOmega = 0.0;
    double cutoff = 1e-10;

};

template <class TTraits>
inline void ChemicalPotentialCalculator5ComponentBoyer::compute(const int k) {
    if (Geometry<typename TTraits::Lattice>::isBulkSolid(k)) return;

    //[[maybe_unused]] static bool isvalid = checkValid<5>();
    //mIsValid=isvalid;
    using Lattice = typename TTraits::Lattice;
    constexpr int N = TTraits::NumberOfComponents;
    double chempot=0.0;
    double chempot_bulk=0.0;
    double chempot_tension=0.0;
    double sumci=0.0;
    double sumcj=0.0;

    double onlylaplacian=0.0;
    double gammalaplacian=0.0;
    double onlylaplaciansum=0.0;
    double gammalaplaciansum=0.0;   

    std::vector<double> CVec = {getInstance<OrderParameter, N - 1, Lattice>(0)[k],
                                getInstance<OrderParameter, N - 1, Lattice>(1)[k],
                                getInstance<OrderParameter, N - 1, Lattice>(2)[k],
                                1-getInstance<OrderParameter, N - 1, Lattice>(0)[k]-getInstance<OrderParameter, N - 1, Lattice>(1)[k]-getInstance<OrderParameter, N - 1, Lattice>(2)[k]};//getInstance<OrderParameter, N - 1, Lattice>(3)[k],
                                //1-getInstance<OrderParameter, N - 1, Lattice>(0)[k]-getInstance<OrderParameter, N - 1, Lattice>(1)[k]-getInstance<OrderParameter, N - 1, Lattice>(2)[k]-getInstance<OrderParameter, N - 1, Lattice>(3)[k]};    

    double product0123 = CVec[0]*CVec[1]*CVec[2]*CVec[3];
    /*double product0124 = CVec[0]*CVec[1]*CVec[2]*CVec[4];
    double product0134 = CVec[0]*CVec[1]*CVec[3]*CVec[4];
    double product0234 = CVec[0]*CVec[2]*CVec[3]*CVec[4];
    double product1234 = CVec[1]*CVec[2]*CVec[3]*CVec[4];*/

    for (int i=0;i<N-1;i++){
        chempot=0.0;
        chempot_bulk=0.0;
        chempot_tension=0.0;
        const double& ci=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
        sumci += ci;

        sumcj=0.0;
        gammalaplaciansum=0.0;
        onlylaplaciansum=0.0;
        double product = 1;
        double P=0;
        for (int j=0;j<N-1;j++){
            
            const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
            sumcj+=cj;
            
            onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];
            
            onlylaplaciansum += onlylaplacian;                
            gammalaplacian=mv_Gamma[i][j]*onlylaplacian;
            gammalaplaciansum += gammalaplacian;

            chempot+=2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
            chempot_bulk += 2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj);
            chempot_tension += - gammalaplacian;
            if(i!=j) {
                product *= cj;
            
                
            }
        }// j
        /*
        for (int j=0;j<N-1;j++){
            if(i!=j){
                const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
                for (int m=j+1;m<N-1;m++){
                    const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
                    if(i!=m) {
                        P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
                        P+=-2*atemp[i][3]/atemp[i][i]*(1-sumcj)*cj*cj*cm*cm;
                    }
                }
                P+=2*ci*pow(cj,2)*pow(1-sumcj,2);
                for (int m=0;m<N-1;m++){
                    const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
                    if(i!=m&&j!=m) {
                        P+=-2*atemp[i][m]/atemp[i][i]*cm*cj*cj*pow(1-sumcj,2);
                    }
                }
            }
        }*/

       /*for (int j=0;j<N-1;j++){
            if(i!=j){
                const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
                for (int m=j+1;m<N-1;m++){
                    const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
                    if(i!=m) {
                        P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
                        
                    }
                }
                P+=2*ci*pow(cj,2)*pow(1-sumcj,2);
            }
        }
        if (i==0) P+=-calcdP(CVec,0,1,2,3)-calcdP(CVec,0,1,2,4)-calcdP(CVec,0,1,3,4)-calcdP(CVec,0,2,3,4);
        if (i==1) P+=-calcdP(CVec,1,0,2,3)-calcdP(CVec,1,0,2,4)-calcdP(CVec,1,0,3,4)-calcdP(CVec,1,2,3,4);
        if (i==2) P+=-calcdP(CVec,2,0,1,3)-calcdP(CVec,2,0,1,4)-calcdP(CVec,2,0,3,4)-calcdP(CVec,2,1,3,4);
        if (i==3) P+=-calcdP(CVec,3,0,1,2)-calcdP(CVec,3,0,1,4)-calcdP(CVec,3,0,2,4)-calcdP(CVec,3,1,2,4);*/

        const double cj=1.0-sumcj;

        product *= cj;

        chempot+=2*mv_Beta[i][N-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) + mv_Gamma[i][N-1]*onlylaplaciansum;
        chempot_bulk += 2*mv_Beta[i][N-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj);
        chempot_tension += mv_Gamma[i][N-1]*onlylaplaciansum;

        //getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot +mv_Lambda_0123[i]*product0123+mv_Lambda_0124[i]*product0124+mv_Lambda_0134[i]*product0134+mv_Lambda_0234[i]*product0234+mv_Lambda_1234[i]*product1234;//+0.5*P;

        if (i==0) getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot+mv_Lambda_0123[i]*CVec[1]*CVec[2]*CVec[3];//+mv_Lambda_0124[i]*CVec[1]*CVec[2]*CVec[4]+mv_Lambda_0134[i]*CVec[1]*CVec[3]*CVec[4]+mv_Lambda_0234[i]*CVec[2]*CVec[3]*CVec[4]+0.25*P;
        if (i==1) getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot+mv_Lambda_0123[i]*CVec[0]*CVec[2]*CVec[3];//+mv_Lambda_0124[i]*CVec[0]*CVec[2]*CVec[4]+mv_Lambda_1234[i]*CVec[2]*CVec[3]*CVec[4]+mv_Lambda_0134[i]*CVec[0]*CVec[3]*CVec[4]+0.25*P;
        if (i==2) getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot+mv_Lambda_0123[i]*CVec[0]*CVec[1]*CVec[3];//+mv_Lambda_0124[i]*CVec[0]*CVec[1]*CVec[4]+mv_Lambda_0234[i]*CVec[0]*CVec[3]*CVec[4]+mv_Lambda_1234[i]*CVec[1]*CVec[3]*CVec[4]+0.25*P;
        //if (i==3) getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot+mv_Lambda_0123[i]*CVec[0]*CVec[1]*CVec[2]+mv_Lambda_0134[i]*CVec[0]*CVec[1]*CVec[4]+mv_Lambda_0234[i]*CVec[0]*CVec[2]*CVec[4]+mv_Lambda_1234[i]*CVec[1]*CVec[2]*CVec[4]+0.25*P;

        double lambdatemp = 0;


        double lambdadH2 = 0;
        for (int j=0;j<N;j++){

            if (i==0) lambdatemp = mv_Lambda_0123[j]*calcdH2(CVec[j],product0123)*CVec[1]*CVec[2]*CVec[3];//+mv_Lambda_0124[j]*calcdH2(CVec[j],product0124)*CVec[1]*CVec[2]*CVec[4]+mv_Lambda_0134[j]*calcdH2(CVec[j],product0134)*CVec[1]*CVec[3]*CVec[4]+mv_Lambda_0234[j]*calcdH2(CVec[j],product0234)*CVec[2]*CVec[3]*CVec[4];
            if (i==1) lambdatemp = mv_Lambda_0123[j]*calcdH2(CVec[j],product0123)*CVec[0]*CVec[2]*CVec[3];//+mv_Lambda_0124[j]*calcdH2(CVec[j],product0124)*CVec[0]*CVec[2]*CVec[4]+mv_Lambda_0134[j]*calcdH2(CVec[j],product0134)*CVec[0]*CVec[3]*CVec[4]+mv_Lambda_1234[j]*calcdH2(CVec[j],product1234)*CVec[2]*CVec[3]*CVec[4];
            if (i==2) lambdatemp = mv_Lambda_0123[j]*calcdH2(CVec[j],product0123)*CVec[0]*CVec[1]*CVec[3];//+mv_Lambda_0124[j]*calcdH2(CVec[j],product0124)*CVec[0]*CVec[1]*CVec[4]+mv_Lambda_0234[j]*calcdH2(CVec[j],product0234)*CVec[0]*CVec[3]*CVec[4]+mv_Lambda_1234[j]*calcdH2(CVec[j],product1234)*CVec[1]*CVec[3]*CVec[4];
            if (i==3) lambdatemp = mv_Lambda_0123[j]*calcdH2(CVec[j],product0123)*CVec[0]*CVec[1]*CVec[2];//+mv_Lambda_0134[j]*calcdH2(CVec[j],product0134)*CVec[0]*CVec[1]*CVec[4]+mv_Lambda_0234[j]*calcdH2(CVec[j],product0234)*CVec[0]*CVec[2]*CVec[4]+mv_Lambda_1234[j]*calcdH2(CVec[j],product1234)*CVec[1]*CVec[2]*CVec[4];
            lambdadH2+=lambdatemp;
            
        }
        
        if (i==0) lambdatemp = mv_Lambda_0123[i]*calcdH1(ci,product0123);//+mv_Lambda_0124[i]*calcdH1(ci,product0124)+mv_Lambda_0134[i]*calcdH1(ci,product0134)+mv_Lambda_0234[i]*calcdH1(ci,product0234);
        if (i==1) lambdatemp = mv_Lambda_0123[i]*calcdH1(ci,product0123);//+mv_Lambda_0124[i]*calcdH1(ci,product0124)+mv_Lambda_0134[i]*calcdH1(ci,product0134)+mv_Lambda_1234[i]*calcdH1(ci,product1234);
        if (i==2) lambdatemp = mv_Lambda_0123[i]*calcdH1(ci,product0123);//+mv_Lambda_0124[i]*calcdH1(ci,product0124)+mv_Lambda_0234[i]*calcdH1(ci,product0234)+mv_Lambda_1234[i]*calcdH1(ci,product1234);
        if (i==3) lambdatemp = mv_Lambda_0123[i]*calcdH1(ci,product0123);//+mv_Lambda_0134[i]*calcdH1(ci,product0134)+mv_Lambda_0234[i]*calcdH1(ci,product0234)+mv_Lambda_1234[i]*calcdH1(ci,product1234);
        //lambdadH2=mv_Lambda[i];
        //getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + 2 * mOmega * (ci < 0) * ci-
        //                                   2 * mOmega * ((1-ci) < 0) * (1-ci) + lambdatemp+(lambdadH2);//+m_P*P;
        

    } // i=0:N-2
    

    int i = N-1;
    chempot=0.0;
    chempot_bulk=0.0;
    chempot_tension=0.0;
    const double& ci=1.0-sumci;
    

    sumcj=0.0;
    gammalaplaciansum=0.0;
    onlylaplaciansum=0.0;
    double product = 1;
    for (int j=0;j<N-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        sumcj+=cj;
        
        onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];                         
        onlylaplaciansum += onlylaplacian;

        gammalaplacian = mv_Gamma[i][j]*onlylaplacian;
        gammalaplaciansum += gammalaplacian;
        
        if (i!=j){
            chempot+=2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
            chempot_bulk += 2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj);
            chempot_tension += - gammalaplacian;
        }
        product *= cj;
    }
    /*
    double P = 0;
    for (int j=0;j<N-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        for (int m=j+1;m<N-1;m++){
            const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
            P+=ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
            for (int n=0;n<N-1;n++){
                const double& cn=getInstance<OrderParameter, N - 1, Lattice>(n)[k];
                if(n!=i&&n!=j&&n!=m) P+=-2*atemp[3][n]/atemp[3][3]*cn*cj*cj*cm*cm;
            }
        }
    }*/

    /*double P = 0;
    for (int j=0;j<N-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        for (int m=j+1;m<N-1;m++){
            const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
            P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
        }
    }
    P+=-calcdP(CVec,4,0,1,2)-calcdP(CVec,4,0,1,3)-calcdP(CVec,4,0,2,3)-calcdP(CVec,4,1,2,3);*/
    
    //getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot+mv_Lambda_1234[i]*CVec[1]*CVec[2]*CVec[3]+mv_Lambda_0124[i]*CVec[0]*CVec[1]*CVec[2]+mv_Lambda_0134[i]*CVec[0]*CVec[1]*CVec[3]+mv_Lambda_0234[i]*CVec[0]*CVec[2]*CVec[3]+0.25*P;
    getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot+mv_Lambda_0123[i]*CVec[0]*CVec[1]*CVec[2];
    double lambdatemp = 0;

    double lambdadH2 = 0;
    for (int j=0;j<N;j++){

        //lambdatemp = mv_Lambda_1234[j]*calcdH2(CVec[j],product1234)*CVec[1]*CVec[2]*CVec[3]+mv_Lambda_0124[j]*calcdH2(CVec[j],product0124)*CVec[0]*CVec[1]*CVec[2]+mv_Lambda_0134[j]*calcdH2(CVec[j],product0134)*CVec[0]*CVec[1]*CVec[3]+mv_Lambda_0234[j]*calcdH2(CVec[j],product0234)*CVec[0]*CVec[2]*CVec[3];
        lambdatemp = mv_Lambda_0123[j]*calcdH2(CVec[j],product0123)*CVec[1]*CVec[2]*CVec[3];
        lambdadH2+=lambdatemp;
        
    }
    
    //lambdatemp = mv_Lambda_1234[i]*calcdH1(ci,product1234)+mv_Lambda_0124[i]*calcdH1(ci,product0124)+mv_Lambda_0134[i]*calcdH1(ci,product0134)+mv_Lambda_0234[i]*calcdH1(ci,product0234);
    lambdatemp = mv_Lambda_0123[i]*calcdH1(ci,product0123);
    //lambdadH2=mv_Lambda[i];
    //getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + 2 * mOmega * (ci < 0) * ci-
    //                                       2 * mOmega * ((1-ci) < 0) * (1-ci) + lambdatemp+(lambdadH2);//+m_P*P;
    
    
}
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^SHOULD BE TEMPORARY^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


template<int TN=5>
class ChemicalPotentialCalculatorNCompBoyer : public AddOnBase {
   public:
    ChemicalPotentialCalculatorNCompBoyer() : mv_Beta({{0}}), mv_Gamma({{0}}) {  
    }

    std::array<std::array<std::array<std::array<double,TN>,TN>,TN>,TN> getTh() {
        std::array<std::array<std::array<std::array<double,TN>,TN>,TN>,TN> _th;
        for (int i=0;i<TN;i++){
            for (int j=0;j<TN;j++){
                for (int k=0;k<TN;k++){
                    if (k==i||k==j) continue;
                    for (int l=0;l<TN;l++){
                        if (l==i||l==j) continue;
                        int _n=0;
                        for (int n=0;n<TN;n++) if (n!=i&&n!=j&&n!=k&&n!=l) _n=n;
                        //This is broken
                        _th[i][j][k][l]=-0.666666666666667;
                        //if ((atemp[l][_n]*atemp[k][k]-atemp[k][_n]*atemp[l][k])==(atemp[k][k]*atemp[l][l]-atemp[l][k]*atemp[k][l])) _th[i][j][k][l]=0;
                        //else _th[i][j][k][l]=-0.666666666666667;
                    }
                }
            }
        }
        return _th;
    }

    std::vector<std::vector<double>> atemp;
    std::vector<std::vector<double>> th2;
    std::array<std::array<std::array<std::array<double,TN>,TN>,TN>,TN> th;
    std::array<std::array<std::array<std::array<double,TN>,TN>,TN>,TN> lambda;

    inline void setLambda(int i,int j,int k,int l,double lambdaijkl) { lambda[i][j][k][l] = lambdaijkl; lambda[i][j][l][k] = lambdaijkl; lambda[i][l][j][k] = lambdaijkl; lambda[i][l][k][j] = lambdaijkl; lambda[i][k][j][l] = lambdaijkl; lambda[i][k][l][j] = lambdaijkl;}
    inline void setAlphas(std::vector<std::vector<double>> a) { atemp=a; 
        th2.push_back({2,   2*atemp[0][1]/atemp[0][0],   2*atemp[0][2]/atemp[0][0],  2*atemp[0][3]/atemp[0][0]});
        th2.push_back({  2*atemp[1][0]/atemp[1][1], 2,    2*atemp[1][2]/atemp[1][1],   2*atemp[1][3]/atemp[1][1]});
        th2.push_back({  2*atemp[2][0]/atemp[2][2],    2*atemp[2][1]/atemp[2][2], 2,   2*atemp[2][3]/atemp[2][2]});
        th2.push_back({ 2*atemp[3][0]/atemp[3][3],   2*atemp[3][1]/atemp[3][3],   2*atemp[3][2]/atemp[3][3], 2});
        th = getTh();
         }

    template <class traits>
    inline void compute(const int k);

    inline void setA(double** A) { ma_Beta = A; }

    inline void setP(double P) { m_P = P; }
    inline void setOmega(double Omega) { mOmega = Omega; }

    inline void setKappa(double** kappa) { ma_Gamma = kappa; }

    inline double calcdH1(double c1, double c2) {
        if (fabs(c1)==0&&fabs(c2)==0) return 0;
        return -2*c1*fabs(c2)*c2/pow(fabs(c2)+c1*c1,2);
    }

    inline double calcdH2(double c1, double c2) {
        if (fabs(c1)==0&&fabs(c2)==0) return 1;
        return (2*fabs(c2)*c1*c1+c2*c2)/pow(fabs(c2)+c1*c1,2);
    }

    inline double calcH(double c1, double c2) {
        if (fabs(c1)==0&&fabs(c2)==0) return 0;
        return (fabs(c2)*c2)/(fabs(c2)+c1*c1);
    }

    inline double calcdP2(std::vector<double> c,int i,int j, int k, int l){
        return pow(c[k],2)*pow(c[l],2)*(th2[i][j]*(calcdH1(c[i], c[i]*c[j]) + c[j]*calcdH2(c[i], c[i]*c[j])) + th2[j][i]*c[j]*calcdH2(c[j], c[i]*c[j])) + pow(c[j],2)*pow(c[l],2)*(th2[i][k]*(calcdH1(c[i], c[i]*c[k]) + c[k]*calcdH2(c[i], c[i]*c[k])) + th2[k][i]*c[k]*calcdH2(c[k], c[i]*c[k])) + pow(c[j],2)*pow(c[k],2)*(th2[i][l]*(calcdH1(c[i], c[i]*c[l]) + c[l]*calcdH2(c[i], c[i]*c[l])) + th2[l][i]*c[l]*calcdH2(c[l], c[i]*c[l])) + 2*c[i]*pow(c[l],2)*(th2[j][k]*calcH(c[j], c[j]*c[k]) + th2[k][j]*calcH(c[k], c[j]*c[k])) + 2*c[i]*pow(c[k],2)*(th2[j][l]*calcH(c[j], c[j]*c[l]) + th2[l][j]*calcH(c[l], c[j]*c[l])) + 2*c[i]*pow(c[j],2)*(th2[k][l]*calcH(c[k], c[k]*c[l]) + th2[l][k]*calcH(c[l], c[k]*c[l]));
    }

    inline double calcdP(std::vector<double> c,int i,int j, int k, int l){
        return pow(c[k],2)*pow(c[l],2)*(th[k][l][i][j]*(calcdH1(c[i], c[i]*c[j]) + c[j]*calcdH2(c[i], c[i]*c[j])) + th[k][l][j][i]*c[j]*calcdH2(c[j], c[i]*c[j])) + pow(c[j],2)*pow(c[l],2)*(th[j][l][i][k]*(calcdH1(c[i], c[i]*c[k]) + c[k]*calcdH2(c[i], c[i]*c[k])) + th[j][l][k][i]*c[k]*calcdH2(c[k], c[i]*c[k])) + pow(c[j],2)*pow(c[k],2)*(th[j][k][i][l]*(calcdH1(c[i], c[i]*c[l]) + c[l]*calcdH2(c[i], c[i]*c[l])) + th[j][k][l][i]*c[l]*calcdH2(c[l], c[i]*c[l])) + 2*c[i]*pow(c[l],2)*(th[i][l][j][k]*calcH(c[j], c[j]*c[k]) + th[i][l][k][j]*calcH(c[k], c[j]*c[k])) + 2*c[i]*pow(c[k],2)*(th[i][k][j][l]*calcH(c[j], c[j]*c[l]) + th[i][k][l][j]*calcH(c[l], c[j]*c[l])) + 2*c[i]*pow(c[j],2)*(th[i][j][k][l]*calcH(c[k], c[k]*c[l]) + th[i][j][l][k]*calcH(c[l], c[k]*c[l]));
    }

    inline void setBeta(int i, int j, double beta) {
        if ((int)mv_Beta.size() - 1 < i || (int)mv_Beta.size() - 1 < j) {
            mv_Beta.resize(mv_Beta.size() + std::max<int>(i - (int)mv_Beta.size() + 1, j - (int)mv_Beta.size() + 1));
            for (int l = 0; l < (int)mv_Beta.size(); l++) {
                mv_Beta[l].resize(mv_Beta[l].size() +
                                  std::max<int>(i - (int)mv_Beta[l].size() + 1, j - (int)mv_Beta[l].size() + 1));
                mv_Beta[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Beta[i][j] = beta;
            mv_Beta[j][i] = beta;
        }
    }
    inline void setGamma(int i, int j, double gamma) {
        if ((int)mv_Gamma.size() - 1 < i || (int)mv_Gamma.size() - 1 < j) {
            mv_Gamma.resize(mv_Gamma.size() +
                            std::max<int>(i - (int)mv_Gamma.size() + 1, j - (int)mv_Gamma.size() + 1));
            for (int l = 0; l < (int)mv_Gamma.size(); l++) {
                mv_Gamma[l].resize(mv_Gamma[l].size() +
                                   std::max<int>(i - (int)mv_Gamma[l].size() + 1, j - (int)mv_Gamma[l].size() + 1));
                mv_Gamma[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Gamma[i][j] = gamma;
            mv_Gamma[j][i] = gamma;
        }
    }

    inline void setBetaAndGamma(int i, int j, double beta, double gamma) {
        setBeta(i, j, beta);
        setGamma(i, j, gamma);
    }

    inline void setBeta(std::vector<std::vector<double>>& beta) { mv_Beta = beta; }
    inline void setGamma(std::vector<std::vector<double>>& gamma) { mv_Gamma = gamma; }

    inline void setBetaAndGamma(std::vector<std::vector<double>>& beta, std::vector<std::vector<double>>& gamma) {
        mv_Beta = beta;
        mv_Gamma = gamma;
    }

    inline void setD(double d) {
        for (int l = 0; l < (int)mv_Gamma.size(); l++) {
            for (int m = 0; m < (int)mv_Gamma.size(); m++) {
                mv_Gamma[l][m] *= d / mD;
            }
        }
        for (int l = 0; l < (int)mv_Beta.size(); l++) {
            for (int m = 0; m < (int)mv_Beta.size(); m++) {
                mv_Beta[l][m] *= mD / d;
            }
        }
        mD = d;
    }

    inline void setSurfaceTension(int i, int j, double surfacetension) {
        setBeta(i, j, 3.0 * surfacetension / mD);
        setGamma(i, j, -3.0 * mD * surfacetension / 4.0);
    }

   public:
    template <int numberofcomponents>
    inline bool checkValid() {
        if ((int)mv_Beta.size() != numberofcomponents || (int)mv_Beta[0].size() != numberofcomponents ||
            (int)mv_Gamma.size() != numberofcomponents || (int)mv_Gamma[0].size() != numberofcomponents) {
            throw std::runtime_error(
                "Number of beta/gamma parameters does not match the number of "
                "components.");
            return false;
        }
        return true;
    }

    bool mIsValid;
    double** ma_Gamma;
    double** ma_Beta;
    std::vector<std::vector<double>> mv_Beta;
    std::vector<std::vector<double>> mv_Gamma;

    double mD = 5;
    double m_P = 2.0;
    double mOmega = 0.0;
    double cutoff = 1e-10;

};

template<int TN>
template <class TTraits>
inline void ChemicalPotentialCalculatorNCompBoyer<TN>::compute(const int k) {
    if (Geometry<typename TTraits::Lattice>::isBulkSolid(k)) return;

    [[maybe_unused]] static bool isvalid = checkValid<TN>();
    mIsValid=isvalid;
    using Lattice = typename TTraits::Lattice;
    constexpr int N = TN;
    double chempot=0.0;
    double chempot_bulk=0.0;
    double chempot_tension=0.0;
    double sumci=0.0;
    double sumcj=0.0;

    double onlylaplacian=0.0;
    double gammalaplacian=0.0;
    double onlylaplaciansum=0.0;
    double gammalaplaciansum=0.0;   

    std::vector<double> CVec = {};
    
    double sum = 0;
    for (int i=0;i<N-1;i++){
        CVec.push_back(getInstance<OrderParameter, N - 1, Lattice>(i)[k]);
        sum+=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
    }
    CVec.push_back(1-sum);

    for (int i=0;i<N-1;i++){
        chempot=0.0;
        chempot_bulk=0.0;
        chempot_tension=0.0;
        const double& ci=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
        sumci += ci;

        sumcj=0.0;
        gammalaplaciansum=0.0;
        onlylaplaciansum=0.0;
        double P=0;
        for (int j=0;j<N-1;j++){
            
            const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
            sumcj+=cj;
            
            onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];
            
            onlylaplaciansum += onlylaplacian;                
            gammalaplacian=mv_Gamma[i][j]*onlylaplacian;
            gammalaplaciansum += gammalaplacian;

            chempot+=2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;

        }

       for (int j=0;j<N-1;j++){
            if(i!=j){
                const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
                for (int m=j+1;m<N-1;m++){
                    const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
                    if(i!=m) {
                        P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
                        
                    }
                }
                P+=2*ci*pow(cj,2)*pow(1-sumcj,2);
            }
        }

        for (int j=0;j<N;j++){
            if(i!=j){
                for (int m=j+1;m<N;m++){
                    if(i!=m){
                        for (int n=m+1;n<N;n++){
                            if(i!=n){
                                P+=-calcdP(CVec,i,j,m,n);
                            }
                        }
                    }
                }
            }
        }
        
        const double cj=1.0-sumcj;

        chempot+=2*mv_Beta[i][N-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) + mv_Gamma[i][N-1]*onlylaplaciansum;

        double lambdadH1 = 0;
        double lambdadH2 = 0;

        for (int j=0;j<N;j++){
            for (int m=j+1;m<N;m++){
                for (int n=m+1;n<N;n++){
                    if (j!=i&&m!=i&&n!=i){
                        lambdadH1 += lambda[i][j][m][n]*calcdH1(ci,CVec[i]*CVec[j]*CVec[m]*CVec[n]);
                    }
                }
            }
        }
        for (int s=0;s<N;s++){
            for (int j=0;j<N;j++){
                for (int m=j+1;m<N;m++){
                    for (int n=m+1;n<N;n++){
                        if (j!=s&&m!=s&&n!=s){
                            if (s==i) lambdadH2 += lambda[s][j][m][n]*calcdH2(CVec[s],CVec[i]*CVec[j]*CVec[m]*CVec[n])*CVec[j]*CVec[m]*CVec[n];
                            else if (j==i) lambdadH2 += lambda[s][j][m][n]*calcdH2(CVec[s],CVec[i]*CVec[s]*CVec[m]*CVec[n])*CVec[s]*CVec[m]*CVec[n];
                            else if (m==i) lambdadH2 += lambda[s][j][m][n]*calcdH2(CVec[s],CVec[i]*CVec[j]*CVec[s]*CVec[n])*CVec[j]*CVec[s]*CVec[n];
                            else if (n==i) lambdadH2 += lambda[s][j][m][n]*calcdH2(CVec[s],CVec[i]*CVec[j]*CVec[m]*CVec[s])*CVec[j]*CVec[m]*CVec[s];
                        }
                    }
                }
            }
        }
        getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + /*2 * mOmega * (ci < 0) * ci-
                                           2 * mOmega * ((1-ci) < 0) * (1-ci) +*/ lambdadH1 + (lambdadH2) + m_P * P; 
        

    }
    

    int i = N-1;
    chempot=0.0;
    chempot_bulk=0.0;
    chempot_tension=0.0;
    const double& ci=1.0-sumci;

    sumcj=0.0;
    gammalaplaciansum=0.0;
    onlylaplaciansum=0.0;

    for (int j=0;j<N-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        sumcj+=cj;
        
        onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];                         
        onlylaplaciansum += onlylaplacian;

        gammalaplacian = mv_Gamma[i][j]*onlylaplacian;
        gammalaplaciansum += gammalaplacian;
        
        if (i!=j){
            chempot+=2*mv_Beta[i][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
        }

    }

    double P = 0;
    for (int j=0;j<N-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        for (int m=j+1;m<N-1;m++){
            const double& cm=getInstance<OrderParameter, N - 1, Lattice>(m)[k];
            P+=2*ci*pow(cj,2)*pow(getInstance<OrderParameter, N - 1, Lattice>(m)[k],2);
        }
    }
    for (int j=0;j<N;j++){
        if(i!=j){
            for (int m=j+1;m<N;m++){
                if(i!=m){
                    for (int n=m+1;n<N;n++){
                        if(i!=n){
                            P+=-calcdP(CVec,i,j,m,n);
                        }
                    }
                }
            }
        }
    }
    
    double lambdadH1 = 0;

    double lambdadH2 = 0;
    for (int j=0;j<N;j++){
        for (int m=j+1;m<N;m++){
            for (int n=m+1;n<N;n++){
                if (j!=i&&m!=i&&n!=i){
                    lambdadH1 += lambda[i][j][m][n]*calcdH1(ci,CVec[i]*CVec[j]*CVec[m]*CVec[n]);
                }
            }
        }
    }
    for (int s=0;s<N;s++){
        for (int j=0;j<N;j++){
            for (int m=j+1;m<N;m++){
                for (int n=m+1;n<N;n++){
                    if (j!=s&&m!=s&&n!=s){
                        if (s==i) lambdadH2 += lambda[s][j][m][n]*calcdH2(CVec[s],CVec[i]*CVec[j]*CVec[m]*CVec[n])*CVec[j]*CVec[m]*CVec[n];
                        else if (j==i) lambdadH2 += lambda[s][j][m][n]*calcdH2(CVec[s],CVec[i]*CVec[s]*CVec[m]*CVec[n])*CVec[s]*CVec[m]*CVec[n];
                        else if (m==i) lambdadH2 += lambda[s][j][m][n]*calcdH2(CVec[s],CVec[i]*CVec[j]*CVec[s]*CVec[n])*CVec[j]*CVec[s]*CVec[n];
                        else if (n==i) lambdadH2 += lambda[s][j][m][n]*calcdH2(CVec[s],CVec[i]*CVec[j]*CVec[m]*CVec[s])*CVec[j]*CVec[m]*CVec[s];
                    }
                }
            }
        }
    }

    getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + /*2 * mOmega * (ci < 0) * ci-
                                           2 * mOmega * ((1-ci) < 0) * (1-ci) +*/ lambdadH1+(lambdadH2)+m_P*P; 
    
    
}
// 


template<int TN>
class ChemicalPotentialCalculatorNCompBoyerConstS : public AddOnBase {
   public:
    ChemicalPotentialCalculatorNCompBoyerConstS() : mv_Beta({{0}}), mv_Gamma({{0}}) {}

    ChemicalPotentialCalculatorNCompBoyerConstS<TN>(const ChemicalPotentialCalculatorNCompBoyerConstS<TN>& other) : mv_Beta(other.mv_Beta), mv_Gamma(other.mv_Gamma), mD(other.mD), mOmega(other.mOmega) {}

    ChemicalPotentialCalculatorNCompBoyerConstS<TN>& operator=(const ChemicalPotentialCalculatorNCompBoyerConstS<TN>& other) {
        mv_Beta=other.mv_Beta;
        mv_Gamma=other.mv_Gamma;
        mD = other.mD;
        mOmega = other.mOmega;
        return *this;
    }

    template <class traits>
    inline void compute(const int k);

    inline void setOmega(double Omega) { mOmega = Omega; }

    inline void setD(double d) {
        for (int l = 0; l < (int)mv_Gamma.size(); l++) {
            for (int m = 0; m < (int)mv_Gamma.size(); m++) {
                mv_Gamma[l][m] *= d / mD;
            }
        }
        for (int l = 0; l < (int)mv_Beta.size(); l++) {
            for (int m = 0; m < (int)mv_Beta.size(); m++) {
                mv_Beta[l][m] *= mD / d;
            }
        }
        mD = d;
    }

   private:
    inline void setBeta(int i, int j, double beta) {
        if ((int)mv_Beta.size() - 1 < i || (int)mv_Beta.size() - 1 < j) {
            mv_Beta.resize(mv_Beta.size() + std::max<int>(i - (int)mv_Beta.size() + 1, j - (int)mv_Beta.size() + 1));
            for (int l = 0; l < (int)mv_Beta.size(); l++) {
                mv_Beta[l].resize(mv_Beta[l].size() +
                                std::max<int>(i - (int)mv_Beta[l].size() + 1, j - (int)mv_Beta[l].size() + 1));
                mv_Beta[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Beta[i][j] = beta;
            mv_Beta[j][i] = beta;
        }
    }
    inline void setGamma(int i, int j, double gamma) {
        if ((int)mv_Gamma.size() - 1 < i || (int)mv_Gamma.size() - 1 < j) {
            mv_Gamma.resize(mv_Gamma.size() +
                            std::max<int>(i - (int)mv_Gamma.size() + 1, j - (int)mv_Gamma.size() + 1));
            for (int l = 0; l < (int)mv_Gamma.size(); l++) {
                mv_Gamma[l].resize(mv_Gamma[l].size() +
                                std::max<int>(i - (int)mv_Gamma[l].size() + 1, j - (int)mv_Gamma[l].size() + 1));
                mv_Gamma[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Gamma[i][j] = gamma;
            mv_Gamma[j][i] = gamma;
        }
    }

   public:

    inline void setSurfaceTension(double surfacetension) {
        for (int i=0;i<TN;i++) {
            for (int j=i;j<TN;j++) {
                setBeta(i, j, 3.0 * surfacetension / mD);
                setGamma(i, j, -3.0 * mD * surfacetension / 4.0);
            }
        }
    }
    template <int numberofcomponents>
    inline bool checkValid() {
        if ((int)mv_Beta.size() != numberofcomponents || (int)mv_Beta[0].size() != numberofcomponents ||
            (int)mv_Gamma.size() != numberofcomponents || (int)mv_Gamma[0].size() != numberofcomponents) {
            std::cout<<numberofcomponents<<" "<<mv_Beta.size()<<" "<<mv_Gamma.size()<<" "<<(int)mv_Beta[0].size()<<" "<<(int)mv_Gamma[0].size()<<std::endl;
            throw std::runtime_error(
                "Number of beta/gamma parameters does not match the number of "
                "components.");
            return false;
        }
        return true;
    }

    bool mIsValid;
    std::vector<std::vector<double>> mv_Beta;
    std::vector<std::vector<double>> mv_Gamma;
    double mD = 5;
    double mOmega = 0.0;

};

template<int TN>
template <class TTraits>
inline void ChemicalPotentialCalculatorNCompBoyerConstS<TN>::compute(const int k) {
    if (Geometry<typename TTraits::Lattice>::isBulkSolid(k)) return;

    [[maybe_unused]] static bool isvalid = checkValid<TN>();
    mIsValid=isvalid;
    using Lattice = typename TTraits::Lattice;
    constexpr int N = TN;
    double chempot=0.0;
    double chempot_bulk=0.0;
    double chempot_tension=0.0;
    double sumci=0.0;
    double sumcj=0.0;

    double onlylaplacian=0.0;
    double gammalaplacian=0.0;
    double onlylaplaciansum=0.0;
    double gammalaplaciansum=0.0;   

    std::vector<double> CVec = {};
    
    double sum = 0;
    for (int i=0;i<N-1;i++){
        CVec.push_back(getInstance<OrderParameter, N - 1, Lattice>(i)[k]);
        sum+=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
    }
    CVec.push_back(1-sum);

    for (int i=0;i<N-1;i++){
        chempot=0.0;
        chempot_bulk=0.0;
        chempot_tension=0.0;
        const double& ci=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
        sumci += ci;

        sumcj=0.0;
        gammalaplaciansum=0.0;
        onlylaplaciansum=0.0;

        for (int j=0;j<N-1;j++){
            
            const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
            sumcj+=cj;
            
            onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];
            
            onlylaplaciansum += onlylaplacian;                
            gammalaplacian=mv_Gamma[i][j]*onlylaplacian;
            gammalaplaciansum += gammalaplacian;

            chempot+= - gammalaplacian;

        }// j

        for (int j=0;j<N;j++){
            if(i!=j){
                for (int m=j+1;m<N;m++){
                    if(i!=m){
                        for (int n=m+1;n<N;n++){
                            if(i!=n){
                                chempot+=8*mv_Beta[i][j]*(CVec[j]*CVec[m]*CVec[n]);
                            }
                        }
                    }
                }
            }
        }
        
        const double cj=1.0-sumcj;

        chempot+=2*mv_Beta[i][N-1]*(2*ci*(2*ci*ci-3*ci+1)) + mv_Gamma[i][N-1]*onlylaplaciansum;
        
        getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + 2 * mOmega * (ci < 0) * ci-
                                           2 * mOmega * ((1-ci) < 0) * (1-ci); 
        

    } // i=0:N-2
    
    int i = N-1;
    
    chempot_bulk=0.0;
    chempot_tension=0.0;
    const double& ci=1.0-sumci;
    chempot=2*mv_Beta[i][0]*(2*ci*(2*ci*ci-3*ci+1)) ;

    sumcj=0.0;
    gammalaplaciansum=0.0;
    onlylaplaciansum=0.0;
    double product = 1;
    for (int j=0;j<N-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        sumcj+=cj;
        
        onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];                         
        onlylaplaciansum += onlylaplacian;

        gammalaplacian = mv_Gamma[i][j]*onlylaplacian;
        gammalaplaciansum += gammalaplacian;
        
        if (i!=j){
            chempot+=- gammalaplacian;
        }
        product *= cj;
    }
    for (int j=0;j<N;j++){
        if(i!=j){
            for (int m=j+1;m<N;m++){
                if(i!=m){
                    for (int n=m+1;n<N;n++){
                        if(i!=n){
                            chempot+=8*mv_Beta[i][j]*(CVec[j]*CVec[m]*CVec[n]);
                        }
                    }
                }
            }
        }
    }
    
    getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + 2 * mOmega * (ci < 0) * ci-
                                           2 * mOmega * ((1-ci) < 0) * (1-ci); 
    
}


template<int TN>
class ChemicalPotentialCalculatorNCompBoyerConstS2 : public AddOnBase {
   public:
    ChemicalPotentialCalculatorNCompBoyerConstS2() : mv_Beta({{0}}), mv_Gamma({{0}}) {}

    ChemicalPotentialCalculatorNCompBoyerConstS2<TN>(const ChemicalPotentialCalculatorNCompBoyerConstS2<TN>& other) : mv_Beta(other.mv_Beta), mv_Gamma(other.mv_Gamma), mD(other.mD), mOmega(other.mOmega) {}

    ChemicalPotentialCalculatorNCompBoyerConstS2<TN>& operator=(const ChemicalPotentialCalculatorNCompBoyerConstS2<TN>& other) {
        mv_Beta=other.mv_Beta;
        mv_Gamma=other.mv_Gamma;
        mD = other.mD;
        mOmega = other.mOmega;
        return *this;
    }

    template <class traits>
    inline void compute(const int k);

    inline void setOmega(double Omega) { mOmega = Omega; }

    inline void setD(double d) {
        for (int l = 0; l < (int)mv_Gamma.size(); l++) {
            for (int m = 0; m < (int)mv_Gamma.size(); m++) {
                mv_Gamma[l][m] *= d / mD;
            }
        }
        for (int l = 0; l < (int)mv_Beta.size(); l++) {
            for (int m = 0; m < (int)mv_Beta.size(); m++) {
                mv_Beta[l][m] *= mD / d;
            }
        }
        mD = d;
    }

   private:
    inline void setBeta(int i, int j, double beta) {
        if ((int)mv_Beta.size() - 1 < i || (int)mv_Beta.size() - 1 < j) {
            mv_Beta.resize(mv_Beta.size() + std::max<int>(i - (int)mv_Beta.size() + 1, j - (int)mv_Beta.size() + 1));
            for (int l = 0; l < (int)mv_Beta.size(); l++) {
                mv_Beta[l].resize(mv_Beta[l].size() +
                                std::max<int>(i - (int)mv_Beta[l].size() + 1, j - (int)mv_Beta[l].size() + 1));
                mv_Beta[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Beta[i][j] = beta;
            mv_Beta[j][i] = beta;
        }
    }
    inline void setGamma(int i, int j, double gamma) {
        if ((int)mv_Gamma.size() - 1 < i || (int)mv_Gamma.size() - 1 < j) {
            mv_Gamma.resize(mv_Gamma.size() +
                            std::max<int>(i - (int)mv_Gamma.size() + 1, j - (int)mv_Gamma.size() + 1));
            for (int l = 0; l < (int)mv_Gamma.size(); l++) {
                mv_Gamma[l].resize(mv_Gamma[l].size() +
                                std::max<int>(i - (int)mv_Gamma[l].size() + 1, j - (int)mv_Gamma[l].size() + 1));
                mv_Gamma[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Gamma[i][j] = gamma;
            mv_Gamma[j][i] = gamma;
        }
    }

   public:

    inline void setSurfaceTension(double surfacetension) {
        for (int i=0;i<TN;i++) {
            for (int j=i;j<TN;j++) {
                setBeta(i, j, 3.0 * surfacetension / mD);
                setGamma(i, j, -3.0 * mD * surfacetension / 4.0);
            }
        }
    }
    template <int numberofcomponents>
    inline bool checkValid() {
        if ((int)mv_Beta.size() != numberofcomponents || (int)mv_Beta[0].size() != numberofcomponents ||
            (int)mv_Gamma.size() != numberofcomponents || (int)mv_Gamma[0].size() != numberofcomponents) {
            std::cout<<numberofcomponents<<" "<<mv_Beta.size()<<" "<<mv_Gamma.size()<<" "<<(int)mv_Beta[0].size()<<" "<<(int)mv_Gamma[0].size()<<std::endl;
            throw std::runtime_error(
                "Number of beta/gamma parameters does not match the number of "
                "components.");
            return false;
        }
        return true;
    }

    bool mIsValid;
    std::vector<std::vector<double>> mv_Beta;
    std::vector<std::vector<double>> mv_Gamma;
    double mD = 5;
    double mOmega = 0.0;

};

template<int TN>
template <class TTraits>
inline void ChemicalPotentialCalculatorNCompBoyerConstS2<TN>::compute(const int k) {
    if (Geometry<typename TTraits::Lattice>::isBulkSolid(k)) return;

    [[maybe_unused]] static bool isvalid = checkValid<TN>();
    mIsValid=isvalid;
    using Lattice = typename TTraits::Lattice;
    constexpr int N = TN;
    double chempot=0.0;
    double chempot_bulk=0.0;
    double chempot_tension=0.0;
    double sumci=0.0;
    double sumcj=0.0;

    double onlylaplacian=0.0;
    double gammalaplacian=0.0;
    double onlylaplaciansum=0.0;
    double gammalaplaciansum=0.0;   

    std::vector<double> CVec = {};
    
    double sum = 0;
    for (int i=0;i<N-1;i++){
        CVec.push_back(getInstance<OrderParameter, N - 1, Lattice>(i)[k]);
        sum+=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
    }
    CVec.push_back(1-sum);

    for (int i=0;i<N-1;i++){
        chempot=0.0;
        chempot_bulk=0.0;
        chempot_tension=0.0;
        const double& ci=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
        sumci += ci;

        sumcj=0.0;
        gammalaplaciansum=0.0;
        onlylaplaciansum=0.0;

        for (int j=0;j<N-1;j++){
            
            const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
            sumcj+=cj;
            
            onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];
            
            onlylaplaciansum += onlylaplacian;                
            gammalaplacian=mv_Gamma[i][j]*onlylaplacian;
            gammalaplaciansum += gammalaplacian;

            if (i!=j) chempot+= 2*mv_Beta[i][N-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
            

        }// j

        for (int j=0;j<N;j++){
            if(i!=j){
                for (int m=j+1;m<N;m++){
                    if(i!=m){
                        for (int n=m+1;n<N;n++){
                            if(i!=n){
                                chempot+=8*7*mv_Beta[i][j]*(CVec[j]*CVec[m]*CVec[n]);
                            }
                        }
                    }
                }
            }
        }
        
        const double cj=1.0-sumcj;

        chempot+=2*mv_Beta[i][N-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj)  + mv_Gamma[i][N-1]*onlylaplaciansum;
        
        getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + 2 * mOmega * (ci < 0) * ci-
                                           2 * mOmega * ((1-ci) < 0) * (1-ci); 
        

    } // i=0:N-2
    
    int i = N-1;
    
    chempot_bulk=0.0;
    chempot_tension=0.0;
    const double& ci=1.0-sumci;
    chempot=0;

    sumcj=0.0;
    gammalaplaciansum=0.0;
    onlylaplaciansum=0.0;
    double product = 1;
    for (int j=0;j<N-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        sumcj+=cj;
        
        onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];                         
        onlylaplaciansum += onlylaplacian;

        gammalaplacian = mv_Gamma[i][j]*onlylaplacian;
        gammalaplaciansum += gammalaplacian;
        
        
        chempot+=2*mv_Beta[0][N-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
        
        product *= cj;
    }
    for (int j=0;j<N;j++){
        if(i!=j){
            for (int m=j+1;m<N;m++){
                if(i!=m){
                    for (int n=m+1;n<N;n++){
                        if(i!=n){
                            chempot+=8*7*mv_Beta[i][j]*(CVec[j]*CVec[m]*CVec[n]);
                        }
                    }
                }
            }
        }
    }
    
    getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + 2 * mOmega * (ci < 0) * ci-
                                           2 * mOmega * ((1-ci) < 0) * (1-ci); 


    /*
    if (Geometry<typename TTraits::Lattice>::isBulkSolid(k)) return;

    [[maybe_unused]] static bool isvalid = checkValid<TTraits::NumberOfComponents>();
    mIsValid=isvalid;
    using Lattice = typename TTraits::Lattice;
    constexpr int N = TTraits::NumberOfComponents;
    double chempot=0.0;
    double sumci=0.0;
    double sumcj=0.0;

    double onlylaplacian=0.0;
    double gammalaplacian=0.0;
    double onlylaplaciansum=0.0;
    double gammalaplaciansum=0.0;       

    for (int i=0;i<TTraits::NumberOfComponents-1;i++){
        chempot=0.0;
        const double& ci=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
        sumci += ci;

        sumcj=0.0;
        gammalaplaciansum=0.0;
        onlylaplaciansum=0.0;

        for (int j=0;j<TTraits::NumberOfComponents-1;j++){
            
            const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
            sumcj+=cj;
            
            onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];
            
            onlylaplaciansum += onlylaplacian;                
            gammalaplacian=mv_Gamma[i][TTraits::NumberOfComponents-1]*onlylaplacian;
            gammalaplaciansum += gammalaplacian;

            if (i!=j) chempot+=2*mv_Beta[i][TTraits::NumberOfComponents-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
        }// j
        
        const double cj=1.0-sumcj;
        
        chempot+=2*mv_Beta[i][TTraits::NumberOfComponents-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) + mv_Gamma[i][TTraits::NumberOfComponents-1]*onlylaplaciansum;

        getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot;
        
    } // i=0:N-2
    

    int i = TTraits::NumberOfComponents-1;
    chempot=0.0;
    const double& ci=1.0-sumci;
    

    sumcj=0.0;
    gammalaplaciansum=0.0;
    onlylaplaciansum=0.0;
    
    for (int j=0;j<TTraits::NumberOfComponents-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        sumcj+=cj;
        
        onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];                         
        onlylaplaciansum += onlylaplacian;

        gammalaplacian = mv_Gamma[0][j]*onlylaplacian;
        gammalaplaciansum += gammalaplacian;
        
        if (i!=j){
            chempot+=2*mv_Beta[0][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
        }
    }

    for (int j=0;j<N;j++){
        if(i!=j){
            for (int m=j+1;m<N;m++){
                if(i!=m){
                    for (int n=m+1;n<N;n++){
                        if(i!=n){
                            chempot+=12*mv_Beta[i][j]*(CVec[j]*CVec[m]*CVec[n]);
                        }
                    }
                }
            }
        }
    }

    getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot;*/
    
}



template<int TN>
class ChemicalPotentialCalculatorNCompBoyerConstSWRONG : public AddOnBase {
   public:
    ChemicalPotentialCalculatorNCompBoyerConstSWRONG() : mv_Beta({{0}}), mv_Gamma({{0}}) {}

    ChemicalPotentialCalculatorNCompBoyerConstSWRONG<TN>(const ChemicalPotentialCalculatorNCompBoyerConstSWRONG<TN>& other) : mv_Beta(other.mv_Beta), mv_Gamma(other.mv_Gamma), mD(other.mD), mOmega(other.mOmega) {}

    ChemicalPotentialCalculatorNCompBoyerConstSWRONG<TN>& operator=(const ChemicalPotentialCalculatorNCompBoyerConstSWRONG<TN>& other) {
        mv_Beta=other.mv_Beta;
        mv_Gamma=other.mv_Gamma;
        mD = other.mD;
        mOmega = other.mOmega;
        return *this;
    }

    template <class traits>
    inline void compute(const int k);

    inline void setOmega(double Omega) { mOmega = Omega; }

    inline void setD(double d) {
        for (int l = 0; l < (int)mv_Gamma.size(); l++) {
            for (int m = 0; m < (int)mv_Gamma.size(); m++) {
                mv_Gamma[l][m] *= d / mD;
            }
        }
        for (int l = 0; l < (int)mv_Beta.size(); l++) {
            for (int m = 0; m < (int)mv_Beta.size(); m++) {
                mv_Beta[l][m] *= mD / d;
            }
        }
        mD = d;
    }

   private:
    inline void setBeta(int i, int j, double beta) {
        if ((int)mv_Beta.size() - 1 < i || (int)mv_Beta.size() - 1 < j) {
            mv_Beta.resize(mv_Beta.size() + std::max<int>(i - (int)mv_Beta.size() + 1, j - (int)mv_Beta.size() + 1));
            for (int l = 0; l < (int)mv_Beta.size(); l++) {
                mv_Beta[l].resize(mv_Beta[l].size() +
                                std::max<int>(i - (int)mv_Beta[l].size() + 1, j - (int)mv_Beta[l].size() + 1));
                mv_Beta[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Beta[i][j] = beta;
            mv_Beta[j][i] = beta;
        }
    }
    inline void setGamma(int i, int j, double gamma) {
        if ((int)mv_Gamma.size() - 1 < i || (int)mv_Gamma.size() - 1 < j) {
            mv_Gamma.resize(mv_Gamma.size() +
                            std::max<int>(i - (int)mv_Gamma.size() + 1, j - (int)mv_Gamma.size() + 1));
            for (int l = 0; l < (int)mv_Gamma.size(); l++) {
                mv_Gamma[l].resize(mv_Gamma[l].size() +
                                std::max<int>(i - (int)mv_Gamma[l].size() + 1, j - (int)mv_Gamma[l].size() + 1));
                mv_Gamma[l][l] = 0;
            }
        }
        if (i != j) {
            mv_Gamma[i][j] = gamma;
            mv_Gamma[j][i] = gamma;
        }
    }

   public:

    inline void setSurfaceTension(double surfacetension) {
        for (int i=0;i<TN;i++) {
            for (int j=i;j<TN;j++) {
                setBeta(i, j, 3.0 * surfacetension / mD);
                setGamma(i, j, -3.0 * mD * surfacetension / 4.0);
            }
        }
    }
    template <int numberofcomponents>
    inline bool checkValid() {
        if ((int)mv_Beta.size() != numberofcomponents || (int)mv_Beta[0].size() != numberofcomponents ||
            (int)mv_Gamma.size() != numberofcomponents || (int)mv_Gamma[0].size() != numberofcomponents) {
            std::cout<<numberofcomponents<<" "<<mv_Beta.size()<<" "<<mv_Gamma.size()<<" "<<(int)mv_Beta[0].size()<<" "<<(int)mv_Gamma[0].size()<<std::endl;
            throw std::runtime_error(
                "Number of beta/gamma parameters does not match the number of "
                "components.");
            return false;
        }
        return true;
    }

    bool mIsValid;
    std::vector<std::vector<double>> mv_Beta;
    std::vector<std::vector<double>> mv_Gamma;
    double mD = 5;
    double mOmega = 0.0;

};

template<int TN>
template <class TTraits>
inline void ChemicalPotentialCalculatorNCompBoyerConstSWRONG<TN>::compute(const int k) {
    if (Geometry<typename TTraits::Lattice>::isBulkSolid(k)) return;

    [[maybe_unused]] static bool isvalid = checkValid<TN>();
    mIsValid=isvalid;
    using Lattice = typename TTraits::Lattice;
    constexpr int N = TN;
    double chempot=0.0;
    double chempot_bulk=0.0;
    double chempot_tension=0.0;
    double sumci=0.0;
    double sumcj=0.0;

    double onlylaplacian=0.0;
    double gammalaplacian=0.0;
    double onlylaplaciansum=0.0;
    double gammalaplaciansum=0.0;   

    std::vector<double> CVec = {};
    
    double sum = 0;
    for (int i=0;i<N-1;i++){
        CVec.push_back(getInstance<OrderParameter, N - 1, Lattice>(i)[k]);
        sum+=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
    }
    CVec.push_back(1-sum);

    for (int i=0;i<N-1;i++){
        chempot=0.0;
        chempot_bulk=0.0;
        chempot_tension=0.0;
        const double& ci=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
        sumci += ci;

        sumcj=0.0;
        gammalaplaciansum=0.0;
        onlylaplaciansum=0.0;

        for (int j=0;j<N-1;j++){
            
            const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
            sumcj+=cj;
            
            onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];
            
            onlylaplaciansum += onlylaplacian;                
            gammalaplacian=mv_Gamma[i][j]*onlylaplacian;
            gammalaplaciansum += gammalaplacian;

            if (i!=j) chempot+= 2*mv_Beta[i][N-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
            

        }// j

        for (int j=0;j<N;j++){
            if(i!=j){
                for (int m=j+1;m<N;m++){
                    if(i!=m){
                        for (int n=m+1;n<N;n++){
                            if(i!=n){
                                //chempot+=8*7*mv_Beta[i][j]*(CVec[j]*CVec[m]*CVec[n]);
                            }
                        }
                    }
                }
            }
        }
        
        const double cj=1.0-sumcj;

        chempot+=2*mv_Beta[i][N-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj)  + mv_Gamma[i][N-1]*onlylaplaciansum;
        
        getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + 2 * mOmega * (ci < 0) * ci-
                                           2 * mOmega * ((1-ci) < 0) * (1-ci); 
        

    } // i=0:N-2
    
    int i = N-1;
    
    chempot_bulk=0.0;
    chempot_tension=0.0;
    const double& ci=1.0-sumci;
    chempot=0;

    sumcj=0.0;
    gammalaplaciansum=0.0;
    onlylaplaciansum=0.0;
    double product = 1;
    for (int j=0;j<N-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        sumcj+=cj;
        
        onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];                         
        onlylaplaciansum += onlylaplacian;

        gammalaplacian = mv_Gamma[i][j]*onlylaplacian;
        gammalaplaciansum += gammalaplacian;
        
        
        chempot+=2*mv_Beta[0][N-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
        
        product *= cj;
    }
    for (int j=0;j<N;j++){
        if(i!=j){
            for (int m=j+1;m<N;m++){
                if(i!=m){
                    for (int n=m+1;n<N;n++){
                        if(i!=n){
                            //chempot+=8*7*mv_Beta[i][j]*(CVec[j]*CVec[m]*CVec[n]);
                        }
                    }
                }
            }
        }
    }
    
    getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot + 2 * mOmega * (ci < 0) * ci-
                                           2 * mOmega * ((1-ci) < 0) * (1-ci); 


    /*
    if (Geometry<typename TTraits::Lattice>::isBulkSolid(k)) return;

    [[maybe_unused]] static bool isvalid = checkValid<TTraits::NumberOfComponents>();
    mIsValid=isvalid;
    using Lattice = typename TTraits::Lattice;
    constexpr int N = TTraits::NumberOfComponents;
    double chempot=0.0;
    double sumci=0.0;
    double sumcj=0.0;

    double onlylaplacian=0.0;
    double gammalaplacian=0.0;
    double onlylaplaciansum=0.0;
    double gammalaplaciansum=0.0;       

    for (int i=0;i<TTraits::NumberOfComponents-1;i++){
        chempot=0.0;
        const double& ci=getInstance<OrderParameter, N - 1, Lattice>(i)[k];
        sumci += ci;

        sumcj=0.0;
        gammalaplaciansum=0.0;
        onlylaplaciansum=0.0;

        for (int j=0;j<TTraits::NumberOfComponents-1;j++){
            
            const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
            sumcj+=cj;
            
            onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];
            
            onlylaplaciansum += onlylaplacian;                
            gammalaplacian=mv_Gamma[i][TTraits::NumberOfComponents-1]*onlylaplacian;
            gammalaplaciansum += gammalaplacian;

            if (i!=j) chempot+=2*mv_Beta[i][TTraits::NumberOfComponents-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
        }// j
        
        const double cj=1.0-sumcj;
        
        chempot+=2*mv_Beta[i][TTraits::NumberOfComponents-1]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) + mv_Gamma[i][TTraits::NumberOfComponents-1]*onlylaplaciansum;

        getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot;
        
    } // i=0:N-2
    

    int i = TTraits::NumberOfComponents-1;
    chempot=0.0;
    const double& ci=1.0-sumci;
    

    sumcj=0.0;
    gammalaplaciansum=0.0;
    onlylaplaciansum=0.0;
    
    for (int j=0;j<TTraits::NumberOfComponents-1;j++){
        const double& cj=getInstance<OrderParameter, N - 1, Lattice>(j)[k];
        sumcj+=cj;
        
        onlylaplacian = getInstance<LaplacianOrderParameter, N - 1, Lattice>(j)[k];                         
        onlylaplaciansum += onlylaplacian;

        gammalaplacian = mv_Gamma[0][j]*onlylaplacian;
        gammalaplaciansum += gammalaplacian;
        
        if (i!=j){
            chempot+=2*mv_Beta[0][j]*(-12*ci*ci*cj-12*ci*cj*cj+12*ci*cj-4*cj*cj*cj+6*cj*cj-2*cj) - gammalaplacian;
        }
    }

    for (int j=0;j<N;j++){
        if(i!=j){
            for (int m=j+1;m<N;m++){
                if(i!=m){
                    for (int n=m+1;n<N;n++){
                        if(i!=n){
                            chempot+=12*mv_Beta[i][j]*(CVec[j]*CVec[m]*CVec[n]);
                        }
                    }
                }
            }
        }
    }

    getInstance<ChemicalPotential, N, Lattice>(i)[k] = chempot;*/
    
}