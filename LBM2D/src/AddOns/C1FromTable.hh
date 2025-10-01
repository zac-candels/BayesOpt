#pragma once

class C1FromTable : public AddOnBase {
   public:
    template <class TTraits>
    inline void compute(int k);

    inline void setTable(std::vector<double> table) {mTable=table;}
    inline void setRhoMinMax(double rhomin, double rhomax) {mRhoMin=rhomin;mRhoMax=rhomax;}
    inline void setElecMinMax(double elecmin, double elecmax) {mElecMin=elecmin;mElecMax=elecmax;}
    inline void setRhoConversion(double rhoconvert) {mRhoConvert=rhoconvert;}

    inline void setElectricField(double (*elec)(int k)) {
        evalElectricField = elec;
    }

   private:
    double mRhoMin;
    double mRhoMax;
    double mElecMin;
    double mElecMax;

    double mRhoConvert;

    static double defaultField(int k) { return 0; }
    double (*evalElectricField)(int k) = &defaultField;

    std::vector<double> mTable;
};

template <class TTraits>
inline void C1FromTable::compute(int k) {

    using Lattice = typename TTraits::Lattice;
    const int NDIM = Lattice::NDIM;

    double density = Density<>::template get<typename TTraits::Lattice>(k)*mRhoConvert;
    double elec = evalElectricField(k);

    int rhoidx = (mTable.size() - 1) / (mRhoMax - mRhoMin) * (density - mRhoMin);
    if (rhoidx < 0) rhoidx=0;
    if (rhoidx > mTable.size() - 1) rhoidx = mTable.size() - 1;

    /*int elecidx = (mTable.size() - 1) / (mElecMax - mElecMin) * (elec - mElecMin);
    if (elecidx < 0) elecidx=0;
    if (elecidx > mTable.size() - 1) elecidx = mTable.size() - 1;*/

    /*#pragma omp critical
    {
    if(TIME==2000) std::cout<<rhoidx<<" "<<elecidx<<std::endl;
    }*/
    C1<>::get<Lattice>(k)=mTable[k];

}
