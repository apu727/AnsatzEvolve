#include "AnsatzManager.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/native_enum.h>
#include <pybind11/operators.h>




namespace py = pybind11;


PYBIND11_MODULE(PyAnsatzEvolve, m) {
    //stateAnsatzManager
    py::class_<stateAnsatzManager>(m,"stateAnsatzManager")
        .def(py::init<>())
        .def("storeOperators",[](stateAnsatzManager& self,const std::vector<std::vector<int>>& operators)
             {
                 std::vector<stateRotate::exc> excs;
                 excs.reserve(operators.size());
                 stateRotate::exc e;
                 for (auto op : operators)
                 {
                     if (op.size() != 4)
                         throw std::length_error("Each operator must be 4 long");
                     for (int8_t i = 0; i < 4; i++)
                     {
                         if (op[i] > std::numeric_limits<int8_t>::max())
                             throw std::out_of_range("Each operator index too big, < 127");
                         e[i] = op[i]-1;
                     }
                     excs.push_back(e);
                 }
                 self.storeOperators(excs);
             })
        .def("storeInitial",&stateAnsatzManager::storeInitial)
        .def("storeHamiltonian",&stateAnsatzManager::storeHamiltonian)
        .def("storeNuclearEnergy",&stateAnsatzManager::storeNuclearEnergy)

        .def("storeParameterDependencies",&stateAnsatzManager::storeParameterDependencies)
        .def("storeRunPath",&stateAnsatzManager::storeRunPath)

        .def("setAngles",&stateAnsatzManager::setAngles)
        .def("getAngles",&stateAnsatzManager::getAngles)

        .def("getExpectationValue",
             [](stateAnsatzManager& self)
             {realNumType ret; self.getExpectationValue(ret); return ret;})
        .def("getFinalState",
             [](stateAnsatzManager& self)
             {vector<numType> ret; self.getFinalState(ret); return static_cast<vector<numType>::EigenVector>(ret);})
        .def("getGradient",
             [](stateAnsatzManager& self)
             {vector<realNumType> ret; self.getGradient(ret); return static_cast<vector<realNumType>::EigenVector>(ret);})
        .def("getGradientComp",
             [](stateAnsatzManager& self)
             {vector<realNumType> ret; self.getGradientComp(ret); return static_cast<vector<realNumType>::EigenVector>(ret);})
        .def("getHessian",
             [](stateAnsatzManager& self)
             {Matrix<realNumType>::EigenMatrix ret; self.getHessian(ret); return ret;})
        .def("getHessianComp",
             [](stateAnsatzManager& self)
             {Matrix<realNumType>::EigenMatrix ret; self.getHessianComp(ret); return ret;})

        .def("getExpectationValue",
             [](stateAnsatzManager& self,const std::vector<realNumType>& angles)
             {realNumType ret; self.getExpectationValue(angles,ret); return ret;})
        .def("getFinalState",
             [](stateAnsatzManager& self,const std::vector<realNumType>& angles)
             {vector<numType> ret; self.getFinalState(angles,ret); return static_cast<vector<numType>::EigenVector>(ret);})
        .def("getGradient",
             [](stateAnsatzManager& self,const std::vector<realNumType>& angles)
             {vector<realNumType> ret; self.getGradient(angles,ret); return static_cast<vector<realNumType>::EigenVector>(ret);})
        .def("getGradientComp",
             [](stateAnsatzManager& self,const std::vector<realNumType>& angles)
             {vector<realNumType> ret; self.getGradientComp(angles,ret); return static_cast<vector<realNumType>::EigenVector>(ret);})
        .def("getHessian",
             [](stateAnsatzManager& self,const std::vector<realNumType>& angles)
             {Matrix<realNumType>::EigenMatrix ret; self.getHessian(angles,ret); return ret;})
        .def("getHessianComp",
             [](stateAnsatzManager& self,const std::vector<realNumType>& angles)
             {Matrix<realNumType>::EigenMatrix ret; self.getHessianComp(angles,ret); return ret;})

        .def("getExpectationValues",
             [](stateAnsatzManager& self,Matrix<realNumType>::EigenMatrix angles)
             {
                vector<realNumType>::EigenVector Es;
                self.getExpectationValues(angles,Es);
                return Es;
             })

        .def("optimise",&stateAnsatzManager::optimise)
        // .def("generatePathsForSubspace",&stateAnsatzManager::generatePathsForSubspace);
        ;

}

