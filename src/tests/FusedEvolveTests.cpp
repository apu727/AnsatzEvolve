/*
 * Test the interface of the fusedEvolve class. There are two methods for computing gradients there, that can be compared against one another.
 * Test that it can evolve a wavefunction correctly. 
 * Test that it does the same thing in complex mode. An easy cheat is to multiply the wavefunction by i at the start and then divide it at the end. 
 * Test that the fusing of gates works correctly, need to produce the same results with fused and non-fused evolutions. 
 * Check that gradients + Hessians are computed correctly. Can just compute the answer once and regression check that it doesnt change.
 * 
 *  Specifically the following functions should be tested
 *  void updateExc(const std::vector<stateRotate::exc>& excs);
 *  void evolve(vector<numType>& dest, const std::vector<realNumType>& angles, vector<numType>* specifiedStart  = nullptr);
 *  void evolveMultiple(Matrix<numType>& dest, const Matrix<realNumType>::EigenMatrix& angles, vector<numType>* specifiedStart  = nullptr);
 *  //finalVector is the result of an evolve operation. Since we dont cache the angles this is up to the user to provide correctly.
 *  void evolveDerivative(const vector<numType>& finalVector,vector<realNumType>& deriv,const std::vector<realNumType>& angles, realNumType* Energy = nullptr);
 *  void evolveHessian(Eigen::MatrixXd& Hessian,vector<realNumType>& derivCompressed,const std::vector<realNumType>& angles, Eigen::Matrix<numType,-1,-1>* Ts = nullptr, realNumType* Energy = nullptr);
 *  realNumType getEnergy(const vector<numType> &psi);
 *  vector<realNumType> getEnergies(const Matrix<numType> &psi);
 *  const vector<numType> &getStart() const { return m_start; }
 * 
 * The ansatz should be small. A couple difficult rotations is enough:
 * E.g. |01> Initial state and E_0^2-E^0_2 and E_1^2-E^1_2 This checks fermionic signs
 * Should also check two commuting rotations produce the right answer:
 * E.g. |02> Initial state and E_0^1-E^0_1 and E_2^3-E^2_3
 * These are simple enough that for nice angles analytic derivs and hessians are possible. 
 * 
 * Make sure the tests are deterministic
 */
int FusedEvolveTests(int argc, char **argv)
{
    return 0;
}