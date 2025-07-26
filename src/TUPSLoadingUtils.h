/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "globals.h"

#include "ansatz.h"
#include <vector>
/********************TUPS Stuff*****************/
//reads a statevector from filepath(csv) into Coeffs. Coeffs[3] corresponds to 00011:x. Length of Coeffs is determined by length of bitstring in file
int readCsvState(std::vector<std::complex<realNumType>>& Coeffs, const std::string& filePath);
int readCsvState(std::vector<realNumType>& Coeffs, const std::string& filePath);

//reads in the nuclear energy
void LoadNuclearEnergy(realNumType &NuclearEnergy, std::string filePath);

bool loadPath(stateRotate& sr, std::string filePath,
              std::vector<ansatz::rotationElement>& rotationPath);

bool loadParameters(std::string filePath,
                    std::vector<ansatz::rotationElement>& rotationPath,
                    std::vector<std::vector<ansatz::rotationElement>> &rotationPaths,
                    std::vector<std::pair<int,realNumType>> &order,
                    int& numberOfUniqueParameters);

