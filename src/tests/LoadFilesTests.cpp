/*
 * Test loading of files and make sure their internal representations are correct. 
 * This is not a great test as it requires knowledge of the internal representation 
 * 
 * The following functions need to be tested:
 * void LoadNuclearEnergy(realNumType &NuclearEnergy, std::string filePath);
 *
 * bool loadPath(std::shared_ptr<stateRotate> sr, std::string filePath,
 *            std::vector<ansatz::rotationElement>& rotationPath);

 * bool loadParameters(std::string filePath,
                    std::vector<ansatz::rotationElement>& rotationPath,
                    std::vector<std::vector<ansatz::rotationElement>> &rotationPaths,
                    std::vector<std::pair<int,realNumType>> &order,
                    int& numberOfUniqueParameters);

 * bool loadParameters(const std::string& orderFilePath, const std::string& parameterFilePath,
                    std::vector<ansatz::rotationElement>& rotationPath,
                    std::vector<std::vector<ansatz::rotationElement>> &rotationPaths,
                    std::vector<std::pair<int,realNumType>> &orders // the fixed relation between parameters
                    ,int& numberOfUniqueParameters);
 * stateRotate::loadOperators(std::string filePath, std::vector<stateRotate::exc>& excs);
 * 
 * Example data is already present and used in EndtoEndTests.cpp - use that. 
 * Inspect the state of NuclearEnergy, rotationPath, rotationPaths, orders, excs, and make sure they were loaded correctly
 */

int LoadFilesTests(int argc, char **argv)
{
    return 0;
}