set (TestsToRun
  EndtoEndTests.cpp
  FusedEvolveTests.cpp
  LoadFilesTests.cpp
)
create_test_sourcelist (Tests linalgTestsCMAKE.cpp ${TestsToRun})

# add the executable
add_executable (AnsatzEvolveTests ${Tests}
  TestUtil.hpp
  )

add_dependencies(AllTests AnsatzEvolveTests)


target_include_directories(AnsatzEvolveTests PUBLIC ../)
target_link_libraries(AnsatzEvolveTests AnsatzSynthInterface)
target_compile_definitions(AnsatzEvolveTests PRIVATE ANSATZEVOLVE_RESOURCE_DIR="${PROJECT_SOURCE_DIR}/tests/Resources")

# Add all the ADD_TEST for each test
foreach (t ${TestsToRun})
  get_filename_component (TName ${t} NAME_WE)
  add_test (NAME ${TName} COMMAND linalgTests ${TName})
endforeach ()