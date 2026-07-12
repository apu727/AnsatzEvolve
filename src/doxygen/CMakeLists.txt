# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

find_package(Doxygen)

if (Doxygen_FOUND)
    set(DOXYGEN_PROJECT_NAME "AnsatzEvolve")
    set(DOXYGEN_GENERATE_HTML YES)   # default
    set(DOXYGEN_RECURSIVE YES)       # default
    set(DOXYGEN_EXCLUDE_PATTERNS */third-party/*)
    set(DOXYGEN_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../docs)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_USE_MATHJAX YES)
    # set(DOXYGEN_FORMULA_MACROFILE macros.tex)
    set(DOXYGEN_MATHJAX_EXTENSIONS braket)
    set(DOXYGEN_MATHJAX_VERSION MathJax_3)

    set(ANSATZEVOLVE_DOC_FILES
            GettingStarted.dox
            Manual.dox
            ExternalInterface.dox
            benchmarks.dox
        )

    set(ANSATZEVOLVE_PYTHON_SRC_FILES
            ${PROJECT_SOURCE_DIR}/../python/HamGen.py
            ${PROJECT_SOURCE_DIR}/../python/makeOperators.py
        )

    doxygen_add_docs(AnsatzEvolve_docs
        ${PROJECT_SOURCE_DIR}
        ${ANSATZEVOLVE_PYTHON_SRC_FILES}
        ${ANSATZEVOLVE_DOC_FILES})
endif()

