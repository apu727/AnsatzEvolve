#include "logger.h"
#include <string>
template<typename valType1, typename valType2>
struct DefaultTestComparisons
{
    static constexpr bool HaveComparison = false;
    DefaultTestComparisons(const valType1 &, const valType2 &, double tol = 1e-13)
    {
        static_assert(dependent_false_v<valType1>, "No default test comparison for this type");
    }
};

template<>
struct DefaultTestComparisons<double, double>
{
    static constexpr bool HaveComparison = true;
    double m_tol;
    DefaultTestComparisons(const double &, const double &, double tol = 1e-13) { m_tol = tol; }

    bool operator()(const double &a, const double &b) const { return (std::abs(a - b) < m_tol || std::abs((a - b) / a) < m_tol); }
};

template<>
struct DefaultTestComparisons<std::complex<double>, std::complex<double>>
{
    static constexpr bool HaveComparison = true;
    double m_tol;
    DefaultTestComparisons(const std::complex<double> &, const std::complex<double> &, double tol = 1e-13) { m_tol = tol; }

    bool operator()(const std::complex<double> &a, const std::complex<double> &b) const
    {
        return (std::abs(a - b) < m_tol || std::abs((a - b) / a) < m_tol);
    }
};

template<typename comparisonFunctor>
struct NotComparison
{
    comparisonFunctor m_functor;
    NotComparison(const comparisonFunctor &f)
        : m_functor(f)
    { }

    template<typename... Args>
    bool operator()(Args &&...args) const
    {
        return !m_functor(std::forward<Args>(args)...);
    }
};

template<
    typename valType1,
    typename valType2,
    size_t N,
    typename comparisonFunctor,
    std::enable_if_t<std::is_invocable_r_v<bool, comparisonFunctor, valType1, valType2>, bool> = true>
inline int expectNear(const valType1 &val, const valType2 &found, const comparisonFunctor &comparison, const char (&funcName)[N], std::string message)
{
    if (comparison(val, found)) return 0;
    logger().printf("Test failed in \"%s\" with message: %s\n", funcName, message.c_str());
    logger().log("Found   ", found);
    logger().log("Expected", val);
    return 1;
}

template<typename valType1, typename valType2, size_t N>
inline int expectNear(const valType1 &val, const valType2 &found, const char (&funcName)[N], std::string message)
{
    static_assert(DefaultTestComparisons<valType1, valType2>::HaveComparison, "No default test comparison for this type");
    DefaultTestComparisons comp(val, found);
    return expectNear(val, found, comp, funcName, message);
}

template<typename valType1, typename valType2, size_t N>
inline int expectNear(const valType1 &val, const valType2 &found, double tol, const char (&funcName)[N], std::string message)
{
    static_assert(DefaultTestComparisons<valType1, valType2>::HaveComparison, "No default test comparison for this type");
    DefaultTestComparisons comp(val, found, tol);
    return expectNear(val, found, comp, funcName, message);
}

template<typename valType1, typename valType2, size_t N>
inline int expectNotNear(const valType1 &val, const valType2 &found, const char (&funcName)[N], std::string message)
{
    static_assert(DefaultTestComparisons<valType1, valType2>::HaveComparison, "No default test comparison for this type");
    NotComparison comp(DefaultTestComparisons(val, found));
    return expectNear(val, found, comp, funcName, message);
}

template<typename valType1, typename valType2, size_t N>
inline int expectNotNear(const valType1 &val, const valType2 &found, double tol, const char (&funcName)[N], std::string message)
{
    static_assert(DefaultTestComparisons<valType1, valType2>::HaveComparison, "No default test comparison for this type");
    NotComparison comp(DefaultTestComparisons(val, found, tol));
    return expectNear(val, found, comp, funcName, message);
}