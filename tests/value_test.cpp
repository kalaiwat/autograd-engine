#include <catch2/catch_test_macros.hpp>
#include "autograd/value.h"
#include <memory>

TEST_CASE("leaf value initializes data and zeroes gradient") {
    const auto x = std::make_shared<Value>(1.0);
    REQUIRE(x->data == 1.0);
    REQUIRE(x->grad == 0.0);
}

TEST_CASE("addition computes correct forward value") {
    const auto x = std::make_shared<Value>(.75);
    const auto y = std::make_shared<Value>(.25);
    const auto out = x + y;
    REQUIRE(out->data == 1.0);
}

TEST_CASE("addition computes correct backward pass") {
    const auto x = std::make_shared<Value>(.75);
    const auto y = std::make_shared<Value>(.25);
    const auto out = x + y;

    out->backward();

    REQUIRE(x->grad == 1.0);
    REQUIRE(y->grad == 1.0);
    REQUIRE(out->grad == 1.0); // not super necessary but doesn't hurt
}

TEST_CASE("addition with same shared_ptr computes correct convergence derivative") {
    const auto x = std::make_shared<Value>(1.0);
    const auto out = x + x;

    out->backward();

    REQUIRE(x->grad == 2.0);
}

TEST_CASE("multiplication computes correct forward value") {
    const auto x = std::make_shared<Value>(0.50);
    const auto y = std::make_shared<Value>(0.25);
    const auto out = x * y;

    REQUIRE(out->data == 0.125);
}

TEST_CASE("multiplication computes correct backward pass") {
    const auto x = std::make_shared<Value>(1.0);
    const auto y = std::make_shared<Value>(0.1);
    const auto out = x * y;

    out->backward();

    REQUIRE(x->grad == y->data);
    REQUIRE(y->grad == x->data);
}

TEST_CASE("multiplication with same shared_ptr computes correct convergence derivative") {
    const auto x = std::make_shared<Value>(10.0);
    const auto out = x * x;

    out->backward();

    REQUIRE(x->grad == (2.0*x->data));
}