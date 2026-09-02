#ifndef AUTOGRAD_ENGINE_VALUE_H
#define AUTOGRAD_ENGINE_VALUE_H

#include <functional>
#include <vector>
#include <memory>
#include <unordered_set>

class Value {
    // shared_ptr keeps operands alive for backward(), even after the
    // expression that built them goes out of scope.
    std::vector<std::shared_ptr<Value>> operands;
    std::function<void()> backward_fn;

    // Raw Value* here: traversal doesn't need to extend lifetime, only
    // the operands vector above does.
    static void build_topology(
        Value* node,
        std::unordered_set<Value*>& visited,
        std::vector<Value*>& topology
    );
public:
    double data;
    double grad;

    Value(double data) : data(data), grad(0.0) {}

    void backward();

    friend std::shared_ptr<Value> operator+(
        const std::shared_ptr<Value>&,
        const std::shared_ptr<Value>&
    );

    friend std::shared_ptr<Value> operator-(
        const std::shared_ptr<Value>&,
        const std::shared_ptr<Value>&
    );

    friend std::shared_ptr<Value> operator*(
        const std::shared_ptr<Value>&,
        const std::shared_ptr<Value>&
    );

    friend std::shared_ptr<Value> operator/(
        const std::shared_ptr<Value>&,
        const std::shared_ptr<Value>&
    );
};

std::shared_ptr<Value> operator+(
    const std::shared_ptr<Value>& lhs,
    const std::shared_ptr<Value>& rhs
);

std::shared_ptr<Value> operator-(
    const std::shared_ptr<Value>& lhs,
    const std::shared_ptr<Value>& rhs
);

std::shared_ptr<Value> operator*(
    const std::shared_ptr<Value>& lhs,
    const std::shared_ptr<Value>& rhs
);

std::shared_ptr<Value> operator/(
    const std::shared_ptr<Value>& lhs,
    const std::shared_ptr<Value>& rhs
);

#endif //AUTOGRAD_ENGINE_VALUE_H