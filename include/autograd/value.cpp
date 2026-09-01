#include "value.h"

void Value::backward() {
    std::vector<Value*> topology;
    std::unordered_set<Value*> visited;
    build_topology(this, visited, topology);

    // reset stale nonzero gradients
    for (const auto& operand: topology)
        operand->grad = 0.0;

    // base case of the chain rule
    this->grad = 1.0;

    // iterate topology list in reverse
    for (auto it = topology.rbegin(); it != topology.rend(); ++it) {
        if (
            const Value* node = *it;
            node->backward_fn
        ) {
            node->backward_fn();
        }
    }
}

void Value::build_topology(
    Value *node,
    std::unordered_set<Value *> &visited,
    std::vector<Value *> &topology
) {
    if (visited.count(node)) {
        return;
    }

    visited.insert(node);

    for (const auto& operand: node->operands) {
        build_topology(operand.get(), visited, topology);
    }

    topology.push_back(node);
}

std::shared_ptr<Value> operator+(
    const std::shared_ptr<Value>& lhs,
    const std::shared_ptr<Value>& rhs
) {
    const double result = lhs->data + rhs->data;
    auto result_node = std::make_shared<Value>(result);

    result_node->operands.push_back(lhs);
    result_node->operands.push_back(rhs);
    result_node->backward_fn = [self = result_node.get()]() {
        self->operands[0]->grad += self->grad;
        self->operands[1]->grad += self->grad;
    };

    return result_node;
}

std::shared_ptr<Value> operator-(
    const std::shared_ptr<Value>& lhs,
    const std::shared_ptr<Value>& rhs
) {
    const auto negated_rhs = std::make_shared<Value>(-rhs->data);
    return lhs + negated_rhs;
}

std::shared_ptr<Value> operator*(
    const std::shared_ptr<Value>& lhs,
    const std::shared_ptr<Value>& rhs
) {
    const double result = lhs->data * rhs->data;
    auto result_node = std::make_shared<Value>(result);

    result_node->operands.push_back(lhs);
    result_node->operands.push_back(rhs);
    result_node->backward_fn = [self = result_node.get()]() {
        self->operands[0]->grad += (self->grad * self->operands[1]->data);
        self->operands[1]->grad += (self->grad * self->operands[0]->data);
    };

    return result_node;
}