#ifdef USE_CUDNN

#include "cudnn_optimizer.h"

#include "cudnn-frontend/include/cudnn_frontend.h"

#include "error.h"
#include "node.h"

#include <functional>
#include <unordered_set>

namespace NHamKaas {

namespace fe = cudnn_frontend;

struct TCudnnCompilationResult
{
    std::shared_ptr<fe::graph::Graph> Graph;
    std::unordered_map<std::string, std::shared_ptr<fe::graph::Tensor_attributes>> InputToTensor;
    std::shared_ptr<fe::graph::Tensor_attributes> OutputTensor;
};

class TCudnnNode
    : public TNodeBase
{
public:
    TCudnnNode(std::vector<TNodeBasePtr> nodes)
        : TNodeBase(nodes.back()->GetMeta())
        , Nodes_(std::move(nodes))
    { }

    void Compile(cudnnHandle_t handle, std::unordered_map<std::string, TCudnnCompilationResult>& compilationCache)
    {
        // (lab5/01): Your code here.
    }

    int64_t GetBufferSize() const override
    {
        // (lab5/01): Your code here.
        return 0;
    }

    void SetBuffer(char* buffer) override
    {
        // (lab5/01): Your code here.
    }

    void Initialize(IDevice* device)
    {
        // (lab5/01): Your code here.
    }

    void EvaluateCpu() override
    {
        THROW("Cudnn is not supported on CPU");
    }

    void EvaluateGpu(const TEvaluationContext& context) override
    {
        // (lab5/01): Your code here.
        THROW("Cudnn is not supported");
    }

private:
    std::vector<TNodeBasePtr> Nodes_;
};

std::vector<TNodeBasePtr> GetAllNodes(const TNodeBasePtr& root)
{
    std::vector<TNodeBasePtr> allNodes;
    std::unordered_set<TNodeBasePtr> visited;

    std::function<void(TNodeBasePtr)> dfs = [&] (TNodeBasePtr node) {
        if (visited.count(node)) {
            return;
        }

        visited.insert(node);
        allNodes.push_back(node);

        for (const auto& input : node->GetInputs()) {
            dfs(input);
        }
    };
    dfs(root);

    return allNodes;
}

std::vector<std::vector<TNodeBasePtr>> GetSubgraphsToFuse(const std::vector<TNodeBasePtr>& nodes)
{
    // For every node, stores a set of nodes that use it as input.
    std::unordered_map<TNodeBasePtr, std::unordered_set<TNodeBasePtr>> reverseEdges;

    for (const auto& node : nodes) {
        for (const auto& input : node->GetInputs()) {
            reverseEdges[input].insert(node);
        }
    }

    auto isMatMul = [&] (TNodeBasePtr node) {
        return dynamic_cast<TMatMulNode*>(node.get()) != nullptr;
    };

    auto isPointwise = [&] (TNodeBasePtr node) {
        // https://github.com/NVIDIA/cudnn-frontend/issues/107, disable broadcasting for now.
        if (dynamic_cast<TSumNode*>(node.get())) {
            return node->GetInputs()[0]->GetShape() == node->GetInputs()[1]->GetShape();
        }
        if (dynamic_cast<THadamardProductNode*>(node.get())) {
            return node->GetInputs()[0]->GetShape() == node->GetInputs()[1]->GetShape();
        }
        if (dynamic_cast<TReLUNode*>(node.get())) {
            return true;
        }
        if (dynamic_cast<TSiLUNode*>(node.get())) {
            return true;
        }

        return false;
    };

    // Set of the nodes that already have been fused to some graph.
    std::unordered_set<TNodeBasePtr> fusedNodes;

    std::vector<std::vector<TNodeBasePtr>> subgraphs;

    // https://github.com/NVIDIA/cudnn-frontend/issues/108
    constexpr int MaxFusedNodes = 30;

    std::vector<TNodeBasePtr> subgraph;
    int subgraphSize = 0;

    std::function<void(TNodeBasePtr)> traversePointwise = [&] (TNodeBasePtr node) {
        if (subgraphSize >= MaxFusedNodes) {
            return;
        }

        if (fusedNodes.count(node)) {
            return;
        }

        fusedNodes.insert(node);
        ++subgraphSize;

        for (const auto& input : node->GetInputs()) {
            if (isPointwise(input)) {
                traversePointwise(input);
            }
        }

        subgraph.push_back(node);
    };

    for (const auto& node : nodes) {
        if (isMatMul(node)) {
            subgraph.clear();
            subgraphSize = 0;

            // Old versions of cuDNN do not support pointwise operations
            // on the second input of MatMul.
            traversePointwise(node->GetInputs()[0]);

            subgraph.push_back(node);

            // Now subgraph contains Matmul(g1(A), B).
            // Now let's add g2.
            while (reverseEdges[subgraph.back()].size() == 1) {
                auto next = *reverseEdges[subgraph.back()].begin();
                if (fusedNodes.count(next)) {
                    break;
                }
                if (subgraphSize >= MaxFusedNodes) {
                    break;
                }

                if (isPointwise(next)) {
                    fusedNodes.insert(next);
                    subgraph.push_back(next);
                    ++subgraphSize;
                } else {
                    break;
                }
            }

            subgraphs.push_back(subgraph);
        }
    }

    for (int64_t index = nodes.size() - 1; index >= 0; --index) {
        const auto& node = nodes[index];
        if (fusedNodes.count(node)) {
            continue;
        }

        if (isPointwise(node)) {
            subgraph.clear();
            subgraphSize = 0;

            traversePointwise(node);
            subgraphs.push_back(subgraph);
        }
    }

    return subgraphs;
}

TNodeBasePtr ReplaceNodes(
    TNodeBasePtr root,
    std::vector<TNodeBasePtr>& allNodes,
    const std::unordered_map<TNodeBasePtr, TNodeBasePtr>& replacements)
{
    for (auto& node : allNodes) {
        for (auto& input : node->GetInputs()) {
            if (replacements.count(input)) {
                assert(input->GetShape() == replacements.at(input)->GetShape());
                node->ReplaceInput(input, replacements.at(input));
            }
        }
    }

    if (replacements.count(root)) {
        return replacements.at(root);
    } else {
        return root;
    }
}

TNodeBasePtr RunCudnnOptimizer(TNodeBasePtr root, const TBootstrap* bootstrap)
{
    auto allNodes = GetAllNodes(root);
    auto subgraphs = GetSubgraphsToFuse(allNodes);

    std::vector<std::shared_ptr<TCudnnNode>> cudnnNodes(subgraphs.size());
    std::unordered_map<TNodeBasePtr, TNodeBasePtr> replacements;
    for (int index = 0; index < subgraphs.size(); ++index) {
        cudnnNodes[index] = std::make_shared<TCudnnNode>(subgraphs[index]);
        replacements[subgraphs[index].back()] = cudnnNodes[index];
    }

    // Right now, graph is the same as before, but we have additional cuDNN nodes
    // whose outputs are not connected to anything.
    root = ReplaceNodes(root, allNodes, replacements);

    // Now the graph has the final shape and also nodes of the original graph that
    // are going to be replaced by cuDNN nodes has proper inputs, so we can run
    // compilation of the cuDNN nodes.
    std::unordered_map<std::string, TCudnnCompilationResult> compilationCache;
    for (auto& node : cudnnNodes) {
        node->Compile(bootstrap->GetCudnnHandle(), compilationCache);
    }

    return root;
}

} // namespace NHamKaas

#endif
