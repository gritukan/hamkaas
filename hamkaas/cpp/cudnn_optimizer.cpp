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
        Graph_ = std::make_shared<fe::graph::Graph>();

        // Nodes are already topologically sorted, so no need to run DFS.
        std::unordered_set<TNodeBasePtr> graphNodes(Nodes_.begin(), Nodes_.end());
        std::unordered_map<std::string, TNodeBasePtr> nameToInput;
        std::unordered_map<TNodeBasePtr, std::shared_ptr<fe::graph::Tensor_attributes>> nodeToTensor;

        int64_t nextNodeIndex = 1;

        auto getTensor = [&] (TNodeBasePtr node) {
            if (nodeToTensor.count(node)) {
                return nodeToTensor[node];
            } else {
                assert(!graphNodes.count(node));
                auto name = "input_" + std::to_string(nextNodeIndex++);
                auto tensor = CreateTensor(node->GetMeta(), name);
                nameToInput[name] = node;
                nodeToTensor[node] = tensor;
                InputTensorToNode_[tensor] = node;
                Inputs_.push_back(node);

                return tensor;
            }
        };

        auto getTensorTrace = [&] (std::shared_ptr<fe::graph::Tensor_attributes> tensor) {
            std::string trace = "(" + tensor->get_name();
            for (auto dim : tensor->get_dim()) {
                trace += ", " + std::to_string(dim);
            }
            trace += ")";
            return trace;
        };

        std::string trace;

        for (const auto& node : Nodes_) {
            std::shared_ptr<fe::graph::Tensor_attributes> tensor;
            if (auto* matmulNode = dynamic_cast<TMulNode*>(node.get())) {
                auto lhs = getTensor(matmulNode->GetInputs()[0]);
                auto rhs = getTensor(matmulNode->GetInputs()[1]);
                std::string name = "matmul_" + std::to_string(nextNodeIndex++);
                tensor = CreateMatMul(lhs, rhs, name);
                trace += name + "(" + getTensorTrace(lhs) + ", " + getTensorTrace(rhs) + ") ";
            } else if (auto* sumNode = dynamic_cast<TSumNode*>(node.get())) {
                auto lhs = getTensor(sumNode->GetInputs()[0]);
                auto rhs = getTensor(sumNode->GetInputs()[1]);
                std::string name = "add_" + std::to_string(nextNodeIndex++);
                tensor = CreateAdd(lhs, rhs, name);
                trace += name + "(" + getTensorTrace(lhs) + ", " + getTensorTrace(rhs) + ") ";
            } else if (auto* hadamardProductNode = dynamic_cast<THadamardProductNode*>(node.get())) {
                auto lhs = getTensor(hadamardProductNode->GetInputs()[0]);
                auto rhs = getTensor(hadamardProductNode->GetInputs()[1]);
                std::string name = "mul_" + std::to_string(nextNodeIndex++);
                tensor = CreateMul(lhs, rhs, name);
                trace += name + "(" + getTensorTrace(lhs) + ", " + getTensorTrace(rhs) + ") ";
            } else if (auto* reluNode = dynamic_cast<TReLUNode*>(node.get())) {
                auto input = getTensor(reluNode->GetInputs()[0]);
                std::string name = "relu_" + std::to_string(nextNodeIndex++);
                tensor = CreateReLU(input, name);
                trace += name + "(" + getTensorTrace(input) + ") ";
            } else if (auto* siluNode = dynamic_cast<TSiLUNode*>(node.get())) {
                auto input = getTensor(siluNode->GetInputs()[0]);
                std::string name = "silu_" + std::to_string(nextNodeIndex++);
                tensor = CreateSiLU(input, name);
                trace += name + "(" + getTensorTrace(input) + ") ";
            } else {
                THROW("Unsupported node type");
            }

            nodeToTensor[node] = tensor;
        }

        OutputTensor_ = nodeToTensor[Nodes_.back()];
        OutputTensor_->set_output(true);

        std::cerr << "Compiling graph: " << trace << std::endl;

        Trace_ = trace;

        if (compilationCache.count(trace)) {
            const auto& compilationResult = compilationCache[trace];
            std::cerr << "Cache hit!" << std::endl;

            Graph_ = compilationResult.Graph;

            for (const auto& [name, node] : nameToInput) {
                assert(compilationResult.InputToTensor.count(name));
                auto tensor = compilationCache[trace].InputToTensor[name];
                InputTensorToNode_[tensor] = node;
            }

            OutputTensor_ = compilationCache[trace].OutputTensor;
            return;
        }

        std::cerr << "Cache miss! Compiling..." << std::endl;

        CUDNN_FE_CHECK_ERROR(Graph_->validate());
        CUDNN_FE_CHECK_ERROR(Graph_->build_operation_graph(handle));
        CUDNN_FE_CHECK_ERROR(Graph_->create_execution_plans({fe::HeurMode_t::A}));
        CUDNN_FE_CHECK_ERROR(Graph_->build_plans(handle, fe::BuildPlanPolicy_t::ALL));

        TCudnnCompilationResult compilationResult;
        compilationResult.Graph = Graph_;
        for (const auto& [name, node] : nameToInput) {
            compilationResult.InputToTensor[name] = nodeToTensor[node];
        }
        compilationResult.OutputTensor = OutputTensor_;
        compilationCache[trace] = compilationResult;

        std::cerr << "Graph compiled" << std::endl;
    }

    int64_t GetBufferSize() const override
    {
        return Graph_->get_workspace_size();
    }

    void SetBuffer(char* buffer) override
    {
        GraphWorkspace_ = buffer;
    }

    void Initialize(IDevice* device)
    {
        for (const auto& [tensor, node] : InputTensorToNode_) {
            assert(TensorMap_.emplace(tensor, node->GetOutput()).second);
        }
        assert(TensorMap_.emplace(OutputTensor_, GetOutput()).second);

        // Clear everything not to keep old nodes in memory.
        Nodes_.clear();
        InputTensorToNode_.clear();
        OutputTensor_.reset();
    }

    void EvaluateCpu() override
    {
        THROW("Cudnn is not supported on CPU");
    }

    void EvaluateGpu(const TEvaluationContext& context) override
    {
        std::cerr << "Running node with trace: " << Trace_ << std::endl;

        auto handle = context.Bootstrap->GetCudnnHandle();
        CUDNN_CHECK_ERROR(cudnnSetStream(handle, context.Stream));

        CUDNN_FE_CHECK_ERROR(Graph_->execute(handle, TensorMap_, GraphWorkspace_));        
    }

private:
    std::shared_ptr<fe::graph::Graph> Graph_;
    char* GraphWorkspace_ = nullptr;

    std::string Trace_;

    // This are needed only between compliation and initialization and cleared after that.
    std::vector<TNodeBasePtr> Nodes_;
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, TNodeBasePtr> InputTensorToNode_;
    std::shared_ptr<fe::graph::Tensor_attributes> OutputTensor_;

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> TensorMap_;

    std::shared_ptr<fe::graph::Tensor_attributes> CreateTensor(const TTensorMeta& meta, const std::string& name)
    {
        if (meta.ValueType != EValueType::Float32) {
            THROW("Only float32 is supported", VAR(meta.ValueType));
        }

        int64_t addDimensions = MaxDimensions - meta.Shape.size();
        std::vector<int64_t> shape(MaxDimensions, -1);
        for (int64_t index = 0; index < addDimensions; ++index) {
            shape[index] = 1;
        }
        for (int64_t index = 0; index < meta.Shape.size(); ++index) {
            shape[addDimensions + index] = meta.Shape[index];
        }

        std::vector<int64_t> strides(MaxDimensions, 1);
        for (int64_t index = MaxDimensions - 2; index >= 0; --index) {
            strides[index] = shape[index + 1] * strides[index + 1];
        }

        return Graph_->tensor(
            fe::graph::Tensor_attributes()
                .set_name(name)
                .set_dim(shape)
                .set_stride(strides)
                .set_data_type(fe::DataType_t::FLOAT));
    }

    std::shared_ptr<fe::graph::Tensor_attributes> CreateMatMul(
        const std::shared_ptr<fe::graph::Tensor_attributes>& lhs,
        const std::shared_ptr<fe::graph::Tensor_attributes>& rhs,
        const std::string& name)
    {
        auto matmul = Graph_->matmul(
            lhs,
            rhs,
            fe::graph::Matmul_attributes()
                .set_name(name)
                .set_compute_data_type(fe::DataType_t::FLOAT));
        matmul->set_data_type(fe::DataType_t::FLOAT);
        return matmul;
    }

    std::shared_ptr<fe::graph::Tensor_attributes> CreateAdd(
        const std::shared_ptr<fe::graph::Tensor_attributes>& lhs,
        const std::shared_ptr<fe::graph::Tensor_attributes>& rhs,
        const std::string& name)
    {
        for (int i = 0; i < lhs->get_dim().size(); i++) {
            std::cerr << lhs->get_dim()[i] << " ";
        }
        std::cerr << std::endl;
        for (int i = 0; i < lhs->get_stride().size(); i++) {
            std::cerr << lhs->get_stride()[i] << " ";
        }
        std::cerr << std::endl;
        for (int i = 0; i < rhs->get_dim().size(); i++) {
            std::cerr << rhs->get_dim()[i] << " ";
        }
        std::cerr << std::endl;
        for (int i = 0; i < rhs->get_stride().size(); i++) {
            std::cerr << rhs->get_stride()[i] << " ";
        }
        std::cerr << std::endl;

        auto add = Graph_->pointwise(
            lhs,
            rhs,
            fe::graph::Pointwise_attributes()
                .set_name(name)
                .set_mode(fe::PointwiseMode_t::ADD)
                .set_compute_data_type(fe::DataType_t::FLOAT));
        add->set_data_type(fe::DataType_t::FLOAT);
        return add;
    }

    std::shared_ptr<fe::graph::Tensor_attributes> CreateMul(
        const std::shared_ptr<fe::graph::Tensor_attributes>& lhs,
        const std::shared_ptr<fe::graph::Tensor_attributes>& rhs,
        const std::string& name)
    {
        auto mul = Graph_->pointwise(
            lhs,
            rhs,
            fe::graph::Pointwise_attributes()
                .set_name(name)
                .set_mode(fe::PointwiseMode_t::MUL)
                .set_compute_data_type(fe::DataType_t::FLOAT));
        mul->set_data_type(fe::DataType_t::FLOAT);
        return mul;
    }

    std::shared_ptr<fe::graph::Tensor_attributes> CreateReLU(
        const std::shared_ptr<fe::graph::Tensor_attributes>& input,
        const std::string& name)
    {
        auto relu = Graph_->pointwise(
            input,
            fe::graph::Pointwise_attributes()
                .set_name(name)
                .set_mode(fe::PointwiseMode_t::RELU_FWD)
                .set_compute_data_type(fe::DataType_t::FLOAT));
        relu->set_data_type(fe::DataType_t::FLOAT);
        return relu;
    }

    std::shared_ptr<fe::graph::Tensor_attributes> CreateSiLU(
        const std::shared_ptr<fe::graph::Tensor_attributes>& input,
        const std::string& name)
    {
        auto silu = Graph_->pointwise(
            input,
            fe::graph::Pointwise_attributes()
                .set_name(name)
                .set_mode(fe::PointwiseMode_t::SWISH_FWD)
                .set_compute_data_type(fe::DataType_t::FLOAT));
        silu->set_data_type(fe::DataType_t::FLOAT);
        return silu;
    }
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
        return dynamic_cast<TMulNode*>(node.get()) != nullptr;
    };

    auto isPointwise = [&] (TNodeBasePtr node) {
        return false;
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

            traversePointwise(node);

            // Now subgraph contains Matmul(g1(A), g2(B)).
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
    std::cerr << "Optimizer: " << subgraphs.size() << " subgraphs to fuse" << std::endl;
    for (int index = 0; index < subgraphs.size(); ++index) {
        cudnnNodes[index] = std::make_shared<TCudnnNode>(subgraphs[index]);
        std::cerr << "Optimizer: subgraph has " << subgraphs[index].size() << " nodes" << std::endl;
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
