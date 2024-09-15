#ifdef USE_CUDNN

#include "bootstrap.h"
#include "node.h"

namespace NHamKaas {

TNodeBasePtr RunCudnnOptimizer(TNodeBasePtr root, const TBootstrap* bootstrap);

} // namespace NHamKaas

#endif
