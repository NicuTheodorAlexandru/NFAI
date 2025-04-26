using NFAI.Core;

namespace NFAI.Models;

public abstract class AbstractModelFactory : IDisposable
{
    public abstract void Dispose();

    public abstract bool TryCreate(Dictionary<string, object> metadata, List<AbstractComputeCollection> tensors, ModelOptions modelOptions, out IInferenceProvider? model);
}
