using NFAI.Core;
using NFAI.Vulkan;
using Silk.NET.Vulkan;

namespace NFAI.Models.Llama3;

public class LlamaModelFactory : AbstractModelFactory
{
    private readonly Vk vk;
    private readonly Device device;
    private readonly VulkanBufferManager vulkanBufferManager;
    private readonly Instance instance;
    private readonly PhysicalDevice physicalDevice;

    public LlamaModelFactory()
    {
        vk = Vk.GetApi();
        instance = VulkanHelper.CreateVulkanInstance(vk);
        physicalDevice = VulkanHelper.PickPhysicalDevice(vk, instance);
        device = VulkanHelper.CreateLogicalDevice(vk, physicalDevice);
        vulkanBufferManager = new VulkanBufferManager(vk, device, physicalDevice);
    }

    public override void Dispose()
    {
        GC.SuppressFinalize(this);
        vulkanBufferManager.Dispose();
        var allocationCallbacks = new AllocationCallbacks();
        vk.DestroyDevice(device, ref allocationCallbacks);
        vk.DestroyInstance(instance, ref allocationCallbacks);
        vk.Dispose();
    }

    public override bool TryCreate(Dictionary<string, object> metadata, List<AbstractComputeCollection> tensors, ModelOptions modelOptions, out IInferenceProvider? model)
    {
        var modelFamily = metadata["general.architecture"].ToString() ?? string.Empty;
        if (modelFamily != "llama")
        {
            model = null;
            return false;
        }
        model = new LlamaModel(vk, device, vulkanBufferManager, metadata, tensors, modelOptions.KVCacheSize);
        return true;
    }
}
