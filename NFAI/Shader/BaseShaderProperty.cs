using Silk.NET.Vulkan;

namespace NFAI.Shader;

public unsafe abstract class BaseShaderProperty
{
    public DescriptorBufferInfo* descriptorBufferInfo;

    public Silk.NET.Vulkan.Buffer* buffer;

    public DeviceMemory* memory;

    public string LayoutName => Name.ToUpper();

    public string VariableName => Name.ToLower();

    public required string Name { get; init; }

    public VulkanTransferType TransferType { get; protected init; }

    public ulong Count { get; set; } = 1;

    public abstract string GetShaderCode(int binding);

    public virtual int TypeSize => throw new NotImplementedException();
}