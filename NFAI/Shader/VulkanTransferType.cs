using System.ComponentModel.DataAnnotations;
using Silk.NET.Vulkan;

namespace NFAI.Shader;

public enum VulkanTransferType
{
    [Display(Name = "uniform")]
    PushConstant = BufferUsageFlags.None,
    [Display(Name = "uniform")]
    Uniform = BufferUsageFlags.UniformBufferBit,
    [Display(Name = "buffer")]
    StorageBuffer = BufferUsageFlags.StorageBufferBit,
    Texture = BufferUsageFlags.None,
    SpecializationConstant = BufferUsageFlags.None
}