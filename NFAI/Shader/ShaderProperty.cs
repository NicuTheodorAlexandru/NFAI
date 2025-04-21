using System.Runtime.InteropServices;
using System.Text;
using NFAI.Internal;
using Silk.NET.Vulkan;

namespace NFAI.Shader;

public class ShaderProperty<T> : BaseShaderProperty
    where T : struct
{
    private readonly VulkanBufferManager vulkanBufferManager;
    public bool IsHostVisible { get; private set; }
    private T constantValue;
    public required bool IsCollection { get; init; }
    public unsafe override int TypeSize => sizeof(T);
    private GCHandle bufferHandle;
    private GCHandle memoryHandle;
    private GCHandle descriptorBufferInfoHandle;

    public void TransferTo(ShaderProperty<T> target, int? start = null)
    {
        if (target.Count > Count)
        {
            target.SetValue(GetValue(), 0, (int)Count);
        }
        else
        {
            target.SetValue([.. GetValue().Take((int)target.Count)], start ?? 0, (int)Count);
        }
    }

    public ShaderProperty(VulkanBufferManager vulkanBufferManager, VulkanTransferType transferType, ulong count = 1, bool hostVisible = false)
    {
        base.TransferType = transferType;
        this.vulkanBufferManager = vulkanBufferManager;
        IsHostVisible = hostVisible;
        this.Count = count;

        if ((BufferUsageFlags)transferType != BufferUsageFlags.None)
        {
            SetupBuffer();
        }
    }

    private unsafe void SetupBuffer()
    {
        var flags = IsHostVisible
            ? MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit
            : MemoryPropertyFlags.DeviceLocalBit;
        Silk.NET.Vulkan.Buffer buffer;
        DeviceMemory memory;
        try
        {
            vulkanBufferManager.CreateBuffer<T>
            (
                Count,
                (BufferUsageFlags)TransferType | BufferUsageFlags.TransferDstBit | BufferUsageFlags.TransferSrcBit,
                flags,
                out buffer,
                out memory
            );
        }
        catch (Exception ex)
        {
            vulkanBufferManager.CreateBuffer<T>
            (
                Count,
                (BufferUsageFlags)TransferType | BufferUsageFlags.TransferDstBit | BufferUsageFlags.TransferSrcBit,
                MemoryPropertyFlags.HostVisibleBit | MemoryPropertyFlags.HostCoherentBit,
                out buffer,
                out memory
            );
            this.IsHostVisible = true;
        }

        bufferHandle = GCHandle.Alloc(buffer, GCHandleType.Pinned);

        base.buffer = (Silk.NET.Vulkan.Buffer*)bufferHandle.AddrOfPinnedObject();

        memoryHandle = GCHandle.Alloc(memory, GCHandleType.Pinned);
        base.memory = (DeviceMemory*)memoryHandle.AddrOfPinnedObject();

        var descInfo = new DescriptorBufferInfo
        {
            Buffer = buffer,
            Offset = 0,
            Range = Count * (ulong)TypeSize
        };

        descriptorBufferInfoHandle = GCHandle.Alloc(descInfo, GCHandleType.Pinned);

        base.descriptorBufferInfo = (DescriptorBufferInfo*)descriptorBufferInfoHandle.AddrOfPinnedObject();
    }

    unsafe public void BindShaderProprty(ShaderProperty<T> shaderProperty)
    {
        vulkanBufferManager.DestoryBuffer(ref *buffer, ref *memory);

        bufferHandle.Free();
        memoryHandle.Free();
        descriptorBufferInfo->Buffer = *shaderProperty.buffer;
        descriptorBufferInfo->Offset = shaderProperty.descriptorBufferInfo->Offset;
        descriptorBufferInfo->Range = shaderProperty.descriptorBufferInfo->Range;
        buffer = shaderProperty.buffer;
        memory = shaderProperty.memory;

        Count = shaderProperty.Count;
    }

    unsafe public void SetValue(T[] value, int start, int count)
    {
        // For buffer backed properties, upload the data to the GPU using the VulkanBufferManager.
        if ((BufferUsageFlags)TransferType != BufferUsageFlags.None)
        {
            if (IsHostVisible)
            {
                vulkanBufferManager.UploadConstants(ref *memory, value, start, count);
            }
            else
            {
                vulkanBufferManager.UploadDeviceConstants(ref *buffer, value, (ulong)start, (ulong)count);
            }
        }
        else
        {
            constantValue = value[0];
        }
    }

    unsafe public void SetValue(T[] value)
    {
        // For buffer backed properties, upload the data to the GPU using the VulkanBufferManager.
        if ((BufferUsageFlags)TransferType != BufferUsageFlags.None)
        {
            SetValue(value, 0, value.Length);
        }
        else
        {
            constantValue = value[0];
        }
    }

    unsafe public void SetValue(ComputeCollection<T> value)
    {
        // For buffer backed properties, upload the data to the GPU using the VulkanBufferManager.
        if ((BufferUsageFlags)TransferType != BufferUsageFlags.None)
        {
            if (IsHostVisible)
            {
                vulkanBufferManager.UploadData(ref *memory, value);
            }
            else
            {
                vulkanBufferManager.UploadDataToDeviceLocal(*buffer, value);
            }
        }
        else
        {
            // Simply store the value locally.
            constantValue = value.GetData().FirstOrDefault();
        }        
    }

    unsafe public T[] GetValue()
    {
        // For buffer backed properties, read the data from the GPU using the VulkanBufferManager.
        if ((BufferUsageFlags)TransferType != BufferUsageFlags.None)
        {
            if (IsHostVisible)
            {
                return vulkanBufferManager.ReadBufferData<T>(ref *memory, Count);
            }
            else
            {
                return vulkanBufferManager.ReadDeviceBufferData<T>(ref *buffer, Count);
            }
        }
        else
        {
            return [constantValue];
        }
    }

    public override string GetShaderCode(int binding)
    {
        if ((BufferUsageFlags)TransferType != BufferUsageFlags.None)
        {
            return GetBufferShaderCode(binding);
        }
        else
        {
            return GetConstantShaderCode();
        }
    }

    private string GetBufferShaderCode(int binding)
    {
        var sb = new StringBuilder();
        var collectionSuffix = IsCollection ? "[]" : "";
        var memoryAlignment = TransferType == VulkanTransferType.Uniform ? "std140" : "std430";
        // Example usage:
        sb.AppendLine($@"
        layout(set = 0, binding = {binding}) {TransferType.GetDisplayName()} {LayoutName}
        {{
            {ShaderProperty<T>.ResolveType()} {VariableName}{collectionSuffix};
        }};");
        return sb.ToString();
    }

    private string GetConstantShaderCode()
    {
        return $"{ShaderProperty<T>.ResolveType()} {VariableName};";
    }

    private static string ResolveType()
    {
        return ShaderProperty<T>.ResolveTypeFor(typeof(T));
    }

    private static string ResolveTypeFor(Type type)
    {
        // If the type is an array, resolve the element type recursively.
        if (type.IsArray)
        {
            var elementType = type.GetElementType();
            return $"{ShaderProperty<T>.ResolveTypeFor(elementType!)}";
        }

        // If the type is a generic type that implements IEnumerable<T>, treat it as an array.
        if (type.IsGenericType)
        {
            // For example, List<float> or IReadOnlyList<float>
            var ienumInterface = type.GetInterfaces()
                .FirstOrDefault(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IEnumerable<>));
            if (ienumInterface != null)
            {
                var elementType = ienumInterface.GetGenericArguments().FirstOrDefault();
                return $"{ShaderProperty<T>.ResolveTypeFor(elementType!)}";
            }
        }

        // Otherwise, use the switch based on type name.
        return type.Name switch
        {
            "Int32"    => "int",
            "UInt32"   => "uint",
            "Int16"    => "int",
            "UInt16"   => "uint",
            "Byte"     => "int8_t", // GLSL doesn't have an 8-bit int type, so adjust as needed.
            "SByte"    => "uint8_t",
            "Single"   => "float",
            "Double"   => "double", // Note: double precision requires GLSL extensions or a specific profile.
            "Boolean"  => "bool",
            "Vector2"  => "vec2",
            "Vector3"  => "vec3",
            "Vector4"  => "vec4",
            "Matrix3x3"=> "mat3",
            "Matrix4x4"=> "mat4",
            "Half" => "float16_t",
            _ => throw new NotSupportedException($"Unsupported type: {type.Name}")
        };
    }
}