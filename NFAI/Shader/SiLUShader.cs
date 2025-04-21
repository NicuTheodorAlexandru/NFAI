using NFAI.Internal;
using Silk.NET.Vulkan;

namespace NFAI.Shader;

public class SiLUShader<T> : ShaderWrapper where T : struct
{
    // Constants for work group sizing
    private const uint LOCAL_SIZE_X = 32; // Work group size

    private readonly ShaderProperty<T> inputData;
    private readonly ShaderProperty<T> outputData;
    private readonly ShaderProperty<uint> dataSize;
    private readonly uint elementsCount;

    public SiLUShader(
        Vk vk,
        Device device,
        VulkanBufferManager vulkanBufferManager,
        uint inputSize) 
        : base(vk, device, vulkanBufferManager, $"SiLU_{nameof(T)}_{inputSize}")
    {
        // Create shader properties
        this.inputData = new ShaderProperty<T>(vulkanBufferManager, VulkanTransferType.StorageBuffer, inputSize)
        {
            Name = "inputData",
            IsCollection = true
        };
        
        this.outputData = new ShaderProperty<T>(vulkanBufferManager, VulkanTransferType.StorageBuffer, inputSize)
        {
            Name = "outputData",
            IsCollection = true
        };
        
        this.dataSize = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "dataSize",
            IsCollection = false
        };

        // Add properties to the shader
        AddProperties(this.inputData, this.outputData, this.dataSize);
        elementsCount = inputSize;
        
        // Set default values
        this.dataSize.SetValue([inputSize]);
    }

    public T[] GetOutputs()
    {
        return outputData.GetValue();
    }

    public ShaderProperty<T> GetInputProperty()
    {
        return inputData;
    }

    public ShaderProperty<T> GetOutputProperty()
    {
        return outputData;
    }

    public override void Compute(AbstractComputeCollection computeCollection)
    {
        if (computeCollection is ComputeCollection<T> compute)
        {
            Compute(compute);
        }
        else if (computeCollection is ShaderProperty<T> property)
        {
            Compute(property);
        }
        else
        {
            throw new ArgumentException("Invalid input type");
        }
    }

    public void Compute(ComputeCollection<T> value)
    {
        inputData.SetValue(value);
        
        // Calculate dispatch dimensions based on input size
        uint elementsCount = (uint)value.Length;
        uint groupsX = (elementsCount + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        
        base.Compute(groupsX, 1, 1);
    }

    public void Compute(ShaderProperty<T>? value = null)
    {
        if (value != null)
        {
            value.TransferTo(inputData);
        }
        // Connect input shader property to our input property
        
        // Calculate dispatch dimensions
        uint groupsX = (elementsCount + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        
        base.Compute(groupsX, 1, 1);
    }

    protected override string GetMainMethodCode()
    {
        return @$"
        // Define local work group sizes
        layout(local_size_x = {LOCAL_SIZE_X}, local_size_y = 1, local_size_z = 1) in;
        
        void main()
        {{
            uint idx = gl_GlobalInvocationID.x;
            
            // Skip if we're out of bounds
            if (idx >= {dataSize.VariableName}) {{
                return;
            }}
            
            // Apply SiLU activation: x * sigmoid(x)
            float x = float({inputData.VariableName}[idx]);
            float sigmoid_x = 1.0 / (1.0 + exp(-x));
            float silu = x * sigmoid_x;
            
            {outputData.VariableName}[idx] = silu;
        }}";
    }
}