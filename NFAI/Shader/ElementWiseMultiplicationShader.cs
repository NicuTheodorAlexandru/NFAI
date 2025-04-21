using NFAI.Internal;
using Silk.NET.Vulkan;

namespace NFAI.Shader;

public class ElementWiseMultiplicationShader<T> : ShaderWrapper where T : struct
{
    // Constants for work group sizing
    private const uint LOCAL_SIZE_X = 32; // Work group size

    private readonly ShaderProperty<T> inputDataA;
    private readonly ShaderProperty<T> inputDataB;
    private readonly ShaderProperty<T> outputData;
    private readonly ShaderProperty<uint> dataSize;
    private readonly uint elementsCount;

    public ElementWiseMultiplicationShader(
        Vk vk,
        Device device,
        VulkanBufferManager vulkanBufferManager,
        uint inputSize) 
        : base(vk, device, vulkanBufferManager, $"ElementWiseMultiplication_{nameof(T)}_{inputSize}")
    {
        // Create shader properties
        this.inputDataA = new ShaderProperty<T>(vulkanBufferManager, VulkanTransferType.StorageBuffer, inputSize)
        {
            Name = "inputDataA",
            IsCollection = true
        };
        
        this.inputDataB = new ShaderProperty<T>(vulkanBufferManager, VulkanTransferType.StorageBuffer, inputSize)
        {
            Name = "inputDataB",
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
        AddProperties(this.inputDataA, this.inputDataB, this.outputData, this.dataSize);
        elementsCount = inputSize;
        
        // Set default values
        this.dataSize.SetValue([inputSize]);
    }

    public ShaderProperty<T> GetInputB()
    {
        return inputDataB;
    }

    public ShaderProperty<T> GetInputA()
    {
        return inputDataA;
    }

    public T[] GetOutputs()
    {
        return outputData.GetValue();
    }

    public ShaderProperty<T> GetOutputProperty()
    {
        return outputData;
    }

    public override void Compute(AbstractComputeCollection computeCollection)
    {
        throw new ArgumentException("Element-wise multiplication requires two inputs. Use the Compute(ComputeCollection<T>, ComputeCollection<T>) method instead.");
    }

    public void Compute(ComputeCollection<T> valueA, ComputeCollection<T> valueB)
    {
        if (valueA.Length != valueB.Length)
        {
            throw new ArgumentException("Input collections must have the same length");
        }

        inputDataA.SetValue(valueA);
        inputDataB.SetValue(valueB);
        
        // Calculate dispatch dimensions based on input size
        uint elementsCount = (uint)valueA.Length;
        uint groupsX = (elementsCount + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        
        base.Compute(groupsX, 1, 1);
    }

    public void Compute(ShaderProperty<T>? valueA = null, ShaderProperty<T>? valueB = null)
    {
        if (valueA != null)
        {
            valueA.TransferTo(inputDataA);
        }
        if (valueB != null)
        {
            valueB.TransferTo(inputDataB);
        }

        if (valueA != null && valueB != null && valueA.Count != valueB.Count)
        {
            throw new ArgumentException("Input properties must have the same size");
        }
        
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
            
            // Perform element-wise multiplication
            {outputData.VariableName}[idx] = {inputDataA.VariableName}[idx] * {inputDataB.VariableName}[idx];
        }}";
    }
}