using NFAI.Core;
using Silk.NET.Vulkan;

namespace NFAI.Vulkan.Shaders;

public class RoPEShader<T> : ShaderWrapper where T : struct
{
    // Constants for work group sizing
    private const uint LOCAL_SIZE_X = 8; // Work group size for sequence dimension
    private const uint LOCAL_SIZE_Y = 4; // Work group size for embedding dimension
    private const uint LOCAL_SIZE_Z = 32; // Work group size for embedding dimension

    public readonly ShaderProperty<T> inputData;
    public readonly ShaderProperty<T> outputData;
    public readonly ShaderProperty<uint> inputSize;
    public readonly ShaderProperty<uint> outputSize;
    public readonly ShaderProperty<float> baseFreq;
    public readonly ShaderProperty<uint> ropeDimensions;
    public readonly ShaderProperty<uint> currentTokenIndex;
    public readonly ShaderProperty<uint> offsetBase;
    public readonly ShaderProperty<uint> headDim;
    public readonly ShaderProperty<uint> maxContextSize;
    public readonly ShaderProperty<uint> noHeads;

    public RoPEShader(Vk vk, Device device, VulkanBufferManager vulkanBufferManager, uint inputSize, uint outputSize, ComputeCollection<float> baseFreq, uint ropeDimensions, uint numHeads, uint maxCacheSize = 1) 
        : base(vk, device, vulkanBufferManager, $"RoPE_{typeof(T).Name}_{inputSize}_{outputSize}_{ropeDimensions}_{numHeads}_{maxCacheSize}")
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
        
        this.inputSize = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "inputSize",
            IsCollection = false
        };
        
        this.outputSize = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "outputSize",
            IsCollection = false
        };
                
        this.baseFreq = new ShaderProperty<float>(vulkanBufferManager, VulkanTransferType.StorageBuffer, ropeDimensions / 2)
        {
            Name = "baseFreq",
            IsCollection = true
        };

        this.ropeDimensions = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "ropeDimensions",
            IsCollection = false
        };

        this.currentTokenIndex = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "currentTokenIndex",
            IsCollection = false
        };

        this.offsetBase = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "offsetBase",
            IsCollection = false
        };

        this.headDim = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "headDim",
            IsCollection = false
        };

        this.maxContextSize = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "maxContextSize",
            IsCollection = false
        };

        this.noHeads = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "noHeads",
            IsCollection = false
        };

        // Add properties to the shader
        AddProperties(
            this.inputData,
            this.outputData,
            this.inputSize,
            this.outputSize,
            this.baseFreq,
            this.ropeDimensions,
            this.currentTokenIndex,
            this.offsetBase,
            this.headDim,
            this.maxContextSize,
            this.noHeads);
        
        // Set default values
        this.inputSize.SetValue([inputSize]);
        this.outputSize.SetValue([inputSize]);
        this.baseFreq.SetValue(baseFreq);
        this.ropeDimensions.SetValue([ropeDimensions]);
        this.currentTokenIndex.SetValue([0]);
        this.noHeads.SetValue([numHeads]);

        var offsetBaseValue = maxCacheSize == 1 ? 0 : inputSize / maxCacheSize;
        this.offsetBase.SetValue([offsetBaseValue]);
        this.maxContextSize.SetValue([maxCacheSize]);

        uint headDim;
        if (offsetBaseValue > 0)
        {
            headDim = offsetBaseValue / numHeads;
        }
        else
        {
            headDim = inputSize / numHeads;
        }
        this.headDim.SetValue([headDim]);
    }

    public ShaderProperty<T> GetInputProperty()
    {
        return inputData;
    }

    public T[] GetOutputs()
    {
        return outputData.GetValue();
    }

    public void SetCurrentTokenIndex(uint index)
    {
        currentTokenIndex.SetValue([index]);
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
            throw new ArgumentException("Invalid input type");
        }
        else
        {
            throw new ArgumentException("Invalid input type");
        }
    }

    public void Compute(ComputeCollection<T> value, uint position)
    {
        inputData.SetValue(value);
        
        // Calculate dispatch dimensions based on sequence length and embedding dimensions
        uint seqLen = inputSize.GetValue()[0];
        uint noHeads = this.noHeads.GetValue()[0];
        if (maxContextSize.GetValue()[0] == 1)
        {
            seqLen = 1;
        }
        
        uint dispatchX = (seqLen  + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        uint dispatchY = (noHeads + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y;

        currentTokenIndex.SetValue([position]);        
        base.Compute(dispatchX, dispatchY, 1);
    }

    public void Compute(uint position, ShaderProperty<T>? value = null)
    {
        // Connect input shader property to our inputData property
        if (value != null)
        {
            value.TransferTo(inputData);
        }
        
        // Calculate dispatch dimensions
        uint seqLen = inputSize.GetValue()[0];
        uint noHeads = this.noHeads.GetValue()[0];
        uint headDim = this.headDim.GetValue()[0];
        if (maxContextSize.GetValue()[0] == 1)
        {
            seqLen = 1;
        }
        
        uint dispatchX = (seqLen  + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        uint dispatchY = (noHeads + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y;
        uint dispatchZ = (headDim / 2 + LOCAL_SIZE_Z - 1) / LOCAL_SIZE_Z;
        
        currentTokenIndex.SetValue([position]);

        base.Compute(dispatchX, dispatchY, dispatchZ);
    }

    public (uint x, uint y, uint z) CalculateDispatchSize()
    {
        uint seqLen = inputSize.GetValue()[0];
        uint noHeads = this.noHeads.GetValue()[0];
        uint headDim = this.headDim.GetValue()[0];
        if (maxContextSize.GetValue()[0] == 1)
        {
            seqLen = 1;
        }
        
        uint dispatchX = (seqLen  + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        uint dispatchY = (noHeads + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y;
        uint dispatchZ = (headDim / 2 + LOCAL_SIZE_Z - 1) / LOCAL_SIZE_Z;

        return (dispatchX, dispatchY, dispatchZ);
    }

    protected override string GetMainMethodCode()
    {
        // TODO - modify this so it actually applies rope for each dimension size slice of the input
        return @$"
        // Define local work group sizes
        layout(local_size_x = {LOCAL_SIZE_X}, local_size_y = {LOCAL_SIZE_Y}, local_size_z = {LOCAL_SIZE_Z}) in;
        
        void main()
        {{
            uint seqIdx = gl_GlobalInvocationID.x;   // Context position in sequence
            uint headIdx = gl_GlobalInvocationID.y;  // Head index
            uint pairIdx = gl_GlobalInvocationID.z * 2;  // Pair index

            if (seqIdx > {currentTokenIndex.VariableName})
            {{
                return;
            }}

            uint index1 = seqIdx * {headDim.VariableName} * {noHeads.VariableName} + headIdx * {headDim.VariableName} + pairIdx;
            uint index2 = index1 + 1;

            if (pairIdx < {ropeDimensions.VariableName} && ({maxContextSize.VariableName} == 1 || seqIdx == {currentTokenIndex.VariableName}))
            {{
                float theta = {baseFreq.VariableName}[pairIdx / 2] * float({currentTokenIndex.VariableName});
                float cosTheta = cos(theta);
                float sinTheta = sin(theta);

                float value1 = cosTheta * {inputData.VariableName}[index1] - sinTheta * {inputData.VariableName}[index2];
                float value2 = sinTheta * {inputData.VariableName}[index1] + cosTheta * {inputData.VariableName}[index2];

                {outputData.VariableName}[index1] = value1;
                {outputData.VariableName}[index2] = value2;
            }}
            else
            {{

                // Copy the input data to the output data
                {outputData.VariableName}[index1] = {inputData.VariableName}[index1];
                {outputData.VariableName}[index2] = {inputData.VariableName}[index2];
            }}
        }}";
    }
}