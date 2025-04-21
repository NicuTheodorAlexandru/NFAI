using NFAI.Internal;
using Silk.NET.Vulkan;

namespace NFAI.Shader;

public class RMSNormShader<TInput, TWeights> : ShaderWrapper 
    where TInput : struct 
    where TWeights : struct
{
    // Constants for work group sizing
    private const uint LOCAL_SIZE_X = 32; // Work group size for normalization dimension

    public readonly ShaderProperty<TInput> inputData;
    public readonly ShaderProperty<TInput> outputData;
    public readonly ShaderProperty<TWeights> gammaData; // Scale parameters
    public readonly ShaderProperty<uint> normDim;
    public readonly ShaderProperty<float> epsilon;
    private readonly uint normalizationDimension;

    public RMSNormShader(
        Vk vk,
        Device device,
        VulkanBufferManager vulkanBufferManager,
        uint inputSize,
        ComputeCollection<TWeights> gamma,
        float epsilon = 1e-5f) 
        : base(vk, device, vulkanBufferManager, $"RMSNorm_{typeof(TInput).Name}_{typeof(TWeights).Name}")
    {
        uint normalizationDimension = (uint)gamma.Shape[0];
        // Create shader properties
        this.inputData = new ShaderProperty<TInput>(vulkanBufferManager, VulkanTransferType.StorageBuffer, inputSize)
        {
            Name = "inputData",
            IsCollection = true
        };
        
        this.outputData = new ShaderProperty<TInput>(vulkanBufferManager, VulkanTransferType.StorageBuffer, inputSize)
        {
            Name = "outputData",
            IsCollection = true
        };
        
        this.gammaData = new ShaderProperty<TWeights>(vulkanBufferManager, VulkanTransferType.StorageBuffer, normalizationDimension)
        {
            Name = "gammaData",
            IsCollection = true
        };
        
        this.normDim = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "normDim",
            IsCollection = false
        };
        
        this.epsilon = new ShaderProperty<float>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "epsilon",
            IsCollection = false
        };

        // Add properties to the shader
        AddProperties(this.inputData, this.outputData, this.gammaData, this.normDim, this.epsilon);
        
        // Set default values
        this.normDim.SetValue([normalizationDimension]);
        this.normalizationDimension = normalizationDimension;
        this.epsilon.SetValue([epsilon]);
        this.gammaData.SetValue(gamma);
    }

    public ShaderProperty<TInput> GetInputProperty()
    {
        return inputData;
    }

    public TInput[] GetOutputs()
    {
        return outputData.GetValue();
    }

    public ShaderProperty<TInput> GetOutputProperty()
    {
        return outputData;
    }

    public override void Compute(AbstractComputeCollection computeCollection)
    {
        if (computeCollection is ComputeCollection<TInput> compute)
        {
            Compute(compute);
        }
        else if (computeCollection is ShaderProperty<TInput> property)
        {
            Compute(property);
        }
        else
        {
            throw new ArgumentException("Invalid input type");
        }
    }

    public void Compute(ComputeCollection<TInput> value)
    {
        inputData.SetValue(value);
        
        uint groupsX = (normalizationDimension + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        
        base.Compute(groupsX, 1, 1);
    }

    public void Compute(ShaderProperty<TInput>? value = null)
    {
        // Connect input shader property to our inputData property
        if (value != null)
        {
            value.TransferTo(inputData);
        }
        
        uint groupsX = (normalizationDimension + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        
        base.Compute(groupsX, 1, 1);
    }

    protected override string GetMainMethodCode()
    {
        return @$"
        layout(local_size_x = {LOCAL_SIZE_X}, local_size_y = 1, local_size_z = 1) in;

        void main()
        {{
            uint idx = gl_GlobalInvocationID.x;
            uint hiddenSize = {normDim.VariableName};

            if (idx >= hiddenSize) return;

            // 1. Calculate mean square of the whole input vector
            float sumSq = 0.0;
            for (uint i = 0; i < hiddenSize; i++)
            {{
                float val = float({inputData.VariableName}[i]);
                sumSq += val * val;
            }}
            float meanSq = sumSq / float(hiddenSize);
            float rms = sqrt(meanSq + {epsilon.VariableName});

            // 2. Normalize and scale by gamma
            float inputVal = float({inputData.VariableName}[idx]);
            float gammaVal = float({gammaData.VariableName}[idx]);
            {outputData.VariableName}[idx] = (inputVal / rms) * gammaVal;
        }}";
    }
}