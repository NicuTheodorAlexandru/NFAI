using NFAI.Internal;
using Silk.NET.Vulkan;

namespace NFAI.Shader;

public class TokenEmbedShader<TInput, TEmbeddings, TOutput> : ShaderWrapper
    where TInput : struct
    where TEmbeddings : struct
    where TOutput : struct
{
    public readonly ShaderProperty<TInput> inputData;
    public readonly ShaderProperty<TEmbeddings> embeddingsData;
    public readonly ShaderProperty<TOutput> outputData;
    public readonly ShaderProperty<uint> outputCount;
    public readonly ShaderProperty<float>? scales;
    public readonly ShaderProperty<uint> vocabSize;

    // Constants for work group sizing
    private const uint LOCAL_SIZE_X = 256; // Work group size for token dimension

    public TokenEmbedShader(
        Vk vk,
        Device device,
        VulkanBufferManager vulkanBufferManager,
        ulong inputSize,
        ulong outputSize,
        ComputeCollection<TEmbeddings> embeddings) : base(vk, device, vulkanBufferManager, "TokenEmbed")
    {
        // actual shader setup
        inputData = new ShaderProperty<TInput>(vulkanBufferManager, VulkanTransferType.StorageBuffer, inputSize)
        {
            Name = "inputData",
            IsCollection = true
        };
        embeddingsData = new ShaderProperty<TEmbeddings>(vulkanBufferManager, VulkanTransferType.StorageBuffer, embeddings.Length)
        {
            Name = "embeddings",
            IsCollection = true
        };
        outputData = new ShaderProperty<TOutput>(vulkanBufferManager, VulkanTransferType.StorageBuffer, outputSize)
        {
            Name = "outputData",
            IsCollection = true
        };
        outputCount = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "outputSize",
            IsCollection = false
        };
        vocabSize = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "vocabSize",
            IsCollection = false
        };

        base.AddProperties(inputData, embeddingsData, outputData, outputCount, vocabSize);

        if (embeddings.ConstantCount > 0)
        {
            scales = new ShaderProperty<float>(vulkanBufferManager, VulkanTransferType.Uniform, embeddings.ConstantCount)
            {
                Name = "scales",
                IsCollection = true
            };
            base.AddProperties(scales);

            scales.SetValue([.. embeddings.GetConstants().ToBlockingEnumerable()]);
        }

        embeddingsData.SetValue(embeddings);
        outputCount.SetValue([(uint)outputSize]);
        vocabSize.SetValue([(uint)embeddings.Length / (uint)outputSize]);
    }

    public ShaderProperty<TEmbeddings> GetWeightProperty()
    {
        return embeddingsData;
    }

    public ShaderProperty<TOutput> GetOutputProperty()
    {
        return outputData;
    }

    public TOutput[] GetOutputs()
    {
        //Thread.Sleep(1000); // Debugging delay
        //var outputSize = this.outputCount.GetValue();
        return outputData.GetValue();
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
        
        // Calculate dispatch dimensions based on input size and embedding dimensions
        uint tokenSize = 1; // Single token ID
        uint embeddingDim = outputCount.GetValue()[0] / tokenSize;  // Calculate embedding dimension
        
        uint groupsX = (embeddingDim + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        
        base.Compute(groupsX, 1, 1);
    }

    public void Compute(ShaderProperty<TInput> value)
    {
        throw new NotImplementedException("Compute with ShaderProperty<TInput> is not implemented yet.");
        // For GPU-to-GPU calls, calculate dimensions from current property values
        uint tokenSize = 1; // Single token ID
        uint embeddingDim = outputCount.GetValue()[0] / tokenSize;  // Calculate embedding dimension
        
        uint groupsX = (embeddingDim + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        
        base.Compute(groupsX, 1, 1);
    }

    protected override string GetMainMethodCode()
    {
        var scaleOp = scales == null ? "" : $" * {scales.VariableName}[0]";
        
        return @$"
        // Define local work group sizes
        layout(local_size_x = {LOCAL_SIZE_X}, local_size_y = 1, local_size_z = 1) in;
        
        void main()
        {{
            uint embeddingDim = gl_GlobalInvocationID.x;
                            
            // Calculate output size per token
            uint dimPerToken = {outputCount.VariableName};
            
            // Check if embedding dimension is within bounds
            if (embeddingDim >= dimPerToken)
                return;
            
            // Get token ID from input 
            uint tokenId = uint({inputData.VariableName}[0]);
                        
            // Calculate embedding index for the token ID and dimension
            uint embeddingIdx = dimPerToken * tokenId + embeddingDim;
            
            // Copy the embedding value to output with optional scaling
            {outputData.VariableName}[embeddingDim] = {embeddingsData.VariableName}[embeddingIdx];
        }}";
    }
}