using NFAI.Core;
using Silk.NET.Vulkan;

namespace NFAI.Vulkan.Shaders;

public class AttentionWeightedValueSumShader<T> : ShaderWrapper where T : struct
{
    // Constants for work group sizing
    private const uint LOCAL_SIZE_X = 1; // Work group size for head dimension
    private const uint LOCAL_SIZE_Y = 1; // Work group size for query heads

    private readonly ShaderProperty<T> attentionWeights;
    private readonly ShaderProperty<T> valueCache;
    private readonly ShaderProperty<T> attentionOutput;
    private readonly ShaderProperty<uint> qHeads;
    private readonly ShaderProperty<uint> kvHeads;
    private readonly ShaderProperty<uint> seqLen;
    private readonly ShaderProperty<uint> headDim;
    private readonly ShaderProperty<uint> maxContextSize;

    public AttentionWeightedValueSumShader(
        Vk vk,
        Device device,
        VulkanBufferManager vulkanBufferManager,
        uint queryHeads,
        uint kvHeads,
        uint maxContextSize,
        uint headDimension) 
        : base(vk, device, vulkanBufferManager, $"AttentionWeightedValueSum_{nameof(T)}_{queryHeads}_{kvHeads}_{maxContextSize}_{headDimension}")
    {
        // Calculate buffer sizes
        uint weightsSize = queryHeads * maxContextSize;
        uint valueCacheSize = kvHeads * maxContextSize * headDimension;
        uint outputSize = queryHeads * headDimension;
        
        // Create shader properties
        this.attentionWeights = new ShaderProperty<T>(vulkanBufferManager, VulkanTransferType.StorageBuffer, weightsSize)
        {
            Name = "attentionWeights",
            IsCollection = true
        };
        
        this.valueCache = new ShaderProperty<T>(vulkanBufferManager, VulkanTransferType.StorageBuffer, valueCacheSize)
        {
            Name = "valueCache",
            IsCollection = true
        };
        
        this.attentionOutput = new ShaderProperty<T>(vulkanBufferManager, VulkanTransferType.StorageBuffer, outputSize)
        {
            Name = "attentionOutput",
            IsCollection = true
        };
        
        this.qHeads = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "qHeads",
            IsCollection = false
        };
        
        this.kvHeads = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "kvHeads",
            IsCollection = false
        };
        
        this.seqLen = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "seqLen",
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

        // Add properties to the shader
        AddProperties(
            this.attentionWeights, 
            this.valueCache, 
            this.attentionOutput, 
            this.qHeads, 
            this.kvHeads, 
            this.seqLen,
            this.headDim,
            this.maxContextSize);
        
        // Set default values
        this.qHeads.SetValue([queryHeads]);
        this.kvHeads.SetValue([kvHeads]);
        this.headDim.SetValue([headDimension]);
        this.maxContextSize.SetValue([maxContextSize]);
    }

    public ShaderProperty<T> GetAttentionWeights()
    {
        return attentionWeights;
    }

    public ShaderProperty<T> GetValueCache()
    {
        return valueCache;
    }

    public T[] GetAttentionOutput()
    {
        return attentionOutput.GetValue();
    }

    public ShaderProperty<T> GetAttentionOutputProperty()
    {
        return attentionOutput;
    }

    public override void Compute(AbstractComputeCollection computeCollection)
    {
        throw new NotSupportedException("Use the ComputeWeightedSum method instead.");
    }

    public void SetSeqLen(uint seqLen)
    {
        this.seqLen.SetValue([seqLen]);
    }

    public void ComputeWeightedSum(ComputeCollection<T> weights, ComputeCollection<T> values, uint seqLen)
    {
        // Connect input shader properties to our properties
        attentionWeights.SetValue(weights);
        valueCache.SetValue(values);
        
        // Calculate dispatch dimensions
        uint queryHeads = qHeads.GetValue()[0];
        uint headDimension = headDim.GetValue()[0];
        
        uint groupsX = (headDimension + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        uint groupsY = (queryHeads + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y;

        this.seqLen.SetValue([seqLen]);
        
        base.Compute(groupsX, groupsY, 1);
    }

    public void ComputeWeightedSum(uint seqLen, ShaderProperty<T>? weights = null, ShaderProperty<T>? values = null)
    {
        // Connect input shader properties to our properties
        if (weights != null)
        {
            weights.TransferTo(attentionWeights);
        }
        if (values != null)
        {
            values.TransferTo(valueCache);
        }
        
        // Calculate dispatch dimensions
        uint queryHeads = qHeads.GetValue()[0];
        uint headDimension = headDim.GetValue()[0];
        
        uint groupsX = (headDimension + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
        uint groupsY = (queryHeads + LOCAL_SIZE_Y - 1) / LOCAL_SIZE_Y;

        this.seqLen.SetValue([seqLen]);
        
        base.Compute(groupsX, groupsY, 1);
    }

    protected override string GetMainMethodCode()
    {
        return @$"
        // Define local work group sizes
        layout(local_size_x = {LOCAL_SIZE_X}, local_size_y = {LOCAL_SIZE_Y}, local_size_z = 1) in;
                
        void main()
        {{
            uint dimIdx = gl_GlobalInvocationID.x;    // Head dimension index
            uint qHeadIdx = gl_GlobalInvocationID.y;  // Query head index
                        
            // Check bounds
            if (dimIdx >= {headDim.VariableName} || qHeadIdx >= {qHeads.VariableName}) {{
                return;
            }}
            
            // Calculate the corresponding KV head for this query head
            //uint kvHeadIdx = qHeadIdx % {kvHeads.VariableName};
            uint kvHeadIdx = qHeadIdx / ({qHeads.VariableName} / {kvHeads.VariableName});

            // Initialize accumulated value
            float accum = 0.0;
            
            // Compute weighted sum across the sequence
            for (int seqIdxx = 0; seqIdxx < int({seqLen.VariableName}); seqIdxx++) {{
                uint seqIdx = uint(seqIdxx);
                // Get attention weight
                float weight = float({attentionWeights.VariableName}[qHeadIdx * {seqLen.VariableName} + seqIdx]);                
                // Get value from value cache
                // Base index for value vector (note: youngest token at index 0)
                uint valueBaseIdx = seqIdx * {kvHeads.VariableName} * {headDim.VariableName} + kvHeadIdx * {headDim.VariableName};
                float value = float({valueCache.VariableName}[valueBaseIdx + dimIdx]);
                
                // Accumulate weighted value
                accum += weight * value;
            }}
            
            // Write the result to attention output
            uint outputIdx = qHeadIdx * {headDim.VariableName} + dimIdx;
            {attentionOutput.VariableName}[outputIdx] = accum;
        }}";
    }
}