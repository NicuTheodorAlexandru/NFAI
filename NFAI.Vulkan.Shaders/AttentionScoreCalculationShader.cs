using NFAI.Core;
using Silk.NET.Vulkan;

namespace NFAI.Vulkan.Shaders;

public class AttentionScoreCalculationShader<T> : ShaderWrapper where T : struct
{
    // Constants for work group sizing
    private const uint LOCAL_SIZE_X = 1; // Work group size for sequence length dimension
    private const uint LOCAL_SIZE_Y = 1; // Work group size for query heads dimension

    private readonly ShaderProperty<T> queryVectors;
    private readonly ShaderProperty<T> keyCache;
    private readonly ShaderProperty<T> attentionScores;
    private readonly ShaderProperty<uint> qHeads;
    private readonly ShaderProperty<uint> kvHeads;
    private readonly ShaderProperty<uint> seqLen;
    private readonly ShaderProperty<uint> headDim;
    private readonly ShaderProperty<float> scaleFactor;
    private readonly ShaderProperty<uint> maxContextSize;

    public AttentionScoreCalculationShader(
        Vk vk,
        Device device,
        VulkanBufferManager vulkanBufferManager,
        uint queryHeads,
        uint kvHeads,
        uint maxContextSize,
        uint headDimension) 
        : base(vk, device, vulkanBufferManager, $"AttentionScoreCalculation_{nameof(T)}_{queryHeads}_{kvHeads}_{maxContextSize}_{headDimension}")
    {
        // Calculate buffer sizes
        uint querySize = queryHeads * headDimension;
        uint keyCacheSize = kvHeads * maxContextSize * headDimension;
        uint scoresSize = queryHeads * maxContextSize;
        
        // Create shader properties
        this.maxContextSize = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "maxContextSize",
            IsCollection = false
        };

        this.queryVectors = new ShaderProperty<T>(vulkanBufferManager, VulkanTransferType.StorageBuffer, querySize)
        {
            Name = "queryVectors",
            IsCollection = true
        };
        
        this.keyCache = new ShaderProperty<T>(vulkanBufferManager, VulkanTransferType.StorageBuffer, keyCacheSize)
        {
            Name = "keyCache",
            IsCollection = true
        };
        
        this.attentionScores = new ShaderProperty<T>(vulkanBufferManager, VulkanTransferType.StorageBuffer, scoresSize)
        {
            Name = "attentionScores",
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
        
        this.scaleFactor = new ShaderProperty<float>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "scaleFactor",
            IsCollection = false
        };

        // Add properties to the shader
        AddProperties(
            this.queryVectors, 
            this.keyCache, 
            this.attentionScores, 
            this.qHeads, 
            this.kvHeads, 
            this.seqLen, 
            this.headDim,
            this.scaleFactor,
            this.maxContextSize);
        
        // Set default values
        this.qHeads.SetValue([queryHeads]);
        this.kvHeads.SetValue([kvHeads]);
        this.seqLen.SetValue([0]);
        this.headDim.SetValue([headDimension]);
        this.maxContextSize.SetValue([maxContextSize]);
        
        // Calculate 1/sqrt(headDim) for scaling
        float scale = 1.0f / MathF.Sqrt(headDimension);
        this.scaleFactor.SetValue([scale]);
    }

    public ShaderProperty<T> GetKeyCacheProperty()
    {
        return keyCache;
    }

    public ShaderProperty<T> GetQueryVectorsProperty()
    {
        return queryVectors;
    }

    public T[] GetAttentionScores()
    {
        return attentionScores.GetValue();
    }

    public ShaderProperty<T> GetAttentionScoresProperty()
    {
        return attentionScores;
    }

    public override void Compute(AbstractComputeCollection computeCollection)
    {
        throw new NotSupportedException("Use the ComputeAttention method instead.");
    }

    public void ComputeAttention(uint seqLen, ShaderProperty<T>? queries = null, ShaderProperty<T>? keys = null)
    {
        // Connect input shader properties to our properties

        if (queries != null)
        {
            queries.TransferTo(queryVectors);
        }
        if (keys != null)
        {
            keys.TransferTo(keyCache);
        }
        // Calculate dispatch dimensions
        uint queryHeads = qHeads.GetValue()[0];
        
        uint groupsX = (maxContextSize.GetValue()[0] + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;
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
            uint seqIdx = gl_GlobalInvocationID.x;   // Context position in sequence
            uint qHeadIdx = gl_GlobalInvocationID.y; // Query head index

            // Check bounds
            if (qHeadIdx >= {qHeads.VariableName} || seqIdx >= {maxContextSize.VariableName}) {{
                return;
            }}

            if (seqIdx >= {seqLen.VariableName}) {{
                // If seqIdx is out of bounds, set attention score to -1e9
                {attentionScores.VariableName}[qHeadIdx * {seqLen.VariableName} + seqIdx] = -1e9;
                return;
            }}

            // Calculate the corresponding KV head for this query head
            //uint kvHeadIdx = qHeadIdx % {kvHeads.VariableName};
            uint kvHeadIdx = qHeadIdx / ({qHeads.VariableName} / {kvHeads.VariableName});

            // Initialize partial dot product
            float dotProduct = 0.0;

            // Base index for query vector
            uint queryBaseIdx = qHeadIdx * {headDim.VariableName};

            uint keyBaseIdx = seqIdx * {kvHeads.VariableName} * {headDim.VariableName} + kvHeadIdx * {headDim.VariableName};
            for (uint i = 0; i < {headDim.VariableName}; i++) {{
                float q = float({queryVectors.VariableName}[queryBaseIdx + i]);
                float k = float({keyCache.VariableName}[keyBaseIdx + i]);
                dotProduct += q * k;
            }}

            // Mask future tokens (adjusted for reversed indexing)
            {attentionScores.VariableName}[qHeadIdx * {seqLen.VariableName} + seqIdx] = dotProduct * {scaleFactor.VariableName};
        }}";
    }
}