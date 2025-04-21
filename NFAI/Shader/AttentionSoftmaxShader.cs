using NFAI.Internal;
using Silk.NET.Vulkan;

namespace NFAI.Shader;

public class AttentionSoftmaxShader<T> : ShaderWrapper where T : struct
{
    // Constants for work group sizing
    private const uint LOCAL_SIZE_X = 1; // Work group size for sequence length dimension

    private readonly ShaderProperty<T> attentionScores;
    private readonly ShaderProperty<T> attentionWeights;
    private readonly ShaderProperty<uint> qHeads;
    private readonly ShaderProperty<uint> seqLen;
    private readonly ShaderProperty<float> softmaxScale;
    private readonly ShaderProperty<float> epsilon;
    private readonly ShaderProperty<uint> maxContextSize;
    
    public AttentionSoftmaxShader(
        Vk vk,
        Device device,
        VulkanBufferManager vulkanBufferManager,
        uint queryHeads,
        uint maxContextSize,
        uint headDimension,
        float epsilon = 1.0e-5f) 
        : base(vk, device, vulkanBufferManager, $"AttentionSoftmax_{nameof(T)}_{queryHeads}_{maxContextSize}_{headDimension}_{epsilon}")
    {
        // Calculate buffer sizes
        uint maxScoresSize = queryHeads * maxContextSize;

        // Create shader properties
        this.epsilon = new ShaderProperty<float>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "epsilon",
            IsCollection = false
        };

        this.attentionScores = new ShaderProperty<T>(vulkanBufferManager, VulkanTransferType.StorageBuffer, maxScoresSize)
        {
            Name = "attentionScores",
            IsCollection = true
        };
        
        this.attentionWeights = new ShaderProperty<T>(vulkanBufferManager, VulkanTransferType.StorageBuffer, maxScoresSize)
        {
            Name = "attentionWeights",
            IsCollection = true
        };
        
        this.qHeads = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "qHeads",
            IsCollection = false
        };
        
        this.seqLen = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "seqLen",
            IsCollection = false
        };

        this.softmaxScale = new ShaderProperty<float>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "softmaxScale",
            IsCollection = false
        };

        this.maxContextSize = new ShaderProperty<uint>(vulkanBufferManager, VulkanTransferType.Uniform)
        {
            Name = "maxContextSize",
            IsCollection = false
        };

        // Add properties to the shader
        AddProperties(
            this.attentionScores, 
            this.attentionWeights, 
            this.qHeads, 
            this.seqLen,
            this.softmaxScale,
            this.epsilon,
            this.maxContextSize);
        
        // Set default values
        this.qHeads.SetValue([queryHeads]);
        this.softmaxScale.SetValue([1f / MathF.Sqrt(headDimension)]); // Default scale
        this.epsilon.SetValue([epsilon]);
        this.maxContextSize.SetValue([maxContextSize]);
    }

    public ShaderProperty<T> GetInputProperty()
    {
        return attentionScores;
    }

    public void SetSeqLen(uint seqLen)
    {
        this.seqLen.SetValue([seqLen]);
    }

    public T[] GetAttentionWeights()
    {
        return attentionWeights.GetValue();
    }

    public ShaderProperty<T> GetAttentionWeightsProperty()
    {
        return attentionWeights;
    }

    public override void Compute(AbstractComputeCollection computeCollection)
    {
        throw new NotSupportedException("Use the ComputeSoftmax method instead.");
    }

    public void ComputeSoftmax(uint seqLen, ShaderProperty<T>? scores = null)
    {
        // Connect input shader property to our property
        if (scores != null)
        {
            scores.TransferTo(attentionScores);
        }
        
        // Calculate dispatch dimensions
        uint queryHeads = qHeads.GetValue()[0];
        uint groupsX = (queryHeads + LOCAL_SIZE_X - 1) / LOCAL_SIZE_X;

        this.seqLen.SetValue([seqLen]);
        
        base.Compute(groupsX, 1, 1);
    }

    public void SetSoftmaxScale(float scale)
    {
        softmaxScale.SetValue([scale]);
    }

    protected override string GetMainMethodCode()
    {
        return @$"
        // Define local work group sizes
        layout(local_size_x = {LOCAL_SIZE_X}, local_size_y = 1, local_size_z = 1) in;

        void main()
        {{
            uint qHeadIdx = gl_GlobalInvocationID.x;

            // Bounds check
            if (qHeadIdx >= {qHeads.VariableName}) {{
                return;
            }}

            // 1. Find max score for numerical stability
            float maxScore = -1.0e38;
            for (uint seqIdx = 0; seqIdx < {seqLen.VariableName}; seqIdx++) {{
                float score = float({attentionScores.VariableName}[qHeadIdx * {seqLen.VariableName} + seqIdx]);
                maxScore = max(maxScore, score);
            }}

            // 2. Apply exp(score - maxScore) with scaling, and compute sum of exp
            float sumExp = 0.0;
            for (uint seqIdx = 0; seqIdx < {seqLen.VariableName}; seqIdx++) {{
                float score = float({attentionScores.VariableName}[qHeadIdx * {seqLen.VariableName} + seqIdx]);
                float scaled = clamp((score - maxScore), -80.0, 80.0);
                float expValue = exp(scaled);
                {attentionWeights.VariableName}[qHeadIdx * {seqLen.VariableName} + seqIdx] = expValue;
                sumExp += expValue;
            }}

            // 3. Normalize
            float invSum = sumExp > {epsilon.VariableName} ? 1.0 / sumExp : 0.0;
            for (uint seqIdx = 0; seqIdx < {seqLen.VariableName}; seqIdx++) {{
                float expValue = float({attentionWeights.VariableName}[qHeadIdx * {seqLen.VariableName} + seqIdx]);
                {attentionWeights.VariableName}[qHeadIdx * {seqLen.VariableName} + seqIdx] = expValue * invSum;
            }}
        }}";
    }
}