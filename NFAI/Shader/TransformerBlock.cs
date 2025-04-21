using NFAI.Internal;
using Silk.NET.Vulkan;

namespace NFAI.Shader;

public class TransformerBlock
{
    private readonly RMSNormShader<float, float> attnNormLayer;
    private readonly MatrixMultiplyShader<float, float, float> attnQueryLayer;
    private readonly MatrixMultiplyShader<float, float, float> attnKeysLayer;
    private readonly MatrixMultiplyShader<float, float, float> attnValuesLayer;
    private readonly RoPEShader<float> ropeQueryLayer;
    private readonly RoPEShader<float> ropeKeysLayer;
    private readonly MatrixMultiplyShader<float, float, float> attentionWeightsLayer;
    private readonly RMSNormShader<float, float> ffnNormLayer;
    private readonly MatrixMultiplyShader<float, float, float> ffnDownLayer;
    private readonly MatrixMultiplyShader<float, float, float> ffnGateLayer;
    private readonly ElementWiseMultiplicationShader<float> ffnProjectionLayer;
    private readonly MatrixMultiplyShader<float, float, float> ffnUpLayer;
    private readonly SiLUShader<float> siluLayer;
    private readonly AttentionScoreCalculationShader<float> attentionScoreCalcLayer;
    private readonly AttentionSoftmaxShader<float> attentionSoftmaxLayer;
    private readonly AttentionWeightedValueSumShader<float> attentionWeightedValueSumLayer;

    private uint currentToken = 0;

    private readonly ComputeCollection<float> attnNormCC;
    private readonly ComputeCollection<float> ffnNorm;
    private readonly uint blockIndex;

    public TransformerBlock(Vk vk, Device device, VulkanBufferManager vulkanBufferManager, List<AbstractComputeCollection> tensors, uint headDim, uint queryHeadCount, uint kvHeadCount, uint contextSize, float epsilon, int index, float ropeFrequency, uint ropeDimensions, uint blockIndex)
    {
        var ropeFreq = 500000.0f;
        var ropeFreqs = new float[ropeDimensions / 2];
        for (var i = 0; i < ropeDimensions / 2; i++)
        {
            ropeFreqs[i] = 1.0f / MathF.Pow(ropeFreq, (float)i / ((float)ropeDimensions / 2));
        }

        this.blockIndex = blockIndex;
        attnNormCC = tensors.Where(x => x.Name.Contains($"blk.{index}.attn_norm")).FirstOrDefault() as ComputeCollection<float>;
        attnNormLayer = new RMSNormShader<float, float>
        (vk, device, vulkanBufferManager, (uint)attnNormCC!.Shape[0], attnNormCC!, epsilon);

        var attnQueryCC = tensors.Where(x => x.Name.Contains($"blk.{index}.attn_q")).FirstOrDefault() as ComputeCollection<float>;
        attnQueryLayer = new MatrixMultiplyShader<float, float, float>
        (vk, device, vulkanBufferManager, 1, (uint)attnQueryCC!.Shape[0], (uint)attnQueryCC!.Shape[1], attnQueryCC!);
        attnQueryLayer.GetInputProperty().BindShaderProprty(attnNormLayer.GetOutputProperty());

        var attnKeysCC = tensors.Where(x => x.Name.Contains($"blk.{index}.attn_k")).FirstOrDefault() as ComputeCollection<float>;
        attnKeysLayer = new MatrixMultiplyShader<float, float, float>
        (vk, device, vulkanBufferManager, 1, (uint)attnKeysCC!.Shape[0], (uint)attnKeysCC!.Shape[1], attnKeysCC!, contextSize);
        attnKeysLayer.GetInputProperty().BindShaderProprty(attnNormLayer.GetOutputProperty());

        var attnValuesCC = tensors.Where(x => x.Name.Contains($"blk.{index}.attn_v")).FirstOrDefault() as ComputeCollection<float>;
        attnValuesLayer = new MatrixMultiplyShader<float, float, float>
        (vk, device, vulkanBufferManager, 1, (uint)attnValuesCC!.Shape[0], (uint)attnValuesCC!.Shape[1], attnValuesCC!, contextSize);
        attnValuesLayer.GetInputProperty().BindShaderProprty(attnNormLayer.GetOutputProperty());

        var memoryStream = new MemoryStream();
        foreach (var r in ropeFreqs)
        {
            memoryStream.Write(BitConverter.GetBytes(r), 0, sizeof(float));
        }
        var ropeCC = new ComputeCollection<float>(memoryStream, 32, 0) { Shape = [32], ConstantCount = 0, Name = "rope" };
        ropeQueryLayer = new RoPEShader<float>
        (vk, device, vulkanBufferManager, (uint)attnQueryCC!.Shape[1], (uint)attnQueryCC!.Shape[1], ropeCC!, ropeDimensions, queryHeadCount);
        ropeQueryLayer.GetInputProperty().BindShaderProprty(attnQueryLayer.GetOutputProperty());

        ropeKeysLayer = new RoPEShader<float>
        (vk, device, vulkanBufferManager, (uint)attnKeysCC!.Shape[1] * contextSize, (uint)attnKeysCC!.Shape[1] * contextSize, ropeCC!, ropeDimensions, kvHeadCount, contextSize);
        attnKeysLayer.GetOutputProperty().BindShaderProprty(ropeKeysLayer.GetOutputProperty());
        ropeKeysLayer.GetInputProperty().BindShaderProprty(attnKeysLayer.GetOutputProperty());

        var attnWeights = tensors.Where(x => x.Name.Contains($"blk.{index}.attn_output.weight")).FirstOrDefault() as ComputeCollection<float>;
        attentionWeightsLayer = new MatrixMultiplyShader<float, float, float>
        (vk, device, vulkanBufferManager, 1, (uint)attnWeights!.Shape[0], (uint)attnWeights!.Shape[1], attnWeights!);
        
        ffnNorm = tensors.Where(x => x.Name.Contains($"blk.{index}.ffn_norm")).FirstOrDefault() as ComputeCollection<float>
            ?? throw new InvalidOperationException($"Tensor with name containing 'blk.{index}.ffn_norm' not found.");
        ffnNormLayer = new RMSNormShader<float, float>
        (vk, device, vulkanBufferManager, (uint)ffnNorm!.Shape[0], ffnNorm!, epsilon);
        ffnNormLayer.GetInputProperty().BindShaderProprty(attentionWeightsLayer.GetOutputProperty());

        var ffnDown = tensors.Where(x => x.Name.Contains($"blk.{index}.ffn_down")).FirstOrDefault() as ComputeCollection<float>;
        ffnDownLayer = new MatrixMultiplyShader<float, float, float>
        (vk, device, vulkanBufferManager, 1, (uint)ffnDown!.Shape[0], (uint)ffnDown!.Shape[1], ffnDown!);

        var ffnGate = tensors.Where(x => x.Name.Contains($"blk.{index}.ffn_gate")).FirstOrDefault() as ComputeCollection<float>;
        ffnGateLayer = new MatrixMultiplyShader<float, float, float>
        (vk, device, vulkanBufferManager, 1, (uint)ffnGate!.Shape[0], (uint)ffnGate!.Shape[1], ffnGate!);
        ffnGateLayer.GetInputProperty().BindShaderProprty(ffnNormLayer.GetOutputProperty());

        ffnProjectionLayer = new ElementWiseMultiplicationShader<float>
        (vk, device, vulkanBufferManager, (uint)ffnDown!.Shape[0]);
        ffnDownLayer.GetInputProperty().BindShaderProprty(ffnProjectionLayer.GetOutputProperty()); 

        var ffnUp = tensors.Where(x => x.Name.Contains($"blk.{index}.ffn_up")).FirstOrDefault() as ComputeCollection<float>;
        ffnUpLayer = new MatrixMultiplyShader<float, float, float>
        (vk, device, vulkanBufferManager, 1, (uint)ffnUp!.Shape[0], (uint)ffnUp!.Shape[1], ffnUp!);
        ffnProjectionLayer.GetInputA().BindShaderProprty(ffnUpLayer.GetOutputProperty());
        ffnUpLayer.GetInputProperty().BindShaderProprty(ffnNormLayer.GetOutputProperty());

        siluLayer = new SiLUShader<float>
        (vk, device, vulkanBufferManager, (uint)ffnGate!.Shape[1]);
        ffnProjectionLayer.GetInputB().BindShaderProprty(siluLayer.GetOutputProperty());
        siluLayer.GetInputProperty().BindShaderProprty(ffnGateLayer.GetOutputProperty());

        attentionScoreCalcLayer = new AttentionScoreCalculationShader<float>
        (vk, device, vulkanBufferManager, queryHeadCount, kvHeadCount, contextSize, headDim);
        attentionScoreCalcLayer.GetQueryVectorsProperty().BindShaderProprty(ropeQueryLayer.GetOutputProperty());
        attentionScoreCalcLayer.GetKeyCacheProperty().BindShaderProprty(ropeKeysLayer.GetOutputProperty());

        attentionSoftmaxLayer = new AttentionSoftmaxShader<float>
        (vk, device, vulkanBufferManager, queryHeadCount, contextSize, headDim, epsilon);
        attentionSoftmaxLayer.GetInputProperty().BindShaderProprty(attentionScoreCalcLayer.GetAttentionScoresProperty());

        attentionWeightedValueSumLayer = new AttentionWeightedValueSumShader<float>
        (vk, device, vulkanBufferManager, queryHeadCount, kvHeadCount, contextSize, headDim);
        attentionWeightedValueSumLayer.GetValueCache().BindShaderProprty(attnValuesLayer.GetOutputProperty());
        attentionWeightedValueSumLayer.GetAttentionWeights().BindShaderProprty(attentionSoftmaxLayer.GetAttentionWeightsProperty());

        attentionWeightsLayer.GetInputProperty().BindShaderProprty(attentionWeightedValueSumLayer.GetAttentionOutputProperty());
    }

    public void Compute(ShaderProperty<float>? embed = null)
    {
        attnNormLayer.Compute(embed);

        attnQueryLayer.Compute();

        attnKeysLayer.Compute();

        attnValuesLayer.Compute();

        // Apply RoPE to queries and keys
        ropeQueryLayer.Compute(currentToken);

        // For keys, use the output property from the MatrixMultiplyShader's caching version
        ropeKeysLayer.Compute(currentToken);

        // Get attention output
        attentionScoreCalcLayer.ComputeAttention(this.currentToken + 1);

        attentionSoftmaxLayer.ComputeSoftmax(this.currentToken + 1);

        attentionWeightedValueSumLayer.ComputeWeightedSum(this.currentToken + 1);

        attentionWeightsLayer.Compute();
        var attnWeightsOutput = attentionWeightsLayer.GetOutputs();

        var input = attnNormLayer.GetInputProperty().GetValue();
        var attnFinal = new float[input.Length];
        for (var i = 0; i < attnFinal.Length; i++)
        {
            attnFinal[i] = input[i] + attnWeightsOutput[i];
        }

        var attnWeightsProp = attentionWeightsLayer.GetOutputProperty();
        attnWeightsProp.SetValue(attnFinal);

        ffnNormLayer.Compute();

        ffnUpLayer.Compute();

        ffnGateLayer.Compute();

        siluLayer.Compute();

        ffnProjectionLayer.Compute();

        ffnDownLayer.Compute();
        var ffnDownOutput = ffnDownLayer.GetOutputs();

        var finalResidual = new float[attnFinal.Length];
        for (var i = 0; i < finalResidual.Length; i++)
        {
            finalResidual[i] = attnFinal[i] + ffnDownOutput[i];
        }
        ffnDownLayer.GetOutputProperty().SetValue(finalResidual);

        this.currentToken++;
    }

    private void DebugLog(string layer, IEnumerable<float> output)
    {
        /*
        Console.WriteLine($"{layer} Output");
        var max = output.Max();
        var min = output.Min();
        var avg = output.Average();
        var sum = output.Sum();
        var count = output.Count(x => x != 0);
        Console.WriteLine($"Max: {max}");
        Console.WriteLine($"Min: {min}");
        Console.WriteLine($"Avg: {avg}");
        Console.WriteLine($"Sum: {sum}");
        Console.WriteLine($"Non zero Count: {count}");
        Console.WriteLine($"Output: {string.Join(", ", output.Take(10))}");
        Console.WriteLine("===================================");*/
    }


    public ShaderProperty<float> GetInputProperty()
    {
        return attnNormLayer.inputData;
    }

    public ShaderProperty<float> GetOutputProperty()
    {
        return ffnDownLayer.GetOutputProperty();
    }
}