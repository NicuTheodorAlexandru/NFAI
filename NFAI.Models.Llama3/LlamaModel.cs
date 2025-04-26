using System.Runtime.CompilerServices;
using NFAI.Vulkan;
using NFAI.Vulkan.Shaders;
using Silk.NET.Vulkan;
using Microsoft.Extensions.AI;
using NFAI.Core;

namespace NFAI.Models.Llama3;

public class LlamaModel : IInferenceProvider
{
    private readonly Tokenizer tokenizer;
    private readonly TokenEmbedShader<uint, float, float> embedShader;
    private readonly RMSNormShader<float, float> outputNormLayer;
    private readonly MatrixMultiplyShader<float, float, float> lmHead;
    private readonly List<TransformerBlock> transformerBlocks = [];
    private bool firstInput = true;

    public string ModelName { get; init; }

    public LlamaModel(Vk vk, Device device, VulkanBufferManager vulkanBufferManager, Dictionary<string, object> metadata, List<AbstractComputeCollection> tensors, uint contextSize = 1024u)
    {
        var bosToken = (UInt32)metadata["tokenizer.ggml.bos_token_id"];
        var eosToken = (UInt32)metadata["tokenizer.ggml.eos_token_id"];
        //var chatTemplate = (string)metadata["tokenizer.chat_template"];
        var ropeFrequency = (float)metadata["llama.rope.freq_base"];
        var ropeDimensions = (uint)metadata["llama.rope.dimension_count"];
        var epsilon = (float)(metadata.Where(x => x.Key.Contains("epsilon")).Select(x => x.Value).FirstOrDefault() ?? 0f);
        var queryHeadCount = (uint)metadata["llama.attention.head_count"];
        var kvHeadCount = (uint)metadata["llama.attention.head_count_kv"];
        var keyLength = (uint)metadata["llama.attention.key_length"];
        var valueLength = (uint)metadata["llama.attention.value_length"];
        var transformerBlockCount = (uint)metadata["llama.block_count"];
        ModelName = metadata["general.name"].ToString() ?? "unknown";
        var alignment = 32u;
        if (metadata.TryGetValue("general.alignment", out object? value))
        {
            alignment = (uint)value;
        }

        tokenizer = new Tokenizer(metadata);

        var embedCC = tensors.Where(x => x.Name.Contains("token")).FirstOrDefault() as ComputeCollection<float>;

        embedShader = new TokenEmbedShader<uint, float, float>
        (vk, device, vulkanBufferManager, 1, embedCC!.Shape[0], embedCC!);

        var headDim = keyLength;
        var lastProp = embedShader.GetOutputProperty();
        for (var i = 0; i < transformerBlockCount; i++)
        {
            var block = new TransformerBlock(vk, device, vulkanBufferManager, tensors, headDim, queryHeadCount, kvHeadCount, contextSize, epsilon, i, ropeFrequency, ropeDimensions, (uint)i);
            transformerBlocks.Add(block);
            block.GetInputProperty().BindShaderProprty(lastProp);
            lastProp = block.GetOutputProperty();
        }

        var outputNormCC = tensors.Where(x => x.Name.Contains("output_norm")).FirstOrDefault() as ComputeCollection<float>;
        outputNormLayer = new RMSNormShader<float, float>
        (vk, device, vulkanBufferManager, (uint)outputNormCC!.Shape[0], outputNormCC!, epsilon);
        outputNormLayer.GetInputProperty().BindShaderProprty(lastProp);
        lastProp = outputNormLayer.GetOutputProperty();

        lmHead = new MatrixMultiplyShader<float, float, float>
        (vk, device, vulkanBufferManager, 1, (uint)embedCC!.Shape[0], (uint)embedCC!.Shape[1], null);
        lmHead.GetInputProperty().BindShaderProprty(lastProp);
        lmHead.GetWeightProperty().BindShaderProprty(embedShader.GetWeightProperty());
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        throw new NotImplementedException();
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var messageId = Guid.NewGuid().ToString();
        var userMessage = messages.FirstOrDefault(x => x.Role == ChatRole.User) ?? throw new ArgumentException("No user message found in the input messages.");
        var prompt = userMessage.Text;

        await foreach (var messagePart in RunAsync(prompt, cancellationToken))
        {
            var chatResponseUpdate = new ChatResponseUpdate
            {
                CreatedAt = DateTime.UtcNow,
                ChatThreadId = null,
                AdditionalProperties = null,
                FinishReason = null,
                ModelId = ModelName,
                MessageId = messageId,
                RawRepresentation = null,
                Contents = [new TextContent(messagePart)],
            };
            yield return chatResponseUpdate;
        }
    }

    public async IAsyncEnumerable<string> RunAsync(string prompt, CancellationToken ct = default)
    {
        var tokenIds = tokenizer.Tokenize(prompt, addBos: firstInput);
        firstInput = false;
        foreach (var token in tokenIds)
        {
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);

            writer.Write(token);

            var input = new ComputeCollection<uint>(ms, 1, 0)
            {
                Shape = [(ulong)tokenIds.Count],
                ConstantCount = 0,
                Name = "Input"
            };
            embedShader.Compute(input);

            foreach (var block in transformerBlocks)
            {
                block.Compute();
            }

            outputNormLayer.Compute();

            lmHead.Compute();
        }

        var lmHeadOutput = lmHead.GetOutputs();

        var tk = SamplingUtils.TopP(lmHeadOutput);
        var outputTokens = new List<uint> { tk };
        yield return tokenizer.Detokenize([tk]);
        // initial input was given, start feeding output tokens back in
        while (tk != tokenizer.EosTokenId)
        {
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write(tk);
            var input = new ComputeCollection<uint>(ms, 1, 0)
            {
                Shape = [1],
                ConstantCount = 0,
                Name = "Input"
            };

            // Create a stable ShaderProperty reference for each inference step
            // to prevent the property from being garbage collected between shader operations
            var embedOutput = embedShader.GetOutputProperty();
            embedShader.Compute(input);

            // Use a stable reference to pass through transformer blocks
            var blockOutput = embedOutput;
            foreach (var block in transformerBlocks)
            {
                block.Compute(blockOutput);
                blockOutput = block.GetOutputProperty();
            }

            // Use the stable output property reference for the next steps
            outputNormLayer.Compute(blockOutput);
            var normOutput = outputNormLayer.GetOutputProperty();

            lmHead.Compute(normOutput);

            var tokenId = SamplingUtils.TopP(lmHead.GetOutputs());
            var predictedToken = tokenizer.Detokenize([tokenId]);
            tk = tokenId;
            
            outputTokens.Add(tk);

            if (tk != tokenizer.EosTokenId)
                yield return predictedToken;
        }
    }
}