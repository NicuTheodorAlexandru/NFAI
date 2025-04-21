using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using NFAI.GGUF;
using NFAI.Shader;
using Silk.NET.Vulkan;

namespace NFAI.Internal;

public class Model : IChatClient
{
    private readonly Tokenizer tokenizer;
    private readonly TokenEmbedShader<uint, float, float> embedShader;
    private readonly RMSNormShader<float, float> outputNormLayer;
    private readonly MatrixMultiplyShader<float, float, float> lmHead;
    private readonly List<TransformerBlock> transformerBlocks = [];
    private bool firstInput = true;
    private readonly string modelName;

    public Model(Vk vk, Device device, VulkanBufferManager vulkanBufferManager, Dictionary<string, object> metadata, List<AbstractComputeCollection> tensors, uint contextSize = 1024u)
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
        modelName = metadata["general.name"].ToString() ?? "unknown";
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

    public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        var res = await GetStreamingResponseAsync(messages, options, cancellationToken).ToChatResponseAsync(cancellationToken);
        return res;
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        throw new NotImplementedException();
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var messageId = Guid.NewGuid().ToString();
        // get only first user message
        /*var userMessage = messages.FirstOrDefault(x => x.Role == ChatRole.User) ?? throw new ArgumentException("No user message found in the input messages.");
        var prompt = userMessage.Text;
        var response = Run(prompt).ToList();
        var chatMessage = new ChatMessage
        {
            Role = ChatRole.Assistant,
            AdditionalProperties = null,
            Contents = [.. response.Select(x => new TextContent(x)).Cast<AIContent>()],
            MessageId = null,
            RawRepresentation = null,
        };
        var chatResponse = new ChatResponse
        {
            CreatedAt = DateTime.UtcNow,
            ChatThreadId = null,
            AdditionalProperties = null,
            FinishReason = ChatFinishReason.Stop,
            ModelId = modelName,
            Usage = null,
            RawRepresentation = null,
        };*/
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
                ModelId = modelName,
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
        uint lastToken = 0u;
        var idx = 0u;
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
            var outputNormOutput = outputNormLayer.GetOutputs();

            lmHead.Compute();
            var lmHeadOutput = lmHead.GetOutputs();

            var tk = SamplingUtils.TopP(lmHeadOutput);
            lastToken = tk;
            idx++;
        }

        var outputTokens = new List<uint> { lastToken };
        yield return tokenizer.Detokenize([lastToken]);
        // initial input was given, start feeding output tokens back in
        while (lastToken != tokenizer.EosTokenId)
        {
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            writer.Write(lastToken);
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
            lastToken = tokenId;
            
            outputTokens.Add(lastToken);

            if (lastToken != tokenizer.EosTokenId)
                yield return predictedToken;
        }
    }
}