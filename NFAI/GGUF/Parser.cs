using NFAI.Internal;
using Silk.NET.Vulkan;
using System.Text;

namespace NFAI.GGUF;

public static class Extensions
{
    public static IEnumerable<IEnumerable<T>> Batch<T>(this IEnumerable<T> source, int size)
    {
        T[] bucket = null;
        var count = 0;

        foreach (var item in source)
        {
            if (bucket == null)
                bucket = new T[size];

            bucket[count++] = item;

            if (count != size)
                continue;

            yield return bucket.Select(x => x);

            bucket = null;
            count = 0;
        }

        if (bucket != null && count > 0)
            yield return bucket.Take(count);
    }
}

public class Parser
{
    private readonly Dictionary<string, object> metadata = [];
    private readonly List<string> tensorNames = [];
    private readonly List<(string Name, List<UInt64> Shape, string DataType, UInt64 offset, int constants)> tensorInfo = [];
    public Tokenizer? Tokenizer { get; private set; }
    private uint alignment = 32u; // Default alignment
    private readonly Vk vk;
    private readonly Device device;
    private readonly VulkanBufferManager vulkanBufferManager;
    private readonly Instance instance;
    private readonly PhysicalDevice physicalDevice;

    public Parser()
    {
        // 1) Obtain the Vulkan API interface
        vk = Vk.GetApi();

        // 2) (Pseudo) Create a Vulkan Instance
        //    In real code you must specify correct layers/extensions. 
        instance = VulkanHelper.CreateVulkanInstance(vk);

        // 3) (Pseudo) Select a Physical Device
        physicalDevice = VulkanHelper.PickPhysicalDevice(vk, instance);

        // 4) (Pseudo) Create a Logical Device (with compute queue)
        device = VulkanHelper.CreateLogicalDevice(vk, physicalDevice);
        // 5) Create our buffer manager (for allocations and command pool)
        vulkanBufferManager = new VulkanBufferManager(vk, device, physicalDevice);
    }

    public Model Parse(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"File not found: {path}");
        
        using var binaryReader = new BinaryReader(File.OpenRead(path));
        var tensorCount = ReadHeader(binaryReader);
        ReadMetadata(binaryReader);
        ReadTensors(binaryReader, tensorCount);

        if (!metadata.TryGetValue("general.alignment", out object? value))
            alignment = value is uint uintValue ? uintValue : alignment;

        return new Model(vk, device, vulkanBufferManager, metadata, ReadTensors(binaryReader), 1024u);

        /*using var binaryReader = new BinaryReader(File.OpenRead(path));

        var tensorCount = ReadHeader(binaryReader);
        ReadMetadata(binaryReader);

        var bosToken = (UInt32)metadata["tokenizer.ggml.bos_token_id"];
        var eosToken = (UInt32)metadata["tokenizer.ggml.eos_token_id"];
        //var chatTemplate = (string)metadata["tokenizer.chat_template"];
        var ropeFrequency = (float)metadata["llama.rope.freq_base"];
        var ropeDimensions = (uint)metadata["llama.rope.dimension_count"];
        var epsilon = (float)metadata.Where(x => x.Key.Contains("epsilon")).Select(x => x.Value).FirstOrDefault();
        var queryHeadCount = (uint)metadata["llama.attention.head_count"];
        var kvHeadCount = (uint)metadata["llama.attention.head_count_kv"];
        var keyLength = (uint)metadata["llama.attention.key_length"];
        var valueLength = (uint)metadata["llama.attention.value_length"];
        var contextSize = 1024u;
        var transformerBlockCount = (uint)metadata["llama.block_count"];
        //transformerBlockCount = 1; // For now, only use the first block

        Tokenizer = new Tokenizer(metadata);

        var messages = new List<ChatMessage>
        {
            new("system", "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"),
            new("user", "What is the capital of France?"),
        };

        //var tokenIds = Tokenizer.BuildChatPrompt(messages);
        var tokenIds = Tokenizer.Tokenize("Hello, how are you?");
        SanityCheck.PromptTokenIds(tokenIds);

        Console.WriteLine($"Tokenized: {string.Join(", ", tokenIds)}");

        //tokenIds = [128000, 128006, 9125, 128007, 198, 198, 38766, 1303, 33025, 2696, 25, 6790, 220, 2366, 18, 11450, 2696, 25, 220, 1627, 10263, 220, 2366, 19, 128009, 128006, 882, 128007, 198, 198, 13347, 128009, 128006, 78191, 128007, 198, 198];
        // tokenIds = [
        //     2028, 
        //     374, 
        //     264, 
        //     1296, 
        //     ];
        //tokenIds = [2028, 374, 264];
        // TODO - debug only, remove later
        //tokenIds = tokenIds.Take(10).ToList();
        var detokenized = Tokenizer.Detokenize(tokenIds);
        Console.WriteLine($"Detokenized: {detokenized}");

        if (metadata.TryGetValue("general.alignment", out object? value))
        {
            alignment = (UInt32)value;
        }
        ReadTensors(binaryReader, tensorCount);

        // return the first tensor for now
        var tensors = ReadTensors(binaryReader);

        var embedCC = tensors.Where(x => x.Name.Contains("token")).FirstOrDefault() as ComputeCollection<float>;

        var embedShader = new TokenEmbedShader<uint, float, float>
        (vk, device, vulkanBufferManager, 1, embedCC!.Shape[0], embedCC!);

        // Create Attention shader for combining query, key, and value calculations
        // The head dimension is derived from keyLength (per key/value pair)
        var headDim = keyLength;
        // AttentionShader takes headDim, qHeads, kvHeads, and contextLength as parameters

        var transformerBlocks = new List<TransformerBlock>();
        for (var i = 0; i < transformerBlockCount; i++)
        {
           transformerBlocks.Add(new TransformerBlock(vk, device, vulkanBufferManager, tensors, headDim, queryHeadCount, kvHeadCount, contextSize, epsilon, i, ropeFrequency, ropeDimensions, (uint)i));
        }

        var outputNormCC = tensors.Where(x => x.Name.Contains("output_norm")).FirstOrDefault() as ComputeCollection<float>;
        var outputNormLayer = new RMSNormShader<float, float>
        (vk, device, vulkanBufferManager, (uint)outputNormCC!.Shape[0], outputNormCC!, epsilon);

        //var lmHead = new MatrixMultiplyShader<float, float, float>
        //(vk, device, vulkanBufferManager, 1, (uint)embedCC!.Shape[0], (uint)embedCC!.Shape[1], embedCC!);

        var lmHead = new MatrixMultiplyShader<float, float, float>
        (vk, device, vulkanBufferManager, 1, (uint)embedCC!.Shape[0], (uint)embedCC!.Shape[1], embedCC);

        uint lastToken = 0u;
        var idx = 0u;
        foreach (var token in tokenIds)
        {
            //ms.Write(BitConverter.GetBytes(token));
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
            //Console.WriteLine("Embed Output data: " + string.Join(", ", output));

            var property = embedShader.GetOutputProperty();
            var embedOutput = property.GetValue();
            //SanityCheck.Embedding(embedOutput, idx);

            var blockIndex = 0;
            foreach (var block in transformerBlocks)
            {
                block.Compute(property);
                property = block.GetOutputProperty();
                // compare block output with python test
                //var expectedBlockOutput = Helper.LoadBlockOutput((uint)blockIndex, token);
                var blockOutput = property.GetValue();
                blockIndex++;
            }

            outputNormLayer.Compute(property);
            var outputNormOutput = outputNormLayer.GetOutputs();

            lmHead.Compute(outputNormLayer.GetOutputProperty());
            var lmHeadOutput = lmHead.GetOutputs();


            var tk = TopP(lmHeadOutput);
            lastToken = tk;
            Console.WriteLine($"Prompt input in progress: Token: {tk}, Detokenized: {Tokenizer.Detokenize([tk])}");
            idx++;
        }

        var outputTokens = new List<uint> { lastToken };
        // initial input was given, start feeding output tokens back in
        while (lastToken != Tokenizer.EosTokenId)
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
            var output = embedOutput.GetValue();

            // Use a stable reference to pass through transformer blocks
            var blockOutput = embedOutput;
            foreach (var block in transformerBlocks)
            {
                block.Compute(blockOutput);
                blockOutput = block.GetOutputProperty();
                //Console.WriteLine($"Block sample: {string.Join(',', blockOutput.GetValue().Take(10))}");
            }

            // Use the stable output property reference for the next steps
            outputNormLayer.Compute(blockOutput);
            var normOutput = outputNormLayer.GetOutputProperty();

            lmHead.Compute(normOutput);

            var tokenId = TopP(lmHead.GetOutputs());
            //var tokenId = ArgMax(lmHead.GetOutputs());
            var predictedToken = Tokenizer.Detokenize([tokenId]);
            lastToken = tokenId;

            //SanityCheck.Logits(lmHead.GetOutputs(), idx);
            
            outputTokens.Add(lastToken);
            Console.WriteLine($"Predicted output: {Tokenizer.Detokenize(outputTokens)}");
            Console.WriteLine($"Predicted token: {predictedToken}, Token id: {tokenId}");
            //Console.WriteLine($"Test: {string.Join(',', output.Take(20))}");
        }

        Console.Read();*/
    }

    private List<AbstractComputeCollection> ReadTensors(BinaryReader reader)
    {
        var data = new List<AbstractComputeCollection>();
        var tensorStart = AlignOffset((ulong)reader.BaseStream.Position);
        foreach (var info in tensorInfo)
        {
            var length = info.Shape.Aggregate((a, b) => a * b);

            AbstractComputeCollection cc;
            var constants = (ulong)info.constants;

            var offset = (long)AlignOffset(info.offset + tensorStart);
            reader.BaseStream.Seek(offset, SeekOrigin.Begin);

            if (info.DataType == "float32")
            {
                cc = new ComputeCollection<float>(reader.BaseStream, length, offset) {
                    Shape = [.. info.Shape],
                    ConstantCount = constants,
                    Name = info.Name
                };
            }
            else if (info.DataType == "int32")
            {
                cc = new ComputeCollection<int>(reader.BaseStream, length, offset) {
                    Shape = [.. info.Shape],
                    ConstantCount = constants,
                    Name = info.Name
                };
            }
            else if (info.DataType == "int64")
            {
                cc = new ComputeCollection<long>(reader.BaseStream, length, offset) {
                    Shape = [.. info.Shape],
                    ConstantCount = constants,
                    Name = info.Name
                };
            }
            else if (info.DataType == "float64")
            {
                cc = new ComputeCollection<double>(reader.BaseStream, length, offset) {
                    Shape = [.. info.Shape],
                    ConstantCount = constants,
                    Name = info.Name
                };
            }
            else if (info.DataType == "Q8_0")
            {
                cc = new ComputeCollection<sbyte>(reader.BaseStream, length, offset) {
                    Shape = [.. info.Shape],
                    ConstantCount = constants,
                    Name = info.Name
                };
            }
            else if (info.DataType == "float16")
            {
                var c = new ComputeCollection<Half>(reader.BaseStream, length, offset) {
                    Shape = [.. info.Shape],
                    ConstantCount = constants,
                    Name = info.Name
                };

                cc = ComputeCollection<float>.Cast(c);
            }
            else
            {
                throw new Exception("Unsupported data type");
            }

            Console.WriteLine($"Loaded Tensor: {info.Name}, Shape: [{string.Join(", ", info.Shape)}]");

            data.Add(cc);
            //return data;
        }

        return data;
    }

    private UInt64 AlignOffset(UInt64 offset)
    {
        return offset + (alignment - (offset % alignment)) % alignment;
    }

    private ulong ReadHeader(BinaryReader reader)
    {
        // GGUF Magic Identifier
        byte[] magic = reader.ReadBytes(4);
        if (Encoding.ASCII.GetString(magic) != "GGUF")
            throw new Exception("Invalid GGUF file format");

        // Read version and tensor count
        var version = reader.ReadUInt32();
        var tensorCount = reader.ReadUInt64();

        Console.WriteLine($"GGUF Version: {version}, Tensor Count: {tensorCount}");
        return tensorCount;
    }

    private void ReadMetadata(BinaryReader reader)
    {
        var metadataCount = reader.ReadUInt64();
        for (ulong i = 0; i < metadataCount; i++)
        {
            var key = ParseString(reader);

            var valueType = reader.ReadUInt32(); // Read value type
            object value;

            switch (valueType)
            {
                case 0: value = reader.ReadByte(); break;  // uint8
                case 1: value = reader.ReadSByte(); break; // int8
                case 2: value = reader.ReadUInt16(); break; // uint16
                case 3: value = reader.ReadInt16(); break; // int16
                case 4: value = reader.ReadUInt32(); break; // uint32
                case 5: value = reader.ReadInt32(); break; // int32
                case 6: value = reader.ReadSingle(); break; // float32
                case 7: value = reader.ReadByte() != 0; break; // bool (1-byte where 0 = false, 1 = true)
                case 8: // String
                    value = ParseString(reader);
                    break;
                case 9: value = ReadArray(reader); break; // Array
                case 10: value = reader.ReadUInt64(); break; // uint64
                case 11: value = reader.ReadInt64(); break; // int64
                case 12: value = reader.ReadDouble(); break; // float64
                default: throw new Exception("Unsupported metadata value type");
            }

            metadata[key] = value;
            Console.WriteLine($"Metadata: {key} = {value}");
        }
    }

    private object ReadData(BinaryReader reader, UInt32 type)
    {
        return type switch
        {
            0 => reader.ReadByte(),
            1 => reader.ReadSByte(),
            2 => reader.ReadUInt16(),
            3 => reader.ReadInt16(),
            4 => reader.ReadUInt32(),
            5 => reader.ReadInt32(),
            6 => reader.ReadSingle(),
            7 => reader.ReadByte() != 0,
            // String
            8 => ParseString(reader),
            9 => ReadArray(reader),
            10 => reader.ReadUInt64(),
            11 => reader.ReadInt64(),
            12 => reader.ReadDouble(),
            _ => throw new Exception("Unsupported metadata value type"),
        };
    }

    private object ReadArray(BinaryReader reader)
    {
        var arrayType = reader.ReadUInt32(); // Type of elements inside the array
        ulong length = reader.ReadUInt64(); // Number of elements
        List<object> array = [];

        for (ulong i = 0; i < length; i++)
        {
            //var key = ParseString(reader);

            object value;

            switch (arrayType)
            {
                case 0: value = reader.ReadByte(); break;  // uint8
                case 1: value = reader.ReadSByte(); break; // int8
                case 2: value = reader.ReadUInt16(); break; // uint16
                case 3: value = reader.ReadInt16(); break; // int16
                case 4: value = reader.ReadUInt32(); break; // uint32
                case 5: value = reader.ReadInt32(); break; // int32
                case 6: value = reader.ReadSingle(); break; // float32
                case 7: value = reader.ReadByte() != 0; break; // bool (1-byte where 0 = false, 1 = true)
                case 8: // String
                    value = ParseString(reader);
                    break;
                case 9: value = ReadArray(reader); break; // Array
                case 10: value = reader.ReadUInt64(); break; // uint64
                case 11: value = reader.ReadInt64(); break; // int64
                case 12: value = reader.ReadDouble(); break; // float64
                default: throw new Exception("Unsupported metadata value type");
            }
            array.Add(value);
        }
        return array;
    }

    private string ParseString(BinaryReader reader)
    {
        var size = reader.ReadUInt64();
        return Encoding.UTF8.GetString(reader.ReadBytes((int)size));
    }

    private void ReadTensors(BinaryReader reader, UInt64 tensorCount)
    {
        for (uint i = 0; i < tensorCount; i++)
        {
            var name = ParseString(reader);
            tensorNames.Add(name);

            // Read tensor shape
            var numberOfDimensions = reader.ReadUInt32();
            //var shapeDimCount = reader.ReadUInt32();
            List<UInt64> shape = [];
            for (int j = 0; j < numberOfDimensions; j++)
            {
                shape.Add(reader.ReadUInt64());
            }

            // Read tensor data type
            uint dataType = reader.ReadUInt32();
            string dataTypeStr = dataType switch
            {
                0 => "float32",
                1 => "float16",
                2 => "Q4_0",
                3 => "Q4_1",
                6 => "Q5_0",
                7 => "Q5_1",
                8 => "Q8_0",
                9 => "Q8_1",
                10 => "Q2_K",
                11 => "Q3_K",
                12 => "Q4_K",
                13 => "Q5_K",
                14 => "Q6_K",
                15 => "Q8_K",
                16 => "IQ2_XXS",
                17 => "IQ2_XS",
                18 => "IQ3_XXS",
                19 => "IQ1_S",
                20 => "IQ4_NL",
                21 => "IQ3_S",
                22 => "IQ2_S",
                23 => "IQ4_XS",
                24 => "int8",
                25 => "int16",
                26 => "int32",
                27 => "int64",
                28 => "float64",
                29 => "IQ1_M",
                _ => "Unknown"
            };

            var constants = dataType switch
            {
                8 => 1,
                _ => 0
            };


            var offset = reader.ReadUInt64();

            tensorInfo.Add((name, shape, dataTypeStr, offset, constants));
            Console.WriteLine($"Tensor: {name}, Shape: [{string.Join(", ", shape)}], Type: {dataTypeStr}, Offset: {offset}");
        }
    }

    public List<string> GetTensorNames() => tensorNames;
    public List<(string Name, List<UInt64> Shape, string DataType, UInt64 offset, int constants)> GetTensorInfo() => tensorInfo;
}
