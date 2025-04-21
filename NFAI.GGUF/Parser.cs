using System.Text;
using NFAI.Vulkan;
using Silk.NET.Vulkan;
using Microsoft.Extensions.AI;
using NFAI.Core;
using NFAI.Models.Llama3;

namespace NFAI.GGUF;

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

    public IChatClient Parse(string path)
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
