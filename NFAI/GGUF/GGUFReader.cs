using System.Text;

namespace NFAI.GGUF;

public class GGUFReader
{
    public static ulong ReadHeader(BinaryReader reader)
    {
        byte[] magic = reader.ReadBytes(4);
        if (Encoding.ASCII.GetString(magic) != "GGUF")
            throw new Exception("Invalid GGUF file format");
        var version = reader.ReadUInt32();
        var tensorCount = reader.ReadUInt64();
        Console.WriteLine($"GGUF Version: {version}, Tensor Count: {tensorCount}");
        return tensorCount;
    }

    public static void ReadMetadata(BinaryReader reader, Dictionary<string, object> metadata)
    {
        var metadataCount = reader.ReadUInt64();
        for (ulong i = 0; i < metadataCount; i++)
        {
            var key = ParseString(reader);
            var valueType = reader.ReadUInt32();
            object value = ReadData(reader, valueType);
            metadata[key] = value;
            Console.WriteLine($"Metadata: {key} = {value}");
        }
    }

    public static object ReadData(BinaryReader reader, UInt32 type)
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
            8 => ParseString(reader),
            9 => ReadArray(reader),
            10 => reader.ReadUInt64(),
            11 => reader.ReadInt64(),
            12 => reader.ReadDouble(),
            _ => throw new Exception("Unsupported metadata value type"),
        };
    }

    public static object ReadArray(BinaryReader reader)
    {
        var arrayType = reader.ReadUInt32();
        ulong length = reader.ReadUInt64();
        List<object> array = [];
        for (ulong i = 0; i < length; i++)
        {
            object value = ReadData(reader, arrayType);
            array.Add(value);
        }
        return array;
    }

    public static string ParseString(BinaryReader reader)
    {
        var size = reader.ReadUInt64();
        return Encoding.UTF8.GetString(reader.ReadBytes((int)size));
    }
}
