using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace NFAI;

public static class Helper
{
    private static readonly SHA256 sha256 = SHA256.Create();
    private static readonly string debugFilePath = @"E:\AiHome\GGUF test";
    
    public static (float[] newData, int[] newShape) TransposeFlattenedArray(float[] data, int[] shape, int dim0, int dim1)
    {
        int rank = shape.Length;
        if (dim0 < 0) dim0 += rank;
        if (dim1 < 0) dim1 += rank;
        
        if (dim0 == dim1)
            return (data, shape); // No-op

        // Compute old strides
        int[] oldStrides = ComputeStrides(shape);

        // Swap dimensions
        int[] newShape = (int[])shape.Clone();
        (newShape[dim0], newShape[dim1]) = (newShape[dim1], newShape[dim0]);

        // Compute new strides for target shape
        int[] newStrides = ComputeStrides(newShape);
        int totalSize = data.Length;
        float[] newData = new float[totalSize];

        // Iterate over all indices of the new shape
        int[] index = new int[rank];
        for (int i = 0; i < totalSize; i++)
        {
            // Convert flat index to multi-dimensional index
            int remaining = i;
            for (int d = 0; d < rank; d++)
            {
                index[d] = remaining / newStrides[d];
                remaining %= newStrides[d];
            }

            // Swap the two dimensions back to get the source index
            (index[dim0], index[dim1]) = (index[dim1], index[dim0]);

            // Compute flat index in original data
            int sourceIndex = 0;
            for (int d = 0; d < rank; d++)
            {
                sourceIndex += index[d] * oldStrides[d];
            }

            newData[i] = data[sourceIndex];
        }

        return (newData, newShape);
    }

    public static int[] ComputeStrides(int[] shape)
    {
        int rank = shape.Length;
        int[] strides = new int[rank];
        int stride = 1;

        for (int i = rank - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }

        return strides;
    }

    public static List<float> LoadTokenOutput(uint tokenId)
    {
        var filePath = Path.Combine(debugFilePath, $"token_{tokenId}_final_output.txt");
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"File not found: {filePath}");
        }
        return LoadFloats(filePath);
    }

    public static List<float> LoadBlockOutput(uint blockId, uint tokenId)
    {
        var filePath = Path.Combine(debugFilePath, $"block_{blockId}_token_{tokenId}_embedding.txt");
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"File not found: {filePath}");
        }
        return LoadFloats(filePath);
    }

    public static List<float> LoadEmbeddingDebug(uint tokenId)
    {
        var filePath = Path.Combine(debugFilePath, $"token_{tokenId}_embedding.txt");
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"File not found: {filePath}");
        }
        return LoadFloats(filePath);
    }

    private static List<float> LoadFloats(string filePath)
    {
        var lines = File.ReadAllLines(filePath);
        var embedding = new List<float>();
        foreach (var line in lines)
        {
            if (float.TryParse(line, out var value))
            {
                embedding.Add(value);
            }
            else
            {
                throw new FormatException($"Invalid float value in file: {line}");
            }
        }
        return embedding;
    }

    public static void Hash<T>(T[] input) where T : struct
    {
        var bytes = MemoryMarshal.AsBytes(input.AsSpan());
        byte[] hash = sha256.ComputeHash(bytes.ToArray());
        Console.WriteLine(BitConverter.ToString(hash).Replace("-", ""));
    }
}