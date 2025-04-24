using System.Diagnostics;
using System.Runtime.InteropServices;

namespace NFAI.Core;

public abstract class AbstractComputeCollection
{
    protected Stream DataStream { get; private init; }
    public virtual ushort TypeSize { get; set; }
    public required virtual ulong[] Shape { get; init; } = [];
    public ulong Length { get; private init; }
    public required ulong ConstantCount { get; init; }
    public long offset { get; private init; }
    public required string Name { get; init; }

    public AbstractComputeCollection(Stream dataStream, ulong length, long offset)
    {
        Length = length;
        //dataStream.CopyTo(newStream, (int)Length + (int)ConstantCount * 4);
        this.offset = offset;
        DataStream = dataStream;
    }

    public async IAsyncEnumerable<float> GetConstants()
    {
        DataStream.Seek(offset + (long)Length * (long)TypeSize, SeekOrigin.Begin);
        //DataStream.Seek(offset, SeekOrigin.Begin);
        var buffer = new byte[4];
        for (var i = 0ul; i < ConstantCount; i++)
        {
            await DataStream.ReadExactlyAsync(buffer.AsMemory(0, 4));
            yield return BitConverter.ToSingle(buffer, 0);
        }
    }

    protected async IAsyncEnumerable<byte[]> GetDataRaw()
    {
        DataStream.Seek(offset, SeekOrigin.Begin);
        const int batchSizeInBytes = 1024 * 1024 * 10; // 100MB batch size, adjust as needed
        int itemsPerBatch = batchSizeInBytes;
        if (itemsPerBatch < 1) itemsPerBatch = 1;
        
        var current = 0ul;
        var total = Length * TypeSize;
        // Read the entire batch
        //Memory<byte> batchMemory  = new Memory<byte>(batchBuffer);
        while (current < total)
        {
            // Calculate remaining items and adjust batch size if needed
            var itemsToRead = Math.Min(itemsPerBatch, (int)(total - current));
            var batchBuffer = new byte[itemsToRead];
            
            try
            {
                DataStream.ReadExactly(batchBuffer, 0, itemsToRead);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
            current += (ulong)itemsToRead;
            if (TypeSize != 2)
            {
                yield return [..batchBuffer];
            }
            else
            {
                var newBatchBuffer = new byte[itemsToRead * 2];
                var shorts = MemoryMarshal.Cast<byte, Half>(batchBuffer).ToArray();
                for (var j = 0; j < itemsToRead / 2; j++)
                {
                    var f = (float)shorts[j];
                    var bytes = BitConverter.GetBytes(f);
                    Array.Copy(bytes, 0, newBatchBuffer, j * 4, bytes.Length); // Copy float bytes to newBatchBuffer
                }
                yield return [..newBatchBuffer];
            }
        }
    }

    public static float HalfToSingle(byte byteLow, byte byteHigh)
    {
        ushort half = (ushort)((byteHigh << 8) | byteLow);

        int sign = (half >> 15) & 0x00000001;
        int exponent = (half >> 10) & 0x0000001F;
        int mantissa = half & 0x000003FF;

        int fSign = sign << 31;
        int fExponent, fMantissa;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                // Zero
                fExponent = 0;
                fMantissa = 0;
            }
            else
            {
                // Denormalized number — normalize it
                int shift = 0;
                while ((mantissa & 0x00000400) == 0)
                {
                    mantissa <<= 1;
                    shift++;
                }

                mantissa &= 0x000003FF;
                exponent = 1 - shift;
                fExponent = (exponent + 127 - 15) << 23;
                fMantissa = mantissa << 13;
            }
        }
        else if (exponent == 0x1F)
        {
            // Inf or NaN
            fExponent = 0xFF << 23;
            fMantissa = mantissa << 13;
        }
        else
        {
            // Normalized number
            fExponent = (exponent + 127 - 15) << 23;
            fMantissa = mantissa << 13;
        }

        int floatBits = fSign | fExponent | fMantissa;
        return BitConverter.Int32BitsToSingle(floatBits);
    }

    protected IEnumerable<T> GetData<T>() where T : struct
    {
        DataStream.Seek(offset, SeekOrigin.Begin);
        const int batchSizeInBytes = 1024 * 1024 * 100; // 4KB batch size, adjust as needed
        int itemsPerBatch = batchSizeInBytes / TypeSize;
        if (itemsPerBatch < 1) itemsPerBatch = 1;
        
        var current = 0ul;
        var total = Length;
        var batchBuffer = new byte[itemsPerBatch * TypeSize];
        // Read the entire batch
        //Memory<byte> batchMemory  = new Memory<byte>(batchBuffer);
        while (current < total)
        {
            if (total > 1000000 && current != 0)
            {
                Console.WriteLine($"Progress: {current / (float)total}");
            }
                
            // Calculate remaining items and adjust batch size if needed
            var itemsToRead = Math.Min(itemsPerBatch, (int)(total - current));
            //var bytesToRead = itemsToRead * TypeSize;
            
            try
            {
                DataStream.ReadExactly(batchBuffer, 0, itemsToRead * TypeSize);
            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }

            var sw = Stopwatch.StartNew();
            T[] outputs;

            if (TypeSize == 2 && typeof(T) == typeof(float))
            {
                var shorts = MemoryMarshal.Cast<byte, Half>(batchBuffer).ToArray();
                outputs = new T[itemsToRead];
                for (var j = 0; j < itemsToRead; j++)
                {
                    outputs[j] = (T)(object)(float)shorts[j];
                }
            }
            else
            {
                outputs = MemoryMarshal.Cast<byte, T>(batchBuffer).ToArray();
            }

            for (var j = 0; j < itemsToRead; j++)
            {
                yield return outputs[j];
            }

            sw.Stop();
            Console.WriteLine($"Batch time: {sw.ElapsedMilliseconds}ms");
            current += (ulong)itemsToRead;
        }
    }
}
